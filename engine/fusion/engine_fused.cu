/* Fused GPU Transformer — all operations on single stream, 1 CPU sync/token
 * Build: hipcc -O3 -o engine_fused engine_fused.cu -lm -lhipblas
 * Run: LD_LIBRARY_PATH=/opt/rocm/lib ./engine_fused -m model.q4nx -n 32 -p "Hello"
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define MAX_CTX 4096
#define I8_ROW_B 5120
#define DEF_MODEL "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };

static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

/* ── GPU kernels ── */
__global__ void rms_norm(float *x, const float *w, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    __shared__ float s[256];
    float v = x[i]; s[threadIdx.x] = isfinite(v) ? v*v : 0;
    __syncthreads();
    for (int s2 = 128; s2 > 0; s2 >>= 1) { if (threadIdx.x < s2) s[threadIdx.x] += s[threadIdx.x+s2]; __syncthreads(); }
    if (threadIdx.x == 0) s[0] = rsqrtf(s[0]/n + 1e-6f);
    __syncthreads();
    x[i] *= s[0] * w[i];
}

__global__ void qk_norm_rope(float *q, float *k, const float *qn, const float *kn,
                              const float *rs, const float *rc, int pos, int nq, int nkv) {
    extern __shared__ float ss[];
    int h = blockIdx.x, d = threadIdx.x, is_k = (h >= nq);
    float *vec = is_k ? (k + (h-nq)*HD) : (q + h*HD);
    const float *nw = is_k ? kn : qn;
    float v = vec[d]; ss[d] = v*v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (d < s) ss[d] += ss[d+s]; __syncthreads(); }
    float ir = rsqrtf(ss[0]/HD + 1e-6f);
    vec[d] *= ir * (nw ? nw[d] : 1.0f);
    // RoPE
    int hd2 = HD/2;
    if (d < hd2) {
        float a = vec[d], b = vec[d+hd2];
        vec[d] = a*rc[pos*HD+d] - b*rs[pos*HD+d];
        vec[d+hd2] = b*rc[pos*HD+d] + a*rs[pos*HD+d];
    }
}

__global__ void attn_kernel(const float *q, const float *kc, const float *vc,
                             float *out, int seq_len) {
    int h = blockIdx.x, d = threadIdx.x, kvh = h/GQA;
    if (h >= NH || d >= HD) return;
    __shared__ float scores[4096];
    for (int p = 0; p < seq_len; p++) {
        if (d == 0) scores[p] = 0;
        __syncthreads();
        atomicAdd(&scores[p], q[h*HD+d] * kc[p*NKV*HD + kvh*HD + d]);
        __syncthreads();
    }
    if (d == 0) {
        float mx = -1e30f;
        for (int p = 0; p < seq_len; p++) if (scores[p] > mx) mx = scores[p];
        float sum = 0;
        for (int p = 0; p < seq_len; p++) { float e = expf(scores[p]-mx); scores[p]=e; sum+=e; }
        float iv = sum > 0 ? 1.0f/sum : 0;
        for (int p = 0; p < seq_len; p++) scores[p] *= iv;
    }
    __syncthreads();
    float acc = 0;
    for (int p = 0; p < seq_len; p++) acc += scores[p] * vc[p*NKV*HD + kvh*HD + d];
    out[h*HD + d] = acc;
}

__global__ void silu_gate(float *g, float *u, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    u[i] = (gv/(1.0f+expf(-gv))) * u[i];
}

__global__ void add_residual(float *h, const float *r, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] += r[i];
}

__global__ void lm_head_gpu(const float *h, const float *emb, float *lg) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= NV) return;
    double dot = 0;
    const float *row = emb + (size_t)i * H;
    for (int j = 0; j < H; j++) dot += (double)h[j] * (double)row[j];
    lg[i] = (float)dot;
}

/* ── Device weight + buffer management ── */
typedef struct { float *q,*k,*v,*o,*g,*u,*d; } DevW;
typedef struct {
    hipblasHandle_t blas; hipStream_t stream;
    float *dh, *dres, *dqkv, *dat, *dgt, *dact, *dlg;
    float *dkv_k, *dkv_v;
    DevW *l; bool ok;
} GPU;

typedef struct {
    float *emb, *fn, *in[NC], *pa[NC], *qn[NC], *kn[NC], *rs, *rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w; GPU gpu;
} Model;

static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc) {
    int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc;
    *or_=ntr*32;*oc=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc),4);
    for(int ir=0;ir<i8r;ir++){const uint8_t *rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;
        const uint16_t *sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);const uint8_t *pk=rd+1024;
        for(int lr=0;lr<32;lr++){int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;const uint8_t *ld=pk+lane*(256*8);
            for(int c=0;c<256;c++){int g=c/32;float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);
                uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
                o[(tr*32+lr)*(*oc)+(tc*256+c)]=(float)cd*s+z;}}}
    return o;
}
static bool up_w(Model *m, uint64_t off, int i8r, int id, float **dw) {
    int or_,oc;float *c=deq_i8(m->map+m->ds+off,i8r,id,&or_,&oc);
    if(!c)return false;hipMalloc(dw,(size_t)or_*oc*4);
    hipMemcpy(*dw,c,(size_t)or_*oc*4,hipMemcpyHostToDevice);free(c);return true;
}
static bool gpu_init(Model *m) {
    GPU *g=&m->gpu; g->ok=false;
    if(hipblasCreate(&g->blas)!=HIPBLAS_STATUS_SUCCESS)return false;
    hipStreamCreate(&g->stream); hipblasSetStream(g->blas, g->stream);
    size_t kv_sz = (size_t)NC*MAX_CTX*NKV*HD;
    hipMalloc(&g->dh, H*4); hipMalloc(&g->dres, H*4);
    hipMalloc(&g->dqkv, (size_t)QT*4); hipMalloc(&g->dat, (size_t)NH*HD*4);
    hipMalloc(&g->dgt, (size_t)IM*4); hipMalloc(&g->dact, (size_t)IM*4);
    hipMalloc(&g->dlg, (size_t)NV*4);
    hipMalloc(&g->dkv_k, kv_sz*4); hipMalloc(&g->dkv_v, kv_sz*4);
    hipMemset(g->dkv_k,0,kv_sz*4); hipMemset(g->dkv_v,0,kv_sz*4);
    g->l=(DevW*)calloc(NC,sizeof(DevW));
    for(int l=0;l<NC;l++){
        if(!up_w(m,m->qo[l],m->qi[l],H,&g->l[l].q))return false;
        if(!up_w(m,m->ko[l],m->ki[l],H,&g->l[l].k))return false;
        if(!up_w(m,m->vo[l],m->vi[l],H,&g->l[l].v))return false;
        if(!up_w(m,m->oo[l],m->oi[l],NH*HD,&g->l[l].o))return false;
        if(!up_w(m,m->go[l],m->gi[l],H,&g->l[l].g))return false;
        if(!up_w(m,m->uo[l],m->ui[l],H,&g->l[l].u))return false;
        if(!up_w(m,m->do_[l],m->di[l],IM,&g->l[l].d))return false;
    }
    g->ok=true; return true;
}

/* ── Fused forward: all 28 layers on GPU stream, 1 CPU sync ── */
static int fused_fwd(Model *m, uint32_t tok, int *pos, uint32_t *out, int oidx) {
    GPU *g = &m->gpu; if (!g->ok) return 0;
    hipStream_t s = g->stream;
    hipblasHandle_t h = g->blas;
    hipMemcpyAsync(g->dh, m->emb+(size_t)tok*H, H*4, hipMemcpyHostToDevice, s);

    for (int l = 0; l < NC; l++) {
        float a=1.0f, b=0.0f;
        // Save residual
        hipMemcpyAsync(g->dres, g->dh, H*4, hipMemcpyDeviceToDevice, s);

        // RMS norm
        hipLaunchKernelGGL(rms_norm, dim3((H+255)/256), dim3(256), 0, s, g->dh, m->in[l], H);

        // QKV
        hipblasSgemv(h, HIPBLAS_OP_T, H, NH*HD, &a, g->l[l].q, H, g->dh, 1, &b, g->dqkv, 1);
        hipblasSgemv(h, HIPBLAS_OP_T, H, NKV*HD, &a, g->l[l].k, H, g->dh, 1, &b, g->dqkv+NH*HD, 1);
        hipblasSgemv(h, HIPBLAS_OP_T, H, NKV*HD, &a, g->l[l].v, H, g->dh, 1, &b, g->dqkv+NH*HD+NKV*HD, 1);

        // Q/K norm + RoPE
        hipLaunchKernelGGL(qk_norm_rope, dim3(NH+NKV), dim3(HD), HD*4, s,
            g->dqkv, g->dqkv+NH*HD, m->qn[l], m->kn[l], m->rs, m->rc, *pos, NH, NKV);

        // KV cache write (copy QKV's K and V portions)
        hipMemcpyAsync(g->dkv_k + (size_t)l*MAX_CTX*NKV*HD + *pos*NKV*HD,
                       g->dqkv + NH*HD, (size_t)NKV*HD*4, hipMemcpyDeviceToDevice, s);
        hipMemcpyAsync(g->dkv_v + (size_t)l*MAX_CTX*NKV*HD + *pos*NKV*HD,
                       g->dqkv + NH*HD + NKV*HD, (size_t)NKV*HD*4, hipMemcpyDeviceToDevice, s);

        // Attention
        hipLaunchKernelGGL(attn_kernel, dim3(NH), dim3(HD), 0, s,
            g->dqkv, g->dkv_k + (size_t)l*MAX_CTX*NKV*HD,
            g->dkv_v + (size_t)l*MAX_CTX*NKV*HD, g->dat, *pos+1);

        // O projection
        hipblasSgemv(h, HIPBLAS_OP_T, NH*HD, H, &a, g->l[l].o, NH*HD, g->dat, 1, &b, g->dh, 1);

        // Residual: h = res + O
        hipLaunchKernelGGL(add_residual, dim3((H+255)/256), dim3(256), 0, s, g->dh, g->dres, H);

        // Save pre-FFN residual
        hipMemcpyAsync(g->dres, g->dh, H*4, hipMemcpyDeviceToDevice, s);

        // Post-attention norm
        hipLaunchKernelGGL(rms_norm, dim3((H+255)/256), dim3(256), 0, s, g->dh, m->pa[l], H);

        // Gate + Up
        hipblasSgemv(h, HIPBLAS_OP_T, H, IM, &a, g->l[l].g, H, g->dh, 1, &b, g->dgt, 1);
        hipblasSgemv(h, HIPBLAS_OP_T, H, IM, &a, g->l[l].u, H, g->dh, 1, &b, g->dact, 1);

        // SiLU(gate) * up
        hipLaunchKernelGGL(silu_gate, dim3((IM+255)/256), dim3(256), 0, s, g->dgt, g->dact, IM);

        // Down projection
        hipblasSgemv(h, HIPBLAS_OP_T, IM, H, &a, g->l[l].d, IM, g->dact, 1, &b, g->dh, 1);

        // FFN residual
        hipLaunchKernelGGL(add_residual, dim3((H+255)/256), dim3(256), 0, s, g->dh, g->dres, H);
    }

    // Final norm
    hipLaunchKernelGGL(rms_norm, dim3((H+255)/256), dim3(256), 0, s, g->dh, m->fn, H);

    // LM head
    hipLaunchKernelGGL(lm_head_gpu, dim3((NV+255)/256), dim3(256), 0, s, g->dh, m->emb, g->dlg);

    // Single CPU sync
    hipStreamSynchronize(s);

    // Read top-1
    float *lg = (float*)malloc(NV*4);
    hipMemcpy(lg, g->dlg, NV*4, hipMemcpyDeviceToHost);
    uint32_t best = 0; float mx = lg[0];
    for (int i = 1; i < NV; i++) if (lg[i] > mx) { mx = lg[i]; best = (uint32_t)i; }
    free(lg);
    out[oidx] = best;
    (*pos)++;
    return 1;
}

/* ── Model loader ── */
static const uint8_t *fk(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;}
static int64_t fo(const uint8_t *j,size_t l,const char *k){const uint8_t *p=fk(j,l,k);if(!p)return -1;const uint8_t *d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return -1;const uint8_t *b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;}
static int si(const uint8_t *j,size_t l,const char *k){size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;}

static bool load_model(const char *path, Model *m) {
    int fd=open(path,O_RDONLY);if(fd<0)return false;struct stat st;fstat(fd,&st);
    m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(!m->map)return false;uint64_t hdr;memcpy(&hdr,m->map,8);m->ds=8+(size_t)hdr;
    const uint8_t *js=m->map+8;size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");
    m->emb=(float*)malloc((size_t)NV*H*4);const uint16_t *eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);
    for(int i=0;i<NV*H;i++)m->emb[i]=bf16(eb[i]);
    char buf[256];
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);
        m->in[l]=(float*)malloc(H*4);const uint16_t *s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);
        m->pa[l]=(float*)malloc(H*4);s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*4);
    const uint16_t *fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4);m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
    for(int l=0;l<NC;l++){snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);
        if(m->qi[l]>0)m->has_w=true;}
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);
    return true;
}
static void free_model(Model *m){
    free(m->emb);free(m->fn);for(int l=0;l<NC;l++){free(m->in[l]);free(m->pa[l]);if(m->qn[l])free(m->qn[l]);if(m->kn[l])free(m->kn[l]);}
    free(m->rs);free(m->rc);if(m->map)munmap(m->map,m->ds?m->ds-8:0);
}

static uint32_t lt(const char *w){
    if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;
    if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;
    if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;
    if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;
    if(!strcmp(w,"2"))return 17;if(!strcmp(w,"+"))return 10;if(!strcmp(w,"the"))return 262;if(!strcmp(w,"is"))return 374;
    return (uint32_t)(unsigned char)w[0]+3;
}
static int tok(const char *t,uint32_t*tk,int mx){
    int n=0;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=872;if(n<mx)tk[n++]=198;
    char w[256];int wl=0;
    for(const char *p=t;*p&&n<mx;p++){unsigned char c=(unsigned char)*p;
        if(c==' '||c=='\n'||c=='\t'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}if(c=='\n'&&n<mx)tk[n++]=198;}
        else if(c==','||c=='.'||c=='?'||c=='!'||c==';'||c==':'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}char pn[2]={(char)c,0};if(n<mx)tk[n++]=lt(pn);}
        else{if(wl<255)w[wl++]=c;}
    }
    if(wl&&n<mx){w[wl]=0;tk[n++]=lt(w);}
    if(n<mx)tk[n++]=151645;if(n<mx)tk[n++]=198;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=77091;if(n<mx)tk[n++]=198;
    return n;
}

int main(int argc, char **argv) {
    const char *mp=DEF_MODEL, *pr="Hello"; int mx=128;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){printf("Fused GPU Engine\n");return 0;}
        if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];
        if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];
        if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);
    }
    fprintf(stderr,"\n=== Fused GPU Engine ===\n");
    Model m; if(!load_model(mp,&m))return 1;
    int pos=0; uint32_t pt[4096]; int npt=tok(pr,pt,4096);
    fprintf(stderr,"Tokens: %d\n",npt);
    gpu_init(&m);
    fprintf(stderr,"GPU: %s\n",m.gpu.ok?"fused":"unavailable");
    if(!m.gpu.ok) return 1;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t *out=(uint32_t*)calloc((size_t)mx,4); int gen=0;
    fprintf(stderr,"Prefill %d...\n",npt);
    for(int pi=0;pi<npt;pi++) fused_fwd(&m,pt[pi],&pos,out,0);
    fprintf(stderr,"Decode %d...\n",mx);
    uint32_t ct=pt[npt-1];
    while(gen<mx){fused_fwd(&m,ct,&pos,out,gen);ct=out[gen];gen++;}
    clock_gettime(CLOCK_MONOTONIC,&ts); double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"%d tokens in %.0fms (%.0f tok/s)\n",gen,ms,ms>0?gen/(ms/1000):0);
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    free(out);free_model(&m); return 0;
}
