/* engine_55t_v3.cu — Direct GPU-resident port of engine_gpu.c
 * 
 * Identical logic to engine_gpu.c. The only change: all data stays on GPU.
 * No brainer: if engine_gpu.c's math is correct, this is correct.
 *
 * Build: hipcc -O3 -ffast-math -o engine_55t_v3 engine_55t_v3.cu -lhipblas
 * Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./engine_55t_v3 -n 64 -p "Hello"
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
#define TILE_R 32
#define TILE_C 256
#define DEF_MODEL "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };
#define CEILDIV(a,b) (((a)+(b)-1)/(b))
static float bf16(uint16_t v) { uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f; }

/* ── GPU Kernels: identical numerical logic to engine_gpu.c CPU functions ── */

__global__ void k_rms_norm(float *x, const float *w, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double ss[256];
    if (i < n) {
        double v = (double)x[i]; ss[threadIdx.x] = isfinite((float)v) ? v*v : 0.0;
    } else { ss[threadIdx.x] = 0.0; }
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ss[0] = 1.0 / sqrt(ss[0] / (double)n + 1e-6);
    __syncthreads();
    if (i < n) x[i] = (float)((double)x[i] * ss[0] * (double)w[i]);
}

__global__ void k_qk_norm(float *q, float *k, const float *qn_w, const float *kn_w) {
    /* For each head vector, compute unit norm then scale by per-dim weight.
     * Block = head (first NH for Q, next NKV for K). Thread = dimension. */
    int h = blockIdx.x, d = threadIdx.x;
    int nq = NH, nk = NKV;
    bool is_k = h >= nq;
    float *vec = is_k ? (k + (h - nq) * HD) : (q + h * HD);
    const float *nw = is_k ? kn_w : qn_w;

    __shared__ double ss[128];
    if (h < nq + nk) ss[d] = (double)vec[d] * (double)vec[d];
    else ss[d] = 0.0;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (d < s) ss[d] += ss[d + s]; __syncthreads(); }
    if (h < nq + nk) {
        double iq = 1.0 / sqrt(ss[0] / (double)HD + 1e-6);
        vec[d] = (float)((double)vec[d] * iq * (nw ? (double)nw[d] : 1.0));
    }
}

__global__ void k_rope(float *x, int pos, const float *s, const float *c, int nheads) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= nheads || d >= HD/2) return;
    float *vec = x + h * HD;
    float a = vec[d], b = vec[d + HD/2];
    vec[d]       = a * c[pos*HD + d] - b * s[pos*HD + d];
    vec[d + HD/2] = b * c[pos*HD + d] + a * s[pos*HD + d];
}

__global__ void k_attn(const float *q, const float *kc, const float *vc,
                        float *out, int seq_len) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= NH || d >= HD) return;
    int kvh = h / GQA;
    const float *qh = q + h * HD;

    __shared__ float scores[4096];

    // Thread 0: compute QK^T scores (double precision dot product)
    if (d == 0) {
        float sc = 1.0f / sqrtf((float)HD);
        float mx = -1e30f;
        for (int p = 0; p < seq_len; p++) {
            const float *kr = kc + p * NKV * HD + kvh * HD;
            double dot = 0.0;
            for (int dd = 0; dd < HD; dd++) dot += (double)qh[dd] * (double)kr[dd];
            scores[p] = (float)(dot * sc);
            if (scores[p] > mx) mx = scores[p];
        }
        double su = 0.0;
        for (int p = 0; p < seq_len; p++) {
            scores[p] = expf(scores[p] - mx);
            su += (double)scores[p];
        }
        float iv = su > 0 ? (float)(1.0 / su) : 0;
        for (int p = 0; p < seq_len; p++) scores[p] *= iv;
    }
    __syncthreads();

    // All threads: weighted sum of V
    float acc = 0;
    for (int p = 0; p < seq_len; p++)
        acc += scores[p] * vc[p * NKV * HD + kvh * HD + d];
    out[h * HD + d] = acc;
}

__global__ void k_silu(float *g, float *u, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float gv = g[i];
    u[i] = (gv / (1.0f + expf(-gv))) * u[i];
}

__global__ void k_add(float *h, const float *r, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    h[i] += r[i];
}

/* ── Data structures ── */
typedef struct { float *q,*k,*v,*o,*g,*u,*d; } DevW;

typedef struct {
    hipblasHandle_t blas; hipStream_t stream;
    /* Device scratch buffers */
    float *dh, *dres, *dout;  /* dout: big enough for QKV(4096) or Gate+Up(6144) */
    float *datt;               /* attention output [NH*HD] */
    float *dlg;                /* logits [NV] */
    float *dkv_k, *dkv_v;      /* KV cache [NC*MAX_CTX*NKV*HD] */
    float *d_one, *d_zero;     /* device scalars for async hipBLAS */
    /* Per-layer norm weights on device */
    float *d_in[NC], *d_pa[NC], *d_qn[NC], *d_kn[NC];
    /* Global data on device */
    float *d_rs, *d_rc, *d_fn, *d_emb;
    /* Weight matrices */
    DevW *l;
    bool ok;
} GPU;

typedef struct {
    float *emb, *fn, *in[NC], *pa[NC], *qn[NC], *kn[NC], *rs, *rc;
    uint8_t *map; size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w; GPU gpu;
} Model;

/* ── I8→FP32 dequant (CPU, init-time) ── */
static float* deq_i8(const uint8_t *d, int i8r, int id, int *or_, int *oc) {
    int ntc=id/256;if(ntc<1)ntc=1;int ntr=i8r/ntc;
    *or_=ntr*32;*oc=ntc*256;
    float *o=(float*)calloc((size_t)(*or_)*(*oc),4);
    for(int ir=0;ir<i8r;ir++){const uint8_t *rd=d+ir*I8_ROW_B;int tr=ir/ntc,tc=ir%ntc;
        const uint16_t *sc=(const uint16_t*)rd,*zp=(const uint16_t*)(rd+512);const uint8_t *pk=rd+1024;
        for(int lr=0;lr<TILE_R;lr++){int lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;const uint8_t *ld=pk+lane*(TILE_C*8);
            for(int c=0;c<TILE_C;c++){int g=c/32;float s=bf16(sc[g*32+lr]),z=bf16(zp[g*32+lr]);
                uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
                o[(tr*TILE_R+lr)*(*oc)+(tc*TILE_C+c)]=(float)cd*s+z;}}}
    return o;
}
static bool upload_w(Model *m, uint64_t off, int i8r, int id, float **dw) {
    int or_,oc;float *c=deq_i8(m->map+m->ds+off,i8r,id,&or_,&oc);
    if(!c)return false; hipMalloc(dw,(size_t)or_*oc*4);
    hipMemcpy(*dw,c,(size_t)or_*oc*4,hipMemcpyHostToDevice); free(c); return true;
}

/* ── GPU init ── */
static bool gpu_init(Model *m) {
    GPU *g=&m->gpu; g->ok=false;
    hipblasCreate(&g->blas); hipStreamCreate(&g->stream);
    hipblasSetStream(g->blas,g->stream);

    /* Device buffers */
    hipMalloc(&g->dh,H*4);
    hipMalloc(&g->dres,H*4);
    hipMalloc(&g->dout,((QT>(2*IM))?QT:(2*IM))*4);  // max of QKV and Gate+Up
    hipMalloc(&g->datt,(size_t)NH*HD*4);
    hipMalloc(&g->dlg,(size_t)NV*4);
    size_t kv_sz=(size_t)NC*MAX_CTX*NKV*HD;
    hipMalloc(&g->dkv_k,kv_sz*4); hipMalloc(&g->dkv_v,kv_sz*4);
    hipMemset(g->dkv_k,0,kv_sz*4); hipMemset(g->dkv_v,0,kv_sz*4);

    /* Device scalars for async hipBLAS */
    hipMalloc(&g->d_one,4); hipMalloc(&g->d_zero,4);
    float one=1.0f,zero=0.0f;
    hipMemcpy(g->d_one,&one,4,hipMemcpyHostToDevice);
    hipMemcpy(g->d_zero,&zero,4,hipMemcpyHostToDevice);
    hipblasSetPointerMode(g->blas,HIPBLAS_POINTER_MODE_DEVICE);

    /* Per-layer norm + QK norm weights */
    for(int l=0;l<NC;l++){
        hipMalloc(&g->d_in[l],H*4); hipMemcpy(g->d_in[l],m->in[l],H*4,hipMemcpyHostToDevice);
        hipMalloc(&g->d_pa[l],H*4); hipMemcpy(g->d_pa[l],m->pa[l],H*4,hipMemcpyHostToDevice);
        if(m->qn[l]){hipMalloc(&g->d_qn[l],HD*4);hipMemcpy(g->d_qn[l],m->qn[l],HD*4,hipMemcpyHostToDevice);}
        else g->d_qn[l]=0;
        if(m->kn[l]){hipMalloc(&g->d_kn[l],HD*4);hipMemcpy(g->d_kn[l],m->kn[l],HD*4,hipMemcpyHostToDevice);}
        else g->d_kn[l]=0;
    }

    /* Global data */
    hipMalloc(&g->d_fn,H*4); hipMemcpy(g->d_fn,m->fn,H*4,hipMemcpyHostToDevice);
    hipMalloc(&g->d_rs,(size_t)MAX_CTX*HD*4); hipMalloc(&g->d_rc,(size_t)MAX_CTX*HD*4);
    hipMemcpy(g->d_rs,m->rs,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(g->d_rc,m->rc,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    hipMalloc(&g->d_emb,(size_t)NV*H*4);
    hipMemcpy(g->d_emb,m->emb,(size_t)NV*H*4,hipMemcpyHostToDevice);

    /* Weight matrices */
    g->l=(DevW*)calloc(NC,sizeof(DevW));
    fprintf(stderr,"Uploading weights...\n");
    for(int l=0;l<NC;l++){
        fprintf(stderr,"  layer %d/%d\r",l+1,NC);
        if(!upload_w(m,m->qo[l],m->qi[l],H,&g->l[l].q))return false;
        if(!upload_w(m,m->ko[l],m->ki[l],H,&g->l[l].k))return false;
        if(!upload_w(m,m->vo[l],m->vi[l],H,&g->l[l].v))return false;
        if(!upload_w(m,m->oo[l],m->oi[l],NH*HD,&g->l[l].o))return false;
        if(!upload_w(m,m->go[l],m->gi[l],H,&g->l[l].g))return false;
        if(!upload_w(m,m->uo[l],m->ui[l],H,&g->l[l].u))return false;
        if(!upload_w(m,m->do_[l],m->di[l],IM,&g->l[l].d))return false;
    }
    fprintf(stderr,"\nGPU ready.\n");
    g->ok=true; return true;
}

/* ── GPU matmul (device→device, async) ── */
static void d_mm(GPU *g, int od, int id, float *din, float *dout, const float *dw) {
    hipblasSgemv(g->blas,HIPBLAS_OP_T,id,od,g->d_one,dw,id,din,1,g->d_zero,dout,1);
}

/* ── Layer forward: GPU kernels + GPU matmuls, all on one stream ── */
static void layer_fwd_gpu(GPU *g, int l, int pos) {
    hipStream_t s=g->stream;
    hipblasHandle_t h=g->blas;
    size_t l_off=(size_t)l*MAX_CTX*NKV*HD;

    /* --- Attention block (mirrors engine_gpu.c line by line) --- */
    // res = h  (save residual)
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);

    // rms(h, in[l], H)
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->d_in[l],H);

    // Q, K, V matmuls → dout
    d_mm(g,NH*HD,H,g->dh,g->dout,g->l[l].q);
    d_mm(g,NKV*HD,H,g->dh,g->dout+NH*HD,g->l[l].k);
    d_mm(g,NKV*HD,H,g->dh,g->dout+NH*HD+NKV*HD,g->l[l].v);

    // QK norm
    hipLaunchKernelGGL(k_qk_norm,dim3(NH+NKV),dim3(HD),0,s,
        g->dout,g->dout+NH*HD,g->d_qn[l],g->d_kn[l]);

    // RoPE on Q and K only (V is NOT rotated)
    hipLaunchKernelGGL(k_rope,dim3(NH),dim3(HD/2),0,s,g->dout,pos,g->d_rs,g->d_rc,NH);
    hipLaunchKernelGGL(k_rope,dim3(NKV),dim3(HD/2),0,s,g->dout+NH*HD,pos,g->d_rs,g->d_rc,NKV);

    // KV cache write
    hipMemcpyAsync(g->dkv_k+l_off+pos*NKV*HD,g->dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);
    hipMemcpyAsync(g->dkv_v+l_off+pos*NKV*HD,g->dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,s);

    // attention(q, kc, vc, attn_out, pos+1)
    hipLaunchKernelGGL(k_attn,dim3(NH),dim3(HD),0,s,
        g->dout,g->dkv_k+l_off,g->dkv_v+l_off,g->datt,pos+1);

    // O projection: h = O @ attn_out
    d_mm(g,H,NH*HD,g->datt,g->dh,g->l[l].o);

    // h = res + h
    hipLaunchKernelGGL(k_add,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->dres,H);

    /* --- FFN block --- */
    // res = h
    hipMemcpyAsync(g->dres,g->dh,H*4,hipMemcpyDeviceToDevice,s);

    // rms(h, pa[l], H)
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->d_pa[l],H);

    // Gate, Up → dout[0..IM-1], dout[IM..2*IM-1]
    d_mm(g,IM,H,g->dh,g->dout,g->l[l].g);
    d_mm(g,IM,H,g->dh,g->dout+IM,g->l[l].u);

    // SiLU(gate) * up → dout[IM..] modified in-place
    hipLaunchKernelGGL(k_silu,dim3(CEILDIV(IM,256)),dim3(256),0,s,g->dout,g->dout+IM,IM);

    // Down: h = W_d @ activated_up
    d_mm(g,H,IM,g->dout+IM,g->dh,g->l[l].d);

    // h = res + h
    hipLaunchKernelGGL(k_add,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->dres,H);
}

/* ── Full decode: 1 embedding copy + 28 layers + final_norm + lm_head + 1 sync ── */
static uint32_t decode_step(Model *m, uint32_t tok, int *pos) {
    GPU *g=&m->gpu;
    hipStream_t s=g->stream;
    hipblasHandle_t h=g->blas;

    // Embedding: dh = emb[tok*H..]
    hipMemcpyAsync(g->dh,g->d_emb+(size_t)tok*H,H*4,hipMemcpyDeviceToDevice,s);

    // 28 layers
    for(int l=0;l<NC;l++) layer_fwd_gpu(g,l,*pos);

    // Final norm: dh = rms_norm(dh, fn, H)
    hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,s,g->dh,g->d_fn,H);

    // LM head: dlg = emb @ dh  (NV×H × H → NV)
    hipblasSgemv(h,HIPBLAS_OP_T,H,NV,g->d_one,g->d_emb,H,g->dh,1,g->d_zero,g->dlg,1);

    // SYNC — the only one per token
    hipStreamSynchronize(s);

    // Read top-1 token
    float *lg=(float*)malloc((size_t)NV*4);
    hipMemcpy(lg,g->dlg,(size_t)NV*4,hipMemcpyDeviceToHost);
    uint32_t best=0; float mx=lg[0];
    for(int i=1;i<NV;i++) if(lg[i]>mx) { mx=lg[i]; best=(uint32_t)i; }
    free(lg);
    (*pos)++;
    return best;
}

/* ── Model loader (identical to engine_gpu.c) ── */
static const uint8_t *fk(const uint8_t *j,size_t l,const char *k){
    size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;
}
static int64_t fo(const uint8_t *j,size_t l,const char *k){
    const uint8_t *p=fk(j,l,k);if(!p)return -1;const uint8_t *d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return -1;
    const uint8_t *b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;
}
static int si(const uint8_t *j,size_t l,const char *k){
    size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;
        if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return (int)strtoul(br+1,0,10);}p+=kl;}return 0;
}
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
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);m->in[l]=(float*)malloc(H*4);
        const uint16_t *s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);m->pa[l]=(float*)malloc(H*4);
        s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);
        if(o>0){m->kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*4);
    const uint16_t *fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*4);m->rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;m->rs[p*HD+d]=sinf(a);m->rc[p*HD+d]=cosf(a);m->rs[p*HD+HD/2+d]=sinf(a);m->rc[p*HD+HD/2+d]=cosf(a);}
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);
        if(m->qi[l]>0)m->has_w=true;
    }
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);
    return true;
}
static void free_model(Model *m){
    free(m->emb);free(m->fn);for(int l=0;l<NC;l++){free(m->in[l]);free(m->pa[l]);if(m->qn[l])free(m->qn[l]);if(m->kn[l])free(m->kn[l]);}
    free(m->rs);free(m->rc);if(m->map)munmap(m->map,m->ds>8?m->ds-8:0);
}

/* ── Tokenizer ── */
static uint32_t lt(const char *w){
    if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;
    if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;
    if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;
    if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;
    if(!strcmp(w,"2"))return 17;if(!strcmp(w,"+"))return 10;if(!strcmp(w,"the"))return 262;if(!strcmp(w,"is"))return 374;
    if(!strcmp(w,"a"))return 247;if(!strcmp(w,"in"))return 253;
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

/* ── Main ── */
int main(int argc,char **argv){
    const char *mp=DEF_MODEL,*pr="Hello"; int mx=128;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-h")){printf("engine_55t_v3 — GPU-resident engine\n");return 0;}
        if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];
        if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];
        if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);
    }
    fprintf(stderr,"\n=== engine_55t_v3 — GPU-resident ===\n");
    Model m; memset(&m,0,sizeof(m));
    if(!load_model(mp,&m)){fprintf(stderr,"Model load failed\n");return 1;}

    int pos=0;
    uint32_t pt[4096]; int npt=tok(pr,pt,4096);
    fprintf(stderr,"Prompt: %d tokens\n",npt);

    if(!gpu_init(&m)){fprintf(stderr,"GPU init failed\n");free_model(&m);return 1;}

    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t *out=(uint32_t*)calloc((size_t)mx,4); int gen=0;

    // Prefill
    fprintf(stderr,"Prefill...\n");
    for(int pi=0;pi<npt;pi++){decode_step(&m,pt[pi],&pos);}

    // Decode
    fprintf(stderr,"Decode %d...\n",mx);
    uint32_t ct=pt[npt-1];
    while(gen<mx){
        out[gen]=decode_step(&m,ct,&pos);
        ct=out[gen]; gen++;
        if(gen%20==0)fprintf(stderr,"  %d/%d\r",gen,mx);
    }

    clock_gettime(CLOCK_MONOTONIC,&ts);
    double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    double tps=ms>0?gen/(ms/1000):0;
    fprintf(stderr,"\n═══ %d tokens in %.0fms — %.0f tok/s ═══\n",gen,ms,tps);
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    free(out);free_model(&m);
    return 0;
}
