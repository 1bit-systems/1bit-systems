/* Compare CPU vs GPU hidden states layer-by-layer */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
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

/* ─── CPU math (exact copy from engine_gpu.c) ─── */
static void cpu_rms(float *x, const float *w, int n) {
    double ss=0; for(int i=0;i<n;i++){if(!isfinite(x[i]))x[i]=0;ss+=(double)x[i]*(double)x[i];}
    float ir=1.0f/sqrtf((float)(ss/(double)n)+1e-6f); for(int i=0;i<n;i++)x[i]*=ir*w[i];
}
static void cpu_rope(float *x, int p, const float *s, const float *c) {
    for(int d=0;d<HD/2;d++){float a=x[d],b=x[d+HD/2];x[d]=a*c[p*HD+d]-b*s[p*HD+d];x[d+HD/2]=b*c[p*HD+d]+a*s[p*HD+d];}
}
static void cpu_attn(const float *q, const float *kc, const float *vc, float *o, int sl) {
    float sc=1.0f/sqrtf((float)HD);
    for(int h=0;h<NH;h++){
        int kvh=h/GQA; const float *qh=q+h*HD; float mx=-INFINITY; float sb[4096];
        for(int p=0;p<sl;p++){const float *kr=kc+p*NKV*HD+kvh*HD; double dot=0;
            for(int d=0;d<HD;d++)dot+=(double)qh[d]*(double)kr[d]; sb[p]=(float)(dot*sc);if(sb[p]>mx)mx=sb[p];}
        double su=0; for(int p=0;p<sl;p++){sb[p]=expf(sb[p]-mx);su+=sb[p];}
        float iv=su>0?(float)(1.0/su):0; float *oh=o+h*HD; memset(oh,0,(size_t)HD*4);
        for(int p=0;p<sl;p++){float w=sb[p]*iv;const float *vr=vc+p*NKV*HD+kvh*HD;for(int d=0;d<HD;d++)oh[d]+=w*vr[d];}
    }
}

/* ─── GPU Kernels (same as engine_55t_v3) ─── */
__global__ void k_rms_norm(float *x, const float *w, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ double ss[256];
    if (i < n) { double v = (double)x[i]; ss[threadIdx.x] = isfinite((float)v) ? v*v : 0.0; }
    else { ss[threadIdx.x] = 0.0; }
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) ss[threadIdx.x] += ss[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) ss[0] = 1.0 / sqrt(ss[0] / (double)n + 1e-6);
    __syncthreads();
    if (i < n) x[i] = (float)((double)x[i] * ss[0] * (double)w[i]);
}
__global__ void k_qk_norm(float *q, float *k, const float *qn_w, const float *kn_w) {
    int h = blockIdx.x, d = threadIdx.x;
    bool is_k = h >= NH;
    float *vec = is_k ? (k + (h - NH) * HD) : (q + h * HD);
    const float *nw = is_k ? kn_w : qn_w;
    __shared__ double ss[128];
    ss[d] = (h < NH+NKV) ? (double)vec[d] * (double)vec[d] : 0.0;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if (d < s) ss[d] += ss[d + s]; __syncthreads(); }
    if (h < NH+NKV) { double iq = 1.0 / sqrt(ss[0] / (double)HD + 1e-6); vec[d] = (float)((double)vec[d] * iq * (nw ? (double)nw[d] : 1.0)); }
}
__global__ void k_rope(float *x, int pos, const float *s, const float *c, int nheads) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= nheads || d >= HD/2) return;
    float *vec = x + h * HD;
    float a = vec[d], b = vec[d + HD/2];
    vec[d] = a * c[pos*HD + d] - b * s[pos*HD + d];
    vec[d + HD/2] = b * c[pos*HD + d] + a * s[pos*HD + d];
}
__global__ void k_attn(const float *q, const float *kc, const float *vc, float *out, int seq_len) {
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= NH || d >= HD) return;
    int kvh = h / GQA;
    const float *qh = q + h * HD;
    __shared__ float scores[4096];
    if (d == 0) {
        float sc = 1.0f / sqrtf((float)HD), mx = -1e30f;
        for (int p = 0; p < seq_len; p++) {
            const float *kr = kc + p * NKV * HD + kvh * HD;
            double dot = 0.0;
            for (int dd = 0; dd < HD; dd++) dot += (double)qh[dd] * (double)kr[dd];
            scores[p] = (float)(dot * sc); if (scores[p] > mx) mx = scores[p];
        }
        double su = 0.0;
        for (int p = 0; p < seq_len; p++) { scores[p] = expf(scores[p] - mx); su += (double)scores[p]; }
        float iv = su > 0 ? (float)(1.0 / su) : 0;
        for (int p = 0; p < seq_len; p++) scores[p] *= iv;
    }
    __syncthreads();
    float acc = 0;
    for (int p = 0; p < seq_len; p++) acc += scores[p] * vc[p * NKV * HD + kvh * HD + d];
    out[h * HD + d] = acc;
}
__global__ void k_silu(float *g, float *u, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float gv = g[i]; u[i] = (gv / (1.0f + expf(-gv))) * u[i];
}
__global__ void k_add(float *h, const float *r, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return; h[i] += r[i];
}

typedef struct { float *q,*k,*v,*o,*g,*u,*d; } DevW;

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
static bool upload_w(const uint8_t *map, size_t ds, uint64_t off, int i8r, int id, float **dw, float **cpu_w) {
    int or_,oc; *cpu_w = deq_i8(map+ds+off,i8r,id,&or_,&oc);
    if(!*cpu_w)return false;
    hipMalloc(dw,(size_t)or_*oc*4);
    hipMemcpy(*dw,*cpu_w,(size_t)or_*oc*4,hipMemcpyHostToDevice);
    return true;
}

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

/* Compare two float arrays, return max difference */
static float cmp_vec(const float *a, const float *b, int n, const char *label) {
    float md=0; int mi=0;
    for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>md){md=d;mi=i;}}
    if(md>0.001f) fprintf(stderr,"  %s: max_diff=%.6f at [%d] (cpu=%.6f gpu=%.6f)\n",label,md,mi,a[mi],b[mi]);
    return md;
}

static void cpu_mm(const float *h, int od, int id, float *out, const float *w) {
    memset(out,0,(size_t)od*4);
    for(int i=0;i<od;i++){double dot=0;for(int j=0;j<id;j++)dot+=(double)w[i*id+j]*(double)h[j];out[i]=(float)dot;}
}

int main(){
    fprintf(stderr,"=== CPU vs GPU comparison ===\n");
    
    // Load model
    int fd=open(DEF_MODEL,O_RDONLY);
    struct stat st; fstat(fd,&st);
    uint8_t *map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    uint64_t hdr; memcpy(&hdr,map,8); size_t ds=8+(size_t)hdr;
    const uint8_t *js=map+8; size_t jl=(size_t)hdr;

    // Load CPU model data
    float *emb=(float*)malloc((size_t)NV*H*4);
    const uint16_t *eb=(const uint16_t*)(map+ds+(size_t)fo(js,jl,"model.embed_tokens.weight"));
    for(int i=0;i<NV*H;i++)emb[i]=bf16(eb[i]);

    float *in[NC],*pa[NC],*qn[NC],*kn[NC],*fn;
    for(int l=0;l<NC;l++){
        char buf[256];
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);
        in[l]=(float*)malloc(H*4); const uint16_t*s=(const uint16_t*)(map+ds+(size_t)fo(js,jl,buf));
        for(int i=0;i<H;i++)in[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);
        pa[l]=(float*)malloc(H*4); s=(const uint16_t*)(map+ds+(size_t)fo(js,jl,buf));
        for(int i=0;i<H;i++)pa[l][i]=bf16(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);
        int64_t o=fo(js,jl,buf);
        if(o>0){qn[l]=(float*)malloc(HD*4);const uint16_t*qs=(const uint16_t*)(map+ds+(size_t)o);for(int i=0;i<HD;i++)qn[l][i]=bf16(qs[i]);}else qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);
        o=fo(js,jl,buf);
        if(o>0){kn[l]=(float*)malloc(HD*4);const uint16_t*ks=(const uint16_t*)(map+ds+(size_t)o);for(int i=0;i<HD;i++)kn[l][i]=bf16(ks[i]);}else kn[l]=0;
    }
    fn=(float*)malloc(H*4);const uint16_t*fs=(const uint16_t*)(map+ds+(size_t)fo(js,jl,"model.norm.weight"));
    for(int i=0;i<H;i++)fn[i]=bf16(fs[i]);

    float *rs=(float*)malloc((size_t)MAX_CTX*HD*4),*rc=(float*)malloc((size_t)MAX_CTX*HD*4);
    float th=1000000.0f; for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float f=1.0f/powf(th,(float)d/(HD/2)),a=(float)p*f;rs[p*HD+d]=sinf(a);rc[p*HD+d]=cosf(a);rs[p*HD+HD/2+d]=sinf(a);rc[p*HD+HD/2+d]=cosf(a);}

    // Init GPU
    hipblasHandle_t blas; hipStream_t stream;
    hipblasCreate(&blas); hipStreamCreate(&stream);
    hipblasSetStream(blas,stream);

    float *d_one,*d_zero;
    hipMalloc(&d_one,4); hipMalloc(&d_zero,4);
    float one=1.0f,zero=0.0f;
    hipMemcpy(d_one,&one,4,hipMemcpyHostToDevice);
    hipMemcpy(d_zero,&zero,4,hipMemcpyHostToDevice);
    hipblasSetPointerMode(blas,HIPBLAS_POINTER_MODE_DEVICE);

    float *dh,*dres,*dout,*datt,*dkv_k,*dkv_v;
    hipMalloc(&dh,H*4); hipMalloc(&dres,H*4);
    hipMalloc(&dout,((QT>(2*IM))?QT:(2*IM))*4);
    hipMalloc(&datt,(size_t)NH*HD*4);
    size_t kv_sz=(size_t)NC*MAX_CTX*NKV*HD;
    hipMalloc(&dkv_k,kv_sz*4); hipMalloc(&dkv_v,kv_sz*4);
    hipMemset(dkv_k,0,kv_sz*4); hipMemset(dkv_v,0,kv_sz*4);

    // Upload per-layer data
    float *d_in[NC],*d_pa[NC],*d_qn[NC],*d_kn[NC];
    for(int l=0;l<NC;l++){
        hipMalloc(&d_in[l],H*4); hipMemcpy(d_in[l],in[l],H*4,hipMemcpyHostToDevice);
        hipMalloc(&d_pa[l],H*4); hipMemcpy(d_pa[l],pa[l],H*4,hipMemcpyHostToDevice);
        if(qn[l]){hipMalloc(&d_qn[l],HD*4);hipMemcpy(d_qn[l],qn[l],HD*4,hipMemcpyHostToDevice);}else d_qn[l]=0;
        if(kn[l]){hipMalloc(&d_kn[l],HD*4);hipMemcpy(d_kn[l],kn[l],HD*4,hipMemcpyHostToDevice);}else d_kn[l]=0;
    }
    float *d_fn; hipMalloc(&d_fn,H*4); hipMemcpy(d_fn,fn,H*4,hipMemcpyHostToDevice);
    float *d_rs,*d_rc;
    hipMalloc(&d_rs,(size_t)MAX_CTX*HD*4); hipMalloc(&d_rc,(size_t)MAX_CTX*HD*4);
    hipMemcpy(d_rs,rs,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    hipMemcpy(d_rc,rc,(size_t)MAX_CTX*HD*4,hipMemcpyHostToDevice);
    float *d_emb; hipMalloc(&d_emb,(size_t)NV*H*4); hipMemcpy(d_emb,emb,(size_t)NV*H*4,hipMemcpyHostToDevice);

    // Upload weights (keep CPU copies for comparison)
    DevW dw[NC]; float *cpu_w[NC][7];
    for(int l=0;l<NC;l++){
        char buf[256];
        snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),H,&dw[l].q,&cpu_w[l][0]);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),H,&dw[l].k,&cpu_w[l][1]);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),H,&dw[l].v,&cpu_w[l][2]);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),NH*HD,&dw[l].o,&cpu_w[l][3]);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),H,&dw[l].g,&cpu_w[l][4]);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),H,&dw[l].u,&cpu_w[l][5]);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l); upload_w(map,ds,fo(js,jl,buf),si(js,jl,buf),IM,&dw[l].d,&cpu_w[l][6]);
    }
    fprintf(stderr,"Weights uploaded.\n");

    // CPU and GPU hidden states (start from same embedding)
    float cpu_h[H], cpu_res[H], cpu_qkv[QT], cpu_at[NH*HD], cpu_gt[IM], cpu_act[IM];
    float cpu_kc[MAX_CTX*NKV*HD], cpu_vc[MAX_CTX*NKV*HD];
    float gpu_h[H];

    // Test with first token: <|im_start|> = 151644
    uint32_t tok = 151644;
    int pos = 0;

    // CPU init: h = emb[tok]
    memcpy(cpu_h, emb+(size_t)tok*H, H*4);
    // GPU init: copy same embedding
    hipMemcpy(dh, d_emb+(size_t)tok*H, H*4, hipMemcpyDeviceToDevice);
    hipStreamSynchronize(stream);
    hipMemcpy(gpu_h, dh, H*4, hipMemcpyDeviceToHost);
    fprintf(stderr,"Initial h: "); cmp_vec(cpu_h,gpu_h,H,"embedding");

    // Process ONE layer with both CPU and GPU
    for(int l=0;l<1;l++){
        fprintf(stderr,"\n--- Layer %d ---\n",l);

        /* === CPU path === */
        memcpy(cpu_res, cpu_h, H*4);
        cpu_rms(cpu_h, in[l], H);
        
        // CPU matmuls
        cpu_mm(cpu_h, NH*HD, H, cpu_qkv, cpu_w[l][0]);           // Q
        cpu_mm(cpu_h, NKV*HD, H, cpu_qkv+NH*HD, cpu_w[l][1]);    // K
        cpu_mm(cpu_h, NKV*HD, H, cpu_qkv+NH*HD+NKV*HD, cpu_w[l][2]); // V

        // QK norm + RoPE
        for(int hh=0;hh<NH;hh++){float *qh=cpu_qkv+hh*HD;double sq=0;
            for(int d=0;d<HD;d++)sq+=(double)qh[d]*(double)qh[d];
            float iq=1.0f/sqrtf((float)(sq/HD)+1e-6f);
            for(int d=0;d<HD;d++)qh[d]*=iq*(qn[l]?qn[l][d]:1);
            cpu_rope(qh,pos,rs,rc);}
        for(int kh=0;kh<NKV;kh++){float *ks=cpu_qkv+NH*HD+kh*HD;double sk=0;
            for(int d=0;d<HD;d++)sk+=(double)ks[d]*(double)ks[d];
            float ik=1.0f/sqrtf((float)(sk/HD)+1e-6f);
            for(int d=0;d<HD;d++)ks[d]*=ik*(kn[l]?kn[l][d]:1);
            cpu_rope(ks,pos,rs,rc);
            memcpy(cpu_kc+pos*NKV*HD+kh*HD,ks,HD*4);}
        for(int kh=0;kh<NKV;kh++){float *vs=cpu_qkv+NH*HD+NKV*HD+kh*HD;memcpy(cpu_vc+pos*NKV*HD+kh*HD,vs,HD*4);}

        // Attention
        cpu_attn(cpu_qkv, cpu_kc, cpu_vc, cpu_at, pos+1);

        // O projection
        cpu_mm(cpu_at, H, NH*HD, cpu_h, cpu_w[l][3]);
        for(int i=0;i<H;i++)cpu_h[i]=cpu_res[i]+cpu_h[i];

        // FFN
        memcpy(cpu_res, cpu_h, H*4);
        cpu_rms(cpu_h, pa[l], H);
        cpu_mm(cpu_h, IM, H, cpu_gt, cpu_w[l][4]);
        cpu_mm(cpu_h, IM, H, cpu_act, cpu_w[l][5]);
        for(int i=0;i<IM;i++){float gv=cpu_gt[i];cpu_act[i]=(gv/(1.0f+expf(-gv)))*cpu_act[i];}
        cpu_mm(cpu_act, H, IM, cpu_h, cpu_w[l][6]);
        for(int i=0;i<H;i++)cpu_h[i]=cpu_res[i]+cpu_h[i];

        /* === GPU path (sync after each step) === */
        hipMemcpyAsync(dh,d_emb+(size_t)tok*H,H*4,hipMemcpyDeviceToDevice,stream);

        // Save residual
        hipMemcpyAsync(dres,dh,H*4,hipMemcpyDeviceToDevice,stream);
        hipStreamSynchronize(stream);
        hipMemcpy(gpu_h,dh,H*4,hipMemcpyDeviceToHost);
        cmp_vec(cpu_res,gpu_h,H,"after residual save");

        // RMS norm
        hipLaunchKernelGGL(k_rms_norm,dim3(CEILDIV(H,256)),dim3(256),0,stream,dh,d_in[l],H);
        hipStreamSynchronize(stream);
        hipMemcpy(gpu_h,dh,H*4,hipMemcpyDeviceToHost);
        cmp_vec(cpu_h,gpu_h,H,"after input rms_norm");

        // Q matmul
        hipblasSgemv(blas,HIPBLAS_OP_T,H,NH*HD,d_one,dw[l].q,H,dh,1,d_zero,dout,1);
        hipStreamSynchronize(stream);
        {float tmp[NH*HD]; hipMemcpy(tmp,dout,(size_t)NH*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv,tmp,NH*HD,"Q matmul");}
        
        // K matmul
        hipblasSgemv(blas,HIPBLAS_OP_T,H,NKV*HD,d_one,dw[l].k,H,dh,1,d_zero,dout+NH*HD,1);
        hipStreamSynchronize(stream);
        {float tmp[NKV*HD]; hipMemcpy(tmp,dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv+NH*HD,tmp,NKV*HD,"K matmul");}
        
        // V matmul
        hipblasSgemv(blas,HIPBLAS_OP_T,H,NKV*HD,d_one,dw[l].v,H,dh,1,d_zero,dout+NH*HD+NKV*HD,1);
        hipStreamSynchronize(stream);
        {float tmp[NKV*HD]; hipMemcpy(tmp,dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv+NH*HD+NKV*HD,tmp,NKV*HD,"V matmul");}

        // QK norm
        hipLaunchKernelGGL(k_qk_norm,dim3(NH+NKV),dim3(HD),0,stream,dout,dout+NH*HD,d_qn[l],d_kn[l]);
        hipStreamSynchronize(stream);
        {float tmp[QT]; hipMemcpy(tmp,dout,(size_t)QT*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv,tmp,QT,"QKV after QK norm");}

        // Q RoPE
        hipLaunchKernelGGL(k_rope,dim3(NH),dim3(HD/2),0,stream,dout,pos,d_rs,d_rc,NH);
        hipStreamSynchronize(stream);
        {float tmp[NH*HD]; hipMemcpy(tmp,dout,(size_t)NH*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv,tmp,NH*HD,"Q after RoPE");}

        // K RoPE
        hipLaunchKernelGGL(k_rope,dim3(NKV),dim3(HD/2),0,stream,dout+NH*HD,pos,d_rs,d_rc,NKV);
        hipStreamSynchronize(stream);
        {float tmp[NKV*HD]; hipMemcpy(tmp,dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_qkv+NH*HD,tmp,NKV*HD,"K after RoPE");}

        // KV cache write
        hipMemcpyAsync(dkv_k+pos*NKV*HD,dout+NH*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,stream);
        hipMemcpyAsync(dkv_v+pos*NKV*HD,dout+NH*HD+NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToDevice,stream);
        hipStreamSynchronize(stream);
        {float tmp[NKV*HD]; hipMemcpy(tmp,dkv_k+pos*NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_kc+pos*NKV*HD,tmp,NKV*HD,"K cache");}
        {float tmp[NKV*HD]; hipMemcpy(tmp,dkv_v+pos*NKV*HD,(size_t)NKV*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_vc+pos*NKV*HD,tmp,NKV*HD,"V cache");}

        // Attention
        hipLaunchKernelGGL(k_attn,dim3(NH),dim3(HD),0,stream,dout,dkv_k,dkv_v,datt,pos+1);
        hipStreamSynchronize(stream);
        {float tmp[NH*HD]; hipMemcpy(tmp,datt,(size_t)NH*HD*4,hipMemcpyDeviceToHost); cmp_vec(cpu_at,tmp,NH*HD,"Attention output");}

        // O projection
        hipblasSgemv(blas,HIPBLAS_OP_T,NH*HD,H,d_one,dw[l].o,NH*HD,datt,1,d_zero,dh,1);
        hipStreamSynchronize(stream);
        {float tmp[H]; hipMemcpy(tmp,dh,H*4,hipMemcpyDeviceToHost); float ref[H]; memcpy(ref,cpu_h,H*4); for(int i=0;i<H;i++)ref[i]-=cpu_res[i]; cmp_vec(ref,tmp,H,"O projection (no residual)");}

        // Residual add
        hipLaunchKernelGGL(k_add,dim3(CEILDIV(H,256)),dim3(256),0,stream,dh,dres,H);
        hipStreamSynchronize(stream);
        hipMemcpy(gpu_h,dh,H*4,hipMemcpyDeviceToHost);
        cmp_vec(cpu_h,gpu_h,H,"After attention residual");

        fprintf(stderr,"  CPU h[0..3] = %.6f %.6f %.6f %.6f\n",cpu_h[0],cpu_h[1],cpu_h[2],cpu_h[3]);
        fprintf(stderr,"  GPU h[0..3] = %.6f %.6f %.6f %.6f\n",gpu_h[0],gpu_h[1],gpu_h[2],gpu_h[3]);
    }

    // Cleanup
    for(int l=0;l<NC;l++){free(cpu_w[l][0]);free(cpu_w[l][1]);free(cpu_w[l][2]);free(cpu_w[l][3]);free(cpu_w[l][4]);free(cpu_w[l][5]);free(cpu_w[l][6]);}
    return 0;
}
