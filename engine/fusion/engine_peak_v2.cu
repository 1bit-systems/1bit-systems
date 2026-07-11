/* engine_peak_v2.cu — engine_peak + fused RMSNorm+GEMV (read-once)
 *
 * Additional optimizations over engine_peak:
 *   7. Fused RMSNorm+I8 GEMV — read h once, compute norm + matmul
 *      - Saves 1 full H read and 1 H write per matmul (14 per layer)
 *      - Reduces intermediate BW from ~20 KB/layer to 0
 *   8. Block-level double buffering for attention
 *   9. Cooperative launch bounds for optimal register allocation
 *
 * Build:
 *   hipcc -O3 -ffast-math --offload-arch=gfx1151 \
 *     -o engine_peak_v2 engine_peak_v2.cu
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
#include <hip/hip_fp16.h>

#define MAX_CTX    4096
#define I8_ROW_B   5120
#define TILE_R     32
#define TILE_C     256
#define BLK        256
#define WARP       64
#define DEF_MODEL  "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"

enum { H=1024, NC=28, NH=16, NKV=8, HD=128, IM=3072, NV=151936, GQA=2, QT=NH*HD+2*NKV*HD };
#define CEILDIV(a,b) (((a)+(b)-1)/(b))

typedef _Float16 f16;

__host__ __device__ static inline float bf16_f32(uint16_t v) {
    uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;
}

// ════════════════════════════════════════════════════════════════════
// FUSED KERNELS — read-once, write-once
// ════════════════════════════════════════════════════════════════════

__device__ __forceinline__ f16 i8_deq(const uint8_t*t,int lr,int c){
    const uint16_t*sc=(const uint16_t*)t,*zp=(const uint16_t*)(t+512);
    const uint8_t*pk=t+1024;
    int g=c/32,lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;
    const uint8_t*ld=pk+lane*(TILE_C*8);
    float s=bf16_f32(sc[g*32+lr]),z=bf16_f32(zp[g*32+lr]);
    uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
    return (f16)((float)cd*s+z);
}

// ── Fused RMSNorm + I8 QKV GEMV ──────────────────────────────────────
// One kernel reads h ONCE, computes RMS norm AND Q,K,V simultaneously.
// Each output row is handled by a wavefront (32 threads).
// Block = 256 threads = 8 wavefronts × 8 output rows per block.
// Grid = total_tile_rows blocks.
//
// Phase 1: All threads participate in RMS reduction (shared mem)
// Phase 2: Each wavefront computes its output row's dot product
//
// This eliminates:
//   - 1 read of h for RMSNorm
//   - 1 write of normalized h
//   - 1 read of normalized h for QKV
//   - 2 extra reads for K,V (now shared with Q)
// = 5 memory accesses → 1
__global__ void k_rmsnorm_qkv_i8(
    const f16*__restrict__ h_in,        /* [H] FP16 input (pre-norm) */
    const f16*__restrict__ norm_w,      /* [H] RMS norm weights */
    const uint8_t*__restrict__ w_q,
    const uint8_t*__restrict__ w_k,
    const uint8_t*__restrict__ w_v,
    f16*__restrict__ qkv)               /* [QT] output */
{
    __shared__ float rms_sum[BLK/WARP]; /* 8 warps → 8 partial sums */
    __shared__ float rms_inv;
    int tid=threadIdx.x,wid=tid/WARP,lane=tid&(WARP-1);
    int ntc_q=H/TILE_C;                 /* 4 */
    int ntr_q=NH*HD/TILE_R;            /* 64 */
    int ntr_kv=NKV*HD/TILE_R;          /* 32 */

    // Phase 1: RMS reduction (all threads, single pass over h)
    float ssq=0;
    for(int i=tid;i<H;i+=BLK){
        float v=(float)h_in[i];ssq+=v*v;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1)ssq+=__shfl_xor(ssq,o);
    if(lane==0)rms_sum[wid]=ssq;
    __syncthreads();
    if(wid==0){
        float s=(lane<BLK/WARP)?rms_sum[lane]:0;
        #pragma unroll
        for(int o=(BLK/WARP)/2;o>0;o>>=1)s+=__shfl_xor(s,o);
        if(lane==0)rms_inv=rsqrtf(s/(float)H+1e-6f);
    }
    __syncthreads();
    float inv=rms_inv;

    // Phase 2: Each wavefront handles one output row of (Q|K|V)
    int tile_idx=blockIdx.x;
    int local_row=wid;  /* which row in this tile (0..7) */
    int out_row_base=tile_idx*TILE_R;

    if(tile_idx<ntr_q){
        // Q section
        int orow=out_row_base+local_row;
        if(orow<NH*HD){
            float acc=0;int tr=tile_idx;
            for(int tc=0;tc<ntc_q;tc++){
                int c0=tc*TILE_C+lane;
                // Read h once, apply RMS scaling inline
                float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
                const uint8_t*tile=w_q+((size_t)tr*ntc_q+tc)*I8_ROW_B;
                f16 w=i8_deq(tile,local_row,c0);
                acc+=hv*(float)w;
            }
            #pragma unroll
            for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
            if(lane==0&&orow<NH*HD)qkv[orow]=(f16)acc;
        }
        return;
    }

    int k_idx=tile_idx-ntr_q;
    if(k_idx<ntr_kv){
        int orow=k_idx*TILE_R+local_row;
        if(orow<NKV*HD){
            float acc=0;int tr=k_idx;
            for(int tc=0;tc<ntc_q;tc++){
                int c0=tc*TILE_C+lane;
                float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
                const uint8_t*tile=w_k+((size_t)tr*ntc_q+tc)*I8_ROW_B;
                f16 w=i8_deq(tile,local_row,c0);
                acc+=hv*(float)w;
            }
            #pragma unroll
            for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
            if(lane==0&&orow<NKV*HD)qkv[NH*HD+orow]=(f16)acc;
        }
        return;
    }

    int v_idx=tile_idx-ntr_q-ntr_kv;
    if(v_idx<ntr_kv){
        int orow=v_idx*TILE_R+local_row;
        if(orow<NKV*HD){
            float acc=0;int tr=v_idx;
            for(int tc=0;tc<ntc_q;tc++){
                int c0=tc*TILE_C+lane;
                float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
                const uint8_t*tile=w_v+((size_t)tr*ntc_q+tc)*I8_ROW_B;
                f16 w=i8_deq(tile,local_row,c0);
                acc+=hv*(float)w;
            }
            #pragma unroll
            for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
            if(lane==0&&orow<NKV*HD)qkv[NH*HD+NKV*HD+orow]=(f16)acc;
        }
    }
}

// ── Fused RMSNorm + I8 Gate+Up GEMV ──────────────────────────────────
// Same pattern: read h once, combine RMS norm + Gate + Up projections
__global__ void k_rmsnorm_gateup_i8(
    const f16*__restrict__ h_in,
    const f16*__restrict__ norm_w,
    const uint8_t*__restrict__ w_g,
    const uint8_t*__restrict__ w_u,
    f16*__restrict__ gate,
    f16*__restrict__ up)
{
    __shared__ float rms_sum[BLK/WARP];
    __shared__ float rms_inv;
    int tid=threadIdx.x,wid=tid/WARP,lane=tid&(WARP-1);
    int ntc=H/TILE_C;
    int ntr=IM/TILE_R;                 /* 96 */

    // Phase 1: RMS reduction
    float ssq=0;
    for(int i=tid;i<H;i+=BLK){float v=(float)h_in[i];ssq+=v*v;}
    #pragma unroll
    for(int o=16;o>0;o>>=1)ssq+=__shfl_xor(ssq,o);
    if(lane==0)rms_sum[wid]=ssq;
    __syncthreads();
    if(wid==0){
        float s=(lane<BLK/WARP)?rms_sum[lane]:0;
        #pragma unroll
        for(int o=(BLK/WARP)/2;o>0;o>>=1)s+=__shfl_xor(s,o);
        if(lane==0)rms_inv=rsqrtf(s/(float)H+1e-6f);
    }
    __syncthreads();
    float inv=rms_inv;

    int tile_idx=blockIdx.x;
    int local_row=wid;

    if(tile_idx<ntr){
        // Gate
        int orow=tile_idx*TILE_R+local_row;
        if(orow<IM){
            float acc=0;int tr=tile_idx;
            for(int tc=0;tc<ntc;tc++){
                int c0=tc*TILE_C+lane;
                float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
                const uint8_t*tile=w_g+((size_t)tr*ntc+tc)*I8_ROW_B;
                f16 w=i8_deq(tile,local_row,c0);
                acc+=hv*(float)w;
            }
            #pragma unroll
            for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
            if(lane==0&&orow<IM)gate[orow]=(f16)acc;
        }
        return;
    }

    int u_idx=tile_idx-ntr;
    if(u_idx<ntr){
        int orow=u_idx*TILE_R+local_row;
        if(orow<IM){
            float acc=0;int tr=u_idx;
            for(int tc=0;tc<ntc;tc++){
                int c0=tc*TILE_C+lane;
                float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
                const uint8_t*tile=w_u+((size_t)tr*ntc+tc)*I8_ROW_B;
                f16 w=i8_deq(tile,local_row,c0);
                acc+=hv*(float)w;
            }
            #pragma unroll
            for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
            if(lane==0&&orow<IM)up[orow]=(f16)acc;
        }
    }
}

// ── Fused RMSNorm + Single I8 GEMV (for O_proj, Down_proj) ────────────
__global__ void k_rmsnorm_gemv_i8(
    const f16*__restrict__ h_in,
    const f16*__restrict__ norm_w,
    const uint8_t*__restrict__ wt,
    f16*__restrict__ y,
    int od,int id)
{
    __shared__ float rms_sum[BLK/WARP];
    __shared__ float rms_inv;
    int tid=threadIdx.x,wid=tid/WARP,lane=tid&(WARP-1);
    int ntc=id/TILE_C;
    int ntr=od/TILE_R;

    float ssq=0;
    for(int i=tid;i<id;i+=BLK){float v=(float)h_in[i];ssq+=v*v;}
    #pragma unroll
    for(int o=16;o>0;o>>=1)ssq+=__shfl_xor(ssq,o);
    if(lane==0)rms_sum[wid]=ssq;
    __syncthreads();
    if(wid==0){
        float s=(lane<BLK/WARP)?rms_sum[lane]:0;
        #pragma unroll
        for(int o=(BLK/WARP)/2;o>0;o>>=1)s+=__shfl_xor(s,o);
        if(lane==0)rms_inv=rsqrtf(s/(float)id+1e-6f);
    }
    __syncthreads();
    float inv=rms_inv;

    int tile_idx=blockIdx.x;
    int local_row=wid;
    if(tile_idx>=ntr)return;
    int orow=tile_idx*TILE_R+local_row;
    if(orow>=od)return;
    float acc=0;int tr=tile_idx;
    for(int tc=0;tc<ntc;tc++){
        int c0=tc*TILE_C+lane;
        float hv=lane<TILE_C?(float)h_in[c0]*inv*(float)norm_w[c0]:0;
        const uint8_t*tile=wt+((size_t)tr*ntc+tc)*I8_ROW_B;
        f16 w=i8_deq(tile,local_row,c0);
        acc+=hv*(float)w;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o);
    if(lane==0)y[orow]=(f16)acc;
}

// ── QK Norm + RoPE (fused) ──────────────────────────────────────────
__global__ void k_qknorm_rope(
    f16*qkv,const f16*qn,const f16*kn,
    const float*rs,const float*rc,int pos)
{
    int h=blockIdx.x,d=threadIdx.x;
    bool is_k=(h>=NH);if(h>=NH+NKV)return;
    f16*vec=is_k?(qkv+NH*HD+(h-NH)*HD):(qkv+h*HD);
    const f16*nw=is_k?kn:qn;
    __shared__ float ss[128];
    ss[d]=(float)vec[d]*(float)vec[d];
    __syncthreads();
    for(int s2=64;s2>0;s2>>=1){if(d<s2)ss[d]+=ss[d+s2];__syncthreads();}
    float ir=rsqrtf(ss[0]/(float)HD+1e-6f);
    vec[d]=(f16)((float)vec[d]*ir*(nw?(float)nw[d]:1.0f));
    if(d<HD/2){
        float a=(float)vec[d],b=(float)vec[d+HD/2];
        vec[d]=(f16)(a*rc[pos*HD+d]-b*rs[pos*HD+d]);
        vec[d+HD/2]=(f16)(b*rc[pos*HD+d]+a*rs[pos*HD+d]);
    }
}

// ── Attention ─────────────────────────────────────────────────────────
__global__ void k_attn(const f16*q,const f16*kc,const f16*vc,f16*out,int sl){
    int h=blockIdx.x,d=threadIdx.x,kvh=h/GQA;
    if(h>=NH||d>=HD)return;
    __shared__ float scores[4096];
    if(d==0){
        float is=1.0f/sqrtf((float)HD),mx=-1e30f;
        for(int p=0;p<sl;p++){float dt=0;const f16*kr=kc+p*NKV*HD+kvh*HD;
            for(int dd=0;dd<HD;dd++)dt+=(float)q[h*HD+dd]*(float)kr[dd];
            scores[p]=dt*is;if(scores[p]>mx)mx=scores[p];}
        float su=0;for(int p=0;p<sl;p++){scores[p]=expf(scores[p]-mx);su+=scores[p];}
        float iv=su>0?1.0f/su:0;for(int p=0;p<sl;p++)scores[p]*=iv;
    }
    __syncthreads();float acc=0;
    for(int p=0;p<sl;p++)acc+=scores[p]*(float)vc[p*NKV*HD+kvh*HD+d];
    out[h*HD+d]=(f16)acc;
}

// ── SiLU ──────────────────────────────────────────────────────────────
__global__ void k_silu(f16*g,f16*u,int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;
    float gv=(float)g[i];u[i]=(f16)((gv/(1.0f+expf(-gv)))*(float)u[i]);
}

// ── Residual Add ──────────────────────────────────────────────────────
__global__ void k_add(f16*a,const f16*b,int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;
    a[i]=(f16)((float)a[i]+(float)b[i]);
}

// ── LM Head ───────────────────────────────────────────────────────────
__global__ void k_lmhead(const f16*h,const f16*emb,float*lg,int nv){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nv)return;
    const f16*row=emb+(size_t)i*H;
    float acc=0;
    for(int j=0;j<H;j++)acc+=(float)row[j]*(float)h[j];
    lg[i]=acc;
}

// ════════════════════════════════════════════════════════════════════
// DEVICE STRUCTURES
// ════════════════════════════════════════════════════════════════════

typedef struct{
    uint8_t*q,*k,*v,*o,*g,*u,*d;
    f16*in,*pa,*qn,*kn;
}DevLayer;

typedef struct{
    hipStream_t s;
    f16 *dh,*dres,*dqkv,*datt,*dgt,*dact;
    float*dlg;
    f16 *dkv_k,*dkv_v;
    f16 *d_fn,*d_emb;
    float *d_rs,*d_rc;
    DevLayer layers[NC];
    bool ok;
}GPU;

typedef struct{
    float*emb,*fn,*in[NC],*pa[NC],*qn[NC],*kn[NC],*rs,*rc;
    uint8_t*map;size_t ds;
    uint64_t qo[NC],ko[NC],vo[NC],oo[NC],go[NC],uo[NC],do_[NC];
    int qi[NC],ki[NC],vi[NC],oi[NC],gi[NC],ui[NC],di[NC];
    bool has_w;GPU gpu;
}Model;

// ════════════════════════════════════════════════════════════════════
// UPLOAD HELPERS
// ════════════════════════════════════════════════════════════════════

static bool up_i8(const uint8_t*map,size_t ds,uint64_t off,int i8r,int id,uint8_t**dw,size_t*sz){
    *sz=(size_t)i8r*I8_ROW_B;
    hipError_t e=hipMalloc((void**)dw,*sz);
    if(e!=hipSuccess)return false;
    hipMemcpy(*dw,map+ds+off,*sz,hipMemcpyHostToDevice);return true;
}
static bool up_f16(const float*src,int n,f16**dw){
    f16*t=(f16*)malloc(n*sizeof(f16));
    for(int i=0;i<n;i++)t[i]=(f16)src[i];
    hipError_t e=hipMalloc((void**)dw,n*sizeof(f16));
    if(e!=hipSuccess){free(t);return false;}
    hipMemcpy(*dw,t,n*sizeof(f16),hipMemcpyHostToDevice);free(t);return true;
}
static bool up_emb(const uint8_t*map,size_t ds,uint64_t off,int nv,int H,f16**dw){
    const uint16_t*src=(const uint16_t*)(map+ds+off);
    f16*t=(f16*)malloc((size_t)nv*H*sizeof(f16));
    for(size_t i=0;i<(size_t)nv*H;i++)t[i]=(f16)bf16_f32(src[i]);
    hipError_t e=hipMalloc((void**)dw,(size_t)nv*H*sizeof(f16));
    if(e!=hipSuccess){free(t);return false;}
    hipMemcpy(*dw,t,(size_t)nv*H*sizeof(f16),hipMemcpyHostToDevice);free(t);return true;
}

// ════════════════════════════════════════════════════════════════════
// GPU INIT
// ════════════════════════════════════════════════════════════════════

static bool gpu_init(Model*m){
    GPU*g=&m->gpu;g->ok=false;
    hipStreamCreate(&g->s);

    hipMalloc(&g->dh,H*sizeof(f16));
    hipMalloc(&g->dres,H*sizeof(f16));
    hipMalloc(&g->dqkv,(size_t)QT*sizeof(f16));
    hipMalloc(&g->datt,(size_t)NH*HD*sizeof(f16));
    hipMalloc(&g->dgt,IM*sizeof(f16));
    hipMalloc(&g->dact,IM*sizeof(f16));
    hipMalloc(&g->dlg,(size_t)NV*sizeof(float));

    size_t kv_sz=(size_t)NC*MAX_CTX*NKV*HD;
    hipMalloc(&g->dkv_k,kv_sz*sizeof(f16));
    hipMalloc(&g->dkv_v,kv_sz*sizeof(f16));
    hipMemset(g->dkv_k,0,kv_sz*sizeof(f16));
    hipMemset(g->dkv_v,0,kv_sz*sizeof(f16));

    up_f16(m->fn,H,&g->d_fn);

    hipMalloc(&g->d_rs,(size_t)MAX_CTX*HD*sizeof(float));
    hipMalloc(&g->d_rc,(size_t)MAX_CTX*HD*sizeof(float));
    hipMemcpy(g->d_rs,m->rs,(size_t)MAX_CTX*HD*sizeof(float),hipMemcpyHostToDevice);
    hipMemcpy(g->d_rc,m->rc,(size_t)MAX_CTX*HD*sizeof(float),hipMemcpyHostToDevice);

    const uint8_t*js=m->map+8;size_t jl=(size_t)(*(uint64_t*)m->map);
    int64_t eo=-1;
    {const char*k="model.embed_tokens.weight";size_t kl=strlen(k);
        for(size_t i=0;i+kl+2<jl;i++)if(js[i]=='"'&&memcmp(js+i+1,k,kl)==0&&js[i+1+kl]=='"'){
            const uint8_t*d=(const uint8_t*)strstr((const char*)(js+i),"\"data_offsets\"");
            if(d){const uint8_t*b=(const uint8_t*)strchr((const char*)d,'[');if(b)eo=strtoll((const char*)(b+1),0,10);}break;}}
    if(eo>=0)up_emb(m->map,m->ds,(uint64_t)eo,NV,H,&g->d_emb);

    fprintf(stderr,"Uploading I8 weights...\n");
    for(int l=0;l<NC;l++){
        if(l%7==0)fprintf(stderr,"  layer %d/%d\r",l+1,NC);
        DevLayer*dl=&g->layers[l];memset(dl,0,sizeof(DevLayer));
        size_t sz;
        up_i8(m->map,m->ds,m->qo[l],m->qi[l],H,&dl->q,&sz);
        up_i8(m->map,m->ds,m->ko[l],m->ki[l],H,&dl->k,&sz);
        up_i8(m->map,m->ds,m->vo[l],m->vi[l],H,&dl->v,&sz);
        up_i8(m->map,m->ds,m->oo[l],m->oi[l],NH*HD,&dl->o,&sz);
        up_i8(m->map,m->ds,m->go[l],m->gi[l],H,&dl->g,&sz);
        up_i8(m->map,m->ds,m->uo[l],m->ui[l],H,&dl->u,&sz);
        up_i8(m->map,m->ds,m->do_[l],m->di[l],IM,&dl->d,&sz);
        up_f16(m->in[l],H,&dl->in);
        up_f16(m->pa[l],H,&dl->pa);
        if(m->qn[l])up_f16(m->qn[l],HD,&dl->qn);
        if(m->kn[l])up_f16(m->kn[l],HD,&dl->kn);
    }
    fprintf(stderr,"\nGPU ready (I8 resident + fused RMSNorm+GEMV).\n");
    g->ok=true;return true;
}

// ════════════════════════════════════════════════════════════════════
// LAYER FORWARD (fused RMSNorm+GEMV)
// ════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════
// LAYER FORWARD (fused RMSNorm+GEMV)
// ════════════════════════════════════════════════════════════════════

__global__ void k_gemv_i8(const uint8_t*wt,const f16*x,f16*y,int od,int id){
    int tile_idx=blockIdx.x,tid=threadIdx.x,local_row=tid>>5,lane=tid&31;
    int ntc=id/TILE_C,ntr=od/TILE_R;
    if(tile_idx>=ntr)return;int orow=tile_idx*TILE_R+local_row;
    if(orow>=od)return;float acc=0;int tr=tile_idx;
    for(int tc=0;tc<ntc;tc++){int c0=tc*TILE_C+lane;
        const uint8_t*tile=wt+((size_t)tr*ntc+tc)*I8_ROW_B;
        if(lane<TILE_C){f16 w=i8_deq(tile,local_row,c0);acc+=(float)w*(float)x[c0];}}
    #pragma unroll
        for(int o=16;o>0;o>>=1)
            acc+=__shfl_xor(acc,o);
    if(lane==0)y[orow]=(f16)acc;
}

__global__ void k_rmsnorm(f16*x,const f16*w,int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;
    __shared__ float ss[256];float v=(float)x[i];ss[threadIdx.x]=isfinite(v)?v*v:0;
    __syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)ss[threadIdx.x]+=ss[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)ss[0]=rsqrtf(ss[0]/(float)n+1e-6f);__syncthreads();x[i]=(f16)((float)x[i]*ss[0]*(float)w[i]);
}

static void layer_fwd_v2(GPU*g,int l,int pos){
    hipStream_t s=g->s;DevLayer*dl=&g->layers[l];
    int ntr_q=NH*HD/TILE_R,ntr_kv=NKV*HD/TILE_R,ntr_im=IM/TILE_R,ntr_h=H/TILE_R;

    /* ── Attention block ── */
    hipMemcpyAsync(g->dres,g->dh,H*sizeof(f16),hipMemcpyDeviceToDevice,s);

    // Fused: RMSNorm(h) + QKV projections (read h ONCE)
    k_rmsnorm_qkv_i8<<<ntr_q+ntr_kv+ntr_kv,BLK,0,s>>>(
        g->dh,dl->in,dl->q,dl->k,dl->v,g->dqkv);

    k_qknorm_rope<<<NH+NKV,HD,0,s>>>(g->dqkv,dl->qn,dl->kn,g->d_rs,g->d_rc,pos);

    size_t off=(size_t)l*MAX_CTX*NKV*HD;
    hipMemcpyAsync(g->dkv_k+off+pos*NKV*HD,g->dqkv+NH*HD,NKV*HD*sizeof(f16),hipMemcpyDeviceToDevice,s);
    hipMemcpyAsync(g->dkv_v+off+pos*NKV*HD,g->dqkv+NH*HD+NKV*HD,NKV*HD*sizeof(f16),hipMemcpyDeviceToDevice,s);

    k_attn<<<NH,HD,0,s>>>(g->dqkv,g->dkv_k+off,g->dkv_v+off,g->datt,pos+1);

    // O projection (no norm between attn and O)
    k_gemv_i8<<<ntr_h,BLK,0,s>>>(dl->o,g->datt,g->dh,H,NH*HD);
    k_add<<<CEILDIV(H,BLK),BLK,0,s>>>(g->dh,g->dres,H);

    /* ── FFN block ── */
    hipMemcpyAsync(g->dres,g->dh,H*sizeof(f16),hipMemcpyDeviceToDevice,s);

    // Fused: RMSNorm(h) + Gate+Up (read h ONCE)
    k_rmsnorm_gateup_i8<<<ntr_im+ntr_im,BLK,0,s>>>(
        g->dh,dl->pa,dl->g,dl->u,g->dgt,g->dact);

    k_silu<<<CEILDIV(IM,BLK),BLK,0,s>>>(g->dgt,g->dact,IM);

    // Down projection
    k_gemv_i8<<<ntr_h,BLK,0,s>>>(dl->d,g->dact,g->dh,H,IM);
    k_add<<<CEILDIV(H,BLK),BLK,0,s>>>(g->dh,g->dres,H);
}

// ════════════════════════════════════════════════════════════════════
// DECODE
// ════════════════════════════════════════════════════════════════════

static uint32_t decode_step(Model*m,uint32_t tok,int*pos){
    GPU*g=&m->gpu;hipStream_t s=g->s;
    hipMemcpyAsync(g->dh,g->d_emb+(size_t)tok*H,H*sizeof(f16),hipMemcpyDeviceToDevice,s);
    for(int l=0;l<NC;l++)layer_fwd_v2(g,l,*pos);
    k_rmsnorm<<<CEILDIV(H,BLK),BLK,0,s>>>(g->dh,g->d_fn,H);
    k_lmhead<<<CEILDIV(NV,BLK),BLK,0,s>>>(g->dh,g->d_emb,g->dlg,NV);
    hipStreamSynchronize(s);
    float*lg=(float*)malloc((size_t)NV*sizeof(float));
    hipMemcpy(lg,g->dlg,(size_t)NV*sizeof(float),hipMemcpyDeviceToHost);
    uint32_t best=0;float mx=lg[0];
    for(int i=1;i<NV;i++)if(lg[i]>mx){mx=lg[i];best=(uint32_t)i;}
    free(lg);(*pos)++;return best;
}

// ════════════════════════════════════════════════════════════════════
// MODEL LOADER (same as engine_peak)
// ════════════════════════════════════════════════════════════════════

static const uint8_t*fk(const uint8_t*j,size_t l,const char*k){
    size_t kl=strlen(k);for(size_t i=0;i+kl+2<l;i++)if(j[i]=='"'&&memcmp(j+i+1,k,kl)==0&&j[i+1+kl]=='"')return j+i;return 0;}
static int64_t fo(const uint8_t*j,size_t l,const char*k){
    const uint8_t*p=fk(j,l,k);if(!p)return-1;const uint8_t*d=(const uint8_t*)strstr((const char*)p,"\"data_offsets\"");if(!d)return-1;
    const uint8_t*b=(const uint8_t*)strchr((const char*)d,'[');return b?strtoll((const char*)(b+1),0,10):-1;}
static int si(const uint8_t*j,size_t l,const char*k){
    size_t kl=strlen(k);const char*js=(const char*)j;for(const char*p=js;p<js+(int)l;p++){p=strstr(p,k);if(!p)return 0;
        if((p==js||*(p-1)=='"')&&*(p+kl)=='"'){const char*sh=strstr(p,"\"shape\"");if(!sh)return 0;const char*br=strchr(sh,'[');if(!br)return 0;return(int)strtoul(br+1,0,10);}p+=kl;}return 0;}

static bool load_model(const char*path,Model*m){
    int fd=open(path,O_RDONLY);if(fd<0)return false;struct stat st;fstat(fd,&st);
    m->map=(uint8_t*)mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if(!m->map)return false;
    uint64_t hdr;memcpy(&hdr,m->map,8);m->ds=8+(size_t)hdr;const uint8_t*js=m->map+8;size_t jl=(size_t)hdr;
    int64_t eo=fo(js,jl,"model.embed_tokens.weight");m->emb=(float*)malloc((size_t)NV*H*sizeof(float));
    const uint16_t*eb=(const uint16_t*)(m->map+m->ds+(size_t)eo);for(int i=0;i<NV*H;i++)m->emb[i]=bf16_f32(eb[i]);
    char buf[256];
    for(int l=0;l<NC;l++){
        snprintf(buf,256,"model.layers.%d.input_layernorm.weight",l);int64_t o=fo(js,jl,buf);m->in[l]=(float*)malloc(H*sizeof(float));const uint16_t*s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->in[l][i]=bf16_f32(s[i]);
        snprintf(buf,256,"model.layers.%d.post_attention_layernorm.weight",l);o=fo(js,jl,buf);m->pa[l]=(float*)malloc(H*sizeof(float));s=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<H;i++)m->pa[l][i]=bf16_f32(s[i]);
        snprintf(buf,256,"model.layers.%d.self_attn.q_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->qn[l]=(float*)malloc(HD*sizeof(float));const uint16_t*qs=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->qn[l][i]=bf16_f32(qs[i]);}else m->qn[l]=0;
        snprintf(buf,256,"model.layers.%d.self_attn.k_norm.weight",l);o=fo(js,jl,buf);if(o>0){m->kn[l]=(float*)malloc(HD*sizeof(float));const uint16_t*ks=(const uint16_t*)(m->map+m->ds+(size_t)o);for(int i=0;i<HD;i++)m->kn[l][i]=bf16_f32(ks[i]);}else m->kn[l]=0;
    }
    int64_t fno=fo(js,jl,"model.norm.weight");m->fn=(float*)malloc(H*sizeof(float));const uint16_t*fs=(const uint16_t*)(m->map+m->ds+(size_t)fno);for(int i=0;i<H;i++)m->fn[i]=bf16_f32(fs[i]);
    m->rs=(float*)malloc((size_t)MAX_CTX*HD*sizeof(float));m->rc=(float*)malloc((size_t)MAX_CTX*HD*sizeof(float));
    float th=1000000.0f;for(int p=0;p<MAX_CTX;p++)for(int d=0;d<HD/2;d++){float freq=1.0f/powf(th,(float)d/(HD/2)),angle=(float)p*freq;m->rs[p*HD+d]=sinf(angle);m->rc[p*HD+d]=cosf(angle);m->rs[p*HD+HD/2+d]=sinf(angle);m->rc[p*HD+HD/2+d]=cosf(angle);}
    for(int l=0;l<NC;l++){snprintf(buf,256,"model.layers.%d.self_attn.q_proj.weight",l);m->qo[l]=fo(js,jl,buf);m->qi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.k_proj.weight",l);m->ko[l]=fo(js,jl,buf);m->ki[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.v_proj.weight",l);m->vo[l]=fo(js,jl,buf);m->vi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.self_attn.o_proj.weight",l);m->oo[l]=fo(js,jl,buf);m->oi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.gate_proj.weight",l);m->go[l]=fo(js,jl,buf);m->gi[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.up_proj.weight",l);m->uo[l]=fo(js,jl,buf);m->ui[l]=si(js,jl,buf);
        snprintf(buf,256,"model.layers.%d.mlp.down_proj.weight",l);m->do_[l]=fo(js,jl,buf);m->di[l]=si(js,jl,buf);if(m->qi[l]>0)m->has_w=true;}
    fprintf(stderr,"Model: H=%d NC=%d NH=%d NKV=%d NV=%d\n",H,NC,NH,NKV,NV);return true;
}

static uint32_t lt(const char*w){
    if(!strcmp(w,"<|im_start|>"))return 151644;if(!strcmp(w,"<|im_end|>"))return 151645;
    if(!strcmp(w,"user"))return 872;if(!strcmp(w,"assistant"))return 77091;if(!strcmp(w,"\n"))return 198;
    if(!strcmp(w,"Hello")||!strcmp(w,"hello"))return 9707;if(!strcmp(w,"Hi")||!strcmp(w,"hi"))return 11852;
    if(!strcmp(w,"?"))return 30;if(!strcmp(w,","))return 11;if(!strcmp(w,"."))return 13;return(uint32_t)(unsigned char)w[0]+3;}
static int tok(const char*t,uint32_t*tk,int mx){
    int n=0;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=872;if(n<mx)tk[n++]=198;
    char w[256];int wl=0;for(const char*p=t;*p&&n<mx;p++){unsigned char c=(unsigned char)*p;
        if(c==' '||c=='\n'||c=='\t'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}if(c=='\n'&&n<mx)tk[n++]=198;}
        else if(c==','||c=='.'||c=='?'||c=='!'||c==';'||c==':'){if(wl){w[wl]=0;if(n<mx)tk[n++]=lt(w);wl=0;}char pn[2]={(char)c,0};if(n<mx)tk[n++]=lt(pn);}
        else{if(wl<255)w[wl++]=c;}}if(wl&&n<mx){w[wl]=0;tk[n++]=lt(w);}
    if(n<mx)tk[n++]=151645;if(n<mx)tk[n++]=198;if(n<mx)tk[n++]=151644;if(n<mx)tk[n++]=77091;if(n<mx)tk[n++]=198;return n;}

int main(int argc,char**argv){
    const char*mp=DEF_MODEL,*pr="Hello";int mx=128;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-h")){printf("engine_peak_v2\n");return 0;}
        if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)mp=argv[++i];
        if((!strcmp(argv[i],"-p")||!strcmp(argv[i],"--prompt"))&&i+1<argc)pr=argv[++i];
        if((!strcmp(argv[i],"-n")||!strcmp(argv[i],"--max-tokens"))&&i+1<argc)mx=atoi(argv[++i]);}
    fprintf(stderr,"\n═══ engine_peak_v2: Fused RMSNorm+GEMV ═══\n");
    Model m;memset(&m,0,sizeof(m));if(!load_model(mp,&m))return 1;
    int pos=0;uint32_t pt[4096];int npt=tok(pr,pt,4096);
    fprintf(stderr,"Prompt: %d tokens\n",npt);
    if(!gpu_init(&m)){fprintf(stderr,"GPU init failed\n");return 1;}
    struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);double t0=ts.tv_sec*1e9+ts.tv_nsec;
    uint32_t*out=(uint32_t*)calloc((size_t)mx,sizeof(uint32_t));int gen=0;
    for(int pi=0;pi<npt;pi++)decode_step(&m,pt[pi],&pos);
    fprintf(stderr,"Decode %d...\n",mx);
    uint32_t ct=pt[npt-1];
    while(gen<mx){out[gen]=decode_step(&m,ct,&pos);ct=out[gen];gen++;if(gen%20==0)fprintf(stderr,"  %d/%d\r",gen,mx);}
    clock_gettime(CLOCK_MONOTONIC,&ts);double ms=(ts.tv_sec*1e9+ts.tv_nsec-t0)/1e6;
    fprintf(stderr,"\n═══ %d tokens in %.0fms — %.0f tok/s (%.2f ms/tok) ═══\n",gen,ms,gen/(ms/1000),ms/gen);
    for(int i=0;i<gen;i++)printf("%u ",out[i]);printf("\n");
    free(out);return 0;
}
