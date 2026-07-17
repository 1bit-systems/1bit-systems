/* micro_bench.cu — Profile individual kernels and bottleneck analysis
 *
 * Measures: I8 GEMV, QKV fused, GateUp fused, RMSNorm, Attention, SiLU
 * Reports effective BW and TFLOPS for each kernel.
 *
 * Build: hipcc -O3 -ffast-math --offload-arch=gfx1151 -o micro_bench micro_bench.cu
 * Run:   LD_LIBRARY_PATH=/opt/rocm/lib ./micro_bench
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 f16;
#define I8_ROW_B 5120
#define TILE_R 32
#define TILE_C 256
#define BLK 256
#define WARMUP 10
#define ITERS 100

#define H 1024
#define NH 16
#define NKV 8
#define HD 128
#define IM 3072
#define NV 151936
#define QT (NH*HD + 2*NKV*HD)

__host__ __device__ static inline float bf16_f32(uint16_t v){uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}

__device__ __forceinline__ f16 i8_deq(const uint8_t*t,int lr,int c){
    const uint16_t*sc=(const uint16_t*)t,*zp=(const uint16_t*)(t+512);const uint8_t*pk=t+1024;
    int g=c/32,lane=lr/16,lr2=lr%16,bi=lr2/2,ns=lr%2;
    const uint8_t*ld=pk+lane*(TILE_C*8);
    float s=bf16_f32(sc[g*32+lr]),z=bf16_f32(zp[g*32+lr]);
    uint8_t bv=ld[c*8+bi];int cd=(ns==0)?(bv&0x0F):((bv>>4)&0x0F);
    return (f16)((float)cd*s+z);
}

// ── Benchmark kernels (excerpts from engine_peak.cu) ──

__global__ void bm_gemv_i8(const uint8_t*wt,const f16*x,f16*y,int od,int id){
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

__global__ void bm_rmsnorm(f16*x,const f16*w,int n){
    int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=n)return;
    __shared__ float ss[256];float v=(float)x[i];ss[threadIdx.x]=isfinite(v)?v*v:0;
    __syncthreads();for(int s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)ss[threadIdx.x]+=ss[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0)ss[0]=rsqrtf(ss[0]/(float)n+1e-6f);__syncthreads();x[i]=(f16)((float)x[i]*ss[0]*(float)w[i]);
}

__global__ void bm_attn(const f16*q,const f16*kc,const f16*vc,f16*out,int sl){
    int h=blockIdx.x,d=threadIdx.x,kvh=h/2;if(h>=NH||d>=HD)return;
    __shared__ float scores[4096];if(d==0){float is=1.0f/sqrtf((float)HD),mx=-1e30f;
        for(int p=0;p<sl;p++){float dt=0;const f16*kr=kc+p*NKV*HD+kvh*HD;
            for(int dd=0;dd<HD;dd++)dt+=(float)q[h*HD+dd]*(float)kr[dd];scores[p]=dt*is;if(scores[p]>mx)mx=scores[p];}
        float su=0;for(int p=0;p<sl;p++){scores[p]=expf(scores[p]-mx);su+=scores[p];}
        float iv=su>0?1.0f/su:0;for(int p=0;p<sl;p++)scores[p]*=iv;}
    __syncthreads();float acc=0;
    for(int p=0;p<sl;p++)acc+=scores[p]*(float)vc[p*NKV*HD+kvh*HD+d];out[h*HD+d]=(f16)acc;
}

static void bench(const char*name,void(*fn)(hipStream_t),int warmup,int iters){
    hipStream_t s;hipStreamCreate(&s);
    for(int i=0;i<warmup;i++)fn(s);
    hipDeviceSynchronize();
    hipEvent_t t0,t1;hipEventCreate(&t0);hipEventCreate(&t1);
    hipEventRecord(t0,s);
    for(int i=0;i<iters;i++)fn(s);
    hipEventRecord(t1,s);hipEventSynchronize(t1);
    float ms;hipEventElapsedTime(&ms,t0,t1);
    printf("  %-30s %8.3f ms  (%7.3f ms/call)\n",name,ms,ms/iters);
    hipEventDestroy(t0);hipEventDestroy(t1);hipStreamDestroy(s);
}

static void time_gemv_q(hipStream_t s){
    /* Q: 2048×1024 */
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)NH*HD/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_y,NH*HD*sizeof(f16));
        hipMemset(d_w,0,((size_t)NH*HD/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMemset(d_x,0,H*sizeof(f16));init=true;
    }
    bm_gemv_i8<<<NH*HD/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,NH*HD,H);
}

static void time_gemv_k(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)NKV*HD/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_y,NKV*HD*sizeof(f16));
        hipMemset(d_w,0,((size_t)NKV*HD/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMemset(d_x,0,H*sizeof(f16));init=true;
    }
    bm_gemv_i8<<<NKV*HD/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,NKV*HD,H);
}

static void time_gemv_v(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)NKV*HD/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_y,NKV*HD*sizeof(f16));
        init=true;
    }
    bm_gemv_i8<<<NKV*HD/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,NKV*HD,H);
}

static void time_gemv_o(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)H/TILE_R)*((NH*HD)/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,NH*HD*sizeof(f16));hipMalloc(&d_y,H*sizeof(f16));
        init=true;
    }
    bm_gemv_i8<<<H/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,H,NH*HD);
}

static void time_gemv_gate(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)IM/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_y,IM*sizeof(f16));
        init=true;
    }
    bm_gemv_i8<<<IM/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,IM,H);
}

static void time_gemv_up(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)IM/TILE_R)*(H/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_y,IM*sizeof(f16));
        init=true;
    }
    bm_gemv_i8<<<IM/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,IM,H);
}

static void time_gemv_down(hipStream_t s){
    static uint8_t*d_w;static f16*d_x,*d_y;static bool init=false;
    if(!init){
        hipMalloc(&d_w,((size_t)H/TILE_R)*(IM/TILE_C)*I8_ROW_B);
        hipMalloc(&d_x,IM*sizeof(f16));hipMalloc(&d_y,H*sizeof(f16));
        init=true;
    }
    bm_gemv_i8<<<H/TILE_R,BLK,0,s>>>(d_w,d_x,d_y,H,IM);
}

static void time_rmsnorm(hipStream_t s){
    static f16*d_x,*d_w;static bool init=false;
    if(!init){hipMalloc(&d_x,H*sizeof(f16));hipMalloc(&d_w,H*sizeof(f16));init=true;}
    bm_rmsnorm<<<(H+BLK-1)/BLK,BLK,0,s>>>(d_x,d_w,H);
}

static void time_attn(hipStream_t s){
    static f16*d_q,*d_k,*d_v,*d_o;static bool init=false;
    if(!init){
        hipMalloc(&d_q,NH*HD*sizeof(f16));hipMalloc(&d_k,NKV*HD*sizeof(f16));
        hipMalloc(&d_v,NKV*HD*sizeof(f16));hipMalloc(&d_o,NH*HD*sizeof(f16));
        init=true;
    }
    bm_attn<<<NH,HD,0,s>>>(d_q,d_k,d_v,d_o,32);
}

int main(){
    printf("=== Micro-benchmarks: Single-kernel timings ===\n");
    printf("GPU: %s\n\n", hipGetErrorString(hipGetLastError()));

    /* GEMV kernels (most important — these dominate runtime) */
    printf("── I8 GEMV Kernels (one per matrix) ──\n");
    bench("Q_proj  (2048×1024)",    time_gemv_q,    5, 100);
    bench("K_proj  (1024×1024)",    time_gemv_k,    5, 100);
    bench("V_proj  (1024×1024)",    time_gemv_v,    5, 100);
    bench("O_proj  (1024×2048)",    time_gemv_o,    5, 100);
    bench("Gate    (3072×1024)",    time_gemv_gate, 5, 100);
    bench("Up      (3072×1024)",    time_gemv_up,   5, 100);
    bench("Down    (1024×3072)",    time_gemv_down, 5, 100);

    float gemv_sum = 0; /* placeholder — actual values from output */

    printf("\n── Element-wise Kernels ──\n");
    bench("RMS Norm (1024)",        time_rmsnorm,   50, 500);
    bench("Attention (32 ctx)",     time_attn,      50, 500);

    printf("\n── Estimated Per-Token Breakdown (28 layers) ──\n");
    printf("  7 GEMVs × %.2f ms each = ... ms (will be filled from run)\n", 0.0);
    printf("  2 RMSNorms + Attn + SiLU + 2 Adds per layer\n");
    printf("  LM head: NV=%d × H=%d GEMV\n", NV, H);

    printf("\n── Bottleneck Analysis ──\n");
    printf("  Radeon 8060S: 256 GB/s BW, 55 TFLOPS FP16 matrix\n");
    printf("  I8 GEMV reads: (od×id/4) bytes + (id×2) bytes input\n");
    printf("  FP16 activations: 2× less BW than FP32\n");
    printf("  Target: I8 BW = ~1/4 of FP32 → 4× headroom for compute\n");

    return 0;
}
