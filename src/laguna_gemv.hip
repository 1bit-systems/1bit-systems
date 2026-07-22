// laguna_gemv.hip — Q4NX GEMV with shared memory reduction
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

constexpr int TR=32, TC=256, GS=32, NG=TC/GS, RB=NG*4+TC/2, TB=TR*RB;

__device__ inline float bf16_f32(uint16_t bf) { return __uint_as_float((uint32_t)bf<<16); }

__global__ void q4nx_gemv(const float* x, const uint8_t* w, float* y,
                           int K, int M, int n_tr, int n_tc) {
    extern __shared__ float tile[];
    
    // Zero tile
    for (int i = threadIdx.x; i < TR * TC; i += blockDim.x) tile[i] = 0.0f;
    __syncthreads();
    
    int trr = blockIdx.y, tcc = blockIdx.x, r = threadIdx.x;
    int rs = trr * TR, cs = tcc * TC;
    int r_eff = min(rs + TR, K) - rs;
    int c_eff = min(cs + TC, M) - cs;
    
    // Dequant row r
    if (r < r_eff) {
        float xr = x[rs + r];
        if (xr != 0.0f) {
            const uint8_t* rp = w + (((size_t)trr * n_tc + tcc) * TB) + r * RB;
            for (int g = 0; g < NG; g++) {
                float sc = bf16_f32(*(const uint16_t*)(rp + g*2));
                float zp = bf16_f32(*(const uint16_t*)(rp + NG*2 + g*2));
                const uint8_t* pk = rp + NG*4;
                for (int i = 0; i < GS; i++) {
                    int bi = g*GS/2 + i/2;
                    int nl = (i&1) ? (pk[bi]>>4) : (pk[bi]&0x0F);
                    float v = (float)nl * sc + zp;
                    tile[r * TC + g*GS + i] = (v==v&&v<1e10f&&v>-1e10f) ? xr * v : 0.0f;
                }
            }
        }
    }
    __syncthreads();
    
    // Reduce across rows: each thread handles columns c = r, r+TR, r+2*TR, ...
    for (int c = r; c < c_eff; c += TR) {
        float sum = 0.0f;
        for (int rr = 0; rr < r_eff; rr++) sum += tile[rr * TC + c];
        if (sum != 0.0f) atomicAdd(&y[cs + c], sum);
    }
}

extern "C" void launch_q4nx_gemv(const float* x, const uint8_t* w, float* y,
                                  int K, int M, hipStream_t stream) {
    int n_tr = (K + TR - 1) / TR, n_tc = (M + TC - 1) / TC;
    size_t sh = TR * TC * sizeof(float);
    hipLaunchKernelGGL(q4nx_gemv, dim3(n_tc, n_tr), TR, sh, stream, x, w, y, K, M, n_tr, n_tc);
}

#ifndef __HIP_DEVICE_COMPILE__
extern "C" int test_q4nx_gemv_hip() {
    float x_h[32]; for(int i=0;i<32;i++) x_h[i]=(i==4)?1.0f:0.0f;
    float *x_d,*y_d; uint8_t*w_d;
    hipMalloc(&x_d,32*4); hipMalloc(&y_d,256*4); hipMalloc(&w_d,TB);
    uint8_t th[TB]; memset(th,0,TB);
    for(int r=0;r<TR;r++){for(int g=0;g<NG;g++){((uint16_t*)(th+r*RB))[g]=0x3F80;}
        uint8_t*pk=th+r*RB+NG*4;for(int i=0;i<256;i++){int bi=i/2;if(i&1)pk[bi]|=(1<<4);else pk[bi]|=1;}}
    hipMemcpy(w_d,th,TB,hipMemcpyHostToDevice);
    hipMemcpy(x_d,x_h,32*4,hipMemcpyHostToDevice);
    hipMemset(y_d,0,256*4);
    size_t sh=TR*TC*4;
    hipLaunchKernelGGL(q4nx_gemv,dim3(1,1),TR,sh,0,x_d,w_d,y_d,32,256,1,1);
    hipDeviceSynchronize();
    float y_h[256]; hipMemcpy(y_h,y_d,256*4,hipMemcpyDeviceToHost);
    bool ok=true; for(int i=0;i<256;i++)if(fabs(y_h[i]-1.0f)>1e-4f){ok=false;printf("y[%d]=%f\n",i,y_h[i]);break;}
    hipFree(x_d);hipFree(y_d);hipFree(w_d);
    printf("Q4NX GEMV: %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}
#endif
