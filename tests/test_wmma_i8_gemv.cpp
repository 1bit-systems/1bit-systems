// Full pipeline test: Hadamard + INT8 quantize + sudot4 GEMV
#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

static void hadamard_cpu(const float* x, float* y, int K, int B) {
    int nb = K / B;
    for (int b = 0; b < nb; ++b) {
        float* blk = y + b*B;
        for (int i = 0; i < B; ++i) blk[i] = x[b*B + i];
        for (int s = 0; (1<<s) < B; ++s) {
            int d = 1<<s;
            for (int i = 0; i < B; ++i)
                if ((i&d)==0) { int j=i|d; float a=blk[i],b=blk[j]; blk[i]=a+b; blk[j]=a-b; }
        }
        float inv = 1.0f/std::sqrt((float)B);
        for (int i = 0; i < B; ++i) blk[i] *= inv;
    }
}

static float quant_i8(const float* x, int8_t* out, int K) {
    float m=0; for(int i=0;i<K;++i){float a=fabsf(x[i]);if(a>m)m=a;}
    float s=m/127.0f; if(s<1e-10f)s=1e-10f;
    for(int i=0;i<K;++i){float v=x[i]/s; v=fminf(fmaxf(roundf(v),-128.f),127.f); out[i]=(int8_t)v;}
    return s;
}

int main() {
    HIP_CHECK(hipSetDevice(0)); hipStream_t s; HIP_CHECK(hipStreamCreate(&s));
    const int M=256, K=2048, B=128;
    printf("Full pipeline: M=%d K=%d B=%d\n", M, K, B);

    srand(42);
    std::vector<float> hW(M*K), hX(K);
    for(auto& v:hW) v=((rand()%2000)-1000)/1000.0f;
    for(auto& v:hX) v=((rand()%2000)-1000)/1000.0f;

    // ── CPU: Hadamard → quant → GEMV ──
    std::vector<float> hXr(K); hadamard_cpu(hX.data(), hXr.data(), K, B);
    std::vector<int8_t> hXi8(K); float xs_cpu = quant_i8(hXr.data(), hXi8.data(), K);
    std::vector<int8_t> hWi8(M*K); std::vector<float> hSc(M);
    for (int m=0; m<M; ++m) {
        std::vector<float> row(K);
        hadamard_cpu(hW.data()+m*K, row.data(), K, B);
        float sc = quant_i8(row.data(), hWi8.data()+m*K, K);
        hSc[m] = sc;
        for (int k=0; k<K; ++k) { float v=row[k]/sc; v=fminf(fmaxf(roundf(v),-128.f),127.f); hWi8[m*K+k]=(int8_t)v; }
    }
    std::vector<float> hRef(M);
    for (int m=0; m<M; ++m) {
        int32_t d=0; for(int k=0;k<K;++k) d+=(int32_t)hWi8[m*K+k]*(int32_t)hXi8[k];
        hRef[m]=(float)d*xs_cpu*hSc[m];
    }

    // ── GPU ──
    _Float16 *dx, *dxr; int8_t *dxi, *dWi; float *dsdev, *dsc; __half *dy;
    HIP_CHECK(hipMalloc(&dx,K*2)); HIP_CHECK(hipMalloc(&dxr,K*2)); HIP_CHECK(hipMalloc(&dxi,K)); HIP_CHECK(hipMalloc(&dsdev,4));
    HIP_CHECK(hipMalloc(&dWi,(size_t)M*K)); HIP_CHECK(hipMalloc(&dsc,M*4)); HIP_CHECK(hipMalloc(&dy,M*2));
    std::vector<_Float16> hXf16(K); for(int i=0;i<K;++i)hXf16[i]=(_Float16)hX[i];
    HIP_CHECK(hipMemcpy(dx,hXf16.data(),K*2,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dWi,hWi8.data(),(size_t)M*K,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsc,hSc.data(),M*4,hipMemcpyHostToDevice));

    rcpp_hadamard_rotate_fp16(dx,dxr,K,s);
    rcpp_quantize_fp16_to_i8(dxr,dxi,dsdev,K,s);
    float xs_gpu; HIP_CHECK(hipMemcpy(&xs_gpu,dsdev,4,hipMemcpyDeviceToHost));
    rcpp_wmma_i8_gemv(dWi,dxi,xs_gpu,dsc,dy,M,K,s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::vector<__half> hY(M); HIP_CHECK(hipMemcpy(hY.data(),dy,M*2,hipMemcpyDeviceToHost));

    // ── Compare ──
    int pass=1; float mae=0,mre=0; int fails=0;
    for(int m=0;m<M;++m){
        float ref=hRef[m],gpu=(float)hY[m],ae=fabsf(ref-gpu),re=fabsf(ref)>1e-6f?ae/fabsf(ref):ae;
        if(ae>mae)mae=ae; if(re>mre)mre=re;
        if(ae>3.0f&&re>0.15f){if(fails<3)printf("FAIL row %d: ref=%.4f gpu=%.4f ae=%.4f\n",m,ref,gpu,ae); fails++;}
    }
    printf("scales: cpu=%.6f gpu=%.6f  mae=%.4f mre=%.4f fails=%d  %s\n",
           xs_cpu,xs_gpu,mae,mre,fails,fails==0?"PASS":"FAIL");

    HIP_CHECK(hipStreamDestroy(s));
    HIP_CHECK(hipFree(dx));HIP_CHECK(hipFree(dxr));HIP_CHECK(hipFree(dxi));HIP_CHECK(hipFree(dsdev));HIP_CHECK(hipFree(dWi));HIP_CHECK(hipFree(dsc));HIP_CHECK(hipFree(dy));
    return fails==0?0:1;
}
