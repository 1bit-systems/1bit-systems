// Phase 1 correctness: standalone HIP kernel vs CK kernel on the SAME packed
// weights. If both produce the same output (within FP16 rounding), the strip
// is valid and we can wire the standalone path into the C API.

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" void rcpp_standalone_launch     (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_lds (const void*, const void*, void*, int, int, int, void*);
extern "C" void rcpp_standalone_launch_wmma(const void*, const void*, void*, int, int, int, void*);

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP err %d %s:%d\n",_s,__FILE__,__LINE__); std::abort();}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d %s:%d\n",(int)_s,__FILE__,__LINE__); std::abort();}} while(0)

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 512;   // small default — naive kernel is slow
    if(argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

    printf("=== standalone (Phase 1 naive) vs CK — same packed weights ===\n");
    printf("Shape: M=%d N=%d K=%d\n", M, N, K);

    std::mt19937 rng(0x1b1fe4e4);  // "1bit fever" bytes
    std::uniform_real_distribution<float> rd(-0.25f, 0.25f);
    std::uniform_int_distribution<int>    rt(-1, 1);

    // Generate random FP16 A + ternary B, pack B via the C API.
    std::vector<_Float16> A((size_t)M * K);
    for(auto& v : A) v = (_Float16)rd(rng);

    std::vector<int8_t> B_ternary((size_t)K * N);
    for(auto& v : B_ternary) v = (int8_t)rt(rng);

    std::vector<int8_t> B_packed((size_t)K * N / 2);
    RC_OK(rcpp_ternary_pack_pk_i4(B_ternary.data(), B_packed.data(), K, N));

    // Device buffers
    _Float16* dA = nullptr;
    int8_t*   dB = nullptr;
    _Float16* dC_ck   = nullptr;
    _Float16* dC_std  = nullptr;
    _Float16* dC_lds  = nullptr;
    _Float16* dC_wmma = nullptr;
    HIP_OK(hipMalloc(&dA,       A.size() * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dB,       B_packed.size()));
    HIP_OK(hipMalloc(&dC_ck,    (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_std,   (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_lds,   (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_wmma,  (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMemcpy(dA, A.data(),        A.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B_packed.data(), B_packed.size(),             hipMemcpyHostToDevice));

    // Run CK path
    rcpp_ck_gemm_handle_t* h = nullptr;
    RC_OK(rcpp_ck_gemm_create(M, N, K, &h));
    RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr));
    HIP_OK(hipDeviceSynchronize());

    // Run standalone paths
    rcpp_standalone_launch     (dA, dB, dC_std,  M, N, K, nullptr);
    rcpp_standalone_launch_lds (dA, dB, dC_lds,  M, N, K, nullptr);
    rcpp_standalone_launch_wmma(dA, dB, dC_wmma, M, N, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    // Diff
    std::vector<_Float16> C_ck((size_t)M * N), C_std((size_t)M * N), C_lds((size_t)M * N), C_wmma((size_t)M * N);
    HIP_OK(hipMemcpy(C_ck.data(),   dC_ck,   C_ck.size()   * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_std.data(),  dC_std,  C_std.size()  * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_lds.data(),  dC_lds,  C_lds.size()  * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_wmma.data(), dC_wmma, C_wmma.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    auto diff = [&](const char* label, const std::vector<_Float16>& ref, const std::vector<_Float16>& got) {
        float max_abs = 0.0f;
        for(size_t i = 0; i < ref.size(); ++i) {
            float d = std::fabs((float)ref[i] - (float)got[i]);
            max_abs = std::max(max_abs, d);
        }
        printf("  %-22s  max abs = %.6f\n", label, max_abs);
        return max_abs;
    };

    printf("Diffs vs CK:\n");
    float max_abs      = diff("Phase 1 naive", C_ck, C_std);
    float max_abs_lds  = diff("Phase 2 LDS",   C_ck, C_lds);
    float max_abs_wmma = diff("Phase 3 WMMA",  C_ck, C_wmma);

    // Perf sanity (perf climbs across phases)
    const int runs = 20;
    hipEvent_t e0, e1; HIP_OK(hipEventCreate(&e0)); HIP_OK(hipEventCreate(&e1));
    double flops = 2.0 * (double)M * N * K;

    auto time_ms = [&](auto launch) -> double {
        // warmup
        for(int w = 0; w < 3; ++w) launch();
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipEventRecord(e0, nullptr));
        for(int r = 0; r < runs; ++r) launch();
        HIP_OK(hipEventRecord(e1, nullptr));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f; HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        return (double)ms / runs;
    };

    double ms_ck   = time_ms([&](){ RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC_ck, nullptr)); });
    double ms_std  = time_ms([&](){ rcpp_standalone_launch     (dA, dB, dC_std,  M, N, K, nullptr); });
    double ms_lds  = time_ms([&](){ rcpp_standalone_launch_lds (dA, dB, dC_lds,  M, N, K, nullptr); });
    double ms_wmma = time_ms([&](){ rcpp_standalone_launch_wmma(dA, dB, dC_wmma, M, N, K, nullptr); });

    printf("Perf:\n");
    printf("  %-22s  %.3f ms  %6.2f TFlops   (1.0x)\n",
           "CK reference",   ms_ck,   flops / (ms_ck   * 1e-3) / 1e12);
    printf("  %-22s  %.3f ms  %6.2f TFlops  (%.1fx vs CK)\n",
           "Phase 1 naive",  ms_std,  flops / (ms_std  * 1e-3) / 1e12, ms_std  / ms_ck);
    printf("  %-22s  %.3f ms  %6.2f TFlops  (%.1fx vs CK)\n",
           "Phase 2 LDS",    ms_lds,  flops / (ms_lds  * 1e-3) / 1e12, ms_lds  / ms_ck);
    printf("  %-22s  %.3f ms  %6.2f TFlops  (%.1fx vs CK)\n",
           "Phase 3 WMMA",   ms_wmma, flops / (ms_wmma * 1e-3) / 1e12, ms_wmma / ms_ck);

    const float pass_abs = 0.25f;
    const int pass = (max_abs < pass_abs) && (max_abs_lds < pass_abs) && (max_abs_wmma < pass_abs);
    printf("Verdict: %s (threshold max_abs < %.3f)\n", pass ? "PASS" : "FAIL", pass_abs);

    rcpp_ck_gemm_destroy(h);
    HIP_OK(hipFree(dA)); HIP_OK(hipFree(dB));
    HIP_OK(hipFree(dC_ck)); HIP_OK(hipFree(dC_std)); HIP_OK(hipFree(dC_lds)); HIP_OK(hipFree(dC_wmma));
    HIP_OK(hipEventDestroy(e0)); HIP_OK(hipEventDestroy(e1));
    return pass ? 0 : 1;
}
