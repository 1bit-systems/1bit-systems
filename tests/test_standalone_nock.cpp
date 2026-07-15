// Standalone kernel test — no CK dependency.
// Validates standalone (Phase 1 naive, LDS, WMMA) kernels against a CPU
// FP16 reference instead of CK. Same packed-pk_i4 format.

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

// CPU reference: fp16 activations x ternary weights (packed pk_i4)
static void cpu_ref(const _Float16* A, const int8_t* B_packed,
                    _Float16* C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                // Unpack pk_i4: byte b = (k/2) encodes two 4-bit values
                // odd k = low nibble, even k = high nibble
                int byte_idx = k / 2;
                int nibble = (k & 1) ? (B_packed[byte_idx] & 0x0F) : ((B_packed[byte_idx] >> 4) & 0x0F);
                // PK_I4 value = nibble - 8: 0x7->-1, 0x8->0, 0x9->+1
                // Matches prefill_standalone.hip:14 convention.
                int w = (int)(nibble) - 8;
                acc += (int32_t)((float)A[(size_t)m * K + k] * w);
            }
            C[(size_t)m * N + n] = (_Float16)(float)acc;
        }
    }
}

int main(int argc, char** argv) {
    int M = 128, N = 128, K = 256;
    if(argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    // Clamp to legal shapes (WMMA requires M%16==0, N%16==0)
    M = (M / 16) * 16; if (M < 16) M = 16;
    N = (N / 16) * 16; if (N < 16) N = 16;

    printf("=== standalone kernels vs CPU reference (no CK) ===\n");
    printf("Shape: M=%d N=%d K=%d\n", M, N, K);

    std::mt19937 rng(0x1b1fe4e4);
    std::uniform_real_distribution<float> rd(-0.25f, 0.25f);
    std::uniform_int_distribution<int>    rt(-1, 1);

    std::vector<_Float16> A((size_t)M * K);
    for(auto& v : A) v = (_Float16)rd(rng);

    std::vector<int8_t> B_ternary((size_t)K * N);
    for(auto& v : B_ternary) v = (int8_t)rt(rng);

    std::vector<int8_t> B_packed((size_t)K * N / 2);
    RC_OK(rcpp_ternary_pack_pk_i4(B_ternary.data(), B_packed.data(), K, N));

    // CPU reference
    std::vector<_Float16> C_cpu((size_t)M * N);
    cpu_ref(A.data(), B_packed.data(), C_cpu.data(), M, N, K);

    // Device buffers
    _Float16 *dA = nullptr, *dC_std = nullptr, *dC_lds = nullptr, *dC_wmma = nullptr;
    int8_t* dB = nullptr;
    HIP_OK(hipMalloc(&dA,      A.size() * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dB,      B_packed.size()));
    HIP_OK(hipMalloc(&dC_std,  (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_lds,  (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dC_wmma, (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMemcpy(dA, A.data(),        A.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B_packed.data(), B_packed.size(),             hipMemcpyHostToDevice));

    // Run standalone paths
    rcpp_standalone_launch     (dA, dB, dC_std,  M, N, K, nullptr);
    rcpp_standalone_launch_lds (dA, dB, dC_lds,  M, N, K, nullptr);
    rcpp_standalone_launch_wmma(dA, dB, dC_wmma, M, N, K, nullptr);
    HIP_OK(hipDeviceSynchronize());

    // Read back
    std::vector<_Float16> C_std((size_t)M * N), C_lds((size_t)M * N), C_wmma((size_t)M * N);
    HIP_OK(hipMemcpy(C_std.data(),  dC_std,  C_std.size()  * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_lds.data(),  dC_lds,  C_lds.size()  * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(C_wmma.data(), dC_wmma, C_wmma.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    auto diff = [&](const char* label, const std::vector<_Float16>& got) {
        float max_abs = 0.0f;
        double sum_sq = 0.0;
        for(size_t i = 0; i < C_cpu.size(); ++i) {
            float d = std::fabs((float)C_cpu[i] - (float)got[i]);
            max_abs = std::max(max_abs, d);
            sum_sq += d * d;
        }
        float rmse = (float)std::sqrt(sum_sq / C_cpu.size());
        printf("  %-22s  max_abs=%.6f  rmse=%.6f\n", label, max_abs, rmse);
        return max_abs;
    };

    printf("Diffs vs CPU reference:\n");
    float e_std  = diff("Phase 1 naive", C_std);
    float e_lds  = diff("Phase 2 LDS",   C_lds);
    float e_wmma = diff("Phase 3 WMMA",  C_wmma);

    // Perf
    const int runs = 20;
    hipEvent_t e0, e1; HIP_OK(hipEventCreate(&e0)); HIP_OK(hipEventCreate(&e1));
    double flops = 2.0 * (double)M * N * K;

    auto time_ms = [&](auto launch) -> double {
        for(int w = 0; w < 3; ++w) launch();
        HIP_OK(hipDeviceSynchronize());
        HIP_OK(hipEventRecord(e0, nullptr));
        for(int r = 0; r < runs; ++r) launch();
        HIP_OK(hipEventRecord(e1, nullptr));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f; HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        return (double)ms / runs;
    };

    double ms_std  = time_ms([&](){ rcpp_standalone_launch     (dA, dB, dC_std,  M, N, K, nullptr); });
    double ms_lds  = time_ms([&](){ rcpp_standalone_launch_lds (dA, dB, dC_lds,  M, N, K, nullptr); });
    double ms_wmma = time_ms([&](){ rcpp_standalone_launch_wmma(dA, dB, dC_wmma, M, N, K, nullptr); });

    printf("Perf:\n");
    printf("  %-22s  %.3f ms  %6.2f TFlops\n",
           "Phase 1 naive",  ms_std,  flops / (ms_std  * 1e-3) / 1e12);
    printf("  %-22s  %.3f ms  %6.2f TFlops\n",
           "Phase 2 LDS",    ms_lds,  flops / (ms_lds  * 1e-3) / 1e12);
    printf("  %-22s  %.3f ms  %6.2f TFlops\n",
           "Phase 3 WMMA",   ms_wmma, flops / (ms_wmma * 1e-3) / 1e12);

    const float pass_abs = 2.0f;  // FP16 accumulation tolerance
    int pass = (e_std < pass_abs) && (e_lds < pass_abs) && (e_wmma < pass_abs);
    printf("Verdict: %s (threshold max_abs < %.3f)\n", pass ? "PASS" : "FAIL", pass_abs);

    HIP_OK(hipFree(dA)); HIP_OK(hipFree(dB));
    HIP_OK(hipFree(dC_std)); HIP_OK(hipFree(dC_lds)); HIP_OK(hipFree(dC_wmma));
    HIP_OK(hipEventDestroy(e0)); HIP_OK(hipEventDestroy(e1));
    return pass ? 0 : 1;
}
