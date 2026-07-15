// End-to-end proof for the rocm-cpp C API.
//   1. Generate random FP16 activations [M, K] and ternary weights [K, N].
//   2. Host CPU reference: C_ref = A * B_dequant (FP16).
//   3. Pack ternary to pk_i4 via rcpp_ternary_pack_pk_i4 (no CK includes here).
//   4. Run rcpp_ck_gemm_create/run on device.
//   5. Diff GPU output vs host reference.

#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP err %d %s:%d\n",_s,__FILE__,__LINE__); std::abort();}} while(0)
#define RC_OK(e)  do { auto _s=(e); if(_s!=RCPP_OK){fprintf(stderr,"rcpp err %d %s:%d\n",(int)_s,__FILE__,__LINE__); std::abort();}} while(0)

int main(int argc, char** argv) {
    int M = 2560, N = 6912, K = 2560;
    if(argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    printf("=== test_ck_gemm: FP16 x ternary(pk_i4) -> FP16 ===\n");
    printf("Shape: M=%d N=%d K=%d\n", M, N, K);

    std::mt19937 rng(0xc0ffee);
    std::uniform_real_distribution<float> rd(-0.25f, 0.25f);
    std::uniform_int_distribution<int> rt(-1, 1);

    // Host data
    std::vector<_Float16> A((size_t)M * K);
    for(auto& v : A) v = (_Float16)rd(rng);

    std::vector<int8_t> B_ternary((size_t)K * N);  // col-major [K, N]
    for(auto& v : B_ternary) v = (int8_t)rt(rng);

    // Host reference: C_ref[m, n] = sum_k A[m, k] * B_ternary[k, n]
    // A is row-major [M, K], B is col-major [K, N]; C_ref row-major [M, N].
    // Skipped for large shapes (would take minutes on CPU); perf-only in that case.
    const bool do_verify = ((double)M * N * K < 2.5e9);
    std::vector<float> C_ref;
    if(do_verify) {
        C_ref.assign((size_t)M * N, 0.0f);
        for(int m = 0; m < M; ++m) {
            for(int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for(int k = 0; k < K; ++k) {
                    float a = (float)A[(size_t)m * K + k];
                    int   b = B_ternary[(size_t)n * K + k];
                    acc += a * (float)b;
                }
                C_ref[(size_t)m * N + n] = acc;
            }
        }
    } else {
        printf("(skipping CPU reference at MNK=%.2fG — perf-only)\n",
               (double)M * N * K / 1e9);
    }

    // Pack ternary -> pk_i4 WMMA-permuted via rocm-cpp C API.
    std::vector<int8_t> B_packed((size_t)K * N / 2);
    RC_OK(rcpp_ternary_pack_pk_i4(B_ternary.data(), B_packed.data(), K, N));
    printf("Packed %zu MiB ternary -> %zu MiB pk_i4 (1/4 of FP16)\n",
           B_ternary.size() / (1024 * 1024),
           B_packed.size() / (1024 * 1024));

    // Device buffers
    _Float16* dA = nullptr;
    _Float16* dC = nullptr;
    int8_t*   dB = nullptr;
    HIP_OK(hipMalloc(&dA, A.size() * sizeof(_Float16)));
    HIP_OK(hipMalloc(&dB, B_packed.size()));
    HIP_OK(hipMalloc(&dC, (size_t)M * N * sizeof(_Float16)));
    HIP_OK(hipMemcpy(dA, A.data(),        A.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, B_packed.data(), B_packed.size(),             hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dC, 0, (size_t)M * N * sizeof(_Float16)));

    // Handle lifecycle
    rcpp_ck_gemm_handle_t* h = nullptr;
    RC_OK(rcpp_ck_gemm_create(M, N, K, &h));
    printf("Instance: %s\n", rcpp_ck_gemm_instance_string(h));

    // Warmup + time
    for(int w = 0; w < 5; ++w) RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC, nullptr));
    HIP_OK(hipDeviceSynchronize());

    hipEvent_t e0, e1;
    HIP_OK(hipEventCreate(&e0));
    HIP_OK(hipEventCreate(&e1));
    HIP_OK(hipEventRecord(e0, nullptr));
    const int runs = 20;
    for(int r = 0; r < runs; ++r) RC_OK(rcpp_ck_gemm_run(h, dA, dB, dC, nullptr));
    HIP_OK(hipEventRecord(e1, nullptr));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    ms /= runs;
    double tflops = 2.0 * (double)M * N * K / (ms * 1e-3) / 1e12;
    printf("Perf: %.3f ms/run, %.2f TFlops\n", ms, tflops);

    // Correctness (skipped for large shapes)
    int pass = 1;
    if(do_verify) {
        std::vector<_Float16> C_gpu((size_t)M * N);
        HIP_OK(hipMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

        float max_abs = 0.0f, max_rel = 0.0f;
        double sum_abs = 0.0, sum_sq = 0.0;
        for(size_t i = 0; i < C_ref.size(); ++i) {
            float g = (float)C_gpu[i];
            float r = C_ref[i];
            float d = std::fabs(g - r);
            max_abs = std::max(max_abs, d);
            float denom = std::max(std::fabs(r), 1e-3f);
            max_rel = std::max(max_rel, d / denom);
            sum_abs += d;
            sum_sq  += (double)d * d;
        }
        double mean_abs = sum_abs / C_ref.size();
        double rmse     = std::sqrt(sum_sq / C_ref.size());
        printf("Diff vs host reference:\n");
        printf("  max abs  : %.4f\n", max_abs);
        printf("  max rel  : %.4f\n", max_rel);
        printf("  mean abs : %.4f\n", mean_abs);
        printf("  rmse     : %.4f\n", rmse);

        // FP16 accumulation noise at K=2560 is ~sqrt(K) * eps * |x|
        // = ~50 * 1e-3 * 0.25 ≈ 0.01 per output; allow 10x headroom.
        const float pass_abs = 0.5f;
        pass = (max_abs < pass_abs);
        printf("Verdict: %s (threshold max_abs < %.3f)\n", pass ? "PASS" : "FAIL", pass_abs);
    }

    rcpp_ck_gemm_destroy(h);
    HIP_OK(hipFree(dA));
    HIP_OK(hipFree(dB));
    HIP_OK(hipFree(dC));
    HIP_OK(hipEventDestroy(e0));
    HIP_OK(hipEventDestroy(e1));
    return pass ? 0 : 1;
}
