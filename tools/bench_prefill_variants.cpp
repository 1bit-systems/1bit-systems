// bench_prefill_variants — benchmark all ternary prefill GEMM kernels
// on the BitNet FFN shapes that matter. Reports timings + TFlops + rankings.
//
// Build:
//   cmake -B build && ninja -C build bench_prefill_variants
// Run:
//   ./build/bench_prefill_variants [M N K] [warmup] [timed]

#include "rocm_cpp/prefill_tuner.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#define HIP_CHECK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

using launch_fn = void (*)(const void*, const void*, void*, int, int, int, void*);

extern "C" {
void rcpp_standalone_launch_wmma_4x4_vec(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_cached(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_lds_pingpong(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x2wave(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_fp16b(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_decode_pk_i4_to_fp16_launch(const void*, void*, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_vec_i8_tiled(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled(const void*, const void*, void*, int, int, int, void*);
}

struct Variant {
    const char* name;
    launch_fn   launch;
    bool        needs_fp16_b;
    bool        geom_ok;    // set to false if shape requirement not met
};

static Variant make_var(const char* n, launch_fn f, bool fp16 = false) {
    return {n, f, fp16, true};
}

static double tflops(double ms, int M, int N, int K) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return ops / (ms * 1e-3) / 1e12;
}

static std::vector<Variant> make_variants(int M, int N, int K) {
    std::vector<Variant> vs;
    vs.push_back(make_var("4i  (64x64 vec A)",        rcpp_standalone_launch_wmma_4x4_vec));
    vs.push_back(make_var("4h  (64x64 cached A+B)",    rcpp_standalone_launch_wmma_4x4_cached));
    vs.push_back(make_var("4c  (32x64)",               rcpp_standalone_launch_wmma_2x4));
    vs.push_back(make_var("4g  (64x64 4x4 acc)",       rcpp_standalone_launch_wmma_4x4));
    if (M >= 64 && N >= 32) {
        vs.push_back(make_var("4k  (LDS ping-pong 64x32)", rcpp_standalone_launch_lds_pingpong));
    } else {
        auto v = make_var("4k  (LDS ping-pong 64x32)", rcpp_standalone_launch_lds_pingpong);
        v.geom_ok = false;
        vs.push_back(v);
    }
    if (M >= 64 && N >= 64) {
        vs.push_back(make_var("4f  (2x2 wave 64x64 LDS)", rcpp_standalone_launch_wmma_2x2wave));
    } else {
        auto v = make_var("4f  (2x2 wave 64x64 LDS)", rcpp_standalone_launch_wmma_2x2wave);
        v.geom_ok = false;
        vs.push_back(v);
    }
    vs.push_back(make_var("4i-I8-T (int8 WMMA tiled)", rcpp_standalone_launch_wmma_4x4_vec_i8_tiled, true));
    vs.push_back(make_var("4i-I8-APRE (I8 A+ tiled B)", rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled, true));
    vs.push_back(make_var("FP16-B (pre-decoded B)",     rcpp_standalone_launch_fp16b, true));
    return vs;
}

double bench_one(launch_fn fn, const void* A, const void* B, void* C,
                 int M, int N, int K, int warmup, int timed, hipStream_t stream) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    for (int i = 0; i < warmup; ++i) fn(A, B, C, M, N, K, stream);
    hipEventRecord(start, stream);
    for (int i = 0; i < timed; ++i) fn(A, B, C, M, N, K, stream);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return (double)ms / timed;
}

struct Result {
    const char* name;
    double ms;
    double tf;
    bool   ok;
};

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2560;
    int N = (argc > 2) ? atoi(argv[2]) : 6912;
    int K = (argc > 3) ? atoi(argv[3]) : 2560;
    int warmup = (argc > 4) ? atoi(argv[4]) : 5;
    int timed  = (argc > 5) ? atoi(argv[5]) : 20;

    printf("=== Prefill Variant Benchmark (%d x %d x %d) ===\n", M, N, K);
    printf("Warmup: %d, Timed: %d\n\n", warmup, timed);

    hipStream_t stream;
    hipStreamCreate(&stream);

    // Allocate
    size_t a_bytes = (size_t)M * K * sizeof(__half);
    size_t b_bytes = (size_t)K * N / 2;   // pk_i4: 2 vals per byte
    size_t c_bytes = (size_t)M * N * sizeof(__half);
    size_t b_fp16_bytes = (size_t)K * N * sizeof(__half);
    size_t b_i8_tiled_bytes = (size_t)(K / 32) * N * 32;  // tiled INT8: [K0, N, 32]

    __half *A_d, *C_d;
    uint8_t *B_d;
    __half *B_fp16_d;
    int8_t *B_i8_tiled_d;
    int8_t *A_i8_d;          // pre-quantized INT8 activations

    HIP_CHECK(hipMalloc(&A_d, a_bytes));
    HIP_CHECK(hipMalloc(&A_i8_d, a_bytes));  // INT8 A same size as FP16 A (same values as bytes)
    HIP_CHECK(hipMalloc(&B_d, b_bytes));
    HIP_CHECK(hipMalloc(&C_d, c_bytes));
    HIP_CHECK(hipMalloc(&B_fp16_d, b_fp16_bytes));
    HIP_CHECK(hipMalloc(&B_i8_tiled_d, b_i8_tiled_bytes));

    // Fill with random data
    std::vector<__half> A_host(M*K);
    std::vector<int8_t> A_i8_host(M*K);  // pre-quantized INT8 A
    std::vector<uint8_t> B_host(K*N/2);
    std::vector<int8_t> B_i8_tiled_host(b_i8_tiled_bytes);
    for (size_t i = 0; i < A_host.size(); ++i) {
        float v = (float)rand()/RAND_MAX - 0.5f;
        A_host[i] = __float2half(v);
        // Pre-quantize to INT8 for the apre variant
        int q = (int)(v * 127.0f + 0.5f);
        if (q > 127) q = 127; if (q < -127) q = -127;
        A_i8_host[i] = (int8_t)q;
    }
    for (size_t i = 0; i < B_host.size(); ++i) B_host[i] = (uint8_t)(rand() & 0xFF);
    // Generate tiled INT8 B: random ternary values {-1, 0, +1}
    for (size_t i = 0; i < b_i8_tiled_bytes; ++i) B_i8_tiled_host[i] = (int8_t)((rand() % 3) - 1);
    hipMemcpy(A_d, A_host.data(), a_bytes, hipMemcpyHostToDevice);
    hipMemcpy(A_i8_d, A_i8_host.data(), a_bytes, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_host.data(), b_bytes, hipMemcpyHostToDevice);
    hipMemcpy(B_i8_tiled_d, B_i8_tiled_host.data(), b_i8_tiled_bytes, hipMemcpyHostToDevice);

    // Decode B to FP16 once for FP16-B variant
    rcpp_decode_pk_i4_to_fp16_launch(B_d, B_fp16_d, K, N, stream);
    hipStreamSynchronize(stream);

    auto variants = make_variants(M, N, K);

    // Dispatch table: which B buffer each variant uses
    // (default B_d, fp16-B = B_fp16_d, i8-tiled = B_i8_tiled_d)
    auto b_for_variant = [&](int idx) -> const void* {
        const char* name = variants[idx].name;
        if (strstr(name, "I8-APRE")) return (const void*)B_i8_tiled_d;
        if (strstr(name, "I8-T"))    return (const void*)B_i8_tiled_d;
        if (variants[idx].needs_fp16_b) return (const void*)B_fp16_d;
        return (const void*)B_d;
    };
    // A-buffer dispatch: I8-APRE variants use A_i8_d, others use A_d
    auto a_for_variant = [&](int idx) -> const void* {
        const char* name = variants[idx].name;
        if (strstr(name, "I8-APRE")) return (const void*)A_i8_d;
        return (const void*)A_d;
    };

    // Warmup all variants (triggers JIT)
    printf("Warming up all variants...\n");
    for (size_t i = 0; i < variants.size(); ++i) {
        auto& v = variants[i];
        if (!v.geom_ok) continue;
        const void* b = b_for_variant((int)i);
        const void* a = a_for_variant((int)i);
        for (int w = 0; w < 2; ++w) v.launch(a, b, C_d, M, N, K, stream);
    }
    hipStreamSynchronize(stream);
    printf("Ready.\n\n");

    // Benchmark
    std::vector<Result> results;
    for (size_t i = 0; i < variants.size(); ++i) {
        auto& v = variants[i];
        if (!v.geom_ok) {
            results.push_back({v.name, 0, 0, false});
            continue;
        }
        const void* b = b_for_variant((int)i);
        const void* a = a_for_variant((int)i);
        double ms = bench_one(v.launch, a, b, C_d, M, N, K, warmup, timed, stream);
        double tf = tflops(ms, M, N, K);
        results.push_back({v.name, ms, tf, true});
    }

    // Sort by TFlops descending
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
        return a.tf > b.tf;
    });

    printf("%-35s %10s %10s\n", "Variant", "Time (ms)", "TFlops");
    printf("%s\n", std::string(57, '-').c_str());
    for (auto& r : results) {
        if (r.ok) {
            printf("%-35s %10.4f %10.2f\n", r.name, r.ms, r.tf);
        } else {
            printf("%-35s %10s %10s  (shape incompatible)\n", r.name, "—", "—");
        }
    }

    // Pick winner from our own results (auto-tuner only knows FP16/pk_i4 variants)
    printf("\n--- Winner ---\n");
    int best_idx = 0;
    double best_tf = 0;
    for (int i = 0; i < (int)results.size(); i++) {
        if (results[i].ok && results[i].tf > best_tf) { best_tf = results[i].tf; best_idx = i; }
    }
    printf("Fastest variant: %d = %s (%.2f TFLOPS)\n", best_idx, results[best_idx].name, best_tf);
    // Stamp into auto-tuner cache so production dispatch also benefits
    rcpp_prefill_tune(A_d, B_d, C_d, M, N, K, warmup, timed, stream);

    hipFree(A_d); hipFree(B_d); hipFree(C_d); hipFree(B_fp16_d);
    hipStreamDestroy(stream);
    return 0;
}
