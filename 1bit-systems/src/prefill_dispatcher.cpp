#include "rocm_cpp/prefill_tuner.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP %d %s:%d\n", _s, __FILE__, __LINE__); std::abort(); } } while(0)

extern "C" {
void rcpp_standalone_launch_wmma_4x4_vec(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_cached(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_lds_pingpong(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x2wave(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_fp16b(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_decode_pk_i4_to_fp16_launch(const void*, void*, int, int, void*);
}

struct ShapeKey { int M, N, K; bool operator==(const ShapeKey& o) const { return M==o.M && N==o.N && K==o.K; } };
struct ShapeHash { size_t operator()(const ShapeKey& k) const { return (size_t)k.M ^ ((size_t)k.N<<16) ^ ((size_t)k.K<<32); } };

static std::unordered_map<ShapeKey, int, ShapeHash> s_best_variant;
static const int kDefaultVariant = RCPP_PREFILL_VARIANT_4H;
using launch_fn = void (*)(const void*, const void*, void*, int, int, int, void*);

extern "C" {
void rcpp_standalone_launch_wmma_4x4_vec_i8_tiled(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled(const void*, const void*, void*, int, int, int, void*);
}

static launch_fn s_variants[RCPP_PREFILL_VARIANT_COUNT] = {
    rcpp_standalone_launch_wmma_4x4_vec,            // 0: 4i
    rcpp_standalone_launch_wmma_4x4_cached,         // 1: 4h
    rcpp_standalone_launch_lds_pingpong,             // 2: 4k
    rcpp_standalone_launch_wmma_2x2wave,             // 3: 4f
    rcpp_standalone_launch_fp16b,                    // 4: FP16-B
    rcpp_standalone_launch_wmma_2x4,                 // 5: 4c
    rcpp_standalone_launch_wmma_4x4,                 // 6: 4g
    rcpp_standalone_launch_wmma_4x4_vec_i8_tiled,    // 7: I8-T
    rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled, // 8: I8-APRE
};

static const char* s_variant_names[RCPP_PREFILL_VARIANT_COUNT] = {
    "4i (64x64 vec)", "4h (64x64 cached)", "4k (LDS ping-pong)",
    "4f (2x2 wave)", "FP16-B", "4c (32x64)", "4g (4x4 acc)",
    "I8-T (int8 tiled)", "I8-APRE (I8 A + tiled B)"
};

static std::vector<int> viable_variants(int M, int N, int K) {
    std::vector<int> out;
    out.push_back(RCPP_PREFILL_VARIANT_4I);
    out.push_back(RCPP_PREFILL_VARIANT_4H);
    out.push_back(RCPP_PREFILL_VARIANT_4C);
    out.push_back(RCPP_PREFILL_VARIANT_4G);
    if (M >= 64 && N >= 32) out.push_back(RCPP_PREFILL_VARIANT_4K);
    if (M >= 64 && N >= 64) out.push_back(RCPP_PREFILL_VARIANT_4F);
    out.push_back(RCPP_PREFILL_VARIANT_FP16B);
    return out;
}

static double time_launch(launch_fn fn, const void* A, const void* B, void* C,
                          int M, int N, int K, int warmup, int timed, hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) fn(A, B, C, M, N, K, stream);
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < timed; ++i) fn(A, B, C, M, N, K, stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms / timed;
}

static const int BITNET_FFN_UP_M   = 2560;
static const int BITNET_FFN_UP_N   = 6912;
static const int BITNET_FFN_UP_K   = 2560;
static const int BITNET_FFN_DN_M   = 2560;
static const int BITNET_FFN_DN_N   = 2560;
static const int BITNET_FFN_DN_K   = 6912;

extern "C" int rcpp_prefill_tune(const void* A_dev, const void* B_packed_dev, void* C_dev,
                      int M, int N, int K,
                      int warmup_iters, int timed_iters,
                      void* stream) {
    ShapeKey key = {M, N, K};
    auto it = s_best_variant.find(key);
    if (it != s_best_variant.end()) return it->second;

    auto viables = viable_variants(M, N, K);

    void* B_fp16_dev = nullptr;
    bool has_fp16b = false;
    int fp16b_idx = -1;
    for (int idx : viables) {
        if (idx == RCPP_PREFILL_VARIANT_FP16B) { fp16b_idx = idx; break; }
    }
    if (fp16b_idx >= 0) {
        size_t b_bytes = (size_t)K * N * sizeof(__half);
        HIP_CHECK(hipMalloc(&B_fp16_dev, b_bytes));
        rcpp_decode_pk_i4_to_fp16_launch(B_packed_dev, B_fp16_dev, K, N, stream);
        HIP_CHECK(hipStreamSynchronize((hipStream_t)stream));
        has_fp16b = true;
    }

    double best_time = 1e30;
    int best_variant = kDefaultVariant;

    for (int idx : viables) {
        if (idx == RCPP_PREFILL_VARIANT_FP16B) continue;
        s_variants[idx](A_dev, B_packed_dev, C_dev, M, N, K, stream);
    }
    HIP_CHECK(hipStreamSynchronize((hipStream_t)stream));

    for (int idx : viables) {
        const void* b_arg = B_packed_dev;
        if (idx == RCPP_PREFILL_VARIANT_FP16B) {
            if (!has_fp16b) continue;
            b_arg = B_fp16_dev;
        }
        double t = time_launch(s_variants[idx], A_dev, b_arg, C_dev, M, N, K, warmup_iters, timed_iters, (hipStream_t)stream);
        if (t < best_time) { best_time = t; best_variant = idx; }
    }

    if (has_fp16b) HIP_CHECK(hipFree(B_fp16_dev));
    s_best_variant[key] = best_variant;
    return best_variant;
}

extern "C" void rcpp_prefill_dispatch(const void* A_dev, const void* B_packed_dev, void* C_dev,
                           int M, int N, int K, void* stream) {
    int variant = rcpp_prefill_tune(A_dev, B_packed_dev, C_dev, M, N, K, 3, 10, stream);
    s_variants[variant](A_dev, B_packed_dev, C_dev, M, N, K, stream);
}