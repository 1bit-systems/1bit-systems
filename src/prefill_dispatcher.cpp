#include "rocm_cpp/prefill_tuner.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>

// Launchers from prefill_standalone.hip
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

struct ShapeKey {
    int M, N, K;
    bool operator==(const ShapeKey& o) const { return M==o.M && N==o.N && K==o.K; }
};
struct ShapeHash { size_t operator()(const ShapeKey& k) const { return (size_t)k.M ^ ((size_t)k.N<<16) ^ ((size_t)k.K<<32); } };

static std::unordered_map<ShapeKey, int, ShapeHash> s_best_variant;
// Default to 4h (cached A+B) for FP16/pk_i4 prefill. I8 variants
// (I8_APRE ~42 TFLOPS) require tiled INT8 weights + pre-quantized A and
// are benchmarked separately. When the model loader stores both formats,
// switch default to RCPP_PREFILL_VARIANT_I8_APRE.
static const int kDefaultVariant = RCPP_PREFILL_VARIANT_4H;

using launch_fn = void (*)(const void*, const void*, void*, int, int, int, void*);

// Forward-declare I8 launchers (defined in prefill_standalone.hip)
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

// Determine which variants are viable for a given shape.
// 4k/F require K % 32 == 0 and M >= 64 and N >= 32.
// 4f requires K % 32 == 0 and M >= 64 and N >= 64.
// FP16-B requires an FP16 B buffer (caller passes B_fp16, not B_packed).
static std::vector<int> viable_variants(int M, int N, int K) {
    std::vector<int> out;
    // 4i, 4h, 4c, 4g always viable (they handle any shape with edge guards)
    out.push_back(RCPP_PREFILL_VARIANT_4I);
    out.push_back(RCPP_PREFILL_VARIANT_4H);
    out.push_back(RCPP_PREFILL_VARIANT_4C);
    out.push_back(RCPP_PREFILL_VARIANT_4G);
    // 4k needs M>=64, N>=32
    if (M >= 64 && N >= 32) out.push_back(RCPP_PREFILL_VARIANT_4K);
    // 4f needs M>=64, N>=64
    if (M >= 64 && N >= 64) out.push_back(RCPP_PREFILL_VARIANT_4F);
    // FP16-B always viable if caller provides FP16 B
    out.push_back(RCPP_PREFILL_VARIANT_FP16B);
    // I8 variants require tiled INT8 weights and pre-quantized INT8 A,
    // which the production prefill path doesn't provide. They're benchmarked
    // separately by bench_prefill_variants which allocates both buffer types.
    // Add here when the model loader stores tiled INT8 weights alongside pk_i4.
    return out;
}

static double time_launch(launch_fn fn, const void* A, const void* B, void* C,
                          int M, int N, int K, int warmup, int timed, hipStream_t stream) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn(A, B, C, M, N, K, stream);
    }
    hipEventRecord(start, stream);
    for (int i = 0; i < timed; ++i) {
        fn(A, B, C, M, N, K, stream);
    }
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return (double)ms / timed;
}

// Weights for the FFN shapes that matter most (BitNet FFN up/down)
static const int BITNET_FFN_UP_M   = 2560;
static const int BITNET_FFN_UP_N   = 6912;
static const int BITNET_FFN_UP_K   = 2560;
static const int BITNET_FFN_DN_M   = 2560;
static const int BITNET_FFN_DN_N   = 2560;
static const int BITNET_FFN_DN_K   = 6912;

int rcpp_prefill_tune(
    const void* A_dev,
    const void* B_packed_dev,
    void* C_dev,
    int M, int N, int K,
    int warmup_iters,
    int timed_iters,
    void* stream)
{
    ShapeKey key{M, N, K};
    auto viables = viable_variants(M, N, K);

    // For FP16-B variant, we need a decoded B buffer
    void* B_fp16_dev = nullptr;
    bool has_fp16b = false;
    int fp16b_idx = -1;
    for (int idx : viables) {
        if (idx == RCPP_PREFILL_VARIANT_FP16B) { fp16b_idx = idx; break; }
    }
    if (fp16b_idx >= 0) {
        size_t b_bytes = (size_t)K * N * sizeof(__half);
        hipMalloc(&B_fp16_dev, b_bytes);
        rcpp_decode_pk_i4_to_fp16_launch(B_packed_dev, B_fp16_dev, K, N, stream);
        hipStreamSynchronize((hipStream_t)stream);
        has_fp16b = true;
    }

    // Fallback: use the constant default (4h), which covers the common case.
    double best_time = 1e30;
    int    best_variant = kDefaultVariant;

    // Ensure all kernels are loaded (first launch triggers JIT)
    for (int idx : viables) {
        if (idx == RCPP_PREFILL_VARIANT_FP16B) continue; // timed separately
        s_variants[idx](A_dev, B_packed_dev, C_dev, M, N, K, stream);
    }
    hipStreamSynchronize((hipStream_t)stream);

    for (int idx : viables) {
        const void* b_arg = B_packed_dev;
        if (idx == RCPP_PREFILL_VARIANT_FP16B) {
            if (!has_fp16b) continue;
            b_arg = B_fp16_dev;
        }
        double t = time_launch(s_variants[idx], A_dev, b_arg, C_dev,
                               M, N, K, warmup_iters, timed_iters,
                               (hipStream_t)stream);
        if (t < best_time) {
            best_time = t;
            best_variant = idx;
        }
    }

    if (has_fp16b) {
        hipFree(B_fp16_dev);
    }

    s_best_variant[key] = best_variant;
    return best_variant;
}

void rcpp_prefill_dispatch(
    const void* A_dev,
    const void* B_packed_dev,
    void* C_dev,
    int M, int N, int K,
    void* stream)
{
    ShapeKey key{M, N, K};
    auto it = s_best_variant.find(key);
    int variant = (it != s_best_variant.end()) ? it->second : kDefaultVariant;
    if (variant < 0 || variant >= RCPP_PREFILL_VARIANT_COUNT || !s_variants[variant]) {
        variant = kDefaultVariant;
    }
    s_variants[variant](A_dev, B_packed_dev, C_dev, M, N, K, stream);
}

void rcpp_pk_i4_to_fp16(
    const void* B_packed_dev,
    void* B_fp16_dev,
    int K, int N,
    void* stream)
{
    rcpp_decode_pk_i4_to_fp16_launch(B_packed_dev, B_fp16_dev, K, N, stream);
}

void rcpp_prefill_fp16b(
    const void* A_dev,
    const void* B_fp16_dev,
    void* C_dev,
    int M, int N, int K,
    void* stream)
{
    rcpp_standalone_launch_fp16b(A_dev, B_fp16_dev, C_dev, M, N, K, stream);
}
