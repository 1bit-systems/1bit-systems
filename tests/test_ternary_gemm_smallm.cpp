// test_ternary_gemm_smallm.cpp — correctness + timing for the tiled small-M
// ternary GEMM kernel (see src/ternary_gemm_smallm.hip).
//
// Sweep: 100 seeds × M ∈ {1, 2, 4, 8, 16}, fixed N = K = 2560.
// Correctness: max_abs_err(tiled, scalar_ref) ≤ 1 bf16 ULP, element-wise.
// Timing: hipEvent elapsed time over the tiled kernel; reported per (M).
//
// PASS if every (seed, M) cell ≤ 1 ULP. FAIL and non-zero exit otherwise.
// No hipBLAS, no Python, stream-based sync only (no hipDeviceSynchronize in
// the timed section — we use hipEventSynchronize on the stop event).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

// ── Kernel launchers (defined in src/ternary_gemm_smallm{,_scalar_ref}.hip) ──
extern "C" void onebit_ternary_gemm_smallm_launch(
    const int8_t* x_i8, float x_scale,
    const uint32_t* w_packed, const float* w_row_scales,
    hip_bfloat16* y_out,
    int M, int N, int K, hipStream_t stream);

extern "C" void onebit_ternary_gemm_smallm_scalar_launch(
    const int8_t* x_i8, float x_scale,
    const uint32_t* w_packed, const float* w_row_scales,
    hip_bfloat16* y_out,
    int M, int N, int K, hipStream_t stream);

// ── bf16 helpers: compute the ULP gap between two bf16 values ────────────────
// bf16 has 8-bit exponent + 7-bit mantissa. Adjacent representable values
// differ by 1 in the u16 bit pattern (modulo sign flips near zero, which we
// handle by using two's-complement distance with a sign-magnitude shift).
static uint16_t bf16_to_u16(hip_bfloat16 v) {
    uint16_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

// ULP distance in "bit-pattern distance" form. Flip sign bit so the
// representation is monotonic over the reals, then absolute difference.
static uint32_t bf16_ulp_diff(hip_bfloat16 a, hip_bfloat16 b) {
    auto flip = [](uint16_t u) -> int32_t {
        if (u & 0x8000u) return (int32_t)(0x8000u - (u & 0x7FFFu));
        return (int32_t)(0x8000u + u);
    };
    int32_t ia = flip(bf16_to_u16(a));
    int32_t ib = flip(bf16_to_u16(b));
    int32_t d = ia - ib;
    if (d < 0) d = -d;
    return (uint32_t)d;
}

// ── Packer: N-major 2-bit codes into (K/16, N) u32 layout ───────────────────
// code ∈ {0, 1, 2} corresponding to {-1, 0, +1}.
static void pack_ternary(const std::vector<int8_t>& w_signs,  // (K, N) row-major, values ∈ {-1,0,+1}
                         std::vector<uint32_t>& w_packed,     // (K/16, N) u32
                         int N, int K)
{
    const int k_blocks = K / 16;
    w_packed.assign((size_t)k_blocks * (size_t)N, 0u);
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            uint32_t word = 0;
            for (int j = 0; j < 16; ++j) {
                const int k = kb * 16 + j;
                const int s = w_signs[(size_t)k * (size_t)N + (size_t)n];
                uint32_t code = 1u;        // 0
                if (s > 0) code = 2u;      // +1
                else if (s < 0) code = 0u; // -1
                word |= (code & 0x3u) << (j * 2);
            }
            w_packed[(size_t)kb * (size_t)N + (size_t)n] = word;
        }
    }
}

// ── One (seed, M) trial. Returns max ULP gap observed on the M*N output. ─────
struct TrialResult {
    uint32_t max_ulp;
    double   tiled_ms;
    double   scalar_ms;
};

static TrialResult run_trial(int M, int N, int K, uint32_t seed,
                             hipStream_t stream)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> x_dist(-127, 127);
    // Ternary distribution matches halo-1bit training target: ~60% zeros,
    // ~20% +1, ~20% -1. Diversity exercises the zero-short-circuit + sign
    // branches of both kernels.
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    // Host buffers
    std::vector<int8_t>  h_x((size_t)M * (size_t)K);
    std::vector<int8_t>  h_w_signs((size_t)K * (size_t)N);
    std::vector<float>   h_scales((size_t)N);
    std::vector<uint32_t> h_w_packed;

    for (auto& v : h_x) v = (int8_t)x_dist(rng);
    for (auto& v : h_w_signs) {
        float r = prob(rng);
        v = (r < 0.2f) ? -1 : (r < 0.4f ? +1 : 0);
    }
    std::uniform_real_distribution<float> scale_dist(0.5e-3f, 2.0e-3f);
    for (auto& s : h_scales) s = scale_dist(rng);

    pack_ternary(h_w_signs, h_w_packed, N, K);

    const float x_scale = 1.0f / 127.0f;

    // Device buffers
    int8_t*         d_x = nullptr;
    uint32_t*       d_w = nullptr;
    float*          d_scales = nullptr;
    hip_bfloat16* d_y_tiled = nullptr;
    hip_bfloat16* d_y_scalar = nullptr;

    HIP_OK(hipMalloc(&d_x,       h_x.size()));
    HIP_OK(hipMalloc(&d_w,       h_w_packed.size() * sizeof(uint32_t)));
    HIP_OK(hipMalloc(&d_scales,  h_scales.size() * sizeof(float)));
    HIP_OK(hipMalloc(&d_y_tiled,  (size_t)M * (size_t)N * sizeof(hip_bfloat16)));
    HIP_OK(hipMalloc(&d_y_scalar, (size_t)M * (size_t)N * sizeof(hip_bfloat16)));

    HIP_OK(hipMemcpyAsync(d_x, h_x.data(), h_x.size(),
                          hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemcpyAsync(d_w, h_w_packed.data(),
                          h_w_packed.size() * sizeof(uint32_t),
                          hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemcpyAsync(d_scales, h_scales.data(),
                          h_scales.size() * sizeof(float),
                          hipMemcpyHostToDevice, stream));

    // Zero-init outputs so skipped lanes stay at a deterministic 0.
    HIP_OK(hipMemsetAsync(d_y_tiled, 0,
                          (size_t)M * (size_t)N * sizeof(hip_bfloat16), stream));
    HIP_OK(hipMemsetAsync(d_y_scalar, 0,
                          (size_t)M * (size_t)N * sizeof(hip_bfloat16), stream));

    hipEvent_t t0, t1, s0, s1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));
    HIP_OK(hipEventCreate(&s0));
    HIP_OK(hipEventCreate(&s1));

    // Warmup once to avoid JIT/first-launch skew on the tiled path.
    onebit_ternary_gemm_smallm_launch(d_x, x_scale, d_w, d_scales,
                                      d_y_tiled, M, N, K, stream);

    HIP_OK(hipEventRecord(t0, stream));
    onebit_ternary_gemm_smallm_launch(d_x, x_scale, d_w, d_scales,
                                      d_y_tiled, M, N, K, stream);
    HIP_OK(hipEventRecord(t1, stream));

    HIP_OK(hipEventRecord(s0, stream));
    onebit_ternary_gemm_smallm_scalar_launch(d_x, x_scale, d_w, d_scales,
                                             d_y_scalar, M, N, K, stream);
    HIP_OK(hipEventRecord(s1, stream));

    HIP_OK(hipEventSynchronize(t1));
    HIP_OK(hipEventSynchronize(s1));

    float tiled_ms = 0.0f, scalar_ms = 0.0f;
    HIP_OK(hipEventElapsedTime(&tiled_ms,  t0, t1));
    HIP_OK(hipEventElapsedTime(&scalar_ms, s0, s1));

    // Readback and ULP-diff
    std::vector<hip_bfloat16> h_y_tiled((size_t)M * (size_t)N);
    std::vector<hip_bfloat16> h_y_scalar((size_t)M * (size_t)N);
    HIP_OK(hipMemcpyAsync(h_y_tiled.data(),  d_y_tiled,
                          h_y_tiled.size()  * sizeof(hip_bfloat16),
                          hipMemcpyDeviceToHost, stream));
    HIP_OK(hipMemcpyAsync(h_y_scalar.data(), d_y_scalar,
                          h_y_scalar.size() * sizeof(hip_bfloat16),
                          hipMemcpyDeviceToHost, stream));
    HIP_OK(hipStreamSynchronize(stream));

    uint32_t max_ulp = 0;
    for (size_t i = 0; i < h_y_tiled.size(); ++i) {
        uint32_t d = bf16_ulp_diff(h_y_tiled[i], h_y_scalar[i]);
        if (d > max_ulp) max_ulp = d;
    }

    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipEventDestroy(s0));
    HIP_OK(hipEventDestroy(s1));
    HIP_OK(hipFree(d_x));
    HIP_OK(hipFree(d_w));
    HIP_OK(hipFree(d_scales));
    HIP_OK(hipFree(d_y_tiled));
    HIP_OK(hipFree(d_y_scalar));

    return TrialResult{max_ulp, (double)tiled_ms, (double)scalar_ms};
}

int main(int argc, char** argv) {
    const int N = 2560;
    const int K = 2560;
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 100;
    const int m_sweep[] = {1, 2, 4, 8, 16};

    printf("test_ternary_gemm_smallm: N=%d K=%d seeds=%d\n", N, K, n_seeds);
    printf("PASS criterion: max|err| ≤ 1 bf16 ULP across all (seed, M)\n\n");

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    bool overall_pass = true;

    for (int mi : m_sweep) {
        uint32_t worst_ulp = 0;
        double   tiled_sum = 0.0;
        double   scalar_sum = 0.0;

        for (int s = 0; s < n_seeds; ++s) {
            TrialResult tr = run_trial(mi, N, K, (uint32_t)(0xDEADBEEF ^ (s * 2654435761u)), stream);
            if (tr.max_ulp > worst_ulp) worst_ulp = tr.max_ulp;
            tiled_sum  += tr.tiled_ms;
            scalar_sum += tr.scalar_ms;
        }

        const double tiled_avg_ms  = tiled_sum  / (double)n_seeds;
        const double scalar_avg_ms = scalar_sum / (double)n_seeds;
        // tok/s-equivalent: M verified tokens per tiled-kernel call.
        const double toks_per_s = (tiled_avg_ms > 0.0)
            ? ((double)mi / (tiled_avg_ms * 1e-3))
            : 0.0;

        const bool pass = (worst_ulp <= 1u);
        if (!pass) overall_pass = false;

        printf("  M=%2d  worst_ulp=%u  tiled_avg=%.4f ms  scalar_avg=%.4f ms  "
               "tok/s-eq=%.1f  %s\n",
               mi, (unsigned)worst_ulp, tiled_avg_ms, scalar_avg_ms,
               toks_per_s, pass ? "PASS" : "FAIL");
    }

    HIP_OK(hipStreamDestroy(stream));

    if (!overall_pass) {
        printf("\nFAIL: at least one (seed, M) exceeded 1 bf16 ULP.\n");
        return 1;
    }
    printf("\nPASS: all seeds × all M within 1 bf16 ULP.\n");
    return 0;
}
