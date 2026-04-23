// test_ternary_gemv_tq2_0.cpp — standalone unit test for the TQ2_0 x Q8_1
// ternary GEMV kernels (kernels/ternary_gemv_tq2_0.hip).
//
// Validates both the fp32-output and fp16-output launches against a pure
// C++ dequantise + matmul reference. The reference mirrors the kernel
// math exactly: per-block fp16 weight scale, per-q8_1-block fp16 activation
// scale, symmetric {-1,0,+1} alphabet, int32 slot accumulators widened to
// fp32 before scaling.
//
// TQ2_0 stride-32 packing (see kernels/ternary_gemv_tq2_0.hip header):
//   sub-block j ∈ {0, 32}, byte qs[j+m] holds 4 weights at
//   k = j*4 + m + l*32  for l ∈ [0,4), encoded in bits (l*2 .. l*2+1) as
//   q ∈ {0,1,2} mapping to signed {-1,0,+1}. (q == 3 is illegal.)
//
// No ggml headers — Rule B: rocm-cpp TU redeclares the block structs. No
// gtest / catch2 — hand-rolled runner that returns 0 on all-pass.
//
// Test cases (ranked by what to run first on the strix-halo box):
//   1. "edge-M8-K256"   — smallest working shape, one block per row, one
//                          weight block per row. Surfaces grid / tail bugs.
//   2. "edge-M8-K512"   — still minimum-grid row-count but two weight
//                          blocks per row; catches the inter-block
//                          accumulator reset bug family.
//   3. "mid-1024x1024"  — realistic but fast; catches wave-reduction
//                          mismatches before the big run.
//   4. "large-4096x4096" — full paper shape; confirms max-abs stays
//                          inside budget across 16 blocks × 4096 rows.
//   5. fp16-output variants of (3) + (4) with widened threshold.

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

// ── Block layouts. Keep in sync with kernels/ternary_gemv_tq2_0.hip. ────────

#define TQ2_0_QK_K      256
#define TQ2_0_QK8_1     32
#define TQ2_0_BLOCKS_PER_K (TQ2_0_QK_K / TQ2_0_QK8_1)

struct block_tq2_0 {
    uint8_t qs[TQ2_0_QK_K / 4];  // 64 bytes
    __half  d;
};
static_assert(sizeof(block_tq2_0) == 66, "block_tq2_0 must be 66 bytes");

struct block_q8_1 {
    __half  d;
    __half  s;
    int8_t  qs[TQ2_0_QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ── HIP error helper ────────────────────────────────────────────────────────

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

// ── Launch symbols under test ───────────────────────────────────────────────

extern "C" void ternary_gemv_tq2_0_q8_1_launch(
    const void* vx, const void* vy, void* dst,
    int32_t ncols, int32_t nrows, void* stream);

extern "C" void ternary_gemv_tq2_0_q8_1_f16_launch(
    const void* vx, const void* vy, void* dst,
    int32_t ncols, int32_t nrows, void* stream);

// ── Random data generators ──────────────────────────────────────────────────

// Build per-row TQ2_0 weight blocks. Signs drawn uniformly from {-1, 0, +1}.
// Scales drawn uniformly in [0.01, 0.5] — matches the paper's post-gradient
// scale range for a trained ternary linear and keeps d_w * d_x * i32
// products inside fp32 dynamic range comfortably.
static void gen_tq2_0_weights(std::vector<block_tq2_0>& wb, int M, int K,
                              std::mt19937& rng)
{
    const int nblocks_k = K / TQ2_0_QK_K;
    wb.assign((size_t)M * nblocks_k, {});

    std::uniform_int_distribution<int> sdist(-1, 1);   // {-1, 0, +1}
    std::uniform_real_distribution<float> ddist(0.01f, 0.5f);

    for (int m = 0; m < M; ++m) {
        for (int b = 0; b < nblocks_k; ++b) {
            block_tq2_0& blk = wb[(size_t)m * nblocks_k + b];
            blk.d = __float2half(ddist(rng));

            // Walk the logical k index and pack into stride-32 layout.
            // For each kk ∈ [0, 256), find (j, lane, l) such that
            //   kk = j*4 + lane + l*32,  with j ∈ {0, 32}, lane ∈ [0,32),
            //   l ∈ [0, 4). That's the unique decomposition used by the
            //   kernel. Bits (l*2, l*2+1) of byte qs[j+lane] hold q.
            for (int kk = 0; kk < TQ2_0_QK_K; ++kk) {
                const int jhalf = (kk >= 128) ? 1 : 0;
                const int j     = jhalf * 32;
                const int k_in_half = kk - jhalf * 128;       // 0..127
                const int l     = k_in_half / 32;             // 0..3
                const int lane  = k_in_half - l * 32;         // 0..31

                const int sign_val = sdist(rng);              // -1, 0, +1
                const uint8_t q = (uint8_t)(sign_val + 1);    // 0,1,2
                // clear + set two bits at (l*2, l*2+1)
                blk.qs[j + lane] &= (uint8_t)~(0x3u << (l * 2));
                blk.qs[j + lane] |= (uint8_t)((q & 0x3u) << (l * 2));
            }
        }
    }
}

// Decode a single logical weight from a packed block_tq2_0 — used only by the
// CPU reference. Mirrors the kernel decode path.
static int tq2_0_decode(const block_tq2_0& blk, int kk)
{
    const int jhalf = (kk >= 128) ? 1 : 0;
    const int j     = jhalf * 32;
    const int k_in_half = kk - jhalf * 128;
    const int l     = k_in_half / 32;
    const int lane  = k_in_half - l * 32;
    const uint8_t byte = blk.qs[j + lane];
    const int q = (int)((byte >> (l * 2)) & 0x3);
    return q - 1;                                // {-1, 0, +1}
}

static void gen_q8_1_acts(std::vector<block_q8_1>& ya, int K, std::mt19937& rng)
{
    const int nb = K / TQ2_0_QK8_1;
    ya.assign((size_t)nb, {});

    std::uniform_int_distribution<int> qdist(-64, 64);
    std::uniform_real_distribution<float> ddist(0.01f, 0.5f);

    for (int b = 0; b < nb; ++b) {
        block_q8_1& a = ya[b];
        a.d = __float2half(ddist(rng));
        a.s = __float2half(0.0f);                // ignored by kernel
        for (int i = 0; i < TQ2_0_QK8_1; ++i) {
            a.qs[i] = (int8_t)qdist(rng);
        }
    }
}

// ── CPU reference ───────────────────────────────────────────────────────────

// ref[m] = sum_{kk} (sign(m,kk) * x_i8[kk]) * d_w_block(m, kk/256) * d_x_block(kk/32)
// Accumulated the same way the kernel does: int32 per q8_1 block, widen to
// fp32, scale, add to row accumulator.
static void cpu_reference(const std::vector<block_tq2_0>& wb,
                          const std::vector<block_q8_1>&  ya,
                          int M, int K,
                          std::vector<float>& ref)
{
    const int nblocks_k = K / TQ2_0_QK_K;
    ref.assign((size_t)M, 0.0f);

    for (int m = 0; m < M; ++m) {
        float row_acc = 0.0f;
        for (int b = 0; b < nblocks_k; ++b) {
            const block_tq2_0& blk = wb[(size_t)m * nblocks_k + b];
            const float d_w = __half2float(blk.d);

            // 8 q8_1 blocks cover one 256-weight block.
            for (int slot = 0; slot < TQ2_0_BLOCKS_PER_K; ++slot) {
                const block_q8_1& ab = ya[(size_t)b * TQ2_0_BLOCKS_PER_K + slot];
                const float d_x = __half2float(ab.d);

                int32_t s = 0;
                for (int i = 0; i < TQ2_0_QK8_1; ++i) {
                    const int kk = slot * TQ2_0_QK8_1 + i;
                    const int w = tq2_0_decode(blk, kk);
                    s += w * (int)ab.qs[i];
                }
                row_acc += d_w * d_x * (float)s;
            }
        }
        ref[m] = row_acc;
    }
}

// ── Compare helper ──────────────────────────────────────────────────────────

struct DiffStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    int    argmax = -1;
};

static DiffStats diff_stats(const std::vector<float>& a, const std::vector<float>& b)
{
    DiffStats s{};
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > s.max_abs) { s.max_abs = d; s.argmax = (int)i; }
        sum += d;
    }
    s.mean_abs = sum / (double)a.size();
    return s;
}

// ── Test driver ─────────────────────────────────────────────────────────────

struct TestSpec {
    const char* name;
    int M;
    int K;
    bool f16_out;
    double max_abs_threshold;
    uint32_t seed;
};

static bool run_one(const TestSpec& t)
{
    // Sanity: both dimensions must respect the kernel's preconditions.
    if (t.K % TQ2_0_QK_K != 0) {
        fprintf(stderr, "TEST: %s ... SKIP (K=%d not a multiple of %d)\n",
                t.name, t.K, TQ2_0_QK_K);
        return false;
    }
    if (t.M % 8 != 0) {
        // kernel grid is ceil(M/8) — non-multiples are fine but the test
        // doesn't exercise the tail-row nrows check path. Flag it loudly.
        fprintf(stderr, "TEST: %s ... NOTE M=%d not a multiple of 8 "
                        "(grid tail path exercised)\n", t.name, t.M);
    }

    std::mt19937 rng(t.seed);

    std::vector<block_tq2_0> wb;
    std::vector<block_q8_1>  ya;
    gen_tq2_0_weights(wb, t.M, t.K, rng);
    gen_q8_1_acts(ya, t.K, rng);

    std::vector<float> ref;
    cpu_reference(wb, ya, t.M, t.K, ref);

    // Device alloc
    block_tq2_0* d_vx = nullptr;
    block_q8_1*  d_vy = nullptr;
    void*        d_dst = nullptr;
    const size_t bytes_wb  = wb.size() * sizeof(block_tq2_0);
    const size_t bytes_ya  = ya.size() * sizeof(block_q8_1);
    const size_t bytes_dst = t.f16_out
                           ? (size_t)t.M * sizeof(__half)
                           : (size_t)t.M * sizeof(float);
    HIP_OK(hipMalloc(&d_vx,  bytes_wb));
    HIP_OK(hipMalloc(&d_vy,  bytes_ya));
    HIP_OK(hipMalloc(&d_dst, bytes_dst));
    HIP_OK(hipMemcpy(d_vx, wb.data(), bytes_wb, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_vy, ya.data(), bytes_ya, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(d_dst, 0, bytes_dst));

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    if (t.f16_out) {
        ternary_gemv_tq2_0_q8_1_f16_launch(d_vx, d_vy, d_dst,
                                           t.K, t.M, (void*)stream);
    } else {
        ternary_gemv_tq2_0_q8_1_launch(d_vx, d_vy, d_dst,
                                       t.K, t.M, (void*)stream);
    }

    // Surface async launch errors before the stream sync hides them.
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        fprintf(stderr, "TEST: %s ... FAIL launch error %d (%s)\n",
                t.name, (int)le, hipGetErrorString(le));
        HIP_OK(hipStreamDestroy(stream));
        HIP_OK(hipFree(d_vx));
        HIP_OK(hipFree(d_vy));
        HIP_OK(hipFree(d_dst));
        return false;
    }
    HIP_OK(hipStreamSynchronize(stream));

    std::vector<float> got((size_t)t.M, 0.0f);
    if (t.f16_out) {
        std::vector<uint16_t> h16((size_t)t.M, 0);
        HIP_OK(hipMemcpy(h16.data(), d_dst, bytes_dst, hipMemcpyDeviceToHost));
        for (int i = 0; i < t.M; ++i) {
            __half h;
            std::memcpy(&h, &h16[i], sizeof(h));
            got[i] = __half2float(h);
        }
    } else {
        HIP_OK(hipMemcpy(got.data(), d_dst, bytes_dst, hipMemcpyDeviceToHost));
    }

    HIP_OK(hipStreamDestroy(stream));
    HIP_OK(hipFree(d_vx));
    HIP_OK(hipFree(d_vy));
    HIP_OK(hipFree(d_dst));

    DiffStats ds = diff_stats(got, ref);
    const bool pass = (ds.max_abs <= t.max_abs_threshold) &&
                      std::isfinite(ds.max_abs);

    printf("TEST: %-24s ... %s (max_abs=%.3e, mean_abs=%.3e, "
           "threshold=%.1e, argmax=%d, ref=%.6f got=%.6f)\n",
           t.name, pass ? "PASS" : "FAIL",
           ds.max_abs, ds.mean_abs, t.max_abs_threshold,
           ds.argmax,
           ds.argmax >= 0 ? ref[ds.argmax] : 0.0,
           ds.argmax >= 0 ? got[ds.argmax] : 0.0);
    return pass;
}

int main()
{
    // Threshold rationale:
    //
    //   fp32 output: max_abs < 5e-3. d_w and d_x are each fp16 (~10-bit
    //   precision); their product rounds to ~2^-10 * |max product| per
    //   summed slot. With nblocks_k=16 and slots=8, worst-case sum grows
    //   as sqrt(n) for random data but the scales are INDEPENDENT per
    //   block — so error is bounded by ~16 * 8 * 64 (max int32) * 2^-10
    //   ≈ 8 absolute. In practice |ref| ≤ ~100 so 5e-3 is comfortable.
    //
    //   fp16 output: 5e-2. One extra fp16 round-trip at the store site
    //   dominates; 10-bit mantissa on |ref| ≤ 100 gives ~0.1 ULP → ~5e-2
    //   absolute. Widen to 5e-2 to keep occasional large-|ref| rows
    //   passing without overfitting the test.
    //
    // Seeds chosen so that adjacent tests don't share RNG state; each call
    // reseeds so reruns are reproducible.

    bool all_pass = true;

    // Ranked order — smallest / most diagnostic first.
    all_pass &= run_one({"edge-M8-K256",      8,    256,  false, 5e-3, 0xC0FFEE01u});
    all_pass &= run_one({"edge-M8-K512",      8,    512,  false, 5e-3, 0xC0FFEE02u});
    all_pass &= run_one({"mid-1024x1024",     1024, 1024, false, 5e-3, 0xC0FFEE03u});
    all_pass &= run_one({"large-4096x4096",   4096, 4096, false, 5e-3, 0xC0FFEE04u});
    all_pass &= run_one({"mid-1024x1024-f16", 1024, 1024, true,  5e-2, 0xC0FFEE05u});
    all_pass &= run_one({"large-4096x4096-f16", 4096, 4096, true, 5e-2, 0xC0FFEE06u});

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
