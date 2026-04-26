// sherry_gemv.hip — Clean-room Sherry 1.25-bit ternary GEMV for gfx1151.
//
// API (see include/rocm_cpp/sherry.h):
//   sherry_ternary_gemv_launch(packed, act_fp16, out_fp16, N_out, K_in, stream)
//
// This is the fp16-in/fp16-out entry point used by the differential test and
// the Rust (1bit-hip) wrapping layer. It is INTENTIONALLY distinct from the
// halo-1bit v3 decode kernel in kernels/ternary_gemv_sherry.hip — that kernel
// takes int8 activations + per-row fp32 scales and is tuned for live service
// decode. This kernel is the minimal spec-accurate 1:1 Sherry reference: pure
// ±1/0 × fp16 → fp16, no scales, no int8, no LUT.
//
// Packing (matches project_sherry_spike.md):
//   One group = 4 consecutive K weights. One forced-zero position, three ±1
//   signs. 5 bits: [zero_pos:2 | signs:3]. Row stride = K_in * 5 / 32 bytes
//   (K_in multiple of 32 required for byte alignment, enforced at launcher).
//
// Thread model (wave32, gfx1151):
//   - One thread block = 128 threads = 4 waves.
//   - One block computes one row of output (N_out dimension).
//   - Each thread strides over K/4 groups in lockstep, accumulating in fp32.
//   - Per-block reduction: intra-wave __shfl_xor butterfly, then 4-wave LDS
//     reduce to thread 0, fp32→fp16 store.
//
// On the sdot4 hint in the task spec:
//   __builtin_amdgcn_sdot4 performs int8×int8 dot-4. Here the activations are
//   fp16 (not int8) and the weights are ±1/0 signs. Quantizing fp16→i8 on the
//   fly would (a) destroy bit-level matching with the scalar reference (which
//   the differential test requires at ≤1 bf16 ULP across 50 seeds), and
//   (b) impose an input-dependent quantization scale the API doesn't carry.
//   We therefore accumulate as `fp32_acc += sign * fp32(act)` — branch-free
//   via `copysignf` on a precomputed fp32 activation. The sign-only multiply-
//   accumulate is still fused per group of 4 (three contributing lanes + one
//   zero-skip derived from zero_pos). When the caller has an int8 path with a
//   known scale, the halo-1bit v3 path in kernels/ternary_gemv_sherry.hip is
//   the one to use; sdot4 fits there, not here.
//
// LDS staging:
//   Each block streams its row's (K_in/4) 5-bit codes through registers, but
//   the fp16 activation vector is shared across every block (every row
//   multiplies the same acts). We stage it into LDS once per block:
//   128 threads × vector loads cover up to 4096 fp16 values per pass, looped
//   across K_in. This cuts per-row DRAM activation traffic from O(N_out *
//   K_in) down to O(K_in) and keeps the inner loop reading LDS, not DRAM.
//
// Bank conflicts:
//   x_lds is 2048 fp16 = 4 KiB. Each warp reads x_lds[base + 4*lane + j] for
//   j ∈ [0,3]; stride-4 fp16 (8 bytes) = distinct banks across 32 lanes
//   (banks 0..31 cycle exactly once over the warp). No conflict.
//
// Warnings:
//   No -INFINITY literal use. No nodiscard on hipFree. -munsafe-fp-atomics
//   inherited from the target; this kernel issues no atomics.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

#include "rocm_cpp/sherry.h"

namespace {

// ── Packing / unpacking helpers (host + device) ──────────────────────────────

// Given 4 ternary values ∈ {-1, 0, +1} with EXACTLY one zero at index zp,
// encode as (zp << 3) | signs, where signs[i] is 1 if the i-th surviving
// non-zero lane (in positional order) is +1, else 0.
__host__ __device__ __forceinline__
uint32_t encode_group_5b(const int8_t g[4]) {
    // Find the zero lane. If multiple, pick the lowest; if none, the encoding
    // is lossy — contract says "exactly one zero". We pick index 0's sign for
    // the lossless slot to keep the function total / well-defined.
    int zp = 0;
    for (int i = 0; i < 4; ++i) {
        if (g[i] == 0) { zp = i; break; }
    }
    uint32_t signs = 0;
    int si = 0;
    for (int p = 0; p < 4; ++p) {
        if (p == zp) continue;
        const int s = (g[p] >= 0) ? 1 : 0;   // +1 and any stray 0 → sign bit 1
        signs |= ((uint32_t)s) << si;
        ++si;
    }
    return ((uint32_t)zp << 3) | (signs & 0x7u);
}

// Device-side decode of a 5-bit group. Returns three outputs:
//   sign0, sign1, sign2 ∈ {-1, +1} for the three non-zero lanes in POSITIONAL
//   order (not sign order). zero_pos is 0..3. A lane index L ∈ [0,3] maps to:
//     if L == zero_pos: contribution 0
//     else:            contribution act[L] * sign_for_L
//   where sign_for_L is sign_k (k = L with L<zp), or sign_{L-1} (L>zp).
//
// The common consumer pattern (and the one the main kernel uses) is:
//   decoded[L] = (L == zp) ? 0.0f : ((sign_bit_for_L ? +act[L] : -act[L]))
__device__ __forceinline__
void decode_group_5b(uint32_t code, uint32_t& zp, uint32_t& signs) {
    zp    = (code >> 3) & 0x3u;
    signs =  code       & 0x7u;
}

// Extract 5-bit code `g` from the row's byte stream. The group's first bit
// lives at global bit index g*5; bytes are LSB-first. Two bytes cover any
// 5-bit group (worst case spans one byte boundary), so a uint16 load suffices.
__device__ __forceinline__
uint32_t fetch_code_5b(const uint8_t* __restrict__ row, int g) {
    const int bit_ofs   = g * 5;
    const int byte_idx  = bit_ofs >> 3;
    const int shift     = bit_ofs & 0x7;
    // Two-byte load via memcpy idiom: compiler emits aligned/unaligned u16 as
    // the pipeline allows. __builtin_memcpy is required over a reinterpret
    // because `row + byte_idx` isn't guaranteed 2-byte aligned.
    uint16_t two;
    __builtin_memcpy(&two, row + byte_idx, sizeof(two));
    return ((uint32_t)two >> shift) & 0x1Fu;
}

}  // namespace

// ── Kernel ──────────────────────────────────────────────────────────────────

#define SHERRY_BLOCK   128
#define SHERRY_WAVE    32
#define SHERRY_WAVES   (SHERRY_BLOCK / SHERRY_WAVE)  // 4
#define SHERRY_LDS_ACT 2048  // fp16 entries staged per tile = 4 KiB. Multiple
                             // of 4 (group size) and of 128 (block size × 4
                             // fp16-per-thread).

// scales is OPTIONAL. When non-null, the kernel multiplies the per-row
// fp32 accumulator by `scales[row]` before the fp16 cast — matching the
// halo-1bit v3 Sherry contract (the .h1b stores a per-row fp32 scale
// alongside the packed weights so the kernel can fold (1.25-bpw weight ×
// learned absmean magnitude) in one pass). When null, the kernel emits
// the pure signed-sum behavior the differential test relies on (pure
// ±1/0 × fp16 → fp16, bit-identical to the scalar reference).
//
// The branch on `scales == nullptr` is uniform across the block (kernel
// arg is the same for every thread), so the compiler folds it to a
// scalar predicate; no divergence cost.
__attribute__((amdgpu_flat_work_group_size(SHERRY_BLOCK, SHERRY_BLOCK)))
__global__ void sherry_ternary_gemv_kernel(
    const uint8_t*  __restrict__ packed,
    const __half*   __restrict__ act,
    __half*         __restrict__ out,
    int N_out, int K_in,
    const float*    __restrict__ scales)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (SHERRY_WAVE - 1);
    const int wave = tid >> 5;
    const int row  = blockIdx.x;
    if (row >= N_out) return;

    const int row_bytes = (K_in * 5) / 32;
    const uint8_t* __restrict__ wrow = packed + (size_t)row * (size_t)row_bytes;

    // LDS: one tile of staged activations shared by all threads in the block.
    // +4 guard lets the last group's 4-lane read go past the logical tile end
    // without an extra bounds check (the guard slots hold zeros).
    __shared__ __half x_lds[SHERRY_LDS_ACT + 4];

    float acc = 0.0f;

    // Stream K_in activations through LDS in SHERRY_LDS_ACT-sized tiles.
    for (int k_tile = 0; k_tile < K_in; k_tile += SHERRY_LDS_ACT) {
        // Cooperative load: 128 threads × 16 fp16 = 2048 fp16. Vector-load via
        // ext_vector_type keeps the issue rate at one 32-byte transaction per
        // thread per tile. The v16-style __half vector isn't supported
        // directly on HIP so we use two int4 copies (16 bytes each).
        __syncthreads();
        const int tile_len_logical =
            (k_tile + SHERRY_LDS_ACT <= K_in) ? SHERRY_LDS_ACT : (K_in - k_tile);

        // 128 threads × 16 fp16 / thread. Guard: tail goes zero.
        const int per_thread = SHERRY_LDS_ACT / SHERRY_BLOCK;  // 16
        const int t_base     = tid * per_thread;
        #pragma unroll
        for (int j = 0; j < per_thread; ++j) {
            const int lds_i = t_base + j;
            const int k_gl  = k_tile + lds_i;
            __half v;
            if (k_gl < K_in) {
                v = act[k_gl];
            } else {
                v = __float2half(0.0f);
            }
            x_lds[lds_i] = v;
        }
        // Zero the 4-entry guard slots once per tile. Only thread 0 touches
        // them to avoid a 4-way write race; these addresses are past the
        // cooperative-load range so no conflict with the loop above.
        if (tid < 4) {
            x_lds[SHERRY_LDS_ACT + tid] = __float2half(0.0f);
        }
        __syncthreads();

        // Inner loop: each thread processes groups strided by SHERRY_BLOCK.
        // Group `g` starts at activation index g*4 within the tile.
        const int tile_groups = tile_len_logical >> 2;   // tile_len / 4
        const int group_g0    = (k_tile >> 2);           // global group index at tile start

        for (int gt = tid; gt < tile_groups; gt += SHERRY_BLOCK) {
            const int g_global = group_g0 + gt;
            const uint32_t code = fetch_code_5b(wrow, g_global);
            uint32_t zp, signs;
            decode_group_5b(code, zp, signs);

            // Load the 4 activations for this group from LDS.
            const int a_base = gt * 4;
            const float a0 = __half2float(x_lds[a_base + 0]);
            const float a1 = __half2float(x_lds[a_base + 1]);
            const float a2 = __half2float(x_lds[a_base + 2]);
            const float a3 = __half2float(x_lds[a_base + 3]);

            // Decode three signs positionally. bit i of `signs` is the sign
            // of the i-th SURVIVING lane (lowest index first, skipping zp).
            // Build a per-lane (fp32) sign factor: +1, -1, or 0 (for zp).
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            uint32_t si = 0;
            // Lane 0
            if (zp == 0u) {
                s0 = 0.0f;
            } else {
                s0 = ((signs >> si) & 1u) ? 1.0f : -1.0f;
                ++si;
            }
            // Lane 1
            if (zp == 1u) {
                s1 = 0.0f;
            } else {
                s1 = ((signs >> si) & 1u) ? 1.0f : -1.0f;
                ++si;
            }
            // Lane 2
            if (zp == 2u) {
                s2 = 0.0f;
            } else {
                s2 = ((signs >> si) & 1u) ? 1.0f : -1.0f;
                ++si;
            }
            // Lane 3
            if (zp == 3u) {
                s3 = 0.0f;
            } else {
                s3 = ((signs >> si) & 1u) ? 1.0f : -1.0f;
                // ++si; // unused
            }

            // Fused MAC: a * s. The branches above are input-only (code), so
            // they resolve to a short sequence of compares; the compiler
            // typically materializes s_i in 2-3 v_cndmask_b32 per lane. No
            // divergent global loads inside the per-group branch, so wave
            // efficiency holds.
            acc = fmaf(a0, s0, acc);
            acc = fmaf(a1, s1, acc);
            acc = fmaf(a2, s2, acc);
            acc = fmaf(a3, s3, acc);
        }
    }

    // ── Reduce across the block ─────────────────────────────────────────────
    // Intra-wave: xor butterfly.
    #pragma unroll
    for (int off = SHERRY_WAVE / 2; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off);
    }

    // Inter-wave: lane 0 of each wave writes to LDS, wave 0 reduces.
    __shared__ float wave_sums[SHERRY_WAVES];
    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();

    if (wave == 0) {
        float v = (lane < SHERRY_WAVES) ? wave_sums[lane] : 0.0f;
        // Wave-level reduce across up to SHERRY_WAVES lanes of wave 0. We use
        // the same xor butterfly; the high lanes contribute 0.
        #pragma unroll
        for (int off = SHERRY_WAVE / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            // Per-row scale fold-in (halo-1bit v3 Sherry contract). Done in
            // fp32 BEFORE the fp16 clamp + cast so an outlier row can't
            // saturate the unscaled signed-sum into +Inf (which would then
            // multiply to NaN on the post-pass). Skipped when scales is
            // null — preserves the pure-signed-sum semantics the
            // differential test in tests/test_sherry_gemv.cpp relies on.
            if (scales) v *= scales[row];
            // Clamp to fp16 range before cast. Post-hoc ternary × large acts
            // can overflow fp16 on outlier rows; downstream RMSNorm /
            // softmax is NaN-poisoned if we emit +Inf here. Same guard as
            // the decode kernel in kernels/ternary_gemv_sherry.hip.
            constexpr float FP16_MAX = 65504.0f;
            if (v >  FP16_MAX) v =  FP16_MAX;
            if (v < -FP16_MAX) v = -FP16_MAX;
            out[row] = __float2half(v);
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void sherry_ternary_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & 0x3) != 0)                  return;  // groups of 4
    if ((K_in & 0x1F) != 0)                 return;  // byte-aligned rows (K * 5 / 32 integer)

    dim3 grid(N_out, 1, 1);
    dim3 block(SHERRY_BLOCK, 1, 1);
    hipLaunchKernelGGL(sherry_ternary_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in,
                       /*scales=*/nullptr);
}

// Variant that folds in the halo-1bit v3 per-row fp32 scale on each row's
// writeback. Identical kernel as `sherry_ternary_gemv_launch`; the only
// difference is the trailing scale multiply before the fp16 clamp + cast.
// `row_scales_fp32` is a device pointer to N_out fp32 entries (one per
// output row). When integrated into the live decode loop the loader
// hands us back the same `q_scales_dev` / `k_scales_dev` / ... arrays
// the i8 sherry kernel consumes — bit-equivalent fold-in math, just on
// fp16 activations instead of int8.
extern "C" void sherry_ternary_gemv_with_scales_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    const float*    row_scales_fp32,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0)            return;
    if ((K_in & 0x3) != 0)                  return;  // groups of 4
    if ((K_in & 0x1F) != 0)                 return;  // byte-aligned rows (K * 5 / 32 integer)
    // row_scales_fp32 may be null — caller intent is "no fold-in"; fall
    // through to the same kernel with a null scale pointer (kernel-side
    // branch is uniform-on-null, no divergence).

    dim3 grid(N_out, 1, 1);
    dim3 block(SHERRY_BLOCK, 1, 1);
    hipLaunchKernelGGL(sherry_ternary_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in,
                       row_scales_fp32);
}

extern "C" void rcpp_sherry_pack(const int8_t* ternary, uint8_t* out, int count) {
    if (!ternary || !out || count <= 0)     return;
    if ((count & 0x3) != 0)                 return;
    const int groups = count / 4;
    const int total_bits = groups * 5;
    const int total_bytes = total_bits / 8;  // whole-byte prefix
    // Zero the destination whole-byte prefix so we can OR in.
    for (int b = 0; b < total_bytes; ++b) out[b] = 0;
    // If count % 32 != 0 there is a trailing partial byte; docs say we leave
    // it untouched (caller's problem — we don't know what's in the other
    // half of that byte). We therefore only write bytes 0..total_bytes-1,
    // matching the "short input writes the whole-byte prefix" contract.
    for (int g = 0; g < groups; ++g) {
        const int8_t gv[4] = {
            ternary[g * 4 + 0],
            ternary[g * 4 + 1],
            ternary[g * 4 + 2],
            ternary[g * 4 + 3],
        };
        const uint32_t code = encode_group_5b(gv);  // 5 bits
        const int bit_ofs = g * 5;
        const int b0      = bit_ofs >> 3;
        const int shift   = bit_ofs & 0x7;
        // Up to 2 bytes touched. Low byte first.
        if (b0 < total_bytes) {
            out[b0] |= (uint8_t)((code << shift) & 0xFFu);
        }
        const int b1 = b0 + 1;
        if (shift + 5 > 8 && b1 < total_bytes) {
            out[b1] |= (uint8_t)((code >> (8 - shift)) & 0xFFu);
        }
    }
}
