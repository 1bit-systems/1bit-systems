#include <cstdio>
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// Zaya MoE Ternary GEMV — expert-sparse ternary compute for Zaya's hybrid
// CCA + MoE architecture with EDA router.
//
// Packing: TQ1 base-3 ternary (1.6 bpw), lossless for ternary weights.
//   5 ternaries per byte: byte = d0 + d1·3 + d2·9 + d3·27 + d4·81
//   d_i = 0 -> -1, 1 -> 0, 2 -> +1
//   Effective: 1.6 bpw, no accuracy cost (unlike Sherry's 3:4 sparsity).
//
// Expert-sparse dispatch:
//   The EDA router selects top-k experts per token. This kernel loads ONLY
//   the selected experts' packed weights, computes ternary GEMV, multiplies
//   by the router's combination weights, and accumulates into the output.
//   Unselected experts contribute nothing (zero).
//
// Block layout:
//   1 block per token × top_k expert. Each wave processes one selected
//   expert for its token. With 128 threads, 4 waves, each wave handles
//   ceil(top_k/4) experts sequentially.
//
// Output: fp16 (fused convert, per-row scale + router weight applied).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <mutex>

#define BLOCK_SIZE             128
#define WARP_SIZE              32
#define ROWS_PER_WAVE          1       // each wave = 1 expert result
#define WEIGHTS_PER_MACROGROUP 20      // 4 bytes = 1 u32
#define LDS_TILE_I8            2560    // 128 × 20 bytes (BLOCK_SIZE × 20)

// ── Base-3 decode LUT (shared with TQ1 kernel) ─────────────────────────────
__constant__ uint32_t zaya_tq1_lut_lo[256];
__constant__ uint32_t zaya_tq1_lut_hi[256];

static uint32_t _zaya_pack4i8(int8_t a, int8_t b, int8_t c, int8_t d) {
    return   (uint32_t)(uint8_t)a
         | ((uint32_t)(uint8_t)b <<  8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

static std::once_flag zaya_tq1_lut_flag;
static void init_zaya_tq1_lut(hipStream_t stream) {
    uint32_t lo[256], hi[256];
    for (int byte = 0; byte < 256; ++byte) {
        int b = byte;
        int8_t v[5] = {0, 0, 0, 0, 0};
        if (b < 243) {
            for (int i = 0; i < 5; ++i) {
                int d = b % 3;
                v[i]  = (int8_t)(d - 1);  // -1, 0, +1
                b    /= 3;
            }
        }
        lo[byte] = _zaya_pack4i8(v[0], v[1], v[2], v[3]);
        hi[byte] = _zaya_pack4i8(v[4], 0, 0, 0);
    }
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(zaya_tq1_lut_lo), lo, sizeof(lo), 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(zaya_tq1_lut_hi), hi, sizeof(hi), 0, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
}

// ── Zaya MoE Ternary GEMV Kernel ────────────────────────────────────────────
//
// For each token, loads the selected experts' TQ1-packed ternary weights,
// computes the dot product against shared input activations, multiplies by
// the per-expert scale AND the router combination weight, and stores fp16.
//
// Parameters match the EDA router output layout:
//   expert_indices[tokens][top_k]  — which experts are active per token
//   expert_weights[tokens][top_k]  — combination weight (fp32, from softmax)
//
// Kernel parameters:
//   packed_experts[num_experts][K/5 bytes viewed as u32s] — TQ1 ternary weights
//   x_i8[K] — INT8 input activations (shared across all experts for this token)
//   x_scale — activation quantization scale
//   expert_scales[num_experts] — per-expert output scale
//   expert_indices — [tokens × top_k] int32, row-major: idx[t*top_k + e]
//   expert_weights — [tokens × top_k] float, row-major: w[t*top_k + e]
//   y_f16[tokens][num_experts] — output buffer (only top_k slots populated)
//   tokens, num_experts, top_k, K_padded (multiple of 20)

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void zaya_moe_ternary_gemv_kernel(
    const uint32_t* __restrict__ packed_experts,  // [num_experts][u32_per_row]
    const int8_t*   __restrict__ x_i8,            // [K_padded]
    float                       x_scale,
    const float*    __restrict__ expert_scales,    // [num_experts]
    const int32_t*  __restrict__ expert_indices,    // [tokens × top_k]
    const float*    __restrict__ expert_weights,    // [tokens × top_k]
    __half*         __restrict__ y_f16,            // [tokens][num_experts]
    int tokens, int num_experts, int top_k, int K_padded)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;  // 0..3

    // Block ID gives us the token index.
    const int token = blockIdx.x;

    // Each block processes one token. Each wave handles one expert selection.
    // If top_k > 4, each wave loops; if top_k < 4, some waves are idle.
    const int experts_per_wave = (top_k + 3) / 4;  // ceil(top_k/4)
    const int u32_per_row = K_padded / WEIGHTS_PER_MACROGROUP;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    for (int kw = 0; kw < experts_per_wave; ++kw) {
        const int expert_slot = wave * experts_per_wave + kw;
        if (expert_slot >= top_k) break;

        const int expert_idx = expert_indices[(size_t)token * top_k + expert_slot];
        const float router_wt = expert_weights[(size_t)token * top_k + expert_slot];
        // Guard: ignore invalid expert indices
        if (expert_idx < 0 || expert_idx >= num_experts) continue;

        const float per_expert_scale = expert_scales[expert_idx];

        int32_t acc = 0;

        // Stream through K in LDS tiles
        for (int k_base = 0; k_base < K_padded; k_base += LDS_TILE_I8) {
            __syncthreads();
            // Load activation tile into LDS
            using v16 = __attribute__((ext_vector_type(16))) int8_t;
            const int lds_off16 = tid * 16;
            const int k_start16 = k_base + lds_off16;
            v16* __restrict__ lds_v = reinterpret_cast<v16*>(&x_shared[lds_off16]);
            if (k_start16 + 16 <= K_padded) {
                const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start16);
                *lds_v = *src_v;
            } else {
                #pragma unroll
                for (int b = 0; b < 16; ++b) {
                    int k = k_start16 + b;
                    x_shared[lds_off16 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
                }
            }
            // Upper 512 bytes: 4 more per thread
            const int lds_off4 = BLOCK_SIZE * 16 + tid * 4;
            const int k_start4 = k_base + lds_off4;
            if (k_start4 + 4 <= K_padded) {
                *reinterpret_cast<uint32_t*>(&x_shared[lds_off4])
                    = *reinterpret_cast<const uint32_t*>(x_i8 + k_start4);
            } else {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int k = k_start4 + b;
                    x_shared[lds_off4 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
                }
            }
            __syncthreads();

            const int tile_macrogroups = LDS_TILE_I8 / WEIGHTS_PER_MACROGROUP;  // 128
            const int tile_mg_start   = k_base / WEIGHTS_PER_MACROGROUP;

            const uint32_t* row_packed = packed_experts
                + (size_t)expert_idx * u32_per_row
                + tile_mg_start;

            for (int mg = lane; mg < tile_macrogroups; mg += WARP_SIZE) {
                const int global_mg = tile_mg_start + mg;
                if (global_mg >= u32_per_row) break;
                const uint32_t packed_word = row_packed[mg];

                // Decode 4 bytes × 5 ternaries via LUT
                const uint32_t b0 = (packed_word >>  0) & 0xFFu;
                const uint32_t b1 = (packed_word >>  8) & 0xFFu;
                const uint32_t b2 = (packed_word >> 16) & 0xFFu;
                const uint32_t b3 = (packed_word >> 24) & 0xFFu;

                const uint32_t lo0 = zaya_tq1_lut_lo[b0], hi0 = zaya_tq1_lut_hi[b0];
                const uint32_t lo1 = zaya_tq1_lut_lo[b1], hi1 = zaya_tq1_lut_hi[b1];
                const uint32_t lo2 = zaya_tq1_lut_lo[b2], hi2 = zaya_tq1_lut_hi[b2];
                const uint32_t lo3 = zaya_tq1_lut_lo[b3], hi3 = zaya_tq1_lut_hi[b3];

                uint32_t w[5];
                w[0] = lo0;
                w[1] = __builtin_amdgcn_perm(lo1, hi0, 0x06050400u);
                {   const uint32_t t = __builtin_amdgcn_perm(hi1, lo1, 0x07060403u);
                    w[2] = __builtin_amdgcn_perm(lo2, t,   0x05040100u); }
                {   const uint32_t t = __builtin_amdgcn_perm(hi2, lo2, 0x07040302u);
                    w[3] = __builtin_amdgcn_perm(lo3, t,   0x04020100u); }
                w[4] = __builtin_amdgcn_perm(hi3, lo3, 0x04030201u);

                const int x_offset = mg * WEIGHTS_PER_MACROGROUP;
                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);

                int32_t local = 0;
                #pragma unroll
                for (int j = 0; j < 5; ++j) {
                    local = __builtin_amdgcn_sudot4(true, w[j], true, xpack[j], local, false);
                }
                acc += local;
            }
        }

        // Warp reduction
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }

        if (lane == 0) {
            // Apply activation scale, per-expert scale, and router weight
            // Clamp to FP16 range to prevent NaN propagation through RMSNorm
            constexpr float FP16_MAX = 65504.0f;
            float out = (float)acc * x_scale * per_expert_scale * router_wt;
            out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
            // Write to y_f16[token][expert_idx]
            y_f16[(size_t)token * num_experts + expert_idx] = __float2half(out);
        }
    }
}

// ── Batched prefill variant ─────────────────────────────────────────────────
// Same TQ1 packing, but reads activations from a batch buffer [B][K].
// Grid: (B × ceil(top_k/4), 1, 1) — one block per token, waves handle experts.
// Output: y_f16[B][M] — accumulated expert outputs.

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void zaya_moe_ternary_gemv_batched_kernel(
    const uint32_t* __restrict__ packed_experts,
    const int8_t*   __restrict__ x_i8_batch,     // [B × K_padded]
    float                       x_scale,
    const float*    __restrict__ expert_scales,
    const int32_t*  __restrict__ expert_indices,   // [B × top_k]
    const float*    __restrict__ expert_weights,   // [B × top_k]
    __half*         __restrict__ y_f16,            // [B][num_experts]
    int B, int num_experts, int top_k, int K_padded)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    // blockIdx.x encodes (token * num_experts + expert_slot_in_wave)
    // Actually simpler: blockIdx.x = token, processed same as single-token variant
    // but each block reads from x_i8_batch[token * K_padded].
    const int token = blockIdx.x;
    if (token >= B) return;

    // Same as single variant, but reads x from batched offset
    const int experts_per_wave = (top_k + 3) / 4;
    const int u32_per_row = K_padded / WEIGHTS_PER_MACROGROUP;
    const int8_t* x_i8 = x_i8_batch + (size_t)token * K_padded;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    // Pre-compute which experts this wave handles (all waves participate in syncthreads)
    int my_experts[4];   // max experts_per_wave = ceil(top_k/4), top_k <= 16
    int my_expert_count = 0;
    for (int kw = 0; kw < experts_per_wave; ++kw) {
        int slot = wave * experts_per_wave + kw;
        if (slot < top_k) {
            int idx = expert_indices[(size_t)token * top_k + slot];
            if (idx >= 0 && idx < num_experts) {
                my_experts[my_expert_count++] = slot;
            }
        }
    }

    for (int k_base = 0; k_base < K_padded; k_base += LDS_TILE_I8) {
        __syncthreads();
            using v16 = __attribute__((ext_vector_type(16))) int8_t;
            const int lds_off16 = tid * 16;
            const int k_start16 = k_base + lds_off16;
            v16* __restrict__ lds_v = reinterpret_cast<v16*>(&x_shared[lds_off16]);
            if (k_start16 + 16 <= K_padded) {
                const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start16);
                *lds_v = *src_v;
            } else {
                #pragma unroll
                for (int b = 0; b < 16; ++b) {
                    int k = k_start16 + b;
                    x_shared[lds_off16 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
                }
            }
            const int lds_off4 = BLOCK_SIZE * 16 + tid * 4;
            const int k_start4 = k_base + lds_off4;
            if (k_start4 + 4 <= K_padded) {
                *reinterpret_cast<uint32_t*>(&x_shared[lds_off4])
                    = *reinterpret_cast<const uint32_t*>(x_i8 + k_start4);
            } else {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    int k = k_start4 + b;
                    x_shared[lds_off4 + b] = (k < K_padded) ? x_i8[k] : (int8_t)0;
                }
            }
            __syncthreads();

            // All waves reach this syncthreads. Each wave processes its experts.
            const int tile_macrogroups = LDS_TILE_I8 / WEIGHTS_PER_MACROGROUP;
            const int tile_mg_start   = k_base / WEIGHTS_PER_MACROGROUP;

            for (int ei = 0; ei < my_expert_count; ++ei) {
                const int expert_slot = my_experts[ei];
                const int expert_idx = expert_indices[(size_t)token * top_k + expert_slot];
                const float router_wt = expert_weights[(size_t)token * top_k + expert_slot];
                const float per_expert_scale = expert_scales[expert_idx];

                const uint32_t* row_packed = packed_experts
                    + (size_t)expert_idx * u32_per_row
                    + tile_mg_start;

                int32_t local_acc = 0;
                for (int mg = lane; mg < tile_macrogroups; mg += WARP_SIZE) {
                    const int global_mg = tile_mg_start + mg;
                    if (global_mg >= u32_per_row) break;
                    const uint32_t packed_word = row_packed[mg];
                    const uint32_t b0 = (packed_word >>  0) & 0xFFu;
                    const uint32_t b1 = (packed_word >>  8) & 0xFFu;
                    const uint32_t b2 = (packed_word >> 16) & 0xFFu;
                    const uint32_t b3 = (packed_word >> 24) & 0xFFu;
                    const uint32_t lo0 = zaya_tq1_lut_lo[b0], hi0 = zaya_tq1_lut_hi[b0];
                    const uint32_t lo1 = zaya_tq1_lut_lo[b1], hi1 = zaya_tq1_lut_hi[b1];
                    const uint32_t lo2 = zaya_tq1_lut_lo[b2], hi2 = zaya_tq1_lut_hi[b2];
                    const uint32_t lo3 = zaya_tq1_lut_lo[b3], hi3 = zaya_tq1_lut_hi[b3];
                    uint32_t w[5];
                    w[0] = lo0;
                    w[1] = __builtin_amdgcn_perm(lo1, hi0, 0x06050400u);
                    {   const uint32_t t = __builtin_amdgcn_perm(hi1, lo1, 0x07060403u);
                        w[2] = __builtin_amdgcn_perm(lo2, t,   0x05040100u); }
                    {   const uint32_t t = __builtin_amdgcn_perm(hi2, lo2, 0x07040302u);
                        w[3] = __builtin_amdgcn_perm(lo3, t,   0x04020100u); }
                    w[4] = __builtin_amdgcn_perm(hi3, lo3, 0x04030201u);
                    const int x_offset = mg * WEIGHTS_PER_MACROGROUP;
                    const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);
                    int32_t local = 0;
                    #pragma unroll
                    for (int j = 0; j < 5; ++j) {
                        local = __builtin_amdgcn_sudot4(true, w[j], true, xpack[j], local, false);
                    }
                    local_acc += local;
                }

                // Warp reduction for this expert
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    local_acc += __shfl_xor(local_acc, offset);
                }

                if (lane == 0) {
                    constexpr float FP16_MAX = 65504.0f;
                    float out = (float)local_acc * x_scale * per_expert_scale * router_wt;
                    out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
                    y_f16[(size_t)token * num_experts + expert_idx] = __float2half(out);
                }
            }
        }
}

// ── C API launchers ─────────────────────────────────────────────────────────
extern "C" {

void zaya_moe_ternary_gemv_launch(
    const void* packed_experts, const void* x_i8,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int tokens, int num_experts, int top_k, int K_padded,
    void* stream)
{
    hipStream_t s = (hipStream_t)stream;
    std::call_once(zaya_tq1_lut_flag, init_zaya_tq1_lut, s);

    // Single token: block per token
    dim3 grid(tokens, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(zaya_moe_ternary_gemv_kernel,
        grid, block, 0, s,
        static_cast<const uint32_t*>(packed_experts),
        static_cast<const int8_t*>(x_i8),
        x_scale,
        static_cast<const float*>(expert_scales),
        static_cast<const int32_t*>(expert_indices),
        static_cast<const float*>(expert_weights),
        static_cast<__half*>(y_f16),
        tokens, num_experts, top_k, K_padded);
}

void zaya_moe_ternary_gemv_batched_launch(
    const void* packed_experts, const void* x_i8_batch,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int B, int num_experts, int top_k, int K_padded,
    void* stream)
{
    hipStream_t s = (hipStream_t)stream;
    std::call_once(zaya_tq1_lut_flag, init_zaya_tq1_lut, s);

    dim3 grid(B, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(zaya_moe_ternary_gemv_batched_kernel,
        grid, block, 0, s,
        static_cast<const uint32_t*>(packed_experts),
        static_cast<const int8_t*>(x_i8_batch),
        x_scale,
        static_cast<const float*>(expert_scales),
        static_cast<const int32_t*>(expert_indices),
        static_cast<const float*>(expert_weights),
        static_cast<__half*>(y_f16),
        B, num_experts, top_k, K_padded);
}

} // extern "C"
