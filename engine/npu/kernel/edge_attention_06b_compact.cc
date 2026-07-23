// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// edge_attention_06b_compact.cc — L2-optimized attention kernel for Qwen3-0.6B
//
// Fixes "Resource allocation blocked (L2 48KB limit)" by reducing per-tile
// buffer footprint:
//   1. kContext=8 (half-block), attention done in 2 passes → K/V window halved
//   2. kHeads=4 (WQH=NH/AW=16/4=4)
//   3. No cycle profiling (saves ~1KB of .data LUT constants)
//
// L2 budget per tile:
//   Q window: 4 × 128 × 2 = 1,024 B
//   K window: 2 × 8 × 128 × 2 = 4,096 B  (was 8,192 with kContext=16)
//   V window: same = 4,096 B             (was 8,192)
//   Carrier: 4 × 8 × 2 + 4 × 2 × 4 = 64 + 32 = 96 B
//   Accum: 4 × 128 × 4 = 2,048 B
//   State: 4 × 2 × 4 = 32 B
//   Scratch/scores[8]: 32 B
//   MLIR DMA descriptors + locks: ~4 KB
//   Total: ~15 KB ✓ (within 48 KB)

#include <aie_api/aie.hpp>
#include <adf/intrinsics.h>
#include <stdint.h>

namespace {

constexpr int32_t kHeads = 4;          // WQH = NH/AW = 16/4
constexpr int32_t kContext = 8;        // Half-block (2 passes for full 16)
constexpr int32_t kHeadDim = 128;
constexpr int32_t kGqaRatio = 2;      // NH/NKV = 16/8
constexpr int32_t kWeightDwords = 32; // kHeads * kContext * 2 + kHeads * 2 = 72 → round up
constexpr int32_t kVecLanes = 32;
constexpr int32_t kAccumLanes = kHeads * kHeadDim;
constexpr int32_t kStateDwords = kHeads * 2;
constexpr float kRsqrtHeadDim = 0.0883883476f;

using VecBf16 = aie::vector<bfloat16, kVecLanes>;
using VecF32 = aie::vector<float, kVecLanes>;
using AccF32 = aie::accum<accfloat, kVecLanes>;

static bfloat16 *as_bf16(int32_t *payload) {
    return reinterpret_cast<bfloat16 *>(payload);
}

static float *as_f32(int32_t *payload) {
    return reinterpret_cast<float *>(payload);
}

// Fast exp approximation for negative inputs in range [-20, 0]
static float fast_exp(float value) {
    if (value <= -20.0f) return 0.0f;
    if (value >= 0.0f) return 1.0f;
    constexpr float inv_ln2 = 1.4426950409f;
    constexpr float ln2 = 0.6931471806f;
    int32_t exponent;
    float reduced = __builtin_frexpf(value * inv_ln2, &exponent);
    // frexp gives mantissa in [0.5, 1.0), but we want floor
    // Use the polynomial on the fractional part
    reduced = value - (float)exponent * ln2;
    float r2 = reduced * reduced;
    float r3 = r2 * reduced;
    float r4 = r3 * reduced;
    float poly = 1.0f + reduced + 0.5f * r2 + 0.16666667f * r3 + 0.04166667f * r4;
    union { uint32_t u; float f; } bits;
    bits.u = (uint32_t)(exponent + 127) << 23;
    return bits.f * poly;
}

static float attention_score_bf16(
    int32_t *q_window,
    int32_t *k_window,
    int32_t q_head,
    int32_t token
) {
    const int32_t kv_head = q_head / kGqaRatio;
    const int32_t q_base = q_head * kHeadDim;
    const int32_t k_base = kv_head * kContext * kHeadDim + token * kHeadDim;
    bfloat16 *__restrict q_values = as_bf16(q_window);
    bfloat16 *__restrict k_values = as_bf16(k_window);
    AccF32 acc = aie::zeros<accfloat, kVecLanes>();
    for (int32_t dim = 0; dim < kHeadDim; dim += kVecLanes)
        chess_prepare_for_pipelining chess_loop_range(4, 4) {
        VecBf16 q_vec = aie::load_v<kVecLanes>(q_values + q_base + dim);
        VecBf16 k_vec = aie::load_v<kVecLanes>(k_values + k_base + dim);
        acc = aie::mac(acc, q_vec, k_vec);
    }
    return aie::reduce_add(acc.template to_vector<float>()) * kRsqrtHeadDim;
}

static void accum_v_block_for_head(
    bfloat16 *__restrict v_values,
    bfloat16 *__restrict weights,
    float *__restrict accum_values,
    int32_t q_head,
    float old_scale,
    float block_scale
) {
    const int32_t kv_head = q_head / kGqaRatio;
    const int32_t q_base = q_head * kHeadDim;
    const int32_t v_base = kv_head * kContext * kHeadDim;
    VecF32 old_scale_vec = aie::broadcast<float, kVecLanes>(old_scale);
    VecF32 block_scale_vec = aie::broadcast<float, kVecLanes>(block_scale);
    for (int32_t dim = 0; dim < kHeadDim; dim += kVecLanes)
        chess_prepare_for_pipelining chess_loop_range(4, 4) {
        AccF32 block_acc = aie::zeros<accfloat, kVecLanes>();
        for (int32_t token = 0; token < kContext; token++)
            chess_prepare_for_pipelining chess_loop_range(kContext, kContext) {
            VecBf16 weight_vec =
                aie::broadcast<bfloat16, kVecLanes>(weights[q_head * kContext + token]);
            VecBf16 v_vec =
                aie::load_v<kVecLanes>(v_values + v_base + token * kHeadDim + dim);
            block_acc = aie::mac(block_acc, v_vec, weight_vec);
        }
        VecF32 old_vec = aie::load_v<kVecLanes>(accum_values + q_base + dim);
        AccF32 old_scaled = aie::mul(old_vec, old_scale_vec);
        AccF32 block_scaled = aie::mul(block_acc.template to_vector<float>(), block_scale_vec);
        old_scaled = aie::add(old_scaled, block_scaled);
        aie::store_v(accum_values + q_base + dim, old_scaled.template to_vector<float>());
    }
}

} // namespace

extern "C" {

void qwen3_attention_bf16_make_carrier_masked(
    int32_t *q_window, int32_t *k_window, int32_t *carrier, int32_t window,
    int32_t block, int32_t blocks, int32_t tail_tokens, int32_t q_dwords,
    int32_t k_dwords, int32_t carrier_dwords) {
    const int32_t valid_tokens = block + 1 == blocks ? tail_tokens : kContext;
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + kWeightDwords);
    for (int32_t q_head = 0; q_head < kHeads; q_head++) {
        float scores[kContext];
        float running_max = attention_score_bf16(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < valid_tokens; token++) {
            scores[token] = attention_score_bf16(q_window, k_window, q_head, token);
            if (scores[token] > running_max) running_max = scores[token];
        }
        float weight_sum = 0.0f;
        for (int32_t token = 0; token < kContext; token++) {
            float weight = 0.0f;
            if (token < valid_tokens) {
                weight = fast_exp(scores[token] - running_max);
                weight_sum += weight;
            }
            weights[q_head * kContext + token] = static_cast<bfloat16>(weight);
        }
        scalars[q_head * 2] = running_max;
        scalars[q_head * 2 + 1] = weight_sum;
    }
    (void)window;
    (void)q_dwords;
    (void)k_dwords;
    (void)carrier_dwords;
}

void qwen3_attention_bf16_init_accum(
    int32_t *accum, int32_t *state, int32_t accum_lanes, int32_t state_dwords) {
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    VecF32 zero_vec = aie::zeros<float, kVecLanes>();
    for (int32_t idx = 0; idx < kAccumLanes; idx += kVecLanes)
        chess_prepare_for_pipelining chess_loop_range(16, 16) {
        aie::store_v(accum_values + idx, zero_vec);
    }
    aie::vector<float, kStateDwords> zero_state = aie::zeros<float, kStateDwords>();
    aie::store_v(state_values, zero_state);
    (void)accum_lanes;
    (void)state_dwords;
}

void qwen3_attention_bf16_accum_block(
    int32_t *v_window, int32_t *carrier, int32_t *accum, int32_t *state,
    int32_t block, int32_t v_dwords, int32_t carrier_dwords,
    int32_t accum_lanes, int32_t state_dwords) {
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + kWeightDwords);
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    bfloat16 *v_values = as_bf16(v_window);
    for (int32_t q_head = 0; q_head < kHeads; q_head++) {
        const float block_max = scalars[q_head * 2];
        const float block_sum = scalars[q_head * 2 + 1];
        if (block_sum == 0.0f) continue;
        const float old_max = state_values[q_head * 2];
        const float old_sum = state_values[q_head * 2 + 1];
        float new_max = block_max;
        float old_scale = 0.0f;
        float block_scale = 1.0f;
        if (old_sum != 0.0f) {
            new_max = old_max > block_max ? old_max : block_max;
            old_scale = fast_exp(old_max - new_max);
            block_scale = fast_exp(block_max - new_max);
        }
        accum_v_block_for_head(v_values, weights, accum_values, q_head, old_scale, block_scale);
        state_values[q_head * 2] = new_max;
        state_values[q_head * 2 + 1] = old_sum * old_scale + block_sum * block_scale;
    }
    (void)block;
    (void)v_dwords;
    (void)carrier_dwords;
    (void)accum_lanes;
    (void)state_dwords;
}

void qwen3_attention_bf16_finish_accum(
    int32_t *accum, int32_t *state, int32_t *output, int32_t output_dwords,
    int32_t accum_lanes, int32_t state_dwords) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(output);
    for (int32_t idx = 0; idx < output_dwords * 2; idx++)
        payload[idx] = static_cast<bfloat16>(0.0f);
    for (int32_t q_head = 0; q_head < kHeads; q_head++) {
        const float weight_sum = state_values[q_head * 2 + 1];
        const VecF32 inv_sum_vec = aie::broadcast<float, kVecLanes>(1.0f / weight_sum);
        for (int32_t dim = 0; dim < kHeadDim; dim += kVecLanes)
            chess_prepare_for_pipelining chess_loop_range(4, 4) {
            const int32_t lane = q_head * kHeadDim + dim;
            VecF32 accum_vec = aie::load_v<kVecLanes>(accum_values + lane);
            AccF32 scaled = aie::mul(accum_vec, inv_sum_vec);
            aie::store_v(payload + lane, scaled.template to_vector<bfloat16>());
        }
    }
    (void)output_dwords;
    (void)accum_lanes;
    (void)state_dwords;
}

} // extern "C"
