#include <aie_api/aie.hpp>
#include <adf/intrinsics.h>
#include <stdint.h>

namespace {

constexpr int32_t kHeads = 8;
constexpr int32_t kContext = 16;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kGqaRatio = 4;
constexpr int32_t kWeightDwords = 64;
constexpr int32_t kVecLanes = 32;
constexpr int32_t kAccumLanes = kHeads * kHeadDim;
constexpr int32_t kStateDwords = kHeads * 2;
constexpr float kRsqrtHeadDim = 0.0883883476f;

using VecBf16 = aie::vector<bfloat16, kVecLanes>;
using VecF32 = aie::vector<float, kVecLanes>;
using AccF32 = aie::accum<accfloat, kVecLanes>;

#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
constexpr int32_t kEdgeProfileMarker = 0x41545450;
static uint64_t g_init_cycles = 0;
static uint64_t g_carrier_cycles = 0;
static uint64_t g_accum_cycles = 0;
static uint64_t g_finish_cycles = 0;
static int32_t g_carrier_calls = 0;
static int32_t g_accum_calls = 0;

__attribute__((always_inline)) static inline uint64_t current_cycles() {
    return ::aie::tile::current().cycles();
}

__attribute__((always_inline)) static inline void store_u64_words(
    int32_t *words,
    int32_t index,
    uint64_t value
) {
    words[index] = static_cast<int32_t>(value & 0xffffffffULL);
    words[index + 1] = static_cast<int32_t>(value >> 32);
}

static void reset_profile() {
    g_init_cycles = 0;
    g_carrier_cycles = 0;
    g_accum_cycles = 0;
    g_finish_cycles = 0;
    g_carrier_calls = 0;
    g_accum_calls = 0;
}

static void emit_profile_summary(int32_t *output) {
    output[0] = kEdgeProfileMarker;
    store_u64_words(output, 1, g_init_cycles);
    store_u64_words(output, 3, g_carrier_cycles);
    store_u64_words(output, 5, g_accum_cycles);
    store_u64_words(output, 7, g_finish_cycles);
    output[9] = g_carrier_calls;
    output[10] = g_accum_calls;
}
#endif

static bfloat16 bf16_lane(int32_t *payload, int32_t lane) {
    bfloat16 *values = reinterpret_cast<bfloat16 *>(payload);
    return values[lane];
}

static bfloat16 *as_bf16(int32_t *payload) {
    return reinterpret_cast<bfloat16 *>(payload);
}

static float *as_f32(int32_t *payload) {
    return reinterpret_cast<float *>(payload);
}

static int32_t floor_i32(float value) {
    const int32_t truncated = static_cast<int32_t>(value);
    return static_cast<float>(truncated) > value ? truncated - 1 : truncated;
}

static float pow2_i32(int32_t exponent) {
    if (exponent < -126) {
        return 0.0f;
    }
    if (exponent > 127) {
        exponent = 127;
    }
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = static_cast<uint32_t>(exponent + 127) << 23;
    return bits.f;
}

static float fast_exp(float value) {
    if (value <= -20.0f) {
        return 0.0f;
    }
    if (value >= 0.0f) {
        return 1.0f;
    }
    constexpr float inv_ln2 = 1.4426950409f;
    constexpr float ln2 = 0.6931471806f;
    const int32_t exponent = floor_i32(value * inv_ln2);
    const float reduced = value - static_cast<float>(exponent) * ln2;
    const float r2 = reduced * reduced;
    const float r3 = r2 * reduced;
    const float r4 = r3 * reduced;
    const float polynomial = 1.0f + reduced + 0.5f * r2 + 0.16666667f * r3 + 0.04166667f * r4;
    return pow2_i32(exponent) * polynomial;
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
    int32_t *q_window,
    int32_t *k_window,
    int32_t *carrier,
    int32_t window,
    int32_t block,
    int32_t blocks,
    int32_t tail_tokens,
    int32_t q_dwords,
    int32_t k_dwords,
    int32_t carrier_dwords
) {
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    const int32_t valid_tokens = block + 1 == blocks ? tail_tokens : kContext;
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + kWeightDwords);
    for (int32_t q_head = 0; q_head < kHeads; q_head++) {
        float scores[kContext];
        float running_max = attention_score_bf16(q_window, k_window, q_head, 0);
        scores[0] = running_max;
        for (int32_t token = 1; token < valid_tokens; token++) {
            scores[token] = attention_score_bf16(q_window, k_window, q_head, token);
            if (scores[token] > running_max) {
                running_max = scores[token];
            }
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
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    g_carrier_cycles += current_cycles() - start;
    g_carrier_calls += 1;
#endif
}

void qwen3_attention_bf16_init_accum(
    int32_t *accum,
    int32_t *state,
    int32_t accum_lanes,
    int32_t state_dwords
) {
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    reset_profile();
    const uint64_t start = current_cycles();
#endif
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    VecF32 zero_vec = aie::zeros<float, kVecLanes>();
    for (int32_t idx = 0; idx < kAccumLanes; idx += kVecLanes)
        chess_prepare_for_pipelining chess_loop_range(32, 32) {
        aie::store_v(accum_values + idx, zero_vec);
    }
    aie::vector<float, kStateDwords> zero_state = aie::zeros<float, kStateDwords>();
    aie::store_v(state_values, zero_state);

    (void)accum_lanes;
    (void)state_dwords;
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    g_init_cycles = current_cycles() - start;
#endif
}

void qwen3_attention_bf16_accum_block(
    int32_t *v_window,
    int32_t *carrier,
    int32_t *accum,
    int32_t *state,
    int32_t block,
    int32_t v_dwords,
    int32_t carrier_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    bfloat16 *weights = reinterpret_cast<bfloat16 *>(carrier);
    float *scalars = as_f32(carrier + kWeightDwords);
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    bfloat16 *v_values = as_bf16(v_window);

    for (int32_t q_head = 0; q_head < kHeads; q_head++) {
        const float block_max = scalars[q_head * 2];
        const float block_sum = scalars[q_head * 2 + 1];
        if (block_sum == 0.0f) {
            continue;
        }
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
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    g_accum_cycles += current_cycles() - start;
    g_accum_calls += 1;
#endif
}

void qwen3_attention_bf16_finish_accum(
    int32_t *accum,
    int32_t *state,
    int32_t *output,
    int32_t output_dwords,
    int32_t accum_lanes,
    int32_t state_dwords
) {
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    float *accum_values = as_f32(accum);
    float *state_values = as_f32(state);
    bfloat16 *payload = reinterpret_cast<bfloat16 *>(output);
    for (int32_t idx = 0; idx < output_dwords * 2; idx++) {
        payload[idx] = static_cast<bfloat16>(0.0f);
    }
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

    (void)accum_lanes;
    (void)state_dwords;
#ifdef QWEN3_ENABLE_EDGE_ATTENTION_CYCLE_PROFILE
    g_finish_cycles = current_cycles() - start;
    emit_profile_summary(output);
#endif
}

} // extern "C"
