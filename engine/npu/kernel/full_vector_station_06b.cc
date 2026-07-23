#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int32_t kRmsVecLanes = 32;
constexpr int32_t kFullVectorBlockLanes = 512;
constexpr int32_t kFullVectorAddLanes = 16;

#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
constexpr int32_t kFullVectorProfileMarker = 0x46564c50;
static uint64_t g_input_norm_cycles = 0;
static uint64_t g_post_norm_cycles = 0;
static uint64_t g_o_cycles = 0;
static uint64_t g_down_cycles = 0;
static int32_t g_o_blocks = 0;
static int32_t g_down_blocks = 0;

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
    g_input_norm_cycles = 0;
    g_post_norm_cycles = 0;
    g_o_cycles = 0;
    g_down_cycles = 0;
    g_o_blocks = 0;
    g_down_blocks = 0;
}

static void emit_profile_summary(int32_t *output) {
    output[0] = kFullVectorProfileMarker;
    store_u64_words(output, 1, g_input_norm_cycles);
    store_u64_words(output, 3, g_post_norm_cycles);
    store_u64_words(output, 5, g_o_cycles);
    store_u64_words(output, 7, g_down_cycles);
    output[9] = g_o_blocks;
    output[10] = g_down_blocks;
}
#endif

static bfloat16 *as_bf16(int32_t *words) {
    return reinterpret_cast<bfloat16 *>(words);
}

static uint16_t bf16_bits(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits = {value};
    const uint32_t lsb = (bits.u32 >> 16) & 1u;
    bits.u32 += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits.u32 >> 16);
}

static bfloat16 bf16_rne(float value) {
    bfloat16 output;
    reinterpret_cast<uint16_t *>(&output)[0] = bf16_bits(value);
    return output;
}

static void store_bf16_rne(bfloat16 *values, int32_t lane, float value) {
    reinterpret_cast<uint16_t *>(values)[lane] = bf16_bits(value);
}

static float fast_rsqrt(float value) {
    if (value <= 0.0f) {
        return 1.0f;
    }
    float y = 1.0f;
    if (value < 0.000244140625f) {
        y = 64.0f;
    } else if (value < 0.0009765625f) {
        y = 32.0f;
    } else if (value < 0.00390625f) {
        y = 16.0f;
    } else if (value < 0.015625f) {
        y = 8.0f;
    } else if (value < 0.0625f) {
        y = 4.0f;
    } else if (value < 0.25f) {
        y = 2.0f;
    } else if (value > 1.0f) {
        y = 0.5f;
    }
    if (value > 4.0f) {
        y = 0.25f;
    }
    if (value > 16.0f) {
        y = 0.125f;
    }
    const float half = value * 0.5f;
    for (int32_t iter = 0; iter < 10; iter++)
        chess_prepare_for_pipelining chess_loop_range(10, 10) {
        y = y * (1.5f - half * y * y);
    }
    return y;
}

static float rms_scale(bfloat16 *__restrict values, int32_t lanes) {
    const int32_t vector_lanes = lanes & ~(kRmsVecLanes - 1);
    aie::accum<accfloat, kRmsVecLanes> sum_acc =
        aie::zeros<accfloat, kRmsVecLanes>();
    for (int32_t lane = 0; lane < vector_lanes; lane += kRmsVecLanes)
        chess_prepare_for_pipelining chess_loop_range(128, 128) {
        aie::vector<bfloat16, kRmsVecLanes> value_vec =
            aie::load_v<kRmsVecLanes>(values + lane);
        sum_acc = aie::mac(sum_acc, value_vec, value_vec);
    }
    float sum_sq = aie::reduce_add(sum_acc.template to_vector<float>());
    for (int32_t lane = vector_lanes; lane < lanes; lane++) {
        const float value = static_cast<float>(values[lane]);
        sum_sq += value * value;
    }
    constexpr float eps = 0.000001f;
    return fast_rsqrt(sum_sq / static_cast<float>(lanes) + eps);
}

static void write_weighted_rms_values(
    bfloat16 *values,
    bfloat16 *weight,
    bfloat16 *payload,
    int32_t lanes
) {
    const float scale = rms_scale(values, lanes);
    for (int32_t lane = 0; lane < lanes; lane++)
        chess_prepare_for_pipelining chess_loop_range(1, 4096) {
        store_bf16_rne(
            payload,
            lane,
            static_cast<float>(values[lane]) * scale * static_cast<float>(weight[lane])
        );
    }
}

static void add_bf16_block_inplace(
    bfloat16 *__restrict values,
    bfloat16 *__restrict rhs,
    int32_t base
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (int32_t lane = 0; lane < kFullVectorBlockLanes; lane += kFullVectorAddLanes)
        chess_prepare_for_pipelining chess_loop_range(
            kFullVectorBlockLanes / kFullVectorAddLanes,
            kFullVectorBlockLanes / kFullVectorAddLanes
        ) {
        const int32_t global_lane = base + lane;
        aie::vector<bfloat16, kFullVectorAddLanes> value_vec =
            aie::load_v<kFullVectorAddLanes, aie_dm_resource::a>(values + global_lane);
        aie::vector<bfloat16, kFullVectorAddLanes> rhs_vec =
            aie::load_unaligned_v<kFullVectorAddLanes, aie_dm_resource::b>(rhs + lane, 2);
        aie::accum<accfloat, kFullVectorAddLanes> value_acc(value_vec);
        aie::accum<accfloat, kFullVectorAddLanes> rhs_acc(rhs_vec);
        aie::accum<accfloat, kFullVectorAddLanes> sum = aie::add(value_acc, rhs_acc);
        aie::store_v<aie_dm_resource::a>(values + global_lane, sum.template to_vector<bfloat16>());
    }
    chess_separator_scheduler();
}

static void add_bf16_block_out(
    bfloat16 *__restrict lhs,
    bfloat16 *__restrict rhs,
    bfloat16 *__restrict output,
    int32_t base
) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (int32_t lane = 0; lane < kFullVectorBlockLanes; lane += kFullVectorAddLanes)
        chess_prepare_for_pipelining chess_loop_range(
            kFullVectorBlockLanes / kFullVectorAddLanes,
            kFullVectorBlockLanes / kFullVectorAddLanes
        ) {
        const int32_t global_lane = base + lane;
        aie::vector<bfloat16, kFullVectorAddLanes> lhs_vec =
            aie::load_v<kFullVectorAddLanes, aie_dm_resource::a>(lhs + global_lane);
        aie::vector<bfloat16, kFullVectorAddLanes> rhs_vec =
            aie::load_unaligned_v<kFullVectorAddLanes, aie_dm_resource::b>(rhs + lane, 2);
        aie::accum<accfloat, kFullVectorAddLanes> lhs_acc(lhs_vec);
        aie::accum<accfloat, kFullVectorAddLanes> rhs_acc(rhs_vec);
        aie::accum<accfloat, kFullVectorAddLanes> sum = aie::add(lhs_acc, rhs_acc);
        aie::store_v(output + global_lane, sum.template to_vector<bfloat16>());
    }
    chess_separator_scheduler();
}

} // namespace

extern "C" {

void full_c1r2_make_input_norm_payload(
    int32_t *hidden,
    int32_t *norm_weight,
    int32_t *payload,
    int32_t payload_dwords
) {
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    reset_profile();
    const uint64_t start = current_cycles();
#endif
    write_weighted_rms_values(as_bf16(hidden), as_bf16(norm_weight), as_bf16(payload), payload_dwords * 2);
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    g_input_norm_cycles = current_cycles() - start;
#endif
}

void full_c1r2_add_o_compact_to_residual(int32_t *hidden, int32_t *compact, int32_t block) {
    bfloat16 *residual = as_bf16(hidden);
    bfloat16 *compact_payload = as_bf16(compact + 1);
    const int32_t base = block * kFullVectorBlockLanes;
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    add_bf16_block_inplace(residual, compact_payload, base);
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    g_o_cycles += current_cycles() - start;
    g_o_blocks += 1;
#endif
}

void full_c1r2_make_post_norm_payload(
    int32_t *residual,
    int32_t *norm_weight,
    int32_t *payload,
    int32_t payload_dwords
) {
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    write_weighted_rms_values(as_bf16(residual), as_bf16(norm_weight), as_bf16(payload), payload_dwords * 2);
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    g_post_norm_cycles = current_cycles() - start;
#endif
}

void full_c1r2_write_down_block(int32_t *residual, int32_t *compact, int32_t *output, int32_t block) {
    bfloat16 *residual_values = as_bf16(residual);
    bfloat16 *compact_payload = as_bf16(compact + 1);
    bfloat16 *output_values = as_bf16(output);
    const int32_t base = block * kFullVectorBlockLanes;
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    const uint64_t start = current_cycles();
#endif
    add_bf16_block_out(residual_values, compact_payload, output_values, base);
#ifdef QWEN3_ENABLE_FULL_VECTOR_CYCLE_PROFILE
    g_down_cycles += current_cycles() - start;
    g_down_blocks += 1;
    if (block == 7) {
        emit_profile_summary(output);
    }
#endif
}

} // extern "C"
