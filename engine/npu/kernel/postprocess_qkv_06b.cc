#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int32_t kRmsVecLanes = 32;
constexpr int32_t kCopyVecDwords = 16;
constexpr int32_t kRecordPayloadDwords = 256;
constexpr int32_t kQRecords = 4;
constexpr int32_t kKvRecords = 2;

__attribute__((always_inline)) static inline void copy_record_payload(
    const int32_t *__restrict record_payload,
    int32_t *__restrict target
) {
    for (int32_t idx = 0; idx < kRecordPayloadDwords; idx += kCopyVecDwords)
        chess_prepare_for_pipelining
        chess_loop_range(kRecordPayloadDwords / kCopyVecDwords, kRecordPayloadDwords / kCopyVecDwords) {
        aie::vector<int32_t, kCopyVecDwords> words =
            aie::load_v<kCopyVecDwords, aie_dm_resource::a>(record_payload + idx);
        aie::store_v(target + idx, words);
    }
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
    for (int32_t iter = 0; iter < 10; iter++) {
        y = y * (1.5f - half * y * y);
    }
    return y;
}

static float head_rms_scale(bfloat16 *body, int32_t head, int32_t head_dim) {
    const int32_t base = head * head_dim;
    const int32_t vector_lanes = head_dim & ~(kRmsVecLanes - 1);
    aie::accum<accfloat, kRmsVecLanes> sum_acc =
        aie::zeros<accfloat, kRmsVecLanes>();
    for (int32_t dim = 0; dim < vector_lanes; dim += kRmsVecLanes)
        chess_prepare_for_pipelining chess_loop_range(4, 4) {
        aie::vector<bfloat16, kRmsVecLanes> value_vec =
            aie::load_v<kRmsVecLanes>(body + base + dim);
        sum_acc = aie::mac(sum_acc, value_vec, value_vec);
    }
    float sum_sq = aie::reduce_add(sum_acc.template to_vector<float>());
    for (int32_t dim = vector_lanes; dim < head_dim; dim++) {
        const float value = static_cast<float>(body[base + dim]);
        sum_sq += value * value;
    }
    constexpr float eps = 0.000001f;
    return fast_rsqrt(sum_sq / static_cast<float>(head_dim) + eps);
}

static bfloat16 normalized_lane(
    bfloat16 *body,
    bfloat16 *norm_weight,
    int32_t lane,
    int32_t dim,
    float scale
) {
    return bf16_rne(
        static_cast<float>(body[lane]) * scale * static_cast<float>(norm_weight[dim])
    );
}

static void write_rope_pair(
    bfloat16 *body,
    bfloat16 *norm_weight,
    bfloat16 *rope_cos,
    bfloat16 *rope_sin,
    bfloat16 *output,
    int32_t head,
    int32_t dim,
    float scale
) {
    constexpr int32_t head_dim = 128;
    const int32_t lane = head * head_dim + dim;
    const bfloat16 normalized_even = normalized_lane(body, norm_weight, lane, dim, scale);
    const bfloat16 normalized_odd = normalized_lane(body, norm_weight, lane + 1, dim + 1, scale);
    const float even = static_cast<float>(normalized_even);
    const float odd = static_cast<float>(normalized_odd);
    const int32_t pair = dim >> 1;
    const float c = static_cast<float>(rope_cos[pair]);
    const float s = static_cast<float>(rope_sin[pair]);
    output[lane] = bf16_rne(even * c - odd * s);
    output[lane + 1] = bf16_rne(even * s + odd * c);
}

static int32_t packed_rope_word(
    bfloat16 *body,
    bfloat16 *norm_weight,
    bfloat16 *rope_cos,
    bfloat16 *rope_sin,
    int32_t logical_word,
    float scale
) {
    constexpr int32_t head_dim = 128;
    const int32_t lane = logical_word * 2;
    const int32_t dim = lane % head_dim;
    const bfloat16 normalized_even = normalized_lane(body, norm_weight, lane, dim, scale);
    const bfloat16 normalized_odd = normalized_lane(body, norm_weight, lane + 1, dim + 1, scale);
    const float even = static_cast<float>(normalized_even);
    const float odd = static_cast<float>(normalized_odd);
    const int32_t pair = dim >> 1;
    const float c = static_cast<float>(rope_cos[pair]);
    const float s = static_cast<float>(rope_sin[pair]);
    bfloat16 packed[2];
    packed[0] = bf16_rne(even * c - odd * s);
    packed[1] = bf16_rne(even * s + odd * c);
    return reinterpret_cast<int32_t *>(packed)[0];
}

static void write_qwen3_current_even_odd(
    int32_t *k_body,
    int32_t *v_body,
    int32_t *qk_rope_side,
    int32_t *current_k,
    int32_t *current_v,
    int32_t current_dwords
) {
    constexpr int32_t kv_heads = 8;
    constexpr int32_t head_dim = 128;
    bfloat16 *k_values = reinterpret_cast<bfloat16 *>(k_body);
    bfloat16 *side = reinterpret_cast<bfloat16 *>(qk_rope_side);
    bfloat16 *k_norm = side + head_dim;
    bfloat16 *rope_cos = side + head_dim * 2;
    bfloat16 *rope_sin = rope_cos + head_dim / 2;
    const int32_t head_dwords = current_dwords / kv_heads;
    const int32_t half_current_dwords = current_dwords / 2;
    const int32_t half_head_dwords = head_dwords / 2;

    float scales[kv_heads];
    for (int32_t head = 0; head < kv_heads; head++) {
        scales[head] = head_rms_scale(k_values, head, head_dim);
    }

    for (int32_t head = 0; head < kv_heads; head++) {
        for (int32_t pair = 0; pair < half_head_dwords; pair++) {
            const int32_t even_idx = head * head_dwords + pair * 2;
            const int32_t odd_idx = even_idx + 1;
            const int32_t even_stream_idx = head * half_head_dwords + pair;
            const int32_t odd_stream_idx = half_current_dwords + even_stream_idx;
            current_k[even_stream_idx] = packed_rope_word(
                k_values,
                k_norm,
                rope_cos,
                rope_sin,
                even_idx,
                scales[head]
            );
            current_k[odd_stream_idx] = packed_rope_word(
                k_values,
                k_norm,
                rope_cos,
                rope_sin,
                odd_idx,
                scales[head]
            );
            current_v[even_stream_idx] = v_body[even_idx];
            current_v[odd_stream_idx] = v_body[odd_idx];
        }
    }
}

} // namespace

extern "C" {

void qwen3_postprocess_absorb_qkv_payload_record(
    int32_t *record_payload,
    int32_t *q_body,
    int32_t *k_body,
    int32_t *v_body,
    int32_t record_index
) {
    int32_t *target = nullptr;
    int32_t block = 0;
    if (record_index < kQRecords) {
        target = q_body;
        block = record_index;
    } else if (record_index < kQRecords + kKvRecords) {
        target = k_body;
        block = record_index - kQRecords;
    } else if (record_index < kQRecords + kKvRecords * 2) {
        target = v_body;
        block = record_index - kQRecords - kKvRecords;
    }
    if (target == nullptr) {
        return;
    }
    copy_record_payload(record_payload, target + block * kRecordPayloadDwords);
}

void qwen3_postprocess_q4nx_body_payload(
    int32_t *q_body,
    int32_t *k_body,
    int32_t *v_body,
    int32_t *qk_rope_side,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    constexpr int32_t q_heads = 16;
    constexpr int32_t head_dim = 128;
    bfloat16 *q_values = reinterpret_cast<bfloat16 *>(q_body);
    bfloat16 *q_output = reinterpret_cast<bfloat16 *>(q_payload);
    bfloat16 *side = reinterpret_cast<bfloat16 *>(qk_rope_side);
    bfloat16 *q_norm = side;
    bfloat16 *rope_cos = side + head_dim * 2;
    bfloat16 *rope_sin = rope_cos + head_dim / 2;

    for (int32_t head = 0; head < q_heads; head++) {
        const float scale = head_rms_scale(q_values, head, head_dim);
        for (int32_t dim = 0; dim < head_dim; dim += 2) {
            write_rope_pair(q_values, q_norm, rope_cos, rope_sin, q_output, head, dim, scale);
        }
    }

    write_qwen3_current_even_odd(k_body, v_body, qk_rope_side, current_k, current_v, current_dwords);
    (void)current_token_buf;
    (void)q_dwords;
}

void qwen3_postprocess_q4nx_qkv_payload(
    int32_t *qkv_body,
    int32_t *qk_rope_side,
    int32_t *q_payload,
    int32_t *current_k,
    int32_t *current_v,
    int32_t *current_token_buf,
    int32_t q_dwords,
    int32_t current_dwords
) {
    int32_t *q_body = qkv_body;
    int32_t *k_body = qkv_body + q_dwords;
    int32_t *v_body = k_body + current_dwords;
    qwen3_postprocess_q4nx_body_payload(
        q_body,
        k_body,
        v_body,
        qk_rope_side,
        q_payload,
        current_k,
        current_v,
        current_token_buf,
        q_dwords,
        current_dwords
    );
}

} // extern "C"
