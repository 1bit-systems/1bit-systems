#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint has_dt_bias;
    uint has_ssm_a;
    uint n_tokens;
    uint alpha_stride;
    uint beta_stride;
    uint conv_stride;
    uint output_stride;
    uint alpha_offset;
    uint beta_offset;
    uint conv_offset;
    uint output_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device float* output [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (head >= p.dt_rank || p.head_v_dim > 128u || p.d_state > 128u) {
        return;
    }

    const uint tg_threads = simd_width * simdgroups_per_tg;
    const uint simd_lane = tid % simd_width;
    const uint simd_idx = tid / simd_width;
    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float partial_q[4];
    threadgroup float partial_k[4];

    if (p.dt_rank == 32u &&
        p.head_v_dim == 128u &&
        p.d_state == 128u &&
        p.n_group == 16u &&
        p.d_inner == 4096u)
    {
        constexpr uint head_v_dim = 128u;
        constexpr uint qk_dim = 2048u;
        constexpr uint v_base0 = 4096u;
        constexpr float inv_sqrt_d_state = 0.08838834764831845f;

        const uint group_exact = head & 15u;
        const uint q_base0 = group_exact * head_v_dim;
        const uint k_base0 = qk_dim + group_exact * head_v_dim;
        const uint v_base_head = v_base0 + head * head_v_dim;
        const uint head_state_base_exact = head * head_v_dim * head_v_dim;
        threadgroup const float4* k_vec = (threadgroup const float4*)k;
        threadgroup const float4* q_vec = (threadgroup const float4*)q;

        for (uint token = 0u; token < p.n_tokens; ++token) {
            const uint conv_token_base = p.conv_offset + token * p.conv_stride;

            float q_ss = 0.0f;
            float k_ss = 0.0f;
            for (uint i = tid; i < head_v_dim; i += tg_threads) {
                const float qv = conv_out[conv_token_base + q_base0 + i];
                const float kv = conv_out[conv_token_base + k_base0 + i];
                q[i] = qv;
                k[i] = kv;
                q_ss = fma(qv, qv, q_ss);
                k_ss = fma(kv, kv, k_ss);
            }

            const float q_sum = simd_sum(q_ss);
            const float k_sum = simd_sum(k_ss);
            if (simd_lane == 0u) {
                partial_q[simd_idx] = q_sum;
                partial_k[simd_idx] = k_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float q_partial = (simd_lane < simdgroups_per_tg) ? partial_q[simd_lane] : 0.0f;
            const float k_partial = (simd_lane < simdgroups_per_tg) ? partial_k[simd_lane] : 0.0f;
            const float q_norm_sq = simd_sum(q_partial);
            const float k_norm_sq = simd_sum(k_partial);

            float q_scale_lane = 0.0f;
            float k_scale_lane = 0.0f;
            float decay_lane = 0.0f;
            float beta_lane = 0.0f;
            if (simd_lane == 0u) {
                const float alpha_raw = alpha[p.alpha_offset + token * p.alpha_stride + head] +
                    ((p.has_dt_bias != 0u) ? dt_bias[head] : 0.0f);
                const float softplus_alpha = log(1.0f + fast::exp(alpha_raw));
                const float decay_arg = (p.has_ssm_a != 0u) ? (softplus_alpha * ssm_a[head]) : (-softplus_alpha);
                q_scale_lane = fast::rsqrt(fast::max(q_norm_sq, 1.0e-13f)) * inv_sqrt_d_state;
                k_scale_lane = fast::rsqrt(fast::max(k_norm_sq, 1.0e-13f));
                decay_lane = fast::exp(decay_arg);
                beta_lane = fast::divide(1.0f, 1.0f + fast::exp(-beta[p.beta_offset + token * p.beta_stride + head]));
            }

            const float q_scale = simd_broadcast(q_scale_lane, 0u);
            const float k_scale = simd_broadcast(k_scale_lane, 0u);
            const float decay = simd_broadcast(decay_lane, 0u);
            const float beta_val = simd_broadcast(beta_lane, 0u);
            const uint head_out_base = p.output_offset + token * p.output_stride + head * head_v_dim;
            const uint v_base = conv_token_base + v_base_head;

            for (uint row = tid; row < head_v_dim; row += tg_threads) {
                const uint row_base = head_state_base_exact + row * head_v_dim;
                device float4* state_vec = (device float4*)(state + row_base);
                float sk_raw = 0.0f;
                #pragma unroll
                for (uint col4 = 0u; col4 < 32u; ++col4) {
                    const float4 old_state = state_vec[col4];
                    const float4 kv = k_vec[col4];
                    sk_raw = fma(old_state.x, kv.x, sk_raw);
                    sk_raw = fma(old_state.y, kv.y, sk_raw);
                    sk_raw = fma(old_state.z, kv.z, sk_raw);
                    sk_raw = fma(old_state.w, kv.w, sk_raw);
                }
                sk_raw *= decay;

                const float v = conv_out[v_base + row];
                const float delta = beta_val * (v - sk_raw * k_scale);
                const float scaled_delta = delta * k_scale;
                float out_v = 0.0f;
                #pragma unroll
                for (uint col4 = 0u; col4 < 32u; ++col4) {
                    const float4 kv = k_vec[col4];
                    const float4 qv = q_vec[col4];
                    const float4 updated = fma(kv, scaled_delta, state_vec[col4] * decay);
                    state_vec[col4] = updated;
                    out_v = fma(updated.x, qv.x, out_v);
                    out_v = fma(updated.y, qv.y, out_v);
                    out_v = fma(updated.z, qv.z, out_v);
                    out_v = fma(updated.w, qv.w, out_v);
                }
                output[head_out_base + row] = out_v * q_scale;
            }

            if (token + 1u < p.n_tokens) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        return;
    }

    const uint qk_dim = p.d_state * p.n_group;
    const uint group = (p.n_group == p.dt_rank) ? head : (head % p.n_group);
    const uint k_len = min(p.d_state, p.head_v_dim);
    const uint head_state_base = head * p.head_v_dim * p.head_v_dim;

    for (uint token = 0u; token < p.n_tokens; ++token) {
        const uint conv_token_base = p.conv_offset + token * p.conv_stride;
        const uint q_base = conv_token_base + group * p.d_state;
        const uint k_base = conv_token_base + qk_dim + group * p.d_state;
        const uint v_base = conv_token_base + 2u * qk_dim + head * p.head_v_dim;

        float q_ss = 0.0f;
        float k_ss = 0.0f;
        for (uint i = tid; i < k_len; i += tg_threads) {
            const float qv = conv_out[q_base + i];
            const float kv = conv_out[k_base + i];
            q[i] = qv;
            k[i] = kv;
            q_ss = fma(qv, qv, q_ss);
            k_ss = fma(kv, kv, k_ss);
        }

        float q_sum = simd_sum(q_ss);
        float k_sum = simd_sum(k_ss);
        if (simd_lane == 0u) {
            partial_q[simd_idx] = q_sum;
            partial_k[simd_idx] = k_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float q_partial = (simd_lane < simdgroups_per_tg) ? partial_q[simd_lane] : 0.0f;
        const float k_partial = (simd_lane < simdgroups_per_tg) ? partial_k[simd_lane] : 0.0f;
        const float q_total = simd_sum(q_partial);
        const float k_total = simd_sum(k_partial);

        const uint dt_index = head;
        float q_scale_lane = 0.0f;
        float k_scale_lane = 0.0f;
        float decay_lane = 0.0f;
        float beta_lane = 0.0f;
        if (simd_lane == 0u) {
            const float alpha_raw = alpha[p.alpha_offset + token * p.alpha_stride + dt_index] +
                ((p.has_dt_bias != 0u) ? dt_bias[head] : 0.0f);
            const float softplus_alpha = log(1.0f + fast::exp(alpha_raw));
            const float decay_arg = (p.has_ssm_a != 0u) ? (softplus_alpha * ssm_a[head]) : (-softplus_alpha);
            q_scale_lane = fast::rsqrt(fast::max(q_total, 1.0e-13f)) / sqrt(float(p.d_state));
            k_scale_lane = fast::rsqrt(fast::max(k_total, 1.0e-13f));
            decay_lane = fast::exp(decay_arg);
            beta_lane = fast::divide(1.0f, 1.0f + fast::exp(-beta[p.beta_offset + token * p.beta_stride + dt_index]));
        }

        const float q_scale = simd_broadcast(q_scale_lane, 0u);
        const float k_scale = simd_broadcast(k_scale_lane, 0u);
        const float decay = simd_broadcast(decay_lane, 0u);
        const float beta_val = simd_broadcast(beta_lane, 0u);
        const uint head_out_base = p.output_offset + token * p.output_stride + head * p.head_v_dim;

        for (uint row = tid; row < p.head_v_dim; row += tg_threads) {
            const uint row_base = head_state_base + row * p.head_v_dim;
            float sk_raw = 0.0f;
            for (uint col = 0u; col < k_len; ++col) {
                const uint state_idx = row_base + col;
                const float decayed = state[state_idx] * decay;
                state[state_idx] = decayed;
                sk_raw = fma(decayed, k[col], sk_raw);
            }
            for (uint col = k_len; col < p.head_v_dim; ++col) {
                state[row_base + col] *= decay;
            }

            const float v = conv_out[v_base + row];
            const float sk = sk_raw * k_scale;
            const float delta = beta_val * (v - sk);
            const float scaled_delta = delta * k_scale;
            for (uint col = 0u; col < k_len; ++col) {
                state[row_base + col] = fma(k[col], scaled_delta, state[row_base + col]);
            }

            float out_v = 0.0f;
            for (uint col = 0u; col < k_len; ++col) {
                out_v = fma(state[row_base + col], q[col], out_v);
            }
            output[head_out_base + row] = out_v * q_scale;
        }

        if (token + 1u < p.n_tokens) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
