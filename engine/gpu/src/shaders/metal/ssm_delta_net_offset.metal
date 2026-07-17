#include <metal_stdlib>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint has_dt_bias;
    uint has_ssm_a;
    uint alpha_offset;
    uint beta_offset;
    uint output_offset;
    uint conv_offset;
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
    uint simd_width [[thread_execution_width]]
) {
    if (head >= p.dt_rank || p.head_v_dim > 128u || p.d_state > 128u) {
        return;
    }

    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float partial[4];

    const uint qk_dim = p.d_state * p.n_group;
    const uint group = (p.n_group == p.dt_rank) ? head : (head % p.n_group);
    const uint q_base = p.conv_offset + group * p.d_state;
    const uint k_base = p.conv_offset + qk_dim + group * p.d_state;
    const uint v_base = p.conv_offset + 2u * qk_dim + head * p.head_v_dim;
    const uint k_len = min(p.d_state, p.head_v_dim);

    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (uint i = tid; i < k_len; i += 64u) {
        const float qv = conv_out[q_base + i];
        const float kv = conv_out[k_base + i];
        q[i] = qv;
        k[i] = kv;
        q_ss = fma(qv, qv, q_ss);
        k_ss = fma(kv, kv, k_ss);
    }

    float q_sum = simd_sum(q_ss);
    if ((tid % simd_width) == 0u) {
        partial[tid / simd_width] = q_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = (64u + simd_width - 1u) / simd_width;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial[i];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_scale = rsqrt(fast::max(partial[0], 1.0e-13f)) / sqrt(float(p.d_state));
    for (uint i = tid; i < k_len; i += 64u) {
        q[i] *= q_scale;
    }

    float k_sum = simd_sum(k_ss);
    if ((tid % simd_width) == 0u) {
        partial[tid / simd_width] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = (64u + simd_width - 1u) / simd_width;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial[i];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_scale = rsqrt(fast::max(partial[0], 1.0e-13f));
    for (uint i = tid; i < k_len; i += 64u) {
        k[i] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha_raw = alpha[p.alpha_offset + head] + ((p.has_dt_bias != 0u) ? dt_bias[head] : 0.0f);
    const float softplus_alpha = log(1.0f + exp(alpha_raw));
    const float decay_arg = (p.has_ssm_a != 0u) ? (softplus_alpha * ssm_a[head]) : (-softplus_alpha);
    const float decay = exp(decay_arg);
    const float beta_val = 1.0f / (1.0f + exp(-beta[p.beta_offset + head]));
    const uint head_state_base = head * p.head_v_dim * p.head_v_dim;
    const uint head_out_base = p.output_offset + head * p.head_v_dim;

    for (uint row = tid; row < p.head_v_dim; row += 64u) {
        const uint row_base = head_state_base + row * p.head_v_dim;
        for (uint col = 0u; col < p.head_v_dim; ++col) {
            state[row_base + col] *= decay;
        }

        float sk = 0.0f;
        for (uint col = 0u; col < k_len; ++col) {
            sk = fma(state[row_base + col], k[col], sk);
        }

        const float v = conv_out[v_base + row];
        const float delta = beta_val * (v - sk);
        for (uint col = 0u; col < k_len; ++col) {
            state[row_base + col] = fma(k[col], delta, state[row_base + col]);
        }

        float out_v = 0.0f;
        for (uint col = 0u; col < k_len; ++col) {
            out_v = fma(state[row_base + col], q[col], out_v);
        }
        output[head_out_base + row] = out_v;
    }
}
