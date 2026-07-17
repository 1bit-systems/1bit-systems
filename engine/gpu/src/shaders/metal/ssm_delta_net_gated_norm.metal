#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint ssm_a_is_f16;
    uint dt_bias_is_f16;
    uint has_dt_bias;
    uint has_ssm_a;
    uint alpha_offset;
    uint beta_offset;
    uint z_offset;
    uint output_offset;
    uint norm_per_head;
};

static inline float load_f32_or_f16(device const char* raw, uint idx, bool is_f16) {
    if (is_f16) {
        device const half* ptr = (device const half*)raw;
        return float(ptr[idx]);
    }
    device const float* ptr = (device const float*)raw;
    return ptr[idx];
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const char* dt_bias [[buffer(3)]],
    device const char* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device const float* z_gate [[buffer(7)]],
    device const float* norm_weight [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (head >= p.dt_rank || p.head_v_dim > 128u || p.d_state > 128u) {
        return;
    }

    const uint tg_threads = simd_width * simdgroups_per_tg;
    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float delta_out[128];
    // Keep each cross-simdgroup reduction in its own scratch lane. This mirrors
    // the reference implementation's Metal matvec reducers, where reduction scratch is not reused
    // until the dependent value is fully consumed; here it lets the RMS pass
    // start without an extra barrier after the state update.
    threadgroup float partial_q[4];
    threadgroup float partial_k[4];
    threadgroup float partial_sq[4];

    const uint qk_dim = p.d_state * p.n_group;
    const uint group = (p.n_group == p.dt_rank) ? head : (head % p.n_group);
    const uint k_len = min(p.d_state, p.head_v_dim);
    const uint head_state_base = head * p.head_v_dim * p.head_v_dim;
    const uint q_base = group * p.d_state;
    const uint k_base = qk_dim + group * p.d_state;
    const uint v_base = 2u * qk_dim + head * p.head_v_dim;

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

    const uint n_sg = simdgroups_per_tg;
    float q_sum = simd_sum(q_ss);
    float k_sum = simd_sum(k_ss);
    if ((tid % simd_width) == 0u) {
        partial_q[tid / simd_width] = q_sum;
        partial_k[tid / simd_width] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float q_total = 0.0f;
        float k_total = 0.0f;
        for (uint i = 0u; i < n_sg; ++i) {
            q_total += partial_q[i];
            k_total += partial_k[i];
        }
        partial_q[0] = q_total;
        partial_k[0] = k_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_scale = rsqrt(fast::max(partial_q[0], 1.0e-13f)) / sqrt(float(p.d_state));
    const float k_scale = rsqrt(fast::max(partial_k[0], 1.0e-13f));
    // Keep q/k unscaled in threadgroup memory and fold scales into the row
    // recurrence. This removes a second per-head threadgroup barrier.
    const float alpha_raw = alpha[p.alpha_offset + head] +
        ((p.has_dt_bias != 0u) ? load_f32_or_f16(dt_bias, head, p.dt_bias_is_f16 != 0u) : 0.0f);
    const float softplus_alpha = log(1.0f + exp(alpha_raw));
    const float decay_arg = (p.has_ssm_a != 0u) ?
        (softplus_alpha * load_f32_or_f16(ssm_a, head, p.ssm_a_is_f16 != 0u)) :
        (-softplus_alpha);
    const float decay = exp(decay_arg);
    const float beta_val = 1.0f / (1.0f + exp(-beta[p.beta_offset + head]));

    float local_sq = 0.0f;
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
        float out_v = 0.0f;
        for (uint col = 0u; col < k_len; ++col) {
            const float updated = fma(k[col], scaled_delta, state[row_base + col]);
            state[row_base + col] = updated;
            out_v = fma(updated, q[col], out_v);
        }
        out_v *= q_scale;
        delta_out[row] = out_v;
        local_sq = fma(out_v, out_v, local_sq);
    }
    float sq_sum = simd_sum(local_sq);
    if ((tid % simd_width) == 0u) {
        partial_sq[tid / simd_width] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        const uint n_sg = simdgroups_per_tg;
        for (uint i = 0u; i < n_sg; ++i) {
            total += partial_sq[i];
        }
        partial_sq[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms = rsqrt((partial_sq[0] / float(p.head_v_dim)) + 1.0e-6f);

    for (uint row = tid; row < p.head_v_dim; row += tg_threads) {
        const uint head_base = head * p.head_v_dim;
        const uint idx = head_base + row;
        const uint weight_idx = (p.norm_per_head != 0u) ? idx : (row % p.d_state);
        const float z = z_gate[p.z_offset + idx];
        const float silu_z = z / (1.0f + exp(-z));
        output[p.output_offset + idx] = delta_out[row] * rms * norm_weight[weight_idx] * silu_z;
    }
}
