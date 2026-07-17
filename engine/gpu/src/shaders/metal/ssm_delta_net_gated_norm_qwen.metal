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

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device const float* z_gate [[buffer(7)]],
    device const float* norm_weight [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_idx [[simdgroup_index_in_threadgroup]]
) {
    if ((p.dt_rank != 32u && p.dt_rank != 48u) ||
        p.head_v_dim != 128u ||
        p.d_state != 128u ||
        p.n_group != 16u ||
        head >= p.dt_rank)
    {
        return;
    }

    constexpr uint head_v_dim = 128u;
    constexpr uint d_state = 128u;
    constexpr uint qk_dim = 2048u;
    constexpr uint v_base0 = 4096u;
    constexpr float inv_sqrt_d_state = 0.08838834764831845f;

    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float partial_q[4];
    threadgroup float partial_k[4];
    threadgroup float partial_sq[4];

    const uint group = head & 15u;
    const uint q_base = group * d_state;
    const uint k_base = qk_dim + group * d_state;
    const uint v_base = v_base0 + head * head_v_dim;
    const uint head_state_base = head * head_v_dim * head_v_dim;

    const float qv = conv_out[q_base + tid];
    const float kv = conv_out[k_base + tid];
    q[tid] = qv;
    k[tid] = kv;
    const float q_ss = qv * qv;
    const float k_ss = kv * kv;

    // Pack the two per-simdgroup RMS reductions for Q and K into a single
    // `simd_sum(float2)` — Apple9's vector `simd_sum` lowers to one
    // log2(32)=5-level butterfly that transfers 64-bit packed lanes per
    // `shuffle_xor` instead of two independent 32-bit trees, cutting
    // cross-lane shuffle traffic ~2× on the per-simdgroup tail of the fused
    // delta-net + gated-norm SSM kernel that fires every SSM layer per decode
    // token (32 or 48 head TGs per SSM layer, depending on the Qwen dense-
    // hybrid shape). Same proven pattern as cycles
    // ~67 (q8_0_pair/conv1d_dual), ~73 (repacked_k2048_qwen), and ~75
    // (repacked_k2048_nr2_qwen). Both per-row simd_sums operate on
    // independent values with identical reduction trees, so the pack is
    // bit-equivalent to the unpacked scalar form.
    const float2 qk_sum = simd_sum(float2(q_ss, k_ss));
    if (simd_lane == 0u) {
        partial_q[simd_idx] = qk_sum.x;
        partial_k[simd_idx] = qk_sum.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every simdgroup folds the tiny cross-simdgroup partial arrays itself,
    // mirroring the final RMS reduction below. Lane 0 computes the scalar
    // setup once per simdgroup and broadcasts within the simdgroup, avoiding a
    // second threadgroup barrier before the 128 row workers enter the state loop.
    // Pack the Q+K cross-simdgroup folds (4-element reductions across simd
    // lanes 0..3 with lanes 4..31 holding the zero identity) into a single
    // `simd_sum(float2)` — same Apple9 butterfly-packing rationale as the
    // Phase-1 pack above.
    const float q_partial = (simd_lane < 4u) ? partial_q[simd_lane] : 0.0f;
    const float k_partial = (simd_lane < 4u) ? partial_k[simd_lane] : 0.0f;
    const float2 qk_norm_sq = simd_sum(float2(q_partial, k_partial));
    const float q_norm_sq = qk_norm_sq.x;
    const float k_norm_sq = qk_norm_sq.y;

    float q_scale_lane = 0.0f;
    float k_scale_lane = 0.0f;
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (simd_lane == 0u) {
        const float alpha_raw = alpha[p.alpha_offset + head] + dt_bias[head];
        const float softplus_alpha = fast::log(1.0f + fast::exp(alpha_raw));
        q_scale_lane = fast::rsqrt(fast::max(q_norm_sq, 1.0e-13f)) * inv_sqrt_d_state;
        k_scale_lane = fast::rsqrt(fast::max(k_norm_sq, 1.0e-13f));
        decay_lane = fast::exp(softplus_alpha * ssm_a[head]);
        beta_lane = fast::divide(1.0f, 1.0f + fast::exp(-beta[p.beta_offset + head]));
    }

    const float q_scale = simd_broadcast(q_scale_lane, 0u);
    const float k_scale = simd_broadcast(k_scale_lane, 0u);
    const float decay = simd_broadcast(decay_lane, 0u);
    const float beta_val = simd_broadcast(beta_lane, 0u);

    float local_sq = 0.0f;
    const uint row = tid;
    const uint row_base = head_state_base + row * head_v_dim;
    device float4* state_vec = (device float4*)(state + row_base);
    threadgroup const float4* k_vec = (threadgroup const float4*)k;
    threadgroup const float4* q_vec = (threadgroup const float4*)q;
    float sk_raw = 0.0f;
    #pragma unroll
    for (uint col4 = 0u; col4 < 32u; ++col4) {
        const float4 old_state = state_vec[col4];
        const float4 kv4 = k_vec[col4];
        sk_raw = fma(old_state.x, kv4.x, sk_raw);
        sk_raw = fma(old_state.y, kv4.y, sk_raw);
        sk_raw = fma(old_state.z, kv4.z, sk_raw);
        sk_raw = fma(old_state.w, kv4.w, sk_raw);
    }
    sk_raw *= decay;

    const float v = conv_out[v_base + row];
    const float delta = beta_val * (v - sk_raw * k_scale);
    const float scaled_delta = delta * k_scale;
    float out_v = 0.0f;
    #pragma unroll
    for (uint col4 = 0u; col4 < 32u; ++col4) {
        const float4 kv4 = k_vec[col4];
        const float4 qv4 = q_vec[col4];
        const float4 updated = fma(kv4, scaled_delta, state_vec[col4] * decay);
        state_vec[col4] = updated;
        out_v = fma(updated.x, qv4.x, out_v);
        out_v = fma(updated.y, qv4.y, out_v);
        out_v = fma(updated.z, qv4.z, out_v);
        out_v = fma(updated.w, qv4.w, out_v);
    }
    out_v *= q_scale;
    local_sq = out_v * out_v;

    const float sq_sum = simd_sum(local_sq);
    if (simd_lane == 0u) {
        partial_sq[simd_idx] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Mirror rms_norm_mul.metal: each simdgroup redundantly folds the tiny
    // partial array, avoiding a second threadgroup barrier just to broadcast
    // the final RMS scalar.
    const float partial = (simd_lane < 4u) ? partial_sq[simd_lane] : 0.0f;
    const float total_sq = simd_sum(partial);
    const float rms = fast::rsqrt(fast::divide(total_sq, float(head_v_dim)) + 1.0e-6f);
    const uint idx = head * head_v_dim + row;
    const uint weight_idx = (p.norm_per_head != 0u) ? idx : row;
    const float z = z_gate[p.z_offset + idx];
    const float silu_z = z * fast::divide(1.0f, 1.0f + fast::exp(-z));
    output[p.output_offset + idx] = out_v * rms * norm_weight[weight_idx] * silu_z;
}
