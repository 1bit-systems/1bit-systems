#include <metal_stdlib>
using namespace metal;

// Fused Qwen route-packed MoE combine with f32 shared gate input.
//
// This keeps the vLLM-style top-k weight/reduce shape used by
// moe_route_scatter_shared_residual, but computes the one-row shared gate dot
// once per token inside the same kernel. That avoids materializing a separate
// [n_tokens][1] gate buffer for Qwen3.6's f32 ffn_gate_inp_shexp.weight.

struct Params {
    uint n_tokens;
    uint hidden_dim;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint gate_weight_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device const float* routed_src [[buffer(2)]],
    device const float* shared_src [[buffer(3)]],
    device const float* norm_src [[buffer(4)]],
    device const char* gate_weight_bytes [[buffer(5)]],
    device float* hidden [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
    if (token >= p.n_tokens || p.hidden_dim == 0u) {
        return;
    }

    threadgroup float partials[8];
    threadgroup float gate_value;

    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);
    device const float* norm = norm_src + token * p.hidden_dim;

    float dot = 0.0f;
    for (uint dim = tid; dim < p.hidden_dim; dim += 256u) {
        dot = fma(gate_weight[dim], norm[dim], dot);
    }

    const float simd_sum_dot = simd_sum(dot);
    if (lane == 0) {
        partials[simdgroup] = simd_sum_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float part = (lane < 8) ? partials[lane] : 0.0f;
        const float group_sum = simd_sum(part);
        if (lane == 0) {
            gate_value = 1.0f / (1.0f + exp(-group_sum));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uint* route_row = routing + token * p.routing_stride;
    const uint base = token * p.hidden_dim;

    for (uint dim = tid; dim < p.hidden_dim; dim += 256u) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < p.k; slot++) {
            const uint expert_id = route_row[slot];
            if (expert_id >= p.n_experts) {
                continue;
            }

            const uint route = token * p.k + slot;
            const float weight = as_type<float>(route_row[p.k + slot]);
            sum += weight * routed_src[route * p.hidden_dim + dim];
        }

        hidden[base + dim] += sum + gate_value * shared_src[base + dim];
    }
}
