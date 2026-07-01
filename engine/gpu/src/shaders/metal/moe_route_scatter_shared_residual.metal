#include <metal_stdlib>
using namespace metal;

// Fused Qwen route-packed MoE combine.
//
// This follows vLLM's topk_weight_and_reduce shape: route outputs are already
// stored in token*k+slot order, so each output element can reduce its top-k
// routes directly. It also folds the shared-expert contribution and residual
// add into the same pass, mirroring the reference implementation Metal's preference for fused
// graph tails over separate elementwise command-barrier pairs.

struct Params {
    uint n_tokens;
    uint hidden_dim;
    uint n_experts;
    uint k;
    uint routing_stride;
    uint has_gate;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const uint* routing [[buffer(1)]],
    device const float* routed_src [[buffer(2)]],
    device const float* shared_src [[buffer(3)]],
    device const float* shared_gate [[buffer(4)]],
    device float* hidden [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = p.n_tokens * p.hidden_dim;
    if (id >= total || p.hidden_dim == 0u) {
        return;
    }

    const uint token = id / p.hidden_dim;
    const uint dim = id - token * p.hidden_dim;
    device const uint* route_row = routing + token * p.routing_stride;

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

    const float gate = (p.has_gate != 0u)
        ? (1.0f / (1.0f + exp(-shared_gate[token])))
        : 1.0f;
    hidden[id] += sum + gate * shared_src[id];
}
