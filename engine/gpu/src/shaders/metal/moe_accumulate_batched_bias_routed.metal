#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint expert_stride;
    uint bias_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* experts [[buffer(2)]],
    device const uint* routing [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    const uint bias_base = p.bias_offset / 4u;
    float sum = 0.0f;
    for (uint slot = 0u; slot < p.n_used; slot++) {
        const uint expert_id = routing[slot];
        const float weight = as_type<float>(routing[p.n_used + slot]);
        sum += weight * (experts[slot * p.expert_stride + id] + bias[bias_base + expert_id * p.n + id]);
    }
    dst[id] += sum;
}
