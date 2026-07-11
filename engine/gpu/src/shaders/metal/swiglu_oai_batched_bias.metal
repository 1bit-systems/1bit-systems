#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint gate_bias_offset;
    uint up_bias_offset;
};

static inline float oai_swiglu(float gate, float up) {
    const float alpha = 1.702f;
    const float limit = 7.0f;
    const float x = min(gate, limit);
    const float y = clamp(up, -limit, limit);
    return (x / (1.0f + exp(alpha * (-x)))) * (y + 1.0f);
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const float* up [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    device const float* gate_bias [[buffer(5)]],
    device const float* up_bias [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.n) return;

    const uint slot = gid.y;
    const uint idx = slot * p.n + gid.x;
    const uint expert_id = expert_ids[slot];
    const uint gate_bias_base = p.gate_bias_offset / 4u;
    const uint up_bias_base = p.up_bias_offset / 4u;

    const uint bias_idx = expert_id * p.n + gid.x;
    const float g = gate[idx] + gate_bias[gate_bias_base + bias_idx];
    const float u = up[idx] + up_bias[up_bias_base + bias_idx];
    out[idx] = oai_swiglu(g, u);
}
