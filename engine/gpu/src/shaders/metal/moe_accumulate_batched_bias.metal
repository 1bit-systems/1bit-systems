#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint expert_stride;
    uint expert_count;
    uint bias_offset;
    float w0;
    float w1;
    float w2;
    float w3;
    float w4;
    float w5;
    float w6;
    float w7;
    float w_sh;
};

static inline float weight_at(constant Params& p, uint slot) {
    switch (slot) {
        case 0: return p.w0;
        case 1: return p.w1;
        case 2: return p.w2;
        case 3: return p.w3;
        case 4: return p.w4;
        case 5: return p.w5;
        case 6: return p.w6;
        case 7: return p.w7;
        default: return 0.0f;
    }
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const float* experts [[buffer(2)]],
    device const float* sh [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    const uint bias_base = p.bias_offset / 4u;
    const uint s = p.expert_stride;
    float sum = 0.0f;
    for (uint slot = 0; slot < p.expert_count && slot < 8u; slot++) {
        const float w = weight_at(p, slot);
        const uint expert_id = expert_ids[slot];
        sum += w * (experts[slot * s + id] + bias[bias_base + expert_id * p.n + id]);
    }
    if (p.w_sh != 0.0f) {
        sum += p.w_sh * sh[id];
    }
    dst[id] += sum;
}
