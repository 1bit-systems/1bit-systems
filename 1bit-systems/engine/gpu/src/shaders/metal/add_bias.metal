#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint bias_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* data [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.n) return;

    data[id] += bias[p.bias_offset / 4u + id];
}
