#include <metal_stdlib>
using namespace metal;

struct DeinterleaveBatchedPush {
    uint head_dim;
    uint n_heads;
    uint n_tokens;
};

kernel void main0(
    constant DeinterleaveBatchedPush& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* q_out [[buffer(2)]],
    device float* gate_out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint q_dim = p.head_dim * p.n_heads;
    const uint total = p.n_tokens * q_dim;
    if (gid >= total) {
        return;
    }

    const uint token = gid / q_dim;
    const uint elem = gid - token * q_dim;
    const uint head = elem / p.head_dim;
    const uint dim = elem - head * p.head_dim;
    const uint packed_base = token * (q_dim * 2u) + head * (p.head_dim * 2u) + dim;

    q_out[gid] = input[packed_base];
    gate_out[gid] = input[packed_base + p.head_dim];
}
