#version 450

// ─────────────────────────────────────────────────────────────────────────────
// GEMM — General Matrix Multiply for f32 tiles
// Grid = (cols/16, rows/16, batches) — each workgroup computes 16×16 tile
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint M;  // rows of A and C
    uint N;  // cols of B and C
    uint K;  // inner dim
    uint lda;
    uint ldb;
    uint ldc;
    uint batch_count;
    uint pad;
} push;

layout(set = 0, binding = 0) buffer A { float data[]; } mat_a;
layout(set = 0, binding = 1) buffer B { float data[]; } mat_b;
layout(set = 0, binding = 2) buffer C { float data[]; } mat_c;

shared float s_a[256];  // 16×16 tile
shared float s_b[256];

void main() {
    uint row = gl_GlobalInvocationID.y * 16;
    uint col = gl_GlobalInvocationID.x * 16;
    uint batch = gl_GlobalInvocationID.z;

    if (row >= push.M || col >= push.N) return;

    float acc[16][16];
    for (uint i = 0; i < 16; i++)
        for (uint j = 0; j < 16; j++)
            acc[i][j] = 0.0;

    uint a_base = batch * push.K * push.M;
    uint b_base = batch * push.K * push.N;

    for (uint tile = 0; tile < push.K; tile += 16) {
        // Load A tile
        barrier();
        for (uint i = gl_LocalInvocationIndex; i < 256; i += gl_WorkGroupSize.x) {
            uint tr = i / 16;
            uint tc = i % 16;
            s_a[i] = (row + tr < push.M && tile + tc < push.K)
                ? mat_a.data[a_base + (row + tr) * push.lda + tile + tc]
                : 0.0;
        }
        // Load B tile
        for (uint i = gl_LocalInvocationIndex; i < 256; i += gl_WorkGroupSize.x) {
            uint tr = i / 16;
            uint tc = i % 16;
            s_b[i] = (tile + tr < push.K && col + tc < push.N)
                ? mat_b.data[b_base + (tile + tr) * push.ldb + col + tc]
                : 0.0;
        }
        barrier();

        // Compute 16×16 product
        for (uint k = 0; k < 16; k++) {
            for (uint i = 0; i < 16; i++) {
                float a_val = s_a[i * 16 + k];
                for (uint j = 0; j < 16; j++) {
                    acc[i][j] += a_val * s_b[k * 16 + j];
                }
            }
        }
        barrier();
    }

    // Write result
    uint c_base = batch * push.N * push.M;
    for (uint i = gl_LocalInvocationIndex; i < 256; i += gl_WorkGroupSize.x) {
        uint tr = i / 16;
        uint tc = i % 16;
        if (row + tr < push.M && col + tc < push.N) {
            mat_c.data[c_base + (row + tr) * push.ldc + col + tc] = acc[tr][tc];
        }
    }
}
