#version 450

// ─────────────────────────────────────────────────────────────────────────────
// RMS Norm — Root Mean Square Layer Normalization
// Grid = (n_rows/64, 1, 1) — each workgroup processes 64 rows
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint n_rows;
    uint n_cols;
    float epsilon;
} push;

layout(set = 0, binding = 0) buffer Input  { float data[]; } input_buf;
layout(set = 0, binding = 1) buffer Weight  { float data[]; } weight_buf;
layout(set = 0, binding = 2) buffer Output  { float data[]; } output_buf;

shared float s_sum;

void main() {
    uint row_base = gl_GlobalInvocationID.x * 64;
    uint n_cols = push.n_cols;
    float eps = push.epsilon;

    for (uint r = 0; r < 64 && row_base + r < push.n_rows; r++) {
        uint row = row_base + r;

        // Compute sum of squares
        float ss = 0.0;
        for (uint c = gl_LocalInvocationIndex; c < n_cols; c += gl_WorkGroupSize.x) {
            float val = input_buf.data[row * n_cols + c];
            ss += val * val;
        }

        // Reduction
        // (warp shuffle not available in GLSL, use atomic)
        if (gl_LocalInvocationIndex == 0) s_sum = 0.0;
        barrier();
        atomicAdd(s_sum, ss);
        barrier();

        float rms = sqrt(s_sum / float(n_cols) + eps);
        float inv_rms = 1.0 / rms;

        // Apply weight and write
        for (uint c = gl_LocalInvocationIndex; c < n_cols; c += gl_WorkGroupSize.x) {
            float w = weight_buf.data[c];
            output_buf.data[row * n_cols + c] = (input_buf.data[row * n_cols + c] / rms) * w;
        }
    }
}
