#version 450

// ─────────────────────────────────────────────────────────────────────────────
// SiLU + Element-wise Multiply (SwiGLU activation)
// Grid = (n/64, 1, 1) — each workgroup processes 64 elements
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint n_elements;
} push;

layout(set = 0, binding = 0) buffer A { float data[]; } a_buf;
layout(set = 0, binding = 1) buffer B { float data[]; } b_buf;
layout(set = 0, binding = 2) buffer Output { float data[]; } out_buf;

void main() {
    uint base = gl_GlobalInvocationID.x * 64;
    for (uint i = gl_LocalInvocationIndex; i < 64 && base + i < push.n_elements; i += gl_WorkGroupSize.x) {
        uint idx = base + i;
        float a_val = a_buf.data[idx];
        float b_val = b_buf.data[idx];
        // SiLU(x) = x * sigmoid(x)  where sigmoid(x) = 1/(1+exp(-x))
        float sig = 1.0 / (1.0 + exp(-a_val));
        out_buf.data[idx] = (a_val * sig) * b_val;
    }
}
