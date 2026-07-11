#version 450

// ─────────────────────────────────────────────────────────────────────────────
// RoPE — Rotary Position Embedding (direct application)
// Grid = (n_tokens, 1, 1) — each workgroup processes one token
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint n_tokens;
    uint n_heads;
    uint head_dim;
    uint rope_dim;  // typically head_dim / 2
    uint seq_len;
    uint pad0;
    uint pad1;
    uint pad2;
} push;

layout(set = 0, binding = 0) buffer Q { float data[]; } q_buf;
layout(set = 0, binding = 1) buffer K { float data[]; } k_buf;
layout(set = 0, binding = 2) buffer Sin { float data[]; } sin_buf;
layout(set = 0, binding = 3) buffer Cos { float data[]; } cos_buf;

void main() {
    uint token_id = gl_GlobalInvocationID.x;
    if (token_id >= push.n_tokens) return;

    uint head_dim = push.head_dim;
    uint rope_dim = push.rope_dim;
    uint stride = push.n_heads * head_dim;

    float sin_t = sin_buf.data[token_id];
    float cos_t = cos_buf.data[token_id];

    for (uint h = 0; h < push.n_heads; h++) {
        uint base = token_id * stride + h * head_dim;

        for (uint d = gl_LocalInvocationIndex; d < rope_dim; d += gl_WorkGroupSize.x) {
            uint i0 = base + d;
            uint i1 = base + d + rope_dim;

            float q0 = q_buf.data[i0];
            float q1 = q_buf.data[i0 + rope_dim];
            q_buf.data[i0] = q0 * cos_t - q1 * sin_t;
            q_buf.data[i0 + rope_dim] = q0 * sin_t + q1 * cos_t;

            float k0 = k_buf.data[i1];
            float k1 = k_buf.data[i1 + rope_dim];
            k_buf.data[i1] = k0 * cos_t - k1 * sin_t;
            k_buf.data[i1 + rope_dim] = k0 * sin_t + k1 * cos_t;
        }
    }
}
