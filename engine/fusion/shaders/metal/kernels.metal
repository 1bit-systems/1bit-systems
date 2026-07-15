// ─────────────────────────────────────────────────────────────────────────────
// Metal Performance Shaders — GPU compute kernels for Apple Silicon
// Compiled with: xcrun -sdk macosx metal -c kernels.metal -o kernels.air
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
using namespace metal;

// ── Flash Attention (single query decode) ──
// Grid: (n_heads, 1, 1), Threads: (head_dim, 1, 1)
kernel void flash_attn(
    device const float* q          [[buffer(0)]],
    device const float* k_cache    [[buffer(1)]],
    device const float* v_cache    [[buffer(2)]],
    device const int*   page_table [[buffer(3)]],
    device float*       output     [[buffer(4)]],
    device const float* sinks      [[buffer(5)]],
    constant int&       head_dim   [[buffer(6)]],
    constant int&       n_heads    [[buffer(7)]],
    constant int&       n_kv_heads [[buffer(8)]],
    constant int&       seq_len    [[buffer(9)]],
    constant int&       page_size  [[buffer(10)]],
    constant float&     attn_scale [[buffer(11)]],
    constant int&       sink_offset[[buffer(12)]],
    uint head_id                  [[threadgroup_position_in_grid]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    if (head_id >= uint(n_heads)) return;

    int kv_head = int(head_id) % n_kv_heads;

    // Load Q into threadgroup memory
    threadgroup float s_q[128];
    if (tid < uint(head_dim)) s_q[tid] = q[head_id * uint(head_dim) + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int num_pages = (seq_len + page_size - 1) / page_size;
    int page_stride = n_kv_heads * page_size * head_dim;

    float max_score = -1e20f;
    float sum_exp = 0.0f;
    // Simplified: each thread accumulates its dimension
    // Full implementation would use threadgroup memory for online softmax

    for (int p = 0; p < num_pages; p++) {
        int phys_page = page_table[p];
        if (phys_page < 0) continue;

        int page_len = min(page_size, seq_len - p * page_size);
        int kv_offset = phys_page * page_stride + kv_head * page_size * head_dim;

        for (int t = 0; t < page_len; t++) {
            float score = 0.0f;
            int kv_idx = kv_offset + t * head_dim;

            for (uint d = tid; d < uint(head_dim); d += 32) {
                score += s_q[d] * k_cache[uint(kv_idx) + d];
            }

            // SIMD-group reduction (warp-level)
            score = simd_sum(score);

            if (tid == 0) {
                score *= attn_scale;
                float new_max = max(max_score, score);
                float exp_shift = exp(max_score - new_max);
                float exp_score = exp(score - new_max);
                max_score = new_max;
                sum_exp = sum_exp * exp_shift + exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0) {
        // Write result (simplified: dimension 0 only)
        // Full implementation writes all head_dim elements
        output[head_id * uint(head_dim)] = sum_exp > 0.0f ? 1.0f : 0.0f;
    }
}

// ── GEMM (f32) ──
kernel void gemm_f32(
    device const float* A        [[buffer(0)]],
    device const float* B        [[buffer(1)]],
    device float*       C        [[buffer(2)]],
    constant int&       M        [[buffer(3)]],
    constant int&       N        [[buffer(4)]],
    constant int&       K        [[buffer(5)]],
    uint2               gid      [[thread_position_in_grid]]
) {
    int row = int(gid.y);
    int col = int(gid.x);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ── RMS Norm ──
kernel void rms_norm(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant int&       n_cols [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint                row    [[thread_position_in_grid]]
) {
    float ss = 0.0f;
    for (int c = 0; c < n_cols; c++) {
        float v = input[row * uint(n_cols) + uint(c)];
        ss += v * v;
    }
    float rms = sqrt(ss / float(n_cols) + eps);
    for (int c = 0; c < n_cols; c++) {
        output[row * uint(n_cols) + uint(c)] = input[row * uint(n_cols) + uint(c)] / rms * weight[c];
    }
}

// ── SiLU + Mul ──
kernel void silu_mul(
    device const float* a     [[buffer(0)]],
    device const float* b     [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    uint                idx   [[thread_position_in_grid]]
) {
    float x = a[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    out[idx] = (x * sig) * b[idx];
}

// ── RoPE ──
kernel void rope(
    device float*       q       [[buffer(0)]],
    device float*       k       [[buffer(1)]],
    device const float* sin_t   [[buffer(2)]],
    device const float* cos_t   [[buffer(3)]],
    constant int&       n_heads [[buffer(4)]],
    constant int&       head_dim[[buffer(5)]],
    constant int&       rope_dim[[buffer(6)]],
    uint                token   [[thread_position_in_grid]]
) {
    float s = sin_t[token];
    float c = cos_t[token];
    int stride = n_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int base = int(token) * stride + h * head_dim;
        for (int d = 0; d < rope_dim; d++) {
            float q0 = q[base + d];
            float q1 = q[base + d + rope_dim];
            q[base + d] = q0 * c - q1 * s;
            q[base + d + rope_dim] = q0 * s + q1 * c;

            float k0 = k[base + d];
            float k1 = k[base + d + rope_dim];
            k[base + d] = k0 * c - k1 * s;
            k[base + d + rope_dim] = k0 * s + k1 * c;
        }
    }
}

// ── Softmax ──
kernel void softmax(
    device float* data [[buffer(0)]],
    constant int& rows [[buffer(1)]],
    constant int& cols [[buffer(2)]],
    uint row           [[thread_position_in_grid]]
) {
    if (int(row) >= rows) return;

    float max_val = -1e20f;
    for (int c = 0; c < cols; c++) {
        max_val = max(max_val, data[row * uint(cols) + uint(c)]);
    }

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        sum += exp(data[row * uint(cols) + uint(c)] - max_val);
    }

    float inv_sum = 1.0f / sum;
    for (int c = 0; c < cols; c++) {
        data[row * uint(cols) + uint(c)] = exp(data[row * uint(cols) + uint(c)] - max_val) * inv_sum;
    }
}
