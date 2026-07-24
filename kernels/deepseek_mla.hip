// deepseek_mla.hip — Multi-Head Latent Attention (MLA) kernels for DeepSeek V2/V3/R1
//
// MLA compressed attention:
//   c = x @ W_kv_a        [hidden] → [kv_lora_rank + qk_rope_dim]
//   k_rope = c[-qk_rope_dim:] with RoPE applied
//   k_nope = c[:kv_lora_rank] @ W_kv_b  [kv_lora_rank] → [n_heads * qk_nope_dim]
//   v       = c[:kv_lora_rank] @ W_kv_b  (second half of W_kv_b output)
//   q_nope = x @ W_q_a @ W_q_b          (compressed Q decompress)
//   q_rope = x @ W_q_a[-qk_rope_dim:] with RoPE
//   score = (q_nope @ k_nope + q_rope @ k_rope) / sqrt(head_dim)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

// ─── MLA compressed KV projection ─────────────────────────────────
// Each block handles one token: c = x @ W_kv_a^T
// where W_kv_a is [hidden, kv_lora_rank + qk_rope_dim]
__global__ void mla_kv_proj_kernel(
    float* __restrict__ c,           // [kv_lora_rank + qk_rope_dim] output latent
    const float* __restrict__ x,     // [hidden] input
    const float* __restrict__ w_kv_a, // [hidden, kv_lora_rank + qk_rope_dim]
    int hidden, int kv_dim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= kv_dim) return;
    
    float sum = 0.0f;
    for (int j = 0; j < hidden; j++) {
        sum += x[j] * w_kv_a[(size_t)j * kv_dim + d];
    }
    c[d] = sum;
}

// ─── MLA KV decompress ────────────────────────────────────────────
// K_nope = c[:kv_lora_rank] @ W_kv_b[:, :n_heads * qk_nope_dim]
// V      = c[:kv_lora_rank] @ W_kv_b[:, n_heads * qk_nope_dim: n_heads * (qk_nope_dim + v_dim)]
// Each block handles one head
__global__ void mla_kv_decompress_kernel(
    float* __restrict__ k_nope,      // [n_heads * qk_nope_dim]
    float* __restrict__ v,           // [n_heads * v_dim]
    const float* __restrict__ c,     // [kv_lora_rank]
    const float* __restrict__ w_kv_b, // [kv_lora_rank, n_heads * (qk_nope_dim + v_dim)]
    int kv_lora_rank, int qk_nope_dim, int v_dim, int n_heads)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;
    
    int head_offset = h * (qk_nope_dim + v_dim);
    
    for (int d = 0; d < qk_nope_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < kv_lora_rank; j++) {
            sum += c[j] * w_kv_b[(size_t)j * n_heads * (qk_nope_dim + v_dim) + head_offset + d];
        }
        k_nope[(size_t)h * qk_nope_dim + d] = sum;
    }
    
    for (int d = 0; d < v_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < kv_lora_rank; j++) {
            sum += c[j] * w_kv_b[(size_t)j * n_heads * (qk_nope_dim + v_dim) + head_offset + qk_nope_dim + d];
        }
        v[(size_t)h * v_dim + d] = sum;
    }
}

// ─── MLA attention (single token decode) ──────────────────────────
// Q attends to all cached K,V. Single query token, full KV cache.
// Each block handles one head.
__global__ void mla_attn_kernel(
    float* __restrict__ out,         // [n_heads * v_dim] output
    const float* __restrict__ q_nope, // [n_heads * qk_nope_dim]
    const float* __restrict__ q_rope, // [n_heads * qk_rope_dim]
    const float* __restrict__ k_cache, // [max_seq, n_heads * (qk_nope_dim + qk_rope_dim + v_dim)]
    int seq_len, int n_heads, int qk_nope_dim, int qk_rope_dim, int v_dim, float scale)
{
    int h = blockIdx.x;
    if (h >= n_heads) return;
    
    int qk_dim = qk_nope_dim + qk_rope_dim;
    int kv_stride = n_heads * (qk_nope_dim + qk_rope_dim + v_dim);
    
    // Compute score = sum(q_nope * k_nope + q_rope * k_rope) for each position
    extern __shared__ float s_scores[];
    float* scores = s_scores;
    
    float max_score = -1e30f;
    for (int s = 0; s < seq_len; s++) {
        float acc = 0.0f;
        // nope part
        for (int d = threadIdx.x; d < qk_nope_dim; d += blockDim.x) {
            acc += q_nope[(size_t)h * qk_nope_dim + d] * k_cache[(size_t)s * kv_stride + (size_t)h * (qk_nope_dim + qk_rope_dim + v_dim) + d];
        }
        // rope part
        for (int d = threadIdx.x; d < qk_rope_dim; d += blockDim.x) {
            acc += q_rope[(size_t)h * qk_rope_dim + d] * k_cache[(size_t)s * kv_stride + (size_t)h * (qk_nope_dim + qk_rope_dim + v_dim) + qk_nope_dim + d];
        }
        // Warp reduce
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
        if (threadIdx.x == 0) {
            scores[s] = acc * scale;
            max_score = fmaxf(max_score, scores[s]);
        }
        __syncthreads();
    }
    
    // Softmax
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) { scores[s] = expf(scores[s] - max_score); sum += scores[s]; }
        float inv = 1.0f / sum;
        for (int s = 0; s < seq_len; s++) scores[s] *= inv;
    }
    __syncthreads();
    
    // Weighted sum of V
    for (int d = threadIdx.x; d < v_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            acc += scores[s] * k_cache[(size_t)s * kv_stride + (size_t)h * (qk_nope_dim + qk_rope_dim + v_dim) + qk_nope_dim + qk_rope_dim + d];
        }
        out[(size_t)h * v_dim + d] = acc;
    }
}

// ─── MoE router kernel ────────────────────────────────────────────
// Computes router logits and selects top-k experts
__global__ void moe_router_kernel(
    int* __restrict__ expert_ids,     // [top_k] selected expert indices
    float* __restrict__ expert_wts,   // [top_k] expert weights
    const float* __restrict__ x,      // [hidden] input
    const float* __restrict__ w_gate, // [hidden, n_routed_experts]
    int hidden, int n_routed, int top_k)
{
    // Compute router scores
    extern __shared__ float s_scores[];
    float* scores = s_scores;
    
    for (int i = threadIdx.x; i < n_routed; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += x[j] * w_gate[(size_t)j * n_routed + i];
        }
        scores[i] = sum;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        // Softmax over all experts
        float mx = scores[0]; for (int i = 1; i < n_routed; i++) mx = fmaxf(mx, scores[i]);
        float sum = 0; for (int i = 0; i < n_routed; i++) { scores[i] = expf(scores[i] - mx); sum += scores[i]; }
        for (int i = 0; i < n_routed; i++) scores[i] /= sum;
        
        // Top-k selection (simple sort for small k)
        for (int k = 0; k < top_k; k++) {
            int best = 0; float best_v = -1e30f;
            for (int i = 0; i < n_routed; i++) {
                if (scores[i] > best_v) { best_v = scores[i]; best = i; }
            }
            expert_ids[k] = best;
            expert_wts[k] = scores[best];
            scores[best] = -1e30f; // mask for next iteration
        }
    }
}

// ─── GEMV kernel (renamed to avoid clash with whisper_kernels) ──
__global__ void ds_gemv_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ w,
    int M, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float sum = 0.0f;
    for (int j = 0; j < K; j++) sum += in[j] * w[(size_t)i * K + j];
    out[i] = sum;
}
