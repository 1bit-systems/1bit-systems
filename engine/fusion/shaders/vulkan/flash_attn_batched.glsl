#version 450

// ─────────────────────────────────────────────────────────────────────────────
// Flash Attention Batched — N-query prefill or decode
// Grid = (n_heads, n_queries, 1); each (head, query) workgroup uses
// causal_len = seq_start + query + 1
// ─────────────────────────────────────────────────────────────────────────────

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint seq_start;
    uint n_queries;
    uint page_size;
    uint attn_scale_bits;
    uint sink_offset;
} push;

layout(set = 0, binding = 0) buffer Q      { float data[]; } q_buf;
layout(set = 0, binding = 1) buffer KCache  { float data[]; } k_cache;
layout(set = 0, binding = 2) buffer VCache  { float data[]; } v_cache;
layout(set = 0, binding = 3) buffer PageTbl { uint data[]; } page_table;
layout(set = 0, binding = 4) buffer Output  { float data[]; } out_buf;
layout(set = 0, binding = 5) buffer Sinks   { float data[]; } sinks;

shared float s_q[128];

void main() {
    uint head_id = gl_GlobalInvocationID.x;
    uint query_id = gl_GlobalInvocationID.y;
    if (head_id >= push.n_heads || query_id >= push.n_queries) return;

    uint head_dim = push.head_dim;
    float attn_scale = uintBitsToFloat(push.attn_scale_bits);
    if (attn_scale == 0.0) attn_scale = 1.0 / sqrt(float(head_dim));

    uint kv_head = head_id % push.n_kv_heads;
    uint causal_len = push.seq_start + query_id + 1;

    // Load Q
    uint q_offset = query_id * push.n_heads * head_dim + head_id * head_dim;
    for (uint i = gl_LocalInvocationIndex; i < head_dim; i += gl_WorkGroupSize.x) {
        s_q[i] = q_buf.data[q_offset + i];
    }
    barrier();

    // Iterate over KV pages up to causal_len
    uint num_pages = (causal_len + push.page_size - 1) / push.page_size;
    uint page_stride = push.n_kv_heads * push.page_size * head_dim;

    float max_score = -1e20;
    float sum_exp = 0.0;
    float acc[128];
    for (uint i = 0; i < head_dim; i++) acc[i] = 0.0;

    for (uint p = 0; p < num_pages; p++) {
        uint phys_page = page_table.data[p];
        if (phys_page == 0xFFFFFFFF) continue;

        uint page_len = min(push.page_size, causal_len - p * push.page_size);
        uint kv_offset = phys_page * page_stride + kv_head * push.page_size * head_dim;

        for (uint t = 0; t < page_len; t++) {
            float score = 0.0;
            for (uint d = gl_LocalInvocationIndex; d < head_dim; d += gl_WorkGroupSize.x) {
                score += s_q[d] * k_cache.data[kv_offset + t * head_dim + d];
            }
            score *= attn_scale;

            float new_max = max(max_score, score);
            float exp_shift = exp(max_score - new_max);
            float exp_score = exp(score - new_max);

            for (uint d = gl_LocalInvocationIndex; d < head_dim; d += gl_WorkGroupSize.x) {
                acc[d] = acc[d] * exp_shift + exp_score * v_cache.data[kv_offset + t * head_dim + d];
            }

            max_score = new_max;
            sum_exp = sum_exp * exp_shift + exp_score;
        }
    }

    uint out_offset = query_id * push.n_heads * head_dim + head_id * head_dim;
    for (uint d = gl_LocalInvocationIndex; d < head_dim; d += gl_WorkGroupSize.x) {
        out_buf.data[out_offset + d] = sum_exp > 0.0 ? acc[d] / sum_exp : 0.0;
    }
}
