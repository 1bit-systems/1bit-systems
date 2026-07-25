// CPU attention stub — replaces rcpp_kv_cache_attn_decode for linking
// The GPU-attn kernel (kv_cache_attn.hip) would be linked in production.
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <omp.h>

extern "C" int rcpp_kv_cache_attn_decode(
    const void* Q_dev, const void* K_dev, const void* V_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale, void* stream)
{
    (void)stream;
    const float* Q = (const float*)Q_dev;
    const float* K = (const float*)K_dev;
    const float* V = (const float*)V_dev;
    float* out = (float*)out_dev;
    int gqa = num_q_heads / num_kv_heads;

    #pragma omp parallel for
    for (int h = 0; h < num_q_heads; h++) {
        int kvh = h / gqa;
        float scores[4096];
        float mx = -1e30f;
        for (int p = 0; p < seq_len; p++) {
            double s = 0;
            for (int d = 0; d < head_dim; d++)
                s += (double)Q[(size_t)h * head_dim + d] *
                     K[(size_t)p * num_kv_heads * head_dim + kvh * head_dim + d];
            scores[p] = (float)(s * scale);
            if (scores[p] > mx) mx = scores[p];
        }
        double sw = 0;
        for (int p = 0; p < seq_len; p++) {
            scores[p] = expf(scores[p] - mx);
            sw += scores[p];
        }
        float isw = sw > 0 ? 1.0f / (float)sw : 1.0f / seq_len;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0;
            for (int p = 0; p < seq_len; p++)
                acc += scores[p] * V[(size_t)p * num_kv_heads * head_dim + kvh * head_dim + d];
            out[(size_t)h * head_dim + d] = acc * isw;
        }
    }
    return 0;
}
