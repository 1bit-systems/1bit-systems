// Test suite for Attention Kernels
#include "kernels/attention.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace specdecode::kernels;

static constexpr float EPS = 1e-3f;

bool approx_equal(float a, float b, float eps = EPS) {
    return std::abs(a - b) < eps;
}

int test_rms_norm() {
    printf("=== RMSNorm ===\n");

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> weight = {0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> y(4);

    rms_norm(x, weight, y);

    // Reference: ss = 1+4+9+16 = 30, mean = 7.5
    // inv = 1/sqrt(7.5+1e-6) ≈ 0.3651
    // y[i] = weight[i] * x[i] * inv
    float ss = 1.0f + 4.0f + 9.0f + 16.0f;
    float inv = 1.0f / std::sqrt(ss / 4.0f + 1e-6f);
    for (int i = 0; i < 4; i++) {
        float expected = weight[i] * x[i] * inv;
        printf("  y[%d] = %.6f (expected %.6f)\n", i, y[i], expected);
        assert(approx_equal(y[i], expected));
    }

    printf("  PASS\n");
    return 0;
}

int test_rope() {
    printf("=== RoPE ===\n");

    std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    int head_dim = 6;

    apply_rope(x, 1, head_dim);

    printf("  RoPE(pos=1): ");
    for (int i = 0; i < 6; i++) printf("%.4f ", x[i]);
    printf("\n");

    // Verify it's not the same as original
    assert(std::abs(x[0] - 1.0f) > 0.1f || std::abs(x[1]) > 0.1f);

    // Verify norm is preserved
    float orig_norm = std::sqrt(1.0f + 0.0f + 0.0f + 1.0f + 1.0f + 0.0f);
    float new_norm = 0.0f;
    for (int i = 0; i < 6; i++) new_norm += x[i] * x[i];
    new_norm = std::sqrt(new_norm);
    printf("  Norm preservation: %.6f -> %.6f\n", orig_norm, new_norm);
    assert(approx_equal(orig_norm, new_norm, 0.01f));

    printf("  PASS\n");
    return 0;
}

int test_swiglu() {
    printf("=== SwiGLU ===\n");

    std::vector<float> gate = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> up = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4);

    swiglu(gate, up, out);

    // swish(x) = x * sigmoid(x)
    // swish(0) = 0, swish(1) ≈ 0.731, swish(-1) ≈ -0.269, swish(2) ≈ 1.762
    auto swish = [](float x) { return x / (1.0f + std::exp(-x)); };

    for (int i = 0; i < 4; i++) {
        float expected = swish(gate[i]) * up[i];
        printf("  out[%d] = %.6f (expected %.6f)\n", i, out[i], expected);
        if (std::abs(out[i]) > 0.001f) {
            assert(approx_equal(out[i], expected));
        }
    }

    printf("  PASS\n");
    return 0;
}

int test_flash_attention_single() {
    printf("=== Flash Attention (Single Query) ===\n");

    AttentionConfig cfg;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.gqa_ratio = 2;
    cfg.attn_scale = 1.0f / std::sqrt(8.0f);

    FlashAttention attn(cfg);

    // Single query, short context
    std::vector<float> q(4 * 8);
    for (int i = 0; i < 32; i++) q[i] = (float)(i % 4) / 4.0f;

    // K/V cache with 3 positions
    std::vector<float> k_cache(3 * 2 * 8);
    std::vector<float> v_cache(3 * 2 * 8);
    for (int p = 0; p < 3; p++) {
        for (int h = 0; h < 2; h++) {
            for (int d = 0; d < 8; d++) {
                k_cache[(size_t)p * 2 * 8 + (size_t)h * 8 + d] = (float)(p + h * 3 + d) / 8.0f;
                v_cache[(size_t)p * 2 * 8 + (size_t)h * 8 + d] = (float)(p * 2 + h + d) / 8.0f;
            }
        }
    }

    std::vector<float> output(4 * 8);
    attn.forward(q, k_cache, v_cache, 3, output);

    printf("  Output: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", output[i]);
    printf("...\n");

    // Verify all values are finite
    for (auto v : output) {
        assert(std::isfinite(v));
    }

    // Verify some values are non-zero (attention happened)
    bool has_nonzero = false;
    for (auto v : output) {
        if (std::abs(v) > 0.001f) { has_nonzero = true; break; }
    }
    assert(has_nonzero);

    printf("  PASS\n");
    return 0;
}

int test_batched_attention() {
    printf("=== Batched Flash Attention ===\n");

    AttentionConfig cfg;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.gqa_ratio = 2;
    cfg.attn_scale = 1.0f / std::sqrt(8.0f);

    BatchedAttention batched_attn(cfg);

    // 2 queries
    std::vector<float> q(2 * 4 * 8);
    for (int i = 0; i < 64; i++) q[i] = (float)(i % 8) / 8.0f;

    // K/V with 4 positions
    std::vector<float> k_cache(4 * 2 * 8);
    std::vector<float> v_cache(4 * 2 * 8);
    for (int i = 0; i < 64; i++) {
        k_cache[i] = (float)(i % 16) / 16.0f;
        v_cache[i] = (float)(i % 8) / 8.0f;
    }

    std::vector<float> output(2 * 4 * 8);
    batched_attn.forward(q, k_cache, v_cache, 2, 4, 0, output);

    printf("  Output (query 0): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", output[i]);
    printf("\n");
    printf("  Output (query 1): ");
    for (int i = 32; i < 40; i++) printf("%.4f ", output[i]);
    printf("\n");

    for (auto v : output) {
        assert(std::isfinite(v));
    }

    printf("  PASS\n");
    return 0;
}

int test_softmax() {
    printf("=== Softmax ===\n");

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    softmax(x);

    float sum = 0.0f;
    for (auto v : x) sum += v;
    printf("  Sum: %.6f (expected 1.0)\n", sum);
    assert(approx_equal(sum, 1.0f));

    // Last element should be largest
    assert(x[4] > x[3]);

    printf("  PASS\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_rms_norm();
    failures += test_rope();
    failures += test_swiglu();
    failures += test_softmax();
    failures += test_flash_attention_single();
    failures += test_batched_attention();

    printf("\n=== Attention Kernel Tests: %d failures ===\n", failures);
    return failures;
}
