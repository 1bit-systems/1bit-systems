// Test suite for INT8 Quantization module
// Compile: c++ -std=c++23 -O2 -I../.. test_int8_quant.cpp -o test_int8_quant -fopenmp

#include "quantization/int8_quant.h"
#include <cstdio>
#include <cmath>
#include <cassert>

using namespace specdecode::quant;

static constexpr float EPS = 1e-3f;

bool approx_equal(float a, float b, float eps = EPS) {
    return std::abs(a - b) < eps;
}

int test_weight_quantization() {
    printf("=== Weight Quantization ===\n");

    Int8QuantConfig cfg;
    Int8WeightQuant quantizer(cfg);

    // Simple 2x3 weight matrix
    std::vector<float> weights = {
        1.0f, -2.0f,  3.0f,
        4.0f, -5.0f,  6.0f
    };

    auto packed = quantizer.quantize(weights, 2, 3);

    assert(packed.rows == 2);
    assert(packed.cols == 3);
    assert(packed.scales.size() == 2);

    printf("  Scales: %.6f, %.6f\n", packed.scales[0], packed.scales[1]);
    assert(approx_equal(packed.scales[0], 3.0f / 127.0f));  // amax = 3
    assert(approx_equal(packed.scales[1], 6.0f / 127.0f));  // amax = 6

    // Dequantize and verify
    std::vector<float> deq(6);
    Int8WeightQuant::dequantize(packed, deq);

    printf("  Dequantized: ");
    for (int i = 0; i < 6; i++) printf("%.2f ", deq[i]);
    printf("\n");

    for (int i = 0; i < 6; i++) {
        assert(approx_equal(deq[i], weights[i], 0.05f));  // INT8 precision
    }

    printf("  PASS\n");
    return 0;
}

int test_activation_quantization() {
    printf("=== Activation Quantization ===\n");

    Int8QuantConfig cfg;
    Int8ActQuant act_quant(cfg);

    // 3 tokens, hidden=4
    std::vector<float> activations = {
        0.5f, -1.2f,  2.3f, -0.8f,
        1.1f,  0.3f, -0.5f,  2.7f,
        -3.1f, 1.8f, -0.2f,  0.9f
    };

    auto qact = act_quant.quantize(activations, 3, 4);

    assert(qact.qdata.size() == 12);
    assert(qact.scales.size() == 3);

    printf("  Activation scales: %.6f, %.6f, %.6f\n",
           qact.scales[0], qact.scales[1], qact.scales[2]);

    // Verify scale: token 0 amax=2.3 -> scale=2.3/127
    float expected_scale0 = 2.3f / 127.0f;
    assert(approx_equal(qact.scales[0], expected_scale0));

    // Verify quantized values
    float inv0 = 1.0f / qact.scales[0];
    for (int i = 0; i < 4; i++) {
        float qf = qact.qdata[i] * qact.scales[0];
        assert(approx_equal(qf, activations[i], 0.05f));
    }

    printf("  PASS\n");
    return 0;
}

int test_calibrator() {
    printf("=== KL Calibrator ===\n");

    Calibrator calib(4, 128);

    // Feed synthetic activations with small values that fit in histogram range
    std::vector<float> data;
    for (int i = 0; i < 2000; i++) {
        // Uniform-like distribution in [-1, 1]
        float v = (float)(std::sin(i * 0.5f) * 0.8f + std::cos(i * 0.3f) * 0.5f);
        for (int h = 0; h < 4; h++) {
            data.push_back(v * (1.0f + h * 0.1f));
        }
    }

    calib.feed(data, 2000);
    auto result = calib.calibrate(30);

    printf("  Calibration scales: ");
    for (int h = 0; h < 4; h++) {
        printf("%.4f ", result.per_channel_scales[h]);
    }
    printf("\n");
    printf("  Max scale: %.4f\n", result.max_scale);
    printf("  Fixed activation scale: %.6f\n", result.fixed_activation_scale());

    assert(result.max_scale > 0.0f);
    assert(result.fixed_activation_scale() > 0.0f);
    // KL divergence should be >= 0 (or 0 if no data variation)

    printf("  PASS\n");
    return 0;
}

int test_int8_gemm() {
    printf("=== INT8 GEMM ===\n");

    // A: 2x4 INT8, B: 3x4 INT8 (transposed weights)
    // C = A @ B^T => 2x3
    const int M = 2, N = 3, K = 4;

    std::vector<int8_t> A_data = {1, 2, -1, 0, -2, 1, 3, -1};
    std::vector<float> A_scales = {1.0f, 1.0f};
    std::vector<int8_t> B_data = {1, 0, -1, 2, 0, 1, -2, 1, 1, -1, 0, 2};
    std::vector<float> B_scales = {1.0f, 1.0f, 1.0f};

    std::vector<float> C(M * N, 0.0f);

    Int8Gemm<>::compute({
        .A = A_data.data(),
        .A_scales = A_scales.data(),
        .B = B_data.data(),
        .B_scales = B_scales.data(),
        .M = M, .N = N, .K = K,
        .alpha = 1.0f
    }, C);

    // Compute reference
    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += A_data[m * K + k] * B_data[n * K + k];
            }
            ref[m * N + n] = (float)acc;
        }
    }

    printf("  GEMM result vs reference:\n");
    for (int m = 0; m < M; m++) {
        printf("    Row %d: ", m);
        for (int n = 0; n < N; n++) {
            printf("%.1f(%.1f) ", C[m * N + n], ref[m * N + n]);
            assert(C[m * N + n] == ref[m * N + n]);
        }
        printf("\n");
    }

    // Test with scaling
    A_scales = {0.5f, 0.25f};
    B_scales = {0.1f, 0.2f, 0.3f};

    std::fill(C.begin(), C.end(), 0.0f);
    Int8Gemm<>::compute({
        .A = A_data.data(),
        .A_scales = A_scales.data(),
        .B = B_data.data(),
        .B_scales = B_scales.data(),
        .M = M, .N = N, .K = K,
        .alpha = 1.0f
    }, C);

    printf("  Scaled GEMM:\n");
    for (int m = 0; m < M; m++) {
        printf("    Row %d: ", m);
        for (int n = 0; n < N; n++) {
            float expected = ref[m * N + n] * A_scales[m] * B_scales[n];
            printf("%.4f(%.4f) ", C[m * N + n], expected);
            assert(approx_equal(C[m * N + n], expected, 0.01f));
        }
        printf("\n");
    }

    printf("  PASS\n");
    return 0;
}

int test_kv_cache_quant() {
    printf("=== KV Cache Quantization ===\n");

    const int N = 8; // 8 KV head dim elements
    std::vector<float> k = {0.5f, -1.2f, 3.4f, -0.8f, 2.1f, -0.3f, 1.7f, -2.5f};
    std::vector<float> v = {1.0f, -0.5f, 2.0f, -1.0f, 0.5f, -2.0f, 1.5f, -0.8f};

    std::vector<int8_t> kq(N), vq(N);
    float k_scale, v_scale;

    KVCacheQuant::quantize_token(k, v, kq, vq, k_scale, v_scale);

    printf("  K scale: %.6f, V scale: %.6f\n", k_scale, v_scale);

    // Dequantize
    std::vector<float> kd(N), vd(N);
    KVCacheQuant::dequantize_token(kq, vq, k_scale, v_scale, kd, vd);

    printf("  Dequantized vs original:\n");
    for (int i = 0; i < N; i++) {
        printf("    [%d]: K %.4f(%.4f) V %.4f(%.4f)\n",
               i, kd[i], k[i], vd[i], v[i]);
        assert(approx_equal(kd[i], k[i], 0.05f));
        assert(approx_equal(vd[i], v[i], 0.05f));
    }

    printf("  PASS\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_weight_quantization();
    failures += test_activation_quantization();
    failures += test_calibrator();
    failures += test_int8_gemm();
    failures += test_kv_cache_quant();

    printf("\n=== INT8 Quant Tests: %d failures ===\n", failures);
    return failures;
}
