#include "model.h"
#include "device.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// RMS normalization
static void rms_norm(float* output, const float* input, const float* weight,
                      int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += input[i] * input[i];
    }
    float rms = sqrtf(ss / n + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}

// SiLU activation: x * sigmoid(x)
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Matrix multiply (M x K) * (K x N) = (M x N)
static void matmul(float* C, const float* A, const float* B,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    const char* model_path = argc > 1 ? argv[1] 
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    
    LOG_INFO("Loading model from: %s", model_path);
    
    ModelWeights* mw = model_load(model_path, QWEN3_0_6B_CONFIG);
    if (!mw) {
        LOG_ERROR("Failed to load model");
        return 1;
    }
    
    // Verify some weight values
    LOG_INFO("\n=== Weight Verification ===");
    
    // Check a few BF16 values from input_layernorm
    LayerWeights* l0 = &mw->layers[0];
    uint16_t* norm_data = (uint16_t*)model_tensor_data(mw, &l0->input_layernorm_weight);
    
    LOG_INFO("Layer 0 input_layernorm (first 8 values):");
    for (int i = 0; i < 8; i++) {
        LOG_INFO("  [%d] = %f", i, bf16_to_float(norm_data[i]));
    }
    
    // Check I8 weights from q_proj
    int8_t* qproj_data = (int8_t*)model_tensor_data(mw, &l0->q_proj_weight);
    LOG_INFO("Layer 0 q_proj (first 8 I8 values):");
    for (int i = 0; i < 8; i++) {
        LOG_INFO("  [%d] = %d", i, qproj_data[i]);
    }
    
    // Verify bfloat16 conversion roundtrip
    uint16_t bf = 0x3F80;  // 1.0 in BF16
    LOG_INFO("BF16 0x3F80 = %f", bf16_to_float(bf));
    
    float test_vals[] = {1.0f, 0.5f, -0.5f, 0.0f, 3.14159f, -2.718f};
    LOG_INFO("BF16 roundtrip test:");
    for (int i = 0; i < 6; i++) {
        uint16_t encoded = float_to_bf16(test_vals[i]);
        float decoded = bf16_to_float(encoded);
        LOG_INFO("  %f -> 0x%04x -> %f", test_vals[i], encoded, decoded);
    }
    
    // Quick RMS norm test
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float output[4];
    rms_norm(output, input, weight, 4, 1e-6f);
    LOG_INFO("RMS Norm test: input=[1,2,3,4], output=[%f, %f, %f, %f]",
             output[0], output[1], output[2], output[3]);
    
    // Verify model weights file sizes
    LOG_INFO("\n=== Model Size Summary ===");
    LOG_INFO("Embed tokens: %lu MB (header overlap: %lu bytes)", 
             (unsigned long)mw->embed_tokens.data_size / 1024 / 1024,
             (unsigned long)(8 + strlen((const char*)(mw->file_data + 8))));
    
    uint64_t total_weight_bytes = mw->embed_tokens.data_size;
    for (int l = 0; l < mw->config.num_layers; l++) {
        LayerWeights* layer = &mw->layers[l];
        total_weight_bytes += layer->input_layernorm_weight.data_size;
        total_weight_bytes += layer->post_attention_layernorm_weight.data_size;
        total_weight_bytes += layer->q_norm_weight.data_size;
        total_weight_bytes += layer->k_norm_weight.data_size;
        total_weight_bytes += layer->q_proj_weight.data_size;
        total_weight_bytes += layer->k_proj_weight.data_size;
        total_weight_bytes += layer->v_proj_weight.data_size;
        total_weight_bytes += layer->o_proj_weight.data_size;
        total_weight_bytes += layer->gate_proj_weight.data_size;
        total_weight_bytes += layer->up_proj_weight.data_size;
        total_weight_bytes += layer->down_proj_weight.data_size;
    }
    total_weight_bytes += mw->norm_weight.data_size;
    total_weight_bytes += mw->lm_head_weight.data_size;
    
    LOG_INFO("Total weight data: %lu MB", (unsigned long)total_weight_bytes / 1024 / 1024);
    LOG_INFO("File size: %lu MB", (unsigned long)mw->file_size / 1024 / 1024);
    
    model_free(mw);
    LOG_INFO("Done!");
    return 0;
}
