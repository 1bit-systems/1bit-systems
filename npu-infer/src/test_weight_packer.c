#include "model.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test: pack a weight block and compare with captured NPU weight data

static void dump_bf16_hex(const char* label, const uint16_t* data, int count) {
    printf("%s:\n", label);
    for (int i = 0; i < count && i < 16; i++) {
        float f = bf16_to_float(data[i]);
        printf("  [%3d] 0x%04x = %.4f%s\n", i, data[i], f,
               data[i] == 0 ? " (ZERO)" : "");
    }
    if (count > 16) {
        printf("  ... (%d values total)\n", count);
    }
}

int main(int argc, char** argv) {
    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    
    printf("=== NPU Weight Packer Test ===\n\n");
    
    // Load model
    ModelWeights* mw = model_load(model_path, QWEN3_0_6B_CONFIG);
    if (!mw) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    printf("\n=== Testing Q_proj layer 0 ===\n");
    LayerWeights* l0 = &mw->layers[0];
    
    printf("Q_proj: dtype=%s shape=[%ld, %ld] data_size=%lu num_blocks=%d\n",
           l0->q_proj_weight.dtype,
           (long)l0->q_proj_weight.shape[0],
           (long)l0->q_proj_weight.shape[1],
           (unsigned long)l0->q_proj_weight.data_size,
           npu_weight_num_blocks(&l0->q_proj_weight, &QWEN3_0_6B_CONFIG));
    
    void* q_proj_data = model_tensor_data(mw, &l0->q_proj_weight);
    printf("Q_proj data ptr: %p\n", q_proj_data);
    
    // Check data bytes (first few)
    const uint8_t* raw = (const uint8_t*)q_proj_data;
    int num_groups = (l0->q_proj_weight.num_elements + GROUP_SIZE - 1) / GROUP_SIZE;
    printf("Num elements: %lu\n", (unsigned long)l0->q_proj_weight.num_elements);
    printf("Num groups: %d\n", num_groups);
    printf("Expected data size: %d (scales) + %lu (I8) = %lu\n",
           num_groups * 2,
           (unsigned long)l0->q_proj_weight.num_elements,
           (unsigned long)(num_groups * 2 + l0->q_proj_weight.num_elements));
    printf("Actual data_size: %lu\n", (unsigned long)l0->q_proj_weight.data_size);
    
    // Check scale format: first group's scale
    uint16_t scale_bf16;
    memcpy(&scale_bf16, raw, 2);
    float scale = bf16_to_float(scale_bf16);
    printf("First group scale (BF16): 0x%04x = %.6f\n", scale_bf16, scale);
    
    // First I8 values
    printf("First 8 I8 values:\n");
    for (int i = 0; i < 8; i++) {
        int8_t v;
        memcpy(&v, raw + num_groups * 2 + i, 1);
        printf("  [%d] %d\n", i, v);
    }
    
    // Check: is the I8 data right after scales?
    // Total data_size should be num_groups*2 (scales) + num_elements*1 (I8)
    size_t expected_size = num_groups * 2 + l0->q_proj_weight.num_elements;
    printf("Expected total: %zu, Actual: %lu, Match: %s\n",
           expected_size,
           (unsigned long)l0->q_proj_weight.data_size,
           expected_size == l0->q_proj_weight.data_size ? "YES" : "NO");
    
    // If it matches, we have the right format
    // Now dequantize block 0
    printf("\n=== Dequantizing block 0 ===\n");
    
    uint16_t block_buf[256 * 1024]; // [256, 1024] BF16
    memset(block_buf, 0, sizeof(block_buf));
    
    int written = npu_dequant_block(block_buf, q_proj_data,
                                     &l0->q_proj_weight, &QWEN3_0_6B_CONFIG, 0);
    printf("Dequantized %d BF16 values\n", written);
    
    if (written > 0) {
        // Show first few values
        dump_bf16_hex("Q_proj block 0 first 16 BF16", block_buf, 16);
        
        // Statistics
        float min_v = 1e10f, max_v = -1e10f, sum_v = 0, sum_sq = 0;
        int count = 0;
        for (int i = 0; i < 256 * 1024; i++) {
            if (block_buf[i] == 0) continue;
            float v = bf16_to_float(block_buf[i]);
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum_v += v;
            sum_sq += v * v;
            count++;
        }
        float mean_v = count > 0 ? sum_v / count : 0;
        float std_v = count > 0 ? sqrtf(sum_sq / count - mean_v * mean_v) : 0;
        printf("\n  Statistics (%d non-zero values):\n", count);
        printf("  Min:  %.4f\n", min_v);
        printf("  Max:  %.4f\n", max_v);
        printf("  Mean: %.6f\n", mean_v);
        printf("  Std:  %.6f\n", std_v);
        
        // Compare with captured weight data statistics
        // Captured: mean≈0.0037, std≈1.819, range [-56.25, +21.625]
        printf("\n  === Matches NPU weight profile? ===\n");
        printf("  Expected: mean≈0.004, std≈1.82, range ~[-56, +22]\n");
        printf("  Got:      mean=%.4f, std=%.4f, range [%.1f, %.1f]\n",
               mean_v, std_v, min_v, max_v);
        printf("  Match: %s\n",
               (fabsf(mean_v) < 1.0f && std_v > 0.5f && std_v < 5.0f) ? "YES ✓" : "NEEDS REVIEW");
    }
    
    // Test: create a 1MB BO buffer
    printf("\n=== Packing BO buffer ===\n");
    uint8_t* bo_buf = malloc(QWEN3_0_6B_CONFIG.npu_weight_bo_size);
    npu_pack_weight_bo(bo_buf, q_proj_data, &l0->q_proj_weight, &QWEN3_0_6B_CONFIG, 0);
    
    // Verify first 512KB has data, second 512KB is zero
    int non_zero_first = 0, non_zero_second = 0;
    for (int i = 0; i < 512 * 1024; i++) {
        if (bo_buf[i] != 0) { non_zero_first = 1; break; }
    }
    for (int i = 512 * 1024; i < 1024 * 1024; i++) {
        if (bo_buf[i] != 0) { non_zero_second = 1; break; }
    }
    printf("First 512KB has data: %s\n", non_zero_first ? "YES ✓" : "NO ✗");
    printf("Second 512KB is zero: %s\n", !non_zero_second ? "YES ✓" : "NO ✗");
    
    // Write BO dump for comparison
    FILE* f = fopen("/tmp/test_weight_bo.bin", "wb");
    if (f) {
        fwrite(bo_buf, 1, QWEN3_0_6B_CONFIG.npu_weight_bo_size, f);
        fclose(f);
        printf("Wrote /tmp/test_weight_bo.bin (%d bytes)\n",
               QWEN3_0_6B_CONFIG.npu_weight_bo_size);
    }
    
    free(bo_buf);
    
    printf("\n=== Testing K_proj layer 0 ===\n");
    printf("K_proj: dtype=%s shape=[%ld, %ld] data_size=%lu num_blocks=%d\n",
           l0->k_proj_weight.dtype,
           (long)l0->k_proj_weight.shape[0],
           (long)l0->k_proj_weight.shape[1],
           (unsigned long)l0->k_proj_weight.data_size,
           npu_weight_num_blocks(&l0->k_proj_weight, &QWEN3_0_6B_CONFIG));
    
    void* k_proj_data = model_tensor_data(mw, &l0->k_proj_weight);
    memset(block_buf, 0, sizeof(block_buf));
    written = npu_dequant_block(block_buf, k_proj_data,
                                 &l0->k_proj_weight, &QWEN3_0_6B_CONFIG, 0);
    printf("Dequantized %d BF16 values\n", written);
    
    if (written > 0) {
        dump_bf16_hex("K_proj block 0 first 16 BF16", block_buf, 16);
    }
    
    printf("\n=== Testing RMS norms (layer 0) ===\n");
    void* rms_data = model_tensor_data(mw, &l0->input_layernorm_weight);
    printf("input_layernorm: dtype=%s shape=[%ld] data_size=%lu\n",
           l0->input_layernorm_weight.dtype,
           (long)l0->input_layernorm_weight.shape[0],
           (unsigned long)l0->input_layernorm_weight.data_size);
    
    // RMS norms should be BF16
    const uint16_t* rms_bf16 = (const uint16_t*)rms_data;
    printf("First 8 RMS BF16 values:\n");
    for (int i = 0; i < 8; i++) {
        printf("  [%d] 0x%04x = %.6f\n", i, rms_bf16[i], bf16_to_float(rms_bf16[i]));
    }
    
    model_free(mw);
    
    printf("\n=== ALL TESTS COMPLETE ===\n");
    return 0;
}
