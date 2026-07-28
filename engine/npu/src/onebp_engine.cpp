/** onebp_engine.cpp — 1BP model loader for the NPU engine.
 *
 *  Bridges OnebpModel (1BP format) with the existing npu_engine_universal's
 *  I8 weight pipeline. Called during model init when a .1bp file is detected.
 *
 *  Flow:
 *    1. Open 1BP file via OnebpModel (mmap)
 *    2. Read model config from binary header
 *    3. For each tensor: dequantize to float32, transpose, quantize to I8,
 *       pack into the engine's per-layer BOs
 *    4. Load norm weights (stored as float32 in the 1BP format)
 *
 *  This replaces the Q4NX mmap+dequant pipeline (lines ~400-550 of
 *  npu_engine_universal.cpp) when running a .1bp model.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "onebp_format.h"
#include "onebp_loader.hpp"

// ─── Dequantize 1BP tensor → float32 → transpose → pack to I8 ─────
//
// Mirrors what the Q4NX path does:
//   i8p = dequant_i8_to_float_ex(q4nx_data) → transpose_pack → packB()
//
// For 1BP: OnebpModel::get_tensor_f32() → transpose_pack → packB()
//
bool load_onebp_weights(
    OnebpModel&           model,
    const char*           tensor_name,
    int                   rows,     // expected rows (out_features)
    int                   cols,     // expected cols (in_features)
    float*                scale_out, // per-layer scale
    void*                 packB_context, // I8Ctx* for calling packB
    void                  (*packB_func)(void*, int, const float*, int, int, float&) // callback
) {
    std::vector<float> f32_data;
    if (!model.get_tensor_f32(tensor_name, f32_data)) {
        return false;
    }
    
    // Reshape: model stores as [rows, cols], need to transpose for packB
    // (packB expects [in_features, out_features] layout)
    int total = rows * cols;
    if ((int)f32_data.size() != total) {
        fprintf(stderr, "  size mismatch for %s: expected %d, got %zu\n",
                tensor_name, total, f32_data.size());
        return false;
    }
    
    // packB expects the caller to handle transposition
    // Transpose: [rows, cols] → [cols, rows] for packB
    std::vector<float> transposed(total);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            transposed[(size_t)c * rows + r] = f32_data[(size_t)r * cols + c];
    
    if (packB_func) {
        packB_func(packB_context, 0, transposed.data(), cols, rows, *scale_out);
    }
    
    return true;
}

// ─── Load 1BP model ────────────────────────────────────────────────
// Returns false on error. Populates engine's weight BOs via callbacks.
//
bool load_onebp_model(
    OnebpModel& model,
    int* out_H, int* out_NC, int* out_NH, int* out_NKV,
    int* out_HD, int* out_IM, int* out_NV,
    float* out_rope_theta
) {
    auto& h = model.header();
    *out_H = h.hidden_size;
    *out_NC = h.num_layers;
    *out_NH = h.num_attention_heads;
    *out_NKV = h.num_kv_heads;
    *out_HD = h.head_dim;
    *out_IM = h.intermediate_size;
    *out_NV = h.vocab_size;
    *out_rope_theta = h.rope_theta();
    
    return h.valid();
}
