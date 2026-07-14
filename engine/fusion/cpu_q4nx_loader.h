// cpu_q4nx_loader.h — Load Q4NX model file into CPU backend FP32 weight arrays.
//
// Q4NX format (from FLM/aie-rt):
//   1. [u64] JSON header size
//   2. [bytes] JSON metadata: tensor name → {dtype, shape, data_offsets}
//   3. [data] Tensor payloads:
//      - BF16: raw bfloat16 values
//      - I8: tiled INT8 format (5120-byte rows, 32×256 tiles)
//         Each tile row = 5120 bytes:
//           [0..512)   = 256 BF16 scales
//           [512..1024) = 256 BF16 zero-points
//           [1024..5120) = 4096 packed I4 data (32 cols × 32 rows × 4 bit)
//
// After loading, all weights are stored as flat FP32 arrays suitable for
// cpu_layer_forward_qwen3().
//
// Build: g++ -O3 -std=c++17 -c cpu_q4nx_loader.cpp
//
// @section Fused Engine
#pragma once

#include "cpu_layer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

// ── BF16 ↔ FP16 conversion ───────────────────────────────────
static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// ── Simple JSON scanner: find value for a key directly in JSON text ──
// Handles nested objects and arrays by brace counting.
// Returns the value substring (with quotes for strings, braces for objects).
static std::string json_find_value(const std::string& js, const std::string& key) {
    std::string target = "\"" + key + "\"";
    size_t p = 0;
    while (p < js.size()) {
        // Scan for the key, but only count it as a real match when it's at the
        // position after `{` or `,` (i.e., it's a key, not a value).
        size_t f = js.find(target, p);
        if (f == std::string::npos) return "";
        
        // Verify it's a key (preceded by { or ,)
        // Walk backwards past whitespace to find the structural character
        size_t pre = f;
        while (pre > 0 && (js[pre-1] == ' ' || js[pre-1] == '\t' || js[pre-1] == '\n')) pre--;
        if (pre == 0 || (js[pre-1] != '{' && js[pre-1] != ',')) {
            p = f + 1;
            continue;
        }
        
        // Find colon after key
        size_t colon = js.find(':', f + target.size());
        if (colon == std::string::npos) return "";
        
        // Skip whitespace after colon
        size_t vs = colon + 1;
        while (vs < js.size() && (js[vs] == ' ' || js[vs] == '\t' || js[vs] == '\n')) vs++;
        if (vs >= js.size()) return "";
        
        // Find end of value by brace/array depth counting
        size_t ve = vs + 1;
        if (js[vs] == '"') {
            // String value: find end quote
            while (ve < js.size()) {
                if (js[ve] == '\\') ve += 2;
                else if (js[ve] == '"') break;
                else ve++;
            }
            if (ve < js.size()) ve++;
        } else if (js[vs] == '{' || js[vs] == '[') {
            // Object or array: track nesting
            char close = (js[vs] == '{') ? '}' : ']';
            int depth = 1;
            ve = vs + 1;
            while (ve < js.size() && depth > 0) {
                if (js[ve] == '\\' && ve + 1 < js.size()) ve += 2;
                else {
                    if (js[ve] == '"') {
                        // Skip string content
                        ve++;
                        while (ve < js.size() && !(js[ve] == '"' && js[ve-1] != '\\')) ve++;
                    } else if (js[ve] == js[vs]) depth++;
                    else if (js[ve] == close) depth--;
                }
                ve++;
            }
        } else {
            // Number or keyword: find end at comma or close brace
            while (ve < js.size() && js[ve] != ',' && js[ve] != '}' && js[ve] != ']' && js[ve] != ' ' && js[ve] != '\n' && js[ve] != '\t')
                ve++;
        }
        
        return js.substr(vs, ve - vs);
    }
    return "";
}

// Unquote a JSON string value: "foo" -> foo
static std::string json_unquote(const std::string& s) {
    if (s.size() >= 2 && s[0] == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// Parse a JSON integer array: [1, 2, 3] -> vector{1,2,3}
static std::vector<int> json_parse_ints(const std::string& s) {
    std::vector<int> result;
    if (s.empty() || s[0] != '[') return result;
    size_t p = 1;
    while (p < s.size()) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == ',' || s[p] == ']'))
            p++;
        if (p >= s.size() || s[p] == ']') break;
        char* end = nullptr;
        long v = strtol(s.c_str() + p, &end, 10);
        if (end && end > s.c_str() + p) {
            result.push_back((int)v);
            p = end - s.c_str();
        } else p++;
    }
    return result;
}

// ── I8 tile dequantizer ──────────────────────────────────────
// Each 5120-byte block is ONE tile of 32×256 I4 values.
// Tile layout:
//   [0..512)     = 256 BF16 scales (g*32 + lr for group g, row lr)
//   [512..1024)  = 256 BF16 zero-points (g*32 + lr)
//   [1024..5120) = 4096 bytes packed I4 data
//     Packed layout per row (lr): lane=lr/16, bi=(lr%16)/2, ns=lr%2
//     Each byte = (upper nibble = row lane*2+1, lower nibble = row lane*2)
//     offset = lane * 2048 + col * 8 + bi
//   Total: 512 + 512 + 4096 = 5120 bytes per block

static void dequant_i8_block(
    const uint8_t* block,    // 5120 bytes = one 32×256 tile
    float* out,              // [out_rows * out_cols] float matrix
    int tr,                  // tile row index (0-based, groups of 32 output rows)
    int tc,                  // tile col index (0-based, groups of 256 input cols)
    int out_rows,            // total output rows
    int out_cols)            // total output cols
{
    const uint16_t* scales = (const uint16_t*)block;
    const uint16_t* zps    = (const uint16_t*)(block + 512);
    const uint8_t*  packed = block + 1024;

    for (int lr = 0; lr < 32; lr++) {
        int row = tr * 32 + lr;
        if (row >= out_rows) continue;
        int lane = lr / 16;
        int lr2 = lr % 16;
        int bi = lr2 / 2;
        int ns = lr % 2;

        for (int g = 0; g < 8; g++) {  // 8 groups of 32 cols = 256
            // FLM Q4NX: 6-byte header at offsets 0-5, real scales start at byte 6
            // scales are uint16 at offsets [6..512), with last 3 entries wrapping to [0..6)
            int scale_idx = g * 32 + lr;
            int scale_byte_off = 6 + scale_idx * 2;
            if (scale_byte_off >= 512) scale_byte_off -= 512;  // wrap to header
            const uint16_t* scales_ptr = (const uint16_t*)block;
            float s = bf16_to_f32(((const uint16_t*)(block + scale_byte_off))[0]);
            float z = bf16_to_f32(zps[g * 32 + lr]);
            for (int c = 0; c < 32; c++) {
                int col = g * 32 + c;
                int abs_col = tc * 256 + col;
                if (abs_col >= out_cols) continue;
                uint8_t bv = packed[lane * 2048 + col * 8 + bi];
                int cd = (ns == 0) ? (bv & 0x0F) : ((bv >> 4) & 0x0F);
                out[row * out_cols + abs_col] = (float)cd * s + z;
            }
        }
    }
}

// ── Q4NX Model Loader ────────────────────────────────────────
// Loaded tensor data stored as flat FP32 arrays.
struct Q4nxTensor {
    std::string name;
    std::string dtype;       // "BF16" or "I8"
    std::vector<int> shape;  // For I8: [tile_rows, 5120]
    size_t data_offset;      // byte offset in file
    size_t data_size;        // bytes
    std::vector<float> fp32; // dequantized FP32 values
};

struct Q4nxModel {
    // Header metadata
    std::unordered_map<std::string, Q4nxTensor> tensors;
    int vocab_size;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int inter_size;

    // Load from file
    bool load(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return false; }

        // Read header size
        uint64_t hdr_size;
        f.read((char*)&hdr_size, 8);
        if (!f || hdr_size == 0) { fprintf(stderr, "Invalid header size\n"); return false; }

        // Read JSON header
        std::string hdr_json(hdr_size, '\0');
        f.read(&hdr_json[0], hdr_size);
        if (!f) { fprintf(stderr, "Cannot read header\n"); return false; }

        // Parse all top-level keys from JSON header
        int tensor_count = 0;
        size_t pos = 0;

        while (pos < hdr_json.size()) {
            // Find next key position: look for '"' that starts a key
            // Skip past '{' or ',' at the start
            size_t ns = hdr_json.find('"', pos);
            if (ns == std::string::npos) break;
            
            // Extract key name
            size_t ne = ns + 1;
            while (ne < hdr_json.size() && hdr_json[ne] != '"') {
                if (hdr_json[ne] == '\\') ne++;
                ne++;
            }
            if (ne >= hdr_json.size()) break;
            std::string key = hdr_json.substr(ns + 1, ne - ns - 1);
            
            // Find ':' after the key
            size_t colon = hdr_json.find(':', ne);
            if (colon == std::string::npos) break;
            
            // Find the value { ... } by brace matching
            size_t vs = colon + 1;
            while (vs < hdr_json.size() && (hdr_json[vs] == ' ' || hdr_json[vs] == '\t' || hdr_json[vs] == '\n')) vs++;
            if (vs >= hdr_json.size() || hdr_json[vs] != '{') { pos = vs; continue; }
            
            int depth = 1;
            size_t ve = vs + 1;
            while (ve < hdr_json.size() && depth > 0) {
                if (hdr_json[ve] == '\\' && ve + 1 < hdr_json.size()) ve += 2;
                else {
                    if (hdr_json[ve] == '"') {
                        ve++;
                        while (ve < hdr_json.size() && !(hdr_json[ve] == '"' && hdr_json[ve-1] != '\\')) ve++;
                    } else if (hdr_json[ve] == '{') depth++;
                    else if (hdr_json[ve] == '}') depth--;
                }
                ve++;
            }
            
            // Extract the value object substring
            std::string val_sub = hdr_json.substr(vs, ve - vs);
            pos = ve;
            
            // Skip non-tensor keys (model config keys without data_offsets)
            if (val_sub.find("data_offsets") == std::string::npos) continue;
            
            // Extract fields from the value substring
            std::string dt = json_unquote(json_find_value(val_sub, "dtype"));
            std::vector<int> shape = json_parse_ints(json_find_value(val_sub, "shape"));
            std::vector<int> offsets = json_parse_ints(json_find_value(val_sub, "data_offsets"));
            
            if (offsets.size() < 2) continue;
            
            Q4nxTensor tensor;
            tensor.name = key;
            tensor.dtype = dt;
            tensor.shape = shape;
            tensor.data_offset = (size_t)offsets[0];
            tensor.data_size = (size_t)(offsets[1] - offsets[0]);
            tensors[key] = tensor;
            tensor_count++;
        }
        printf("[q4nx] %d tensors in header (%zu bytes)\n", tensor_count, hdr_json.size());

        // Get model dimensions from embed_tokens
        auto it = tensors.find("model.embed_tokens.weight");
        if (it != tensors.end()) {
            if (it->second.shape.size() >= 2) {
                vocab_size = it->second.shape[0];
                hidden_dim = it->second.shape[1];
            }
        }

        // Count layers
        n_layers = 0;
        for (auto& [name, t] : tensors) {
            if (name.find("model.layers.") == 0 && name.find("input_layernorm.weight") != std::string::npos) {
                int layer;
                if (sscanf(name.c_str(), "model.layers.%d.", &layer) == 1) {
                    if (layer + 1 > n_layers) n_layers = layer + 1;
                }
            }
        }

        // Derive model dimensions from tile shapes
        // shape[0] = ntr * ntc where ntr = ceil(OUT/32), ntc = ceil(IN/256)
        auto qit = tensors.find("model.layers.0.self_attn.q_proj.weight");
        if (qit != tensors.end() && qit->second.shape.size() >= 1 && hidden_dim > 0) {
            int n_blocks = qit->second.shape[0];
            int ntc = (hidden_dim + 255) / 256;
            int ntr = n_blocks / ntc;
            int q_out = ntr * 32;         // NH * HD
            int hd_guess = 128;
            n_heads = q_out / hd_guess;
            head_dim = hd_guess;
        }

        auto kit = tensors.find("model.layers.0.self_attn.k_proj.weight");
        if (kit != tensors.end() && kit->second.shape.size() >= 1 && hidden_dim > 0) {
            int n_blocks = kit->second.shape[0];
            int ntc = (hidden_dim + 255) / 256;
            int ntr = n_blocks / ntc;
            int k_out = ntr * 32;         // NKV * HD
            n_kv_heads = k_out / head_dim;
        }

        auto git = tensors.find("model.layers.0.mlp.gate_proj.weight");
        if (git != tensors.end() && git->second.shape.size() >= 1 && hidden_dim > 0) {
            int n_blocks = git->second.shape[0];
            int ntc = (hidden_dim + 255) / 256;
            int ntr = n_blocks / ntc;
            inter_size = ntr * 32;          // IM from tile rows
        }

        printf("[q4nx] Model: V=%d H=%d L=%d NH=%d NKV=%d HD=%d IM=%d\n",
               vocab_size, hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, inter_size);

        // Read and dequantize each tensor
        for (auto& [name, tensor] : tensors) {
            // Seek to data
            f.clear();
            f.seekg(8 + hdr_size + tensor.data_offset);
            if (!f) {
                fprintf(stderr, "  [q4nx] seek failed for %s\n", name.c_str());
                continue;
            }

            if (tensor.dtype == "BF16") {
                // Read BF16 → convert to FP32
                int n = (int)(tensor.data_size / 2);
                std::vector<uint16_t> bf16(n);
                f.read((char*)bf16.data(), tensor.data_size);
                tensor.fp32.resize(n);
                for (int i = 0; i < n; i++) {
                    tensor.fp32[i] = bf16_to_f32(bf16[i]);
                }
            } else if (tensor.dtype == "f32") {
                // Raw F32
                int n = (int)(tensor.data_size / 4);
                tensor.fp32.resize(n);
                f.read((char*)tensor.fp32.data(), tensor.data_size);
            } else if (tensor.dtype == "I8" || tensor.dtype == "I4" || tensor.dtype == "i8_tiled") {
                // Dequantize I8 tiled format → FP32
                // Each block row in the shape represents tiles in a specific arrangement.
                // The shape[0] is the number of 5120-byte blocks, not the tile-row count.
                // shape = [N_blocks, 5120]
                //
                // For a weight matrix of [OUT, IN]:
                //   Each block = 2 sub-tiles of 32×128 = 32×256 total output region
                //   N_blocks = ceil(OUT/32) * ceil(IN/256)
                //   But actually shape[0] already IS N_blocks
                //   
                // Example: q_proj shape=[256, 5120], OUT=NH*HD, IN=H
                //   actual_OUT = uint later from model dims
                
                int n_blocks = tensor.shape[0];
                int actual_IN = hidden_dim;
                int ntc = (actual_IN + 255) / 256;  // tile columns
                int ntr = n_blocks / ntc;            // tile rows
                int actual_OUT = ntr * 32;           // output dimension

                tensor.fp32.resize(actual_OUT * actual_IN, 0.0f);

                size_t byte_pos = 8 + hdr_size + tensor.data_offset;
                f.clear();
                f.seekg(byte_pos);

                for (int bi = 0; bi < n_blocks; bi++) {
                    std::vector<uint8_t> block(5120);
                    f.read((char*)block.data(), 5120);
                    if (!f) { fprintf(stderr, "  read error at block %d\n", bi); break; }

                    int tr = bi / ntc;   // tile row (32 output rows each)
                    int tc = bi % ntc;   // tile column (256 input cols each)

                    dequant_i8_block(block.data(), tensor.fp32.data(),
                                    tr, tc, actual_OUT, actual_IN);
                }
            }

            if (tensor.fp32.empty()) {
                printf("  [q4nx] %s: %s %s → %zu bytes, %zu floats\n",
                       name.c_str(), tensor.dtype.c_str(),
                       (tensor.shape.size() >= 2)
                           ? (std::to_string(tensor.shape[0]) + "x" + std::to_string(tensor.shape[1])).c_str()
                           : "?",
                       tensor.data_size, tensor.fp32.size());
            } else {
                printf("  [q4nx] %s: %s %s → dequant %zu floats\n",
                       name.c_str(), tensor.dtype.c_str(),
                       (tensor.shape.size() >= 2)
                           ? (std::to_string(tensor.shape[0]) + "x" + std::to_string(tensor.shape[1])).c_str()
                           : "?",
                       tensor.fp32.size());
            }
        }

        f.close();
        return true;
    }
};
