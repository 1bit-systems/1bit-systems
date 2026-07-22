#pragma once
/// Q4NX model header parser - reads the JSON metadata section of a Q4NX model file
/// and derives ModelConfig including architecture dimensions and xclbin parameters.
/// Ported from engine/npu/src/model_reader.zig
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ─── BF16 -> F32 conversion ──────────────────────────────────────
inline float bf16ToF32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline float bf16ToF32Safe(uint16_t v) {
    if ((v & 0x7F80) == 0x7F80) return 0.0f;
    return bf16ToF32(v);
}

// ─── ModelConfig ─────────────────────────────────────────────────

struct ModelConfig {
    uint32_t H = 0;
    uint32_t NC = 0;  // number of layers
    uint32_t NH = 0;
    uint32_t NKV = 0;
    uint32_t HD = 0;
    uint32_t IM = 0;
    uint32_t NV = 0;
    uint32_t GQA = 0;
    uint32_t AW = 4;
    uint32_t XM = 128;

    // QKV fused offsets
    uint32_t qkv_k_offset = 0;
    uint32_t qkv_v_offset = 0;
    uint32_t qkv_total = 0;

    // XCLBIN dimensions
    uint32_t xclbin_qkv_k = 0;
    uint32_t xclbin_qkv_n = 0;
    uint32_t xclbin_o_k = 0;
    uint32_t xclbin_o_n = 0;
    uint32_t xclbin_g_k = 0;
    uint32_t xclbin_g_n = 0;
    uint32_t xclbin_u_k = 0;
    uint32_t xclbin_u_n = 0;
    uint32_t xclbin_gu_k = 0;
    uint32_t xclbin_gu_n = 0;
    uint32_t xclbin_d_k = 0;
    uint32_t xclbin_d_n = 0;

    bool has_q_norm = false;
    bool has_k_norm = false;
    bool has_rope_freqs_file = false;
    bool has_lm_head = false;
    bool gu_split = false;

    float rope_theta = 1000000.0f;
    float rope_factor = 1.0f;

    std::string model_tag;
    std::string model_dir;

    bool valid() const {
        return H > 0 && NC > 0 && NH > 0 && NKV > 0 && HD > 0 && IM > 0 && NV > 0;
    }
};

// ─── JSON scanner helpers ────────────────────────────────────────

/// Find a numeric value associated with a JSON key.
/// Searches for `"key"` followed by `"data_offsets"` and returns the first integer.
inline uint32_t findTensorInfo(const char* js, size_t js_len, const char* key) {
    size_t key_len = strlen(key);
    size_t pos = 0;
    while (pos < js_len) {
        const char* found = (const char*)memmem(js + pos, js_len - pos, key, key_len);
        if (!found) return 0;

        size_t q = found - js;
        // Verify it's a JSON key
        if (q > 0 && js[q-1] == '"' && js[q + key_len] == '"') {
            size_t after_key = q + key_len;
            // Look for "data_offsets" after this key
            const char* offs_pos = (const char*)memmem(js + after_key, js_len - after_key,
                                                        "\"data_offsets\"", 14);
            if (offs_pos) {
                size_t offs_idx = offs_pos - js;
                // Find opening bracket
                const char* bracket = (const char*)memmem(js + offs_idx, js_len - offs_idx, "[", 1);
                if (bracket) {
                    // Parse first integer
                    const char* start = bracket + 1;
                    char* end = nullptr;
                    long val = strtol(start, &end, 10);
                    if (end > start) {
                        return (uint32_t)val;
                    }
                }
            }
        }
        pos = q + 1;
    }
    return 0;
}

/// Find tile_rows (shape[0]) for a given tensor key.
inline uint32_t findTileRows(const char* js, size_t js_len, const char* key) {
    size_t key_len = strlen(key);
    size_t pos = 0;
    while (pos < js_len) {
        const char* found = (const char*)memmem(js + pos, js_len - pos, key, key_len);
        if (!found) return 0;

        size_t q = found - js;
        if (q > 0 && js[q-1] == '"' && js[q + key_len] == '"') {
            size_t after_key = q + key_len;
            const char* shape_pos = (const char*)memmem(js + after_key, js_len - after_key,
                                                         "\"shape\"", 7);
            if (shape_pos) {
                size_t sp = shape_pos - js;
                const char* bracket = (const char*)memmem(js + sp, js_len - sp, "[", 1);
                if (bracket) {
                    const char* start = bracket + 1;
                    char* end = nullptr;
                    long val = strtol(start, &end, 10);
                    if (end > start) {
                        return (uint32_t)val;
                    }
                }
            }
        }
        pos = q + 1;
    }
    return 0;
}

/// Get shape[1] of a tensor.
inline uint32_t getShapeDim1(const char* js, size_t js_len, const char* key) {
    size_t key_len = strlen(key);
    size_t pos = 0;
    while (pos < js_len) {
        const char* found = (const char*)memmem(js + pos, js_len - pos, key, key_len);
        if (!found) return 0;

        size_t q = found - js;
        if (q > 0 && js[q-1] == '"' && js[q + key_len] == '"') {
            size_t after_key = q + key_len;
            const char* shape_pos = (const char*)memmem(js + after_key, js_len - after_key,
                                                         "\"shape\"", 7);
            if (shape_pos) {
                size_t sp = shape_pos - js;
                const char* bracket = (const char*)memmem(js + sp, js_len - sp, "[", 1);
                if (bracket) {
                    // Parse first dim
                    const char* p = bracket + 1;
                    char* end = nullptr;
                    strtol(p, &end, 10);
                    // Skip comma and whitespace
                    p = end;
                    while (*p == ',' || *p == ' ') p++;
                    long val = strtol(p, &end, 10);
                    if (end > p) return (uint32_t)val;
                }
            }
        }
        pos = q + 1;
    }
    return 0;
}

/// Check if a tensor key exists in the JSON header.
inline bool keyExists(const char* js, size_t js_len, const char* key) {
    return findTileRows(js, js_len, key) != 0;
}

/// Count the number of transformer layers by scanning for "model.layers.N".
inline uint32_t countLayers(const char* js, size_t js_len) {
    int max_layer = -1;
    const char* target = "model.layers.";
    size_t target_len = 13;
    size_t pos = 0;
    while (pos < js_len) {
        const char* found = (const char*)memmem(js + pos, js_len - pos, target, target_len);
        if (!found) break;
        size_t after = (found - js) + target_len;
        // Parse the layer number
        char* end = nullptr;
        long layer = strtol(js + after, &end, 10);
        if (end > (js + after)) {
            if (layer > max_layer) max_layer = (int)layer;
        }
        pos = after;
    }
    return (uint32_t)(std::max(max_layer, -1) + 1);
}

// ─── Main parser ─────────────────────────────────────────────────

/// Parse the Q4NX model file and return a fully derived ModelConfig.
inline ModelConfig parseQ4nxHeader(const std::string& model_path, const std::string& model_tag) {
    ModelConfig cfg;
    cfg.model_tag = model_tag;

    // Extract model_dir from path
    auto slash = model_path.find_last_of('/');
    if (slash != std::string::npos) {
        cfg.model_dir = model_path.substr(0, slash);
    }

    // Open and mmap the model file
    int fd = open(model_path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open model file: " + model_path);

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); throw std::runtime_error("Cannot stat model file"); }
    size_t file_size = st.st_size;

    void* mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) throw std::runtime_error("Cannot mmap model file");

    const uint8_t* data = (const uint8_t*)mapping;
    if (file_size < 8) { munmap(mapping, file_size); throw std::runtime_error("Invalid model file"); }

    // Read 8-byte header size (little-endian u64)
    uint64_t hdr_size;
    std::memcpy(&hdr_size, data, 8);
    if (hdr_size == 0 || 8 + hdr_size > file_size) {
        munmap(mapping, file_size);
        throw std::runtime_error("Invalid model header");
    }

    const char* js = (const char*)(data + 8);
    size_t js_len = (size_t)hdr_size;

    // ────────────────────────────────────────────────────────────
    // Step 1: Get NV and H from embed_tokens.weight
    // ────────────────────────────────────────────────────────────
    {
        const char* key = "model.embed_tokens.weight";
        size_t key_len = strlen(key);
        size_t pos = 0;
        while (pos < js_len) {
            const char* found = (const char*)memmem(js + pos, js_len - pos, key, key_len);
            if (!found) break;
            size_t q = found - js;
            if (q > 0 && js[q-1] == '"' && js[q + key_len] == '"') {
                size_t after = q + key_len;
                const char* sp = (const char*)memmem(js + after, js_len - after, "\"shape\"", 7);
                if (sp) {
                    size_t sp_idx = sp - js;
                    const char* br = (const char*)memmem(js + sp_idx, js_len - sp_idx, "[", 1);
                    if (br) {
                        const char* p = br + 1;
                        char* end = nullptr;
                        cfg.NV = (uint32_t)strtol(p, &end, 10);
                        p = end;
                        while (*p == ',' || *p == ' ') p++;
                        cfg.H = (uint32_t)strtol(p, &end, 10);
                    }
                }
                break;
            }
            pos = q + 1;
        }
    }

    if (cfg.H == 0 || cfg.NV == 0) {
        munmap(mapping, file_size);
        throw std::runtime_error("Could not determine H or NV from model file");
    }

    // ────────────────────────────────────────────────────────────
    // Step 2: Get I8 tile row counts for each weight
    // ────────────────────────────────────────────────────────────
    const char* prefix_q = "model.layers.0.self_attn.q_proj.weight";
    const char* prefix_k = "model.layers.0.self_attn.k_proj.weight";
    const char* prefix_v = "model.layers.0.self_attn.v_proj.weight";
    const char* prefix_o = "model.layers.0.self_attn.o_proj.weight";
    const char* prefix_g = "model.layers.0.mlp.gate_proj.weight";
    const char* prefix_d = "model.layers.0.mlp.down_proj.weight";

    uint32_t q_tr = findTileRows(js, js_len, prefix_q);
    uint32_t k_tr = findTileRows(js, js_len, prefix_k);
    uint32_t v_tr = findTileRows(js, js_len, prefix_v);
    uint32_t o_tr = findTileRows(js, js_len, prefix_o);
    uint32_t g_tr = findTileRows(js, js_len, prefix_g);
    uint32_t d_tr = findTileRows(js, js_len, prefix_d);

    if (q_tr == 0) { munmap(mapping, file_size); throw std::runtime_error("Missing Q projection"); }
    if (k_tr == 0) { munmap(mapping, file_size); throw std::runtime_error("Missing K projection"); }
    if (g_tr == 0) { munmap(mapping, file_size); throw std::runtime_error("Missing gate projection"); }
    if (d_tr == 0) { munmap(mapping, file_size); throw std::runtime_error("Missing down projection"); }

    // ────────────────────────────────────────────────────────────
    // Step 3: Detect architecture features
    // ────────────────────────────────────────────────────────────
    uint32_t qn_hd = findTileRows(js, js_len, "model.layers.0.self_attn.q_norm.weight");
    if (qn_hd > 0) {
        cfg.has_q_norm = true;
        cfg.HD = qn_hd * 32;
    }
    cfg.has_k_norm = keyExists(js, js_len, "model.layers.0.self_attn.k_norm.weight");
    cfg.has_rope_freqs_file = keyExists(js, js_len, "rope_freqs.weight");
    cfg.has_lm_head = keyExists(js, js_len, "lm_head.weight");

    // ────────────────────────────────────────────────────────────
    // Step 4: Count layers
    // ────────────────────────────────────────────────────────────
    cfg.NC = countLayers(js, js_len);

    // ────────────────────────────────────────────────────────────
    // Step 5: Derive remaining dimensions from I8 tile rows
    // ────────────────────────────────────────────────────────────
    uint32_t col_tiles_h = (cfg.H + 255) / 256;  // ceil(H/256)
    if (col_tiles_h > 0 && q_tr > 0) {
        uint32_t nh_hd = (q_tr / col_tiles_h) * 32;  // NH * HD
        if (cfg.HD == 0) {
            if (nh_hd % 128 == 0) {
                cfg.HD = 128;
                cfg.NH = nh_hd / 128;
            } else if (nh_hd % 256 == 0) {
                cfg.HD = 256;
                cfg.NH = nh_hd / 256;
            } else {
                cfg.HD = 128;
                cfg.NH = nh_hd / 128;
            }
        } else {
            cfg.NH = nh_hd / cfg.HD;
        }
    }

    // k_proj: in_features=H, out_features=NKV*HD
    if (col_tiles_h > 0 && k_tr > 0) {
        uint32_t nkv_hd = (k_tr / col_tiles_h) * 32;
        cfg.NKV = nkv_hd / cfg.HD;
    }

    // gate_proj: in_features=H, out_features=IM
    if (col_tiles_h > 0 && g_tr > 0) {
        cfg.IM = (g_tr / col_tiles_h) * 32;
    }

    // ────────────────────────────────────────────────────────────
    // Step 6: Compute derived values
    // ────────────────────────────────────────────────────────────
    if (cfg.NH > 0 && cfg.NKV > 0) cfg.GQA = cfg.NH / cfg.NKV;

    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;

    cfg.xclbin_qkv_k = cfg.H;
    cfg.xclbin_qkv_n = cfg.qkv_total;
    cfg.xclbin_o_k = cfg.NH * cfg.HD;
    cfg.xclbin_o_n = cfg.H;

    // GU split decision
    cfg.gu_split = (cfg.IM * 2 > 14336);
    if (cfg.gu_split) {
        cfg.xclbin_g_k = cfg.H;
        cfg.xclbin_g_n = cfg.IM;
        cfg.xclbin_u_k = cfg.H;
        cfg.xclbin_u_n = cfg.IM;
    } else {
        cfg.xclbin_gu_k = cfg.H;
        cfg.xclbin_gu_n = cfg.IM * 2;
    }
    cfg.xclbin_d_k = cfg.IM;
    cfg.xclbin_d_n = cfg.H;

    munmap(mapping, file_size);
    return cfg;
}
