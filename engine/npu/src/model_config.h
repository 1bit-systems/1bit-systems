#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct ModelConfig {
    int H = 0, NC = 0, NH = 0, NKV = 0, HD = 0, IM = 0, NV = 0;
    int GQA = 0, AW = 4, WQH = 0, WKVH = 0, XM = 128;
    int qkv_k_offset = 0, qkv_v_offset = 0, qkv_total = 0;
    int xclbin_qkv_k = 0, xclbin_qkv_n = 0;
    int xclbin_o_k = 0, xclbin_o_n = 0;
    int xclbin_g_k = 0, xclbin_g_n = 0;
    int xclbin_u_k = 0, xclbin_u_n = 0;
    int xclbin_gu_k = 0, xclbin_gu_n = 0;
    int xclbin_d_k = 0, xclbin_d_n = 0;
    bool has_q_norm = false, has_k_norm = false;
    bool has_rope_freqs_file = false, has_lm_head = false;
    bool gu_split = false;
    float rope_theta = 1000000.0f;
    float rope_factor = 1.0f;
    std::string model_tag;
    std::string model_dir;

    bool valid() const { return H > 0 && NC > 0 && NH > 0 && NKV > 0 && HD > 0 && IM > 0 && NV > 0; }
};

// Find a JSON key and extract shape[0] + data_offsets[0]
static int find_tensor_info(const char* js, size_t jl, const char* key, int* out_tile_rows) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    *out_tile_rows = (int)strtoul(bracket + 1, nullptr, 10);
                }
            }
            auto offs_loc = strstr(q, "\"data_offsets\"");
            if (offs_loc) {
                auto bracket = strchr(offs_loc, '[');
                if (bracket) return (int)strtoul(bracket + 1, nullptr, 10);
            }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}

// Count layers by scanning for model.layers.N.self_attn.q_proj.weight
static int count_layers(const char* js, size_t jl) {
    int max_layer = -1;
    const char* p = js;
    const char* e = js + jl;
    const char target[] = "model.layers.";
    size_t tlen = strlen(target);
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, target, tlen);
        if (!q) break;
        int layer_num = (int)strtoul(q + tlen, nullptr, 10);
        if (layer_num > max_layer) max_layer = layer_num;
        p = q + tlen;
    }
    return max_layer + 1;
}

// Check if a JSON key pattern exists (for detecting q_norm, lm_head, etc.)
static bool key_exists(const char* js, size_t jl, const char* key) {
    int dummy = 0;
    return find_tensor_info(js, jl, key, &dummy) > 0;
}

static int get_shape_dim1(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    // Parse [dim0, dim1]
                    int dim0 = (int)strtoul(bracket + 1, nullptr, 10);
                    auto comma = strchr(bracket + 1, ',');
                    if (comma) {
                        return (int)strtoul(comma + 1, nullptr, 10);
                    }
                    return dim0;
                }
            }
            return 0;
        }
        p = q + kl;
    }
    return 0;
}

// Parse Q4NX JSON header and derive ModelConfig
inline ModelConfig parse_q4nx_header(const char* model_path, const char* model_tag) {
    ModelConfig cfg;
    cfg.model_tag = model_tag ? model_tag : "unknown";
    
    // Extract model_dir from path
    cfg.model_dir = model_path;
    auto slash = cfg.model_dir.rfind('/');
    if (slash != std::string::npos) cfg.model_dir = cfg.model_dir.substr(0, slash);
    
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "[ModelConfig] Cannot open %s\n", model_path); return cfg; }
    struct stat st;
    fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { fprintf(stderr, "[ModelConfig] mmap failed\n"); return cfg; }
    
    uint64_t hdr_size;
    memcpy(&hdr_size, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hdr_size;
    
    // Step 1: Get H and NV from embed_tokens
    // embed_tokens shape = [NV, H] in the JSON (logical dims, not tiles)
    int emb_nv = get_shape_dim1(js, jl, "model.embed_tokens.weight");
    if (emb_nv > 0) {
        // shape has 2 elements: [NV, H]
        // get_shape_dim1 returns the second element (H)
        // But we also need NV from shape[0]. Let's parse more carefully.
        // Actually 'get_shape_dim1' returns the second dim, so H=embed_tokens.shape[1]
        // NV = embed_tokens.shape[0]
        size_t kl = strlen("model.embed_tokens.weight");
        const char* p = js;
        const char* e = js + jl;
        while (p < e) {
            auto q = (const char*)memmem(p, e - p, "model.embed_tokens.weight", kl);
            if (!q) break;
            if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
                auto shape_loc = strstr(q, "\"shape\"");
                if (shape_loc) {
                    auto bracket = strchr(shape_loc, '[');
                    if (bracket) {
                        cfg.NV = (int)strtoul(bracket + 1, nullptr, 10);
                        auto comma = strchr(bracket + 1, ',');
                        if (comma) {
                            while (*comma == ',' || *comma == ' ') comma++;
                            cfg.H = (int)strtoul(comma, nullptr, 10);
                        }
                    }
                }
                break;
            }
            p = q + kl;
        }
    }
    
    // Step 2: Read I8 tile row counts for each weight
    int q_tr = 0, k_tr = 0, o_tr = 0, g_tr = 0, d_tr = 0;
    int q_off = 0;
    q_off = find_tensor_info(js, jl, "model.layers.0.self_attn.q_proj.weight", &q_tr);
    find_tensor_info(js, jl, "model.layers.0.self_attn.k_proj.weight", &k_tr);
    find_tensor_info(js, jl, "model.layers.0.self_attn.o_proj.weight", &o_tr);
    find_tensor_info(js, jl, "model.layers.0.mlp.gate_proj.weight", &g_tr);
    find_tensor_info(js, jl, "model.layers.0.mlp.down_proj.weight", &d_tr);
    
    // Step 3: Detect architecture features
    int qn_hd = 0;
    cfg.has_q_norm = (find_tensor_info(js, jl, "model.layers.0.self_attn.q_norm.weight", &qn_hd) > 0);
    if (cfg.has_q_norm && qn_hd > 0) cfg.HD = qn_hd;  // q_norm shape = [HD]
    
    cfg.has_k_norm = key_exists(js, jl, "model.layers.0.self_attn.k_norm.weight");
    cfg.has_rope_freqs_file = key_exists(js, jl, "rope_freqs.weight");
    cfg.has_lm_head = key_exists(js, jl, "lm_head.weight");
    
    // Step 4: Count layers
    cfg.NC = count_layers(js, jl);
    
    // Step 5: Derive remaining dimensions from I8 tile rows
    // n_tile_cols = ceil(H / 256) for weight with in_features=H
    // e.g., q_proj: in_features=H, out_features=NH*HD
    // tile_rows_q = ceil(NH*HD/32) * ceil(H/256)
    // tile_rows_o = ceil(H/32) * ceil(NH*HD/256)
    
    if (cfg.H > 0 && q_tr > 0) {
        int A = (cfg.H + 255) / 256;  // ceil(H/256) = n_tile_cols for q_proj input
        if (A > 0) {
            int tile_rows_q = q_tr / A;  // ceil(NH*HD/32)
            if (tile_rows_q > 0) {
                int nh_hd = tile_rows_q * 32;  // NH * HD
                
                // Determine NH and HD
                if (cfg.HD > 0) {
                    cfg.NH = nh_hd / cfg.HD;
                } else {
                    // Assume HD=128 (most common)
                    cfg.HD = 128;
                    cfg.NH = nh_hd / cfg.HD;
                    // Check if it divides evenly; if not try HD=256
                    if (cfg.NH * cfg.HD != nh_hd) {
                        cfg.HD = 256;
                        cfg.NH = nh_hd / cfg.HD;
                        if (cfg.NH * cfg.HD != nh_hd) {
                            // Fallback: try to detect from k_proj
                            cfg.HD = 128;
                            cfg.NH = nh_hd / 128;
                        }
                    }
                }
            }
        }
    }
    
    // NKV from k_proj
    if (cfg.H > 0 && k_tr > 0) {
        int A = (cfg.H + 255) / 256;
        if (A > 0) {
            int tile_rows_k = k_tr / A;
            if (tile_rows_k > 0) {
                int nkv_hd = tile_rows_k * 32;
                cfg.NKV = cfg.HD > 0 ? nkv_hd / cfg.HD : nkv_hd / 128;
            }
        }
    }
    
    // IM from gate_proj
    if (cfg.H > 0 && g_tr > 0) {
        int A = (cfg.H + 255) / 256;
        if (A > 0) {
            int tile_rows_g = g_tr / A;
            if (tile_rows_g > 0) {
                cfg.IM = tile_rows_g * 32;
            }
        }
    }
    // Also verify IM from down_proj
    // down_proj in_features=IM, out_features=H
    // tile_rows_d = ceil(H/32) * ceil(IM/256)
    
    // Step 6: Compute derived values
    if (cfg.NH > 0 && cfg.NKV > 0) cfg.GQA = cfg.NH / cfg.NKV;
    cfg.WQH = cfg.NH / cfg.AW;
    cfg.WKVH = cfg.NKV / cfg.AW;
    
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
    
    munmap(md, st.st_size);
    return cfg;
}
