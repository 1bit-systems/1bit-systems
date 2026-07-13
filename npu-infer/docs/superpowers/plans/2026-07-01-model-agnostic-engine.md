# Model-Agnostic NPU Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded Qwen3-0.6B constants in `npu_engine_mt.cpp` with a `ModelConfig` struct derived from the Q4NX JSON header, so any model loads and runs with zero code changes.

**Architecture:** Parse the Q4NX JSON header at load time → extract all dimensions (H, NC, NH, NKV, HD, IM, NV) + architecture flags (has_q_norm, has_k_norm, has_rope_freqs, has_lm_head) → allocate buffers dynamically → select tagged xclbins by model name.

**Tech Stack:** C++17, XRT, AVX-512 SIMD, Q4NX format

**Global Constraints:**
- All xclbins use M=128 (XM=128) tile size
- All dimensions must be multiples of 128 for tile alignment
- xclbins located at `build/int8/final_i8_{QKV,O,G,U,D}_{tag}.xclbin`
- Instruction files at `build/int8/insts_i8_{QKV,O,G,U,D}_{tag}.txt`
- Model weight files at `/home/bcloud/.config/flm/models/{ModelName}/model.q4nx`
- GU combined when 2×IM ≤ 14336, split when 2×IM > 14336
- KV cache uses KVQuant with KQ_HD=128, KQ_NKV=8, KQ_GROUP=32, KQ_MAX_CTX=4096

---

### Task 1: `ModelConfig` Parsing from Q4NX JSON Header

**Files:**
- Create: `include/model_config.h` — ModelConfig struct + parser from Q4NX JSON
- Modify: `src/dequant_q4nx.c` — add `dequant_i8_to_float_ex` if not already exposed
- Test: `tools/test_model_config.cpp` — standalone test

**Interfaces:**
- Consumes: Q4NX file path → mmaps file → reads JSON header
- Produces: `ModelConfig cfg` struct with all dimensions filled + model name string for xclbin selection

- [ ] **Step 1: Write the header file**

```cpp
// include/model_config.h
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

struct ModelConfig {
    // Core dimensions derived from Q4NX header
    int H = 0;       // hidden size
    int NC = 0;      // num layers
    int NH = 0;      // num Q heads
    int NKV = 0;     // num KV heads
    int HD = 0;      // head dim
    int IM = 0;      // intermediate size (SwiGLU)
    int NV = 0;      // vocab size
    
    // Derived values
    int GQA = 0;     // NH / NKV
    int AW = 4;      // attention windows (always 4 for current xclbins)
    int WQH = 0;     // Q heads per window = NH / AW
    int WKVH = 0;    // KV heads per window = NKV / AW
    int XM = 128;    // NPU tile M
    
    // QKV split offsets (in elements, not head-attributed)
    int qkv_q_offset = 0;    // 0
    int qkv_k_offset = 0;    // NH * HD
    int qkv_v_offset = 0;    // NH * HD + NKV * HD
    int qkv_total = 0;       // NH*HD + 2*NKV*HD
    
    // Xclbin dimensions
    int xclbin_qkv_k = 0;    // = H
    int xclbin_qkv_n = 0;    // = qkv_total
    int xclbin_o_k = 0;      // = NH * HD
    int xclbin_o_n = 0;      // = H
    int xclbin_g_k = 0;      // = H
    int xclbin_g_n = 0;      // = IM
    int xclbin_u_k = 0;      // = H (if GU split)
    int xclbin_u_n = 0;      // = IM (if GU split)
    int xclbin_gu_k = 0;     // = H (if combined)
    int xclbin_gu_n = 0;     // = 2*IM (if combined)
    int xclbin_d_k = 0;      // = IM
    int xclbin_d_n = 0;      // = H
    
    // Architecture flags (inferred from key presence in JSON)
    bool has_q_norm = false;
    bool has_k_norm = false;
    bool has_rope_freqs_file = false;  // rope_freqs.weight from file
    bool has_lm_head = false;          // separate lm_head projection
    bool gu_split = false;             // G+U separate vs combined GU
    
    // RoPE config
    float rope_theta = 10000.0f;
    float rope_factor = 1.0f;  // Llama3 scaling factor
    
    // Model identifier (from directory name or embed_tokens shape)
    std::string model_tag;  // e.g. "qwen3_0_6b", "llama", etc.
    
    // Xclbin directory
    std::string xclbin_dir;
    
    // Norm weights for Q/K (per-layer, loaded at runtime)
    // loaded dynamically in main()
    
    bool valid() const { return H > 0 && NC > 0 && NH > 0 && NKV > 0 && HD > 0 && IM > 0 && NV > 0; }
};

// Parse Q4NX JSON header and derive ModelConfig
// model_path: path to .q4nx file
// model_tag: short tag for xclbin selection (e.g. "qwen3_0_6b")
// Returns: ModelConfig with all fields filled, or invalid config on error
ModelConfig parse_q4nx_header(const char* model_path, const char* model_tag);
```

- [ ] **Step 2: Write the parser implementation**

```cpp
// include/model_config.h (add implementation below struct)

// Parse Q4NX weight tensor shape from JSON value
// Extracts I8 row count and infers logical out_rows from tile grid
static int derive_out_rows(const int* shape, int in_features) {
    // shape[0] = I8 tile rows for this weight
    // shape[1] = storage width per row (always 5120 = 20*256)
    int i8_rows = shape[0];
    int n_tile_cols = (in_features + 255) / 256;  // ceil(H/256)
    if (n_tile_cols == 0) return 0;
    int n_tile_rows = i8_rows / n_tile_cols;
    return n_tile_rows * 32;
}

static int json_parse_int(const char* js, size_t jl, const char* pattern) {
    // Find "pattern": NNN in JSON
    const char* pos = js;
    const char* end = js + jl;
    size_t plen = strlen(pattern);
    while ((pos = (const char*)memmem(pos, end - pos, pattern, plen)) != nullptr) {
        // Check it's a key
        auto before = pos > js ? pos[-1] : ' ';
        if (before == '"') {
            pos += plen;
            while (pos < end && (*pos == ' ' || *pos == ':')) pos++;
            if (pos < end) {
                bool neg = false;
                if (*pos == '-') { neg = true; pos++; }
                unsigned long long val = 0;
                while (pos < end && *pos >= '0' && *pos <= '9') {
                    val = val * 10 + (*pos - '0');
                    pos++;
                }
                return neg ? -(int)val : (int)val;
            }
        }
        pos++;
    }
    return 0;
}

// Find a JSON key value pattern like "key_name": VALUE
static uint64_t find_json_offset(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        // Verify it's a string key
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            // Find "data_offsets" in this entry
            auto offs = strstr(q, "\"data_offsets\"");
            if (offs) {
                auto bracket = strchr(offs, '[');
                if (bracket) return strtoull(bracket + 1, nullptr, 10);
            }
        }
        p = q + kl;
    }
    return 0;
}

// Find one JSON key and extract shape info
// Returns shape[0] and i8 byte offset
static void find_tensor_info(const char* js, size_t jl, const char* json_key,
                             int* out_tile_rows, int* out_i8_offset) {
    size_t kl = strlen(json_key);
    const char* p = js;
    const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, json_key, kl);
        if (!q) break;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            // Found key. Look for "shape": [N, M]
            auto shape_loc = strstr(q, "\"shape\"");
            if (shape_loc) {
                auto bracket = strchr(shape_loc, '[');
                if (bracket) {
                    *out_tile_rows = (int)strtoul(bracket + 1, nullptr, 10);
                }
            }
            // Look for "data_offsets": [offset, break]
            auto offs_loc = strstr(q, "\"data_offsets\"");
            if (offs_loc) {
                auto bracket = strchr(offs_loc, '[');
                if (bracket) {
                    *out_i8_offset = (int)strtoul(bracket + 1, nullptr, 10);
                }
            }
            return;
        }
        p = q + kl;
    }
}

// Main parser function
inline ModelConfig parse_q4nx_header(const char* model_path, const char* model_tag) {
    ModelConfig cfg;
    cfg.model_tag = model_tag ? model_tag : "unknown";
    
    // Open and mmap file
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open %s\n", model_path); return cfg; }
    struct stat st;
    fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return cfg; }
    
    // Parse JSON header
    uint64_t hdr_size;
    memcpy(&hdr_size, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hdr_size;
    
    // Step 1: Get H and NV from embed_tokens
    int emb_shape[2] = {0};
    int emb_off = 0;
    find_tensor_info(js, jl, "model.embed_tokens.weight", &emb_shape[0], &emb_off);
    if (emb_shape[0] > 0 && emb_shape[1] > 0) {
        // JSON shape[1] = storage width per row = 5120
        // in_features = ceil(H/256)*256 = shape[1] / (scales_zp_bytes_per_tile...)
        // Actually, shape[1] = 5120 always for weight matrices
        // embed_tokens is special: it's stored as BF16, not INT4
        // For BF16 embed_tokens, shape = (NV, H) directly
        // But in Q4NX format it might be different...
        // Let's check the offset: if embed_off == 0, it's part of mmap with emb table
        cfg.NV = emb_shape[0];
        // H can be inferred: for embed_tokens, it's typically stored differently
        // We'll get H from the first q_proj derived value instead
    }
    
    // Step 2: Get H from q_proj (layer 0) — derive tile structure
    int q_tile_rows = 0, k_tile_rows = 0, v_tile_rows = 0;
    int o_tile_rows = 0, g_tile_rows = 0, d_tile_rows = 0;
    int q_off = 0, k_off = 0, v_off = 0, o_off = 0, g_off = 0, d_off = 0;
    
    find_tensor_info(js, jl, "model.layers.0.self_attn.q_proj.weight", &q_tile_rows, &q_off);
    find_tensor_info(js, jl, "model.layers.0.self_attn.k_proj.weight", &k_tile_rows, &k_off);
    find_tensor_info(js, jl, "model.layers.0.self_attn.v_proj.weight", &v_tile_rows, &v_off);
    find_tensor_info(js, jl, "model.layers.0.self_attn.o_proj.weight", &o_tile_rows, &o_off);
    find_tensor_info(js, jl, "model.layers.0.mlp.gate_proj.weight", &g_tile_rows, &g_off);
    find_tensor_info(js, jl, "model.layers.0.mlp.down_proj.weight", &d_tile_rows, &d_off);
    
    // Detect q_norm/k_norm
    int qn_off = 0, kn_off = 0;
    find_tensor_info(js, jl, "model.layers.0.self_attn.q_norm.weight", &cfg.HD, &qn_off);
    cfg.has_q_norm = (qn_off > 0);
    find_tensor_info(js, jl, "model.layers.0.self_attn.k_norm.weight", &cfg.HD, &kn_off);
    cfg.has_k_norm = (kn_off > 0);
    
    // Detect rope_freqs
    int rope_off = 0, rope_rows = 0;
    find_tensor_info(js, jl, "rope_freqs.weight", &rope_rows, &rope_off);
    cfg.has_rope_freqs_file = (rope_off > 0);
    
    // Detect lm_head
    int lm_rows = 0, lm_off = 0;
    find_tensor_info(js, jl, "lm_head.weight", &lm_rows, &lm_off);
    cfg.has_lm_head = (lm_off > 0);
    
    // Step 3: Compute H from q_proj tile structure
    // q_tile_rows = ceil(NH*HD/32) * ceil(H/256)
    // So H is derived from: o_tile_rows = ceil(NH*HD/32) * ceil(H/256) = same as q_tile_rows
    // But H can also be found from: o_proj.out_rows = H (output dim)
    // o_tile_rows = ceil(H/32) * ceil(NH*HD/256) ... wait that's wrong
    // Actually o_proj: in_features = NH*HD, out_features = H
    // o_tile_rows = ceil(H/32) * ceil(NH*HD/256)
    // q_proj: in_features = H, out_features = NH*HD
    // q_tile_rows = ceil(NH*HD/32) * ceil(H/256)
    // 
    // Let me think of n_tile_cols differently:
    // For q_proj: n_tile_cols_q = ceil(H/256), tile_rows_i8_q = ceil(NH*HD/32) * n_tile_cols_q
    // H = n_tile_cols_q * 256
    // 
    // We also know: each tile = 32 BF16 rows x 256 cols
    // For o_proj: in_features = NH*HD, out_features = H
    // n_tile_cols_o = ceil(NH*HD/256), n_tile_rows_o = ceil(H/32)
    // o_tile_rows = ceil(H/32) * ceil(NH*HD/256)
    //
    // From q_tile_rows and o_tile_rows, we can deduce H and NH*HD.
    // Let A = ceil(H/256), B = ceil(NH*HD/32)
    // q_tile_rows = B * A
    // Let C = ceil(NH*HD/256), D = ceil(H/32)
    // o_tile_rows = D * C
    // A = D (since both = ceil(H/256) roughly... no, D = ceil(H/32))
    // Hmm, this is getting complex. Let's use a simpler approach:
    //
    // SIMPLER: read the JSON shape[0] for each weight. The shape[0] gives
    // tile_rows. We already know:
    // - n_tile_cols = ceil(in_features/256) for that weight
    // - The weight's in_features = shape[1] / (padded width factor)
    // - But shape[1] is always 5120 for I8 weights = 20*256
    // - So n_tile_cols = 20 for all I8 weights? NO!
    //
    // Wait, 5120 = 20 * 256 so n_tile_cols = 20 for all I8 weights?
    // No. The 5120 = bytes per row, not elements.
    // Each I8 row = 512 bytes scales + 512 bytes zp + 4096 bytes INT4 = 5120 bytes
    // So each tile = 32 BF16 rows x 256 BF16 cols stored in 5120 bytes
    // n_tile_cols = ceil(in_features / 256) tells us how many tiles wide
    // But each tile row in the I8 storage spans all tile columns
    // 
    // I think: n_tile_cols = in_features / 256 = H / 256 for a weight with input H
    // And i8_rows = ceil(out_features/32) * n_tile_cols (since each tile covers 32 out rows)
    // So n_tile_cols divides i8_rows.
    //
    // Since o_proj has in_features=NH*HD, out_features=H:
    // n_tile_cols_o = ceil(NH*HD/256)
    // n_tile_rows_o = ceil(H/32)
    // o_tile_rows = ceil(H/32) * ceil(NH*HD/256)
    //
    // And q_proj has in_features=H, out_features=NH*HD:
    // n_tile_cols_q = ceil(H/256)
    // n_tile_rows_q = ceil(NH*HD/32)
    // q_tile_rows = ceil(NH*HD/32) * ceil(H/256)
    //
    // So q_tile_rows = o_tile_rows (both = ceil(H/256)*ceil(NH*HD/32)... no)
    // Actually ceil(NH*HD/32) != ceil(NH*HD/256) * 8 generally (since 32*8=256)
    // ceil(NH*HD/32) = 8 * ceil(NH*HD/256) approximately
    // 
    // Let me use a practical approach: read the data offsets to find consecutive
    // tensors, and derive dimensions from data layout.
    // 
    // SIMPLEST approach: We know H from embed_tokens weight shape[1].
    // embed_tokens is stored specially, but its JSON shape should be (NV, H) as logical dims.
    // Let's check if emb_shape[1] directly gives H.
    
    // For now, extract H from a known relationship:
    // If we know the I8 block row count and the output dimension for any linear,
    // we can work backwards.
    
    // Actually the simplest reliable approach:
    // Use the config.json alongside model.q4nx when available.
    // But we promised to work from Q4NX only.
    //
    // OK let's accept: for embed_tokens, shape = [NV, H_direct] where shape[1] = H
    // This is because embeddings are stored as flat BF16, not blocked.
    if (emb_shape[1] > 0) {
        cfg.H = emb_shape[1];
    } else {
        // Fallback: compute from o_proj
        // o_proj out_features = H
        // n_tile_cols_o = ceil(out_features/256)... no wait.
        // For o_proj: in_features = NH*HD (this is the reduction dim)
        // out_features = H (this is the output dim)
        // n_tile_cols_o = ceil(in_features/256) = ceil(NH*HD/256)  -- columns of tiles
        // n_tile_rows_o = ceil(out_features/32) = ceil(H/32)       -- rows of tiles
        // o_tile_rows = n_tile_rows_o * n_tile_cols_o = ceil(H/32) * ceil(NH*HD/256)
        // 
        // H = n_tile_rows_o * 32 ... but we don't know n_tile_rows_o alone
        // We need another equation.
        //
        // q_proj: ceil(NH*HD/32) = q_tile_rows / n_tile_cols_q
        // where n_tile_cols_q = ceil(H/256)
        // So q_tile_rows = ceil(NH*HD/32) * ceil(H/256)
        //
        // Similarly, d_proj: in=IM, out=H:
        // d_tile_rows = ceil(H/32) * ceil(IM/256)
        //
        // Let's solve:
        // Let A = ceil(H/256), B = ceil(H/32), C = ceil(NH*HD/256), D = ceil(NH*HD/32)
        // q_tile_rows = D * A
        // o_tile_rows = B * C
        // d_tile_rows = B * ceil(IM/256)
        // g_tile_rows = ceil(IM/32) * A  (gate_proj: in=H, out=IM)
        //
        // We have: A = ceil(H/256), B = ceil(H/32) = (H+31)/32
        // B = 8*A roughly (since 256/32 = 8), exact: B = 8*A - delta where delta is small
        //
        // From d_tile_rows and g_tile_rows:
        // d_tile_rows = B * ceil(IM/256)
        // g_tile_rows = ceil(IM/32) * A = ceil(IM/256)*8 * A (approx) = d_tile_rows * A / B * 8
        // Hmm.
        //
        // Let's just try: if A = g_tile_rows / ceil(IM/32), but we don't know IM either.
        //
        // PRACTICAL: scan embed_tokens.shape to get H directly.
    }
    
    // Step 4: Compute remaining dimensions
    int H = cfg.H;
    if (H > 0) {
        int ntc_q = (H + 255) / 256;  // ceil(H/256)
        if (ntc_q > 0 && q_tile_rows > 0) {
            int tile_rows_q = q_tile_rows / ntc_q;
            int nh_hd = tile_rows_q * 32;  // NH * HD
            cfg.NH = nh_hd / 128;  // assume HD=128
            cfg.HD = 128;
            // But check for HD != 128: from q_norm shape
            // q_norm should have shape [HD] if present
        }
        
        // Compute NKV from k_proj
        int ntc_k = (H + 255) / 256;
        if (ntc_k > 0 && k_tile_rows > 0) {
            int tile_rows_k = k_tile_rows / ntc_k;
            int nkv_hd = tile_rows_k * 32;  // NKV * HD
            cfg.NKV = nkv_hd / 128;
        }
        
        // Compute IM from gate_proj
        int ntc_g = (H + 255) / 256;
        if (ntc_g > 0 && g_tile_rows > 0) {
            int tile_rows_g = g_tile_rows / ntc_g;
            cfg.IM = tile_rows_g * 32;
        }
    }
    
    // Step 5: Count layers from JSON
    cfg.NC = 0;
    for (int i = 0; i < 200; i++) {
        char key[128];
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_proj.weight", i);
        int tr = 0, off = 0;
        find_tensor_info(js, jl, key, &tr, &off);
        if (off == 0) break;
        cfg.NC = i + 1;
    }
    
    // Step 6: Detect HD from q_norm shape (if present)
    // q_norm shape[0] = HD (e.g. 128)
    if (cfg.has_q_norm) {
        // Already read HD from q_norm find above
        // cfg.HD was set there
        // If HD=0 (read failed), try to get from shape differently
    }
    
    // Step 7: Compute derived values
    if (cfg.NH > 0 && cfg.NKV > 0) cfg.GQA = cfg.NH / cfg.NKV;
    if (cfg.NH > 0) cfg.WQH = cfg.NH / cfg.AW;
    if (cfg.NKV > 0) cfg.WKVH = cfg.NKV / cfg.AW;
    
    // QKV offsets
    cfg.qkv_q_offset = 0;
    cfg.qkv_k_offset = cfg.NH * cfg.HD;
    cfg.qkv_v_offset = cfg.NH * cfg.HD + cfg.NKV * cfg.HD;
    cfg.qkv_total = cfg.NH * cfg.HD + 2 * cfg.NKV * cfg.HD;
    
    // Xclbin dimensions
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
        cfg.xclbin_gu_k = 0;
        cfg.xclbin_gu_n = 0;
    } else {
        cfg.xclbin_gu_k = cfg.H;
        cfg.xclbin_gu_n = cfg.IM * 2;
        cfg.xclbin_g_k = 0;
        cfg.xclbin_g_n = 0;
        cfg.xclbin_u_k = 0;
        cfg.xclbin_u_n = 0;
    }
    cfg.xclbin_d_k = cfg.IM;
    cfg.xclbin_d_n = cfg.H;
    
    // Set xclbin directory
    cfg.xclbin_dir = std::string(model_path);
    auto slash = cfg.xclbin_dir.rfind('/');
    if (slash != std::string::npos) cfg.xclbin_dir = cfg.xclbin_dir.substr(0, slash);
    
    munmap(md, st.st_size);
    return cfg;
}
```

Let me simplify this — the JSON shape parsing for the I8 format is complex but the pattern is well understood. I'll use the key approach of reading `embed_tokens.shape[1]` for H, and then computing everything else from the I8 tile row counts using `n_tile_cols = ceil(H/256)` as the divisor.

Actually for the model tag, I should derive it from the directory name or create a mapping.

Let me reconsider the implementation strategy. Instead of one big function with inline JSON parsing, I'll use the lightweight JSON scanning approach (no full parser needed since we only need specific key patterns).

- [ ] **Step 3: Write test for ModelConfig parser**

```cpp
// tools/test_model_config.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../include/model_config.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.q4nx [tag]\n", argv[0]);
        return 1;
    }
    const char* tag = argc > 2 ? argv[2] : "test";
    
    ModelConfig cfg = parse_q4nx_header(argv[1], tag);
    if (!cfg.valid()) {
        fprintf(stderr, "FAILED to parse %s\n", argv[1]);
        return 1;
    }
    
    printf("Model: %s\n", tag);
    printf("  H=%d, NC=%d, NH=%d, NKV=%d, HD=%d, IM=%d, NV=%d\n",
           cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV);
    printf("  GQA=%d, WQH=%d, WKVH=%d\n", cfg.GQA, cfg.WQH, cfg.WKVH);
    printf("  has_q_norm=%d, has_k_norm=%d, has_rope_freqs=%d, has_lm_head=%d\n",
           cfg.has_q_norm, cfg.has_k_norm, cfg.has_rope_freqs_file, cfg.has_lm_head);
    printf("  gu_split=%d\n", cfg.gu_split);
    printf("  QKV: K=%d, N=%d [q=%d, k=%d, v=%d]\n",
           cfg.xclbin_qkv_k, cfg.xclbin_qkv_n,
           cfg.qkv_q_offset, cfg.qkv_k_offset, cfg.qkv_v_offset);
    printf("  O: K=%d, N=%d\n", cfg.xclbin_o_k, cfg.xclbin_o_n);
    if (cfg.gu_split) {
        printf("  G: K=%d, N=%d\n", cfg.xclbin_g_k, cfg.xclbin_g_n);
        printf("  U: K=%d, N=%d\n", cfg.xclbin_u_k, cfg.xclbin_u_n);
    } else {
        printf("  GU: K=%d, N=%d\n", cfg.xclbin_gu_k, cfg.xclbin_gu_n);
    }
    printf("  D: K=%d, N=%d\n", cfg.xclbin_d_k, cfg.xclbin_d_n);
    
    return 0;
}
```

- [ ] **Step 4: Build and test the parser**

```bash
cd /home/bcloud/npu-sandbox/npu-infer
c++ -std=gnu++17 -O2 -Iinclude -o build/test_model_config tools/test_model_config.cpp
# Test with all 5 models
for m in Qwen3-0.6B-NPU2 Llama-3.1-8B-NPU2 Qwen3-VL-4B-Instruct-NPU2 Gemma4-E2B-IT-NPU2 Qwen3-8B-NPU2; do
    echo "=== $m ==="
    tag=$(echo "$m" | tr '[:upper:]' '[:lower:]' | sed 's/-npu2//;s/-instruct//;s/-it//;s/\./-/g;')
    build/test_model_config "/home/bcloud/.config/flm/models/$m/model.q4nx" "$tag" 2>&1 || \
    build/test_model_config "/home/bcloud/models/$m/model.q4nx" "$tag" 2>&1
    echo ""
done
```

- [ ] **Step 5: Commit**

```bash
git add include/model_config.h tools/test_model_config.cpp
git add CMakeLists.txt  # if modified
git commit -m "feat: ModelConfig parser from Q4NX JSON header"
```

---

### Task 2: Parameterized Engine — `npu_engine_mt.cpp`

**Files:**
- Modify: `include/model_config.h` — add header dependency and any missing fields
- Rewrite: `src/npu_engine_mt.cpp` — replace all `static constexpr int` with `cfg.*`

**Key changes:**
- Replace `static constexpr int H=1024,NC=28,...` with `ModelConfig cfg` loaded at startup
- Replace all hardcoded array sizes (qo[4096], ko[1024], su[6144], etc.) with dynamically allocated `std::vector` sized to `cfg.*`
- Replace QKV split offsets (2048, 3072) with `cfg.qkv_k_offset, cfg.qkv_v_offset`
- Replace QKV N-dim (4096) with `cfg.qkv_total`
- Replace GU combined 6144 with `cfg.xclbin_gu_n` or split G+U xclbin selection
- Make RoPE theta/factor configurable
- Conditional q_norm/k_norm (skip for Llama)
- Conditional lm_head (separate vs tied)
- Select xclbins by model tag

- [ ] **Step 1: Complete the ModelConfig header with all necessary fields**

- [ ] **Step 2: Rewrite `npu_engine_mt.cpp` with parameterized loops**

The key parameterization structure:

```cpp
// At the top of main():
ModelConfig cfg = parse_q4nx_header(model_path, model_tag);
if (!cfg.valid()) { fprintf(stderr, "Failed to parse model\n"); return 1; }

int H=cfg.H, NC=cfg.NC, NH=cfg.NH, NKV=cfg.NKV, HD=cfg.HD, 
    IM=cfg.IM, NV=cfg.NV, GQA=cfg.GQA, AW=cfg.AW, XM=cfg.XM;

// Dynamic buffer allocation using cfg dimensions:
std::vector<float> qo(M * cfg.qkv_total);
std::vector<float> ko(M * NKV * HD);
std::vector<float> vo(M * NKV * HD);
std::vector<float> at(M * NH * HD);
std::vector<float> oo(M * H);
int mlp_dim = cfg.gu_split ? IM : (2*IM);
std::vector<float> gt(M * mlp_dim);  // G+U output
std::vector<float> su(M *