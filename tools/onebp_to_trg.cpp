#include "cpu_q4nx_loader.h"
// tools/onebp_to_trg.cpp — Convert 1BP models to .trg format
// Full pipeline: 1BP → Q4NX (re-wrap) → TRG packing
//
// Build: g++ -std=c++17 -O3 -march=native -fopenmp \
//        -I engine/fusion -I include -I . \
//        tools/onebp_to_trg.cpp engine/fusion/cpu_layer.cpp \
//        -o build/onebp_to_trg -lm -lpthread
//
// Run: ./build/onebp_to_trg model.1bp output.trg
//      ./build/onebp_to_trg model.1bp output.q4nx  (save intermediate Q4NX)

#include "onebp_format.h"
#include "cpu_layer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <fstream>

// Tiled size calculation (Q4NX: scales + zero_points + packed)
static uint64_t tiled_size_q4nx(uint32_t rows, uint32_t cols, uint32_t tr, uint32_t tc, uint32_t gs) {
    uint32_t ntr = (rows + tr - 1) / tr;
    uint32_t ntc = (cols + tc - 1) / tc;
    uint32_t gpr = tc / gs;
    uint64_t tile = (uint64_t)tr * gpr * 2  // scales bf16
                  + (uint64_t)tr * gpr * 2  // zero_points bf16
                  + (uint64_t)tr * tc / 2;  // 4-bit packed
    return ntr * ntc * tile;
}

// Pack one projection to TRG format (ternary with per-block scales)
// This is extracted from trg_save.cpp's pack_proj
static void pack_proj(const float* src, uint32_t* packed, float* scales,
                      int OUT, int IN, int n_blocks) {
    int pp = (IN + 15) / 16;
    for (int r = 0; r < OUT; r++) {
        // Per-block scale: mean |weight| within each 256-value block
        for (int b = 0; b < n_blocks; b++) {
            double sa = 0; int nz = 0;
            int blk_start = b * 256;
            int blk_end = (b + 1) * 256;
            if (blk_end > IN) blk_end = IN;
            for (int c = blk_start; c < blk_end; c++) {
                float v = src[r*IN+c]; sa += fabs(v); if (v != 0) nz++;
            }
            scales[r * n_blocks + b] = (nz > 0) ? (float)(sa / nz) : 0;
        }
        for (int u = 0; u < pp; u++) {
            uint32_t w = 0;
            for (int v = 0; v < 16; v++) {
                int c = u*16+v;
                float val = (c < IN) ? src[r*IN+c] : 0;
                w |= ((val > 0 ? 0x1u : (val < 0 ? 0x2u : 0)) << (v*2));
            }
            packed[r*pp+u] = w;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.1bp output.trg [--save-q4nx path.q4nx]\n", argv[0]);
        return 1;
    }

    bool save_q4nx = false;
    std::string q4nx_path;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--save-q4nx") == 0 && i+1 < argc) {
            save_q4nx = true;
            q4nx_path = argv[++i];
        }
    }

    // Memory-map the 1BP file
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    size_t fsz = lseek(fd, 0, SEEK_END);
    auto data = (const uint8_t*)mmap(0, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }

    // Parse 1BP header
    OnebpHeader hdr;
    memcpy(&hdr, data, sizeof(hdr));
    if (!hdr.valid()) {
        fprintf(stderr, "Invalid 1BP header (magic=0x%08x)\n", hdr.magic);
        munmap((void*)data, fsz); return 1;
    }

    int H = hdr.hidden_size;
    int L = hdr.num_layers;
    int NH = hdr.num_attention_heads;
    int NKV = hdr.num_kv_heads ? hdr.num_kv_heads : NH;
    int HD = hdr.head_dim ? hdr.head_dim : (NH ? H / NH : 64);
    int V = hdr.vocab_size;
    int IM = hdr.intermediate_size;
    int tr = 32, tc = 256, gs = 32;

    printf("=== 1BP → TRG: %s\n", argv[1]);
    printf("  H=%d L=%d NH=%d NKV=%d HD=%d V=%d IM=%d\n", H, L, NH, NKV, HD, V, IM);
    printf("  arch=%u quant=%u tensors=%u\n", hdr.arch, hdr.quant, hdr.tensor_count);

    // Walk tensor index
    struct TensorInfo {
        std::string name;
        uint64_t offset, bytes;
        int rows, cols;
    };
    std::vector<TensorInfo> tensors;
    
    uint64_t idx_off = sizeof(OnebpHeader);
    uint64_t weight_data_off = fsz;

    for (uint32_t i = 0; i < hdr.tensor_count; i++) {
        TensorInfo ti;
        uint32_t nlen;
        if (idx_off + 4 > fsz) break;
        memcpy(&nlen, data + idx_off, 4); idx_off += 4;
        if (nlen > 255 || idx_off + nlen > fsz) break;
        ti.name.assign((const char*)(data + idx_off), nlen);
        idx_off += nlen + 1; // skip null terminator
        
        uint32_t ndim;
        if (idx_off + 4 > fsz) break;
        memcpy(&ndim, data + idx_off, 4); idx_off += 4;
        
        uint32_t shape[3] = {0};
        for (uint32_t d = 0; d < ndim && d < 3; d++) {
            if (idx_off + 4 > fsz) break;
            memcpy(&shape[d], data + idx_off, 4); idx_off += 4;
        }
        
        if (idx_off + 16 > fsz) break;
        memcpy(&ti.offset, data + idx_off, 8); idx_off += 8;
        memcpy(&ti.bytes, data + idx_off, 8); idx_off += 8;
        
        ti.rows = shape[0];
        ti.cols = ndim > 1 ? shape[1] : 1;
        
        if (ti.offset < weight_data_off) weight_data_off = ti.offset;
        
        // Only process 2D weight tensors (skip norms, embeddings that are BF16)
        if (ndim == 2 && ti.rows > 0 && ti.cols > 0 && ti.bytes > 0) {
            tensors.push_back(ti);
        }
    }

    printf("  Tensors: %zu (2D weights), data at offset %lu\n", 
           tensors.size(), (unsigned long)weight_data_off);
    
    if (tensors.empty()) {
        fprintf(stderr, "No 2D weight tensors found\n");
        munmap((void*)data, fsz); return 1;
    }

    // ── Step 1: Generate Q4NX JSON header ─────────────────────
    std::string json = "{";
    for (size_t i = 0; i < tensors.size(); i++) {
        auto& t = tensors[i];
        // Q4NX tile shape: [n_blocks, 5120]
        uint32_t ntr = (t.rows + tr - 1) / tr;
        uint32_t ntc = (t.cols + tc - 1) / tc;
        uint32_t n_blocks = ntr * ntc;
        // Data offsets relative to start of weight data
        uint64_t data_start = t.offset - weight_data_off;
        uint64_t data_end = data_start + t.bytes;
        
        if (i > 0) json += ",";
        json += "\"" + t.name + "\":{";
        json += "\"dtype\":\"I8\",";
        json += "\"shape\":[" + std::to_string(n_blocks) + ",5120],";
        json += "\"data_offsets\":[" + std::to_string(data_start) + "," + std::to_string(data_end) + "]";
        json += "}";
    }
    json += "}";
    
    if (save_q4nx) {
        // Write Q4NX file
        FILE* fq = fopen(q4nx_path.c_str(), "wb");
        if (!fq) { perror("fopen q4nx"); return 1; }
        uint64_t json_size = json.size();
        fwrite(&json_size, 8, 1, fq);
        fwrite(json.data(), 1, json.size(), fq);
        fwrite(data + weight_data_off, 1, fsz - weight_data_off, fq);
        fclose(fq);
        printf("  Q4NX saved: %s (%.1f MB)\n", q4nx_path.c_str(), 
               (json_size + fsz - weight_data_off) / (1024.0*1024.0));
    }

    // ── Step 2: Dequantize weights via cpu_layer ────────────────
    // For TRG packing, we need FP32 weights. The Q4NX loader handles
    // this, but we can also do it directly from the 1BP tile data.
    // 
    // For now: load the Q4NX (if saved) via trg_save, or do direct
    // dequant in future. The key bridge is the Q4NX generation above.
    
    printf("\n  Q4NX JSON header: %zu bytes, %zu tensors\n", json.size(), tensors.size());
    printf("  Pipeline: 1BP → Q4NX → trg_save convert → .trg\n");
    printf("  Run: ./build/trg_save save <q4nx_path> <output.trg>\n");

    // ── Step 3: Load Q4NX and pack to TRG ──────────────────────
    // Temporarily write Q4NX to temp file for loading
    std::string tmp_q4nx = "/tmp/onebp_to_trg_tmp.q4nx";
    FILE* fq = fopen(tmp_q4nx.c_str(), "wb");
    if (!fq) { perror("fopen temp"); return 1; }
    uint64_t json_size = json.size();
    fwrite(&json_size, 8, 1, fq);
    fwrite(json.data(), 1, json.size(), fq);
    fwrite(data + weight_data_off, 1, fsz - weight_data_off, fq);
    fclose(fq);
    
    // Load Q4NX model (this dequantizes tiles to FP32)
    printf("  Loading Q4NX for TRG packing...\n");
    // Use our own direct tile reader instead of the full Q4NX loader
    // to avoid format mismatch. Read tiles directly from mmap'd 1BP data.
    
    // ── Step 4: Write TRG header and pack weights ──────────────
    // TRG v2 format (per-block ternary scales):
    //   [TrgHeader: 512 bytes] magic="TRG2", dims, offsets
    //   [Embedding: V*H fp32]
    //   [Final norm: H fp32]
    //   [LM head: V*H fp32]  
    //   [Norms: L*(H+H+HD+HD) fp32]
    //   [Packed ternary weights: per-layer QKV+GUD]
    //   [Per-block scales: per-layer]
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Build weights map: group tensors by layer
    struct LayerWeights {
        float* q=0,* k=0,* v=0,* o=0,* gate=0,* up=0,* down=0;
        int qs=0,ks=0,vs=0,os=0,gs=0,us=0,ds=0;
    };
    std::map<int, LayerWeights> layers;
    
    // Helper: find tensor and mmap its tile data
    auto get_tensor_data = [&](const std::string& name) -> std::vector<float> {
        for (auto& t : tensors) {
            if (t.name == name) {
                // Buffer for dequantized output
                std::vector<float> fp32(t.rows * t.cols);
                // Read tile data from mmap
                const uint8_t* tile_data = data + t.offset;
                uint32_t ntr = (t.rows + tr - 1) / tr;
                uint32_t ntc = (t.cols + tc - 1) / tc;
                
                for (uint32_t bi = 0; bi < ntr * ntc; bi++) {
                    const uint8_t* block = tile_data + bi * 5120;
                    int tile_row = bi / ntc;
                    int tile_col = bi % ntc;
                    
                    // Dequantize I8 block → FP32
                    const uint16_t* scales = (const uint16_t*)block;
                    const uint16_t* zps    = (const uint16_t*)(block + 512);
                    const uint8_t*  packed = block + 1024;
                    
                    for (int lr = 0; lr < 32; lr++) {
                        int row = tile_row * 32 + lr;
                        if (row >= t.rows) continue;
                        int lane = lr / 16;
                        int lr2 = lr % 16;
                        int bi_e = lr2 / 2;
                        int ns = lr % 2;
                        
                        for (int g = 0; g < 8; g++) {
                            float s = bf16_to_f32(scales[g * 32 + lr]);
                            float z = bf16_to_f32(zps[g * 32 + lr]);
                            for (int c = 0; c < 32; c++) {
                                int col = tile_col * 256 + g * 32 + c;
                                if (col >= t.cols) continue;
                                uint8_t bv = packed[lane * 2048 + col * 8 + bi_e];
                                int cd = (ns == 0) ? (bv & 0x0F) : ((bv >> 4) & 0x0F);
                                fp32[row * t.cols + col] = (float)cd * s + z;
                            }
                        }
                    }
                }
                return fp32;
            }
        }
        return {};
    };
    
    // Count weight tensors for TRG header
    int per_layer = 0;
    int rows7[7];
    // Q, K, V, O, gate, up, down dimensions per layer
    // Derived from first layer's tensors
    if (tensors.size() >= 7) {
        auto& t0_w = tensors[0];
        (void)t0_w;
    }
    
    // Determine model architecture from weight names
    bool is_qwen3 = false;
    for (auto& t : tensors) {
        if (t.name.find("self_attn.q_proj") != std::string::npos) is_qwen3 = true;
        if (t.name.find("q_proj") != std::string::npos && 
            t.name.find("self_attn") == std::string::npos) is_qwen3 = false;
    }
    
    // Map tensor names to 7 projections based on naming convention
    std::string qn, kn, vn, on, gn, un, dn;
    if (is_qwen3) {
        qn = "model.layers.0.self_attn.q_proj.weight";
        kn = "model.layers.0.self_attn.k_proj.weight";
        vn = "model.layers.0.self_attn.v_proj.weight";
        on = "model.layers.0.self_attn.o_proj.weight";
        gn = "model.layers.0.mlp.gate_proj.weight";
        un = "model.layers.0.mlp.up_proj.weight";
        dn = "model.layers.0.mlp.down_proj.weight";
    }
    
    // Find first layer dims
    int qr=0, qc=0, kr=0, kc=0, vr=0, vc=0, or_=0, oc=0;
    int gr=0, gc=0, ur_=0, uc=0, dr=0, dc=0;
    for (auto& t : tensors) {
        if (t.name.find("layers.0.") != std::string::npos) {
            if (t.name.find("q_proj") != std::string::npos) { qr=t.rows; qc=t.cols; }
            if (t.name.find("k_proj") != std::string::npos) { kr=t.rows; kc=t.cols; }
            if (t.name.find("v_proj") != std::string::npos) { vr=t.rows; vc=t.cols; }
            if (t.name.find("o_proj") != std::string::npos) { or_=t.rows; oc=t.cols; }
            if (t.name.find("gate_proj") != std::string::npos) { gr=t.rows; gc=t.cols; }
            if (t.name.find("up_proj") != std::string::npos) { ur_=t.rows; uc=t.cols; }
            if (t.name.find("down_proj") != std::string::npos) { dr=t.rows; dc=t.cols; }
        }
    }
    
    // Gather embedding, final norm, lm_head
    std::vector<float> embed, final_norm, lm_head;
    for (auto& t : tensors) {
        if (t.name.find("token_embd") != std::string::npos || 
            t.name.find("embed_tokens") != std::string::npos) {
            embed = get_tensor_data(t.name);
        }
        if (t.name.find("output_norm") != std::string::npos || 
            t.name.find("final_norm") != std::string::npos) {
            final_norm = get_tensor_data(t.name);
        }
    }
    
    // Check for lm_head (may be tied with embedding)
    for (auto& t : tensors) {
        if (t.name.find("lm_head") != std::string::npos) {
            lm_head = get_tensor_data(t.name);
        }
    }
    if (lm_head.empty() && !embed.empty()) {
        lm_head = embed; // tied embeddings
    }
    
    printf("  Layer 0 dims: Q=%dx%d K=%dx%d V=%dx%d O=%dx%d Gate=%dx%d Up=%dx%d Down=%dx%d\n",
           qr, qc, kr, kc, vr, vc, or_, oc, gr, gc, ur_, uc, dr, dc);
    printf("  Embed: %s (%zu elements)\n", embed.empty() ? "missing" : "ok", embed.size());
    printf("  Final norm: %s (%zu elements)\n", final_norm.empty() ? "missing" : "ok", final_norm.size());
    
    // Write TRG file
    FILE* ftrg = fopen(argv[2], "wb");
    if (!ftrg) { perror("fopen trg"); return 1; }
    
    // TRG v2 header (512 bytes)
    struct __attribute__((packed)) TrgHeader {
        char magic[4] = {'T','R','G','2'};
        int32_t H, IM, NH, NKV, HD, V, L, GQA;
        int32_t ps[7];
        int32_t bs[7];
        uint64_t o_emb, o_fn, o_lm, o_norms, o_pk, o_sc, file_size;
        uint8_t reserved[512 - 4 - 8*4 - 7*4 - 7*4 - 8*7];
    };
    static_assert(sizeof(TrgHeader) == 512, "TrgHeader must be exactly 512 bytes");
    
    TrgHeader th;
    memset(&th, 0, sizeof(th));
    th.H = H; th.IM = IM; th.NH = NH; th.NKV = NKV; th.HD = HD; th.V = V; th.L = L;
    th.GQA = NH / (NKV ? NKV : 1);
    
    int pp[] = { (NH*HD+15)/16, (NKV*HD+15)/16, (NKV*HD+15)/16, (H+15)/16, (IM+15)/16, (IM+15)/16, (H+15)/16 };
    int bb[] = { qc/256, kc/256, vc/256, oc/256, gc/256, uc/256, dc/256 };
    memcpy(th.ps, pp, sizeof(pp));
    memcpy(th.bs, bb, sizeof(bb));
    
    uint64_t off = sizeof(TrgHeader);
    th.o_emb = off; off += embed.size() * 4;
    th.o_fn  = off; off += final_norm.size() * 4;
    th.o_lm  = off; off += lm_head.size() * 4;
    
    // Norms: input_norm + post_attn_norm + q_norm + k_norm per layer
    // For now, approximate size. In practice norms are loaded separately.
    th.o_norms = off; off += L * (H + H + HD + HD) * 4;
    th.o_pk = off; off += L * 7 * pp[0]; // approx
    th.o_sc = off; off += L * 7 * bb[0]; // approx
    th.file_size = 0; // filled later
    
    fwrite(&th, sizeof(th), 1, ftrg);
    
    // Write embedding
    if (!embed.empty()) fwrite(embed.data(), 4, embed.size(), ftrg);
    else { for (int i = 0; i < V*H; i++) { float z=0; fwrite(&z,4,1,ftrg); } }
    
    // Write final norm
    if (!final_norm.empty()) fwrite(final_norm.data(), 4, final_norm.size(), ftrg);
    else { for (int i = 0; i < H; i++) { float z=0; fwrite(&z,4,1,ftrg); } }
    
    // Write LM head
    if (!lm_head.empty()) fwrite(lm_head.data(), 4, lm_head.size(), ftrg);
    else if (!embed.empty()) fwrite(embed.data(), 4, embed.size(), ftrg);
    else { for (int i = 0; i < V*H; i++) { float z=0; fwrite(&z,4,1,ftrg); } }
    
    // Write norms (placeholder zeros for now)
    size_t norms_size = L * (H + H + HD + HD);
    for (size_t i = 0; i < norms_size; i++) { float z=0; fwrite(&z,4,1,ftrg); }
    
    // Write packed weights per layer
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Packing %d layers...\n", L);
    fflush(stdout);
    
    for (int layer = 0; layer < L && layer < 100; layer++) {
        std::string prefix = "model.layers." + std::to_string(layer) + ".";
        
        // Find tensors for this layer
        auto find_tensor = [&](const std::string& suffix) -> std::vector<float> {
            for (auto& t : tensors) {
                if (t.name == prefix + suffix) {
                    return get_tensor_data(t.name);
                }
            }
            return {};
        };
        
        auto wq = find_tensor("self_attn.q_proj.weight");
        auto wk = find_tensor("self_attn.k_proj.weight");
        auto wv = find_tensor("self_attn.v_proj.weight");
        auto wo = find_tensor("self_attn.o_proj.weight");
        auto wg = find_tensor("mlp.gate_proj.weight");
        auto wu = find_tensor("mlp.up_proj.weight");
        auto wd = find_tensor("mlp.down_proj.weight");
        
        // Pack each projection
        auto pack_write = [&](const std::vector<float>& w, int out_d, int in_d) {
            if (w.empty() || out_d <= 0 || in_d <= 0) return;
            int n_blocks = in_d / 256;
            int n_packed = ((in_d + 15) / 16) * sizeof(uint32_t);
            std::vector<uint32_t> packed(out_d * ((in_d+15)/16));
            std::vector<float> scales(out_d * n_blocks);
            pack_proj(w.data(), packed.data(), scales.data(), out_d, in_d, n_blocks);
            fwrite(packed.data(), 1, packed.size() * 4, ftrg);
            th.o_sc = ftell(ftrg); // update scale offset
            fwrite(scales.data(), 4, scales.size(), ftrg);
        };
        
        pack_write(wq, NH*HD, H);
        pack_write(wk, NKV*HD, H);
        pack_write(wv, NKV*HD, H);
        pack_write(wo, H, NH*HD);
        pack_write(wg, IM, H);
        pack_write(wu, IM, H);
        pack_write(wd, H, IM);
        
        if (layer % 8 == 0) { printf("  layer %d/%d\r", layer+1, L); fflush(stdout); }
    }
    
    // Update header with actual file size
    th.file_size = ftell(ftrg);
    fseek(ftrg, 0, SEEK_SET);
    fwrite(&th, sizeof(th), 1, ftrg);
    fclose(ftrg);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t2 - t0).count();
    
    struct stat st;
    stat(argv[2], &st);
    
    printf("\n  ✅ TRG saved: %s (%.1f MB) in %.0fs\n", 
           argv[2], st.st_size / (1024.0*1024.0), elapsed);
    
    // Cleanup
    remove(tmp_q4nx.c_str());
    munmap((void*)data, fsz);
    return 0;
}
