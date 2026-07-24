/** tq2_to_q4nx.cpp — Convert 1BP TQ2 model to Q4NX format for NPU inference.
 *
 * Q4NX is the NPU engine's native format: safetensors-style container with
 * float32 weights in the NPU's 32×256 Q4NX tile layout. The NPU engine
 * reads float32 weights and INT8-quantizes them on-the-fly during weight
 * upload (npu_engine_i8.cpp I8Ctx::packB()).
 *
 * This tool bridges the ternary world to the NPU world:
 *   1. Reads 1BP TQ2 weights
 *   2. Dequantizes to float32 (ternary {-1,0,+1} × per-group scale)
 *   3. Writes in Q4NX float32 format
 *
 * Run:  ./build/tq2_to_q4nx model.1bp model.q4nx
 *
 * Build: g++ -std=c++17 -O3 -I include -I src tools/tq2_to_q4nx.cpp \
 *            src/onebp_model.cpp -o build/tq2_to_q4nx -lpthread
 *
 * Then:  NPU_MODEL_PATH=model.q4nx ./build/unified_server -w models/ -p 8088
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include "onebp_format.h"
#include "onebp_loader.h"
#include <chrono>

// ─── FP16 <-> FP32 conversion ────────────────────────────────────
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t b = (uint32_t)bf << 16;
    float f; memcpy(&f, &b, 4); return f;
}

// ─── Q4NX writer ─────────────────────────────────────────────────
// Writes a safetensors-style Q4NX file with float32 weights.
struct Q4nxWriter {
    FILE* f = nullptr;
    uint64_t data_offset = 0;  // current write position in data section
    std::string json;           // accumulating JSON header

    bool open(const char* path) {
        f = fopen(path, "wb");
        if (!f) { perror("fopen"); return false; }
        // Reserve 8 bytes for header length
        uint64_t dummy = 0;
        fwrite(&dummy, 8, 1, f);
        json = "{";
        data_offset = 8;  // after header length
        return true;
    }

    // Write a Q4NX tensor: name, float32 data pointer, element count
    void write_tensor(const char* name, const float* data, uint64_t numel,
                      int dim0, int dim1) {
        if (json.size() > 1) json += ",";
        json += "\"" + std::string(name) + "\":{";
        json += "\"dtype\":\"F32\",";
        json += "\"shape\":[" + std::to_string(dim0) + "," + std::to_string(dim1) + "],";
        uint64_t end = data_offset + numel * 4;
        json += "\"data_offsets\":[" + std::to_string(data_offset) + "," + std::to_string(end) + "]";
        json += "}";
        
        fwrite(data, 4, numel, f);
        data_offset = end;
    }

    bool close() {
        json += "}";
        uint64_t header_len = json.size();
        // Write header length at offset 0
        fseek(f, 0, SEEK_SET);
        fwrite(&header_len, 8, 1, f);
        // Write JSON header right after the length
        fwrite(json.data(), 1, json.size(), f);
        fclose(f);
        return true;
    }
};

// ─── Dequantize one TQ2 tile row (32×256) to float32 ─────────────
// TQ2 tile: [scales: 32×8×2=512 bytes bf16][codes: 32×256/4=2048 bytes]
// Groups of 32: 8 per tile row, bf16 scale per group
// codes: 2-bit LSB-first, 0=-scale, 1=0, 2=+scale, 3=unused
static void dequant_tq2_tile(const uint8_t* tile, float* out,
                              int rows, int cols,
                              int trow, int tcol) {
    constexpr int TR = 32, TC = 256, GS = 32;
    int grps = TC / GS;  // 8
    int scales_bytes = TR * grps * 2;
    int codes_bytes  = TR * TC / 4;
    
    const uint16_t* scales_bf16 = (const uint16_t*)tile;
    const uint8_t*  codes       = tile + scales_bytes;
    
    for (int rr = 0; rr < TR && (trow + rr) < rows; rr++) {
        for (int g = 0; g < grps; g++) {
            float scale = bf16_to_f32(scales_bf16[rr * grps + g]);
            for (int i = 0; i < GS; i += 4) {
                int byte_idx = rr * (TC / 4) + (g * GS + i) / 4;
                uint8_t byte_ = codes[byte_idx];
                for (int j = 0; j < 4; j++) {
                    int ac = tcol + g * GS + i + j;
                    if (ac >= cols) break;
                    uint8_t code = (byte_ >> (j * 2)) & 3;
                    float val;
                    if (code == 0)      val = -scale;
                    else if (code == 2) val =  scale;
                    else                val = 0.0f;
                    out[(size_t)(trow + rr) * cols + ac] = val;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.1bp output.q4nx\n", argv[0]);
        return 1;
    }

    // ── Load 1BP model ──────────────────────────────────────────
    OnebpModel model;
    if (!model.load(argv[1])) {
        fprintf(stderr, "Failed to load 1BP: %s\n", argv[1]);
        return 1;
    }

    auto& h = model.header;
    if (h.quant != ONEBP_TQ2) {
        fprintf(stderr, "Only TQ2 (ternary 2-bit) quant supported. Got quant=%u\n", h.quant);
        return 1;
    }

    printf("1BP loaded: arch=%u H=%d L=%d NH=%d V=%d tensors=%zu\n",
           h.arch, h.hidden_size, h.num_layers,
           h.num_attention_heads, h.vocab_size, model.tensors.size());

    // ── Open Q4NX output ────────────────────────────────────────
    Q4nxWriter qw;
    if (!qw.open(argv[2])) return 1;

    // ── Convert each tensor ─────────────────────────────────────
    constexpr int TR = 32, TC = 256, GS = 32;
    int grps = TC / GS;
    int scales_bytes = TR * grps * 2;
    int codes_bytes  = TR * TC / 4;
    int tile_bytes   = scales_bytes + codes_bytes;

    int total_tiles = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (auto& t : model.tensors) {
        if (t.ndim < 2) continue;
        int rows = (int)t.dims[0];
        int cols = (int)t.dims[1];
        
        // Allocate float32 output
        std::vector<float> f32((size_t)rows * cols, 0.0f);
        
        // Dequantize TQ2 tile by tile
        int ntr = (rows + TR - 1) / TR;
        int ntc = (cols + TC - 1) / TC;
        
        for (int tr = 0; tr < ntr; tr++) {
            for (int tc = 0; tc < ntc; tc++) {
                int tile_idx = tr * ntc + tc;
                const uint8_t* tile_data = model.tensor_data(t);
                tile_data += (size_t)tile_idx * tile_bytes;
                dequant_tq2_tile(tile_data, f32.data(), rows, cols,
                                 tr * TR, tc * TC);
                total_tiles++;
            }
        }
        
        // Write to Q4NX
        qw.write_tensor(t.name.c_str(), f32.data(),
                        (uint64_t)rows * cols, rows, cols);
        
        if (model.tensors.size() <= 10 || total_tiles % 100 == 0) {
            printf("  %-40s %dx%d -> %.1f MB\n",
                   t.name.c_str(), rows, cols,
                   (double)rows * cols * 4 / 1e6);
        }
    }

    qw.close();

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    
    // Get output file size
    FILE* fc = fopen(argv[2], "rb"); fseek(fc, 0, SEEK_END);
    long fsz = ftell(fc); fclose(fc);
    
    printf("\n=== DONE: %s (%.1f MB, %d tiles) in %.1f seconds ===\n",
           argv[2], fsz / 1e6, total_tiles, sec);
    printf("Run: NPU_MODEL_PATH=%s ./build/unified_server -w models/ -p 8088\n", argv[2]);
    return 0;
}
