#ifndef ONEBP_FORMAT_H
#define ONEBP_FORMAT_H

/** 1BP — 1bit-systems binary package format.
 *
 *  Unified model format combining Q4NX NPU-tiled layout with self-contained
 *  metadata. One file, no external config.json needed. Memory-mappable.
 *
 *  Layout on disk:
 *    [Header: 256 bytes]
 *    [Tensor Index: variable]
 *    [Weight Data: tiled arrays]
 *
 *  Tile layout (Q4NX-compatible, 32×256):
 *    Each tile = 32 rows × 256 cols of quantized data
 *    Per tile row:
 *      [0..511]:   256 BF16 scales (8 groups × 32 rows)
 *      [512..1023]: 256 BF16 zero_points (same layout)
 *      [1024..5119]: 4096 bytes packed INT4 (2 per byte)
 *    Total per tile: 5120 bytes
 */

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t ONEBP_MAGIC        = 0x00504231;  // "1BP\0"
static constexpr uint32_t ONEBP_VERSION      = 1;

// ─── Quantization types ────────────────────────────────────────────
enum OnebpQuant : uint32_t {
    ONEBP_Q4NX = 0,   // 4-bit block quant, bf16 scales (Q4NX compatible)
    ONEBP_I8   = 1,   // INT8 per-tensor quant, f32 scales
    ONEBP_TQ1  = 2,   // Ternary TQ1 (1.58-bit)
    ONEBP_TQ2  = 3,   // Ternary TQ2
    ONEBP_F16  = 4,   // Float16 (no quant)
    ONEBP_F32  = 5,   // Float32 (no quant)
};

// ─── Scale types ───────────────────────────────────────────────────
enum OnebpScale : uint32_t {
    ONEBP_SCALE_BF16 = 0,
    ONEBP_SCALE_F32  = 1,
    ONEBP_SCALE_F16  = 2,
};

// ─── Model architecture types ──────────────────────────────────────
enum OnebpArch : uint32_t {
    ONEBP_DENSE  = 0,  // Dense transformer (Qwen3, Llama, Phi, etc.)
    ONEBP_MOE    = 1,  // Mixture of Experts
    ONEBP_VISION = 2,  // Vision-language
    ONEBP_AUDIO  = 3,  // Audio/speech (Whisper)
    ONEBP_TERNARY= 4,  // Ternary/1-bit
    ONEBP_MAMBA  = 5,  // Mamba-style SSM
};

// ─── File header (256 bytes) ──────────────────────────────────────
#pragma pack(push, 1)
struct OnebpHeader {
    uint32_t magic;           // 0x00504231 = "1BP\0"
    uint32_t version;         // format version
    uint32_t arch;            // OnebpArch
    uint32_t quant;           // OnebpQuant
    uint32_t scale_type;      // OnebpScale
    int32_t  hidden_size;
    int32_t  num_layers;
    int32_t  num_attention_heads;
    int32_t  num_kv_heads;
    int32_t  head_dim;
    int32_t  intermediate_size;
    int32_t  vocab_size;
    int32_t  max_seq_len;
    uint32_t tile_rows;       // 32 (Q4NX compatible)
    uint32_t tile_cols;       // 256 (Q4NX compatible)
    uint32_t group_size;      // 32
    uint32_t has_q_norm;      // bool
    uint32_t has_k_norm;      // bool
    uint32_t has_bias;        // bool
    uint32_t rope_theta_f;    // rope_theta * 1000 (stored as fixed-point)
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t tensor_count;
    uint8_t  reserved[100];   // pad to 256 bytes
    char     model_tag[64];   // model identifier string
    
    bool valid() const {
        return magic == ONEBP_MAGIC && version == ONEBP_VERSION
            && hidden_size > 0 && num_layers > 0 && vocab_size > 0;
    }
    
    void init() {
        memset(this, 0, sizeof(*this));
        magic = ONEBP_MAGIC;
        version = ONEBP_VERSION;
        tile_rows = 32;
        tile_cols = 256;
        group_size = 32;
    }
    
    float rope_theta() const { return (float)rope_theta_f / 1000.0f; }
    void set_rope_theta(float v) { rope_theta_f = (uint32_t)(v * 1000.0f); }
};
#pragma pack(pop)

// Verify header size at compile time
static_assert(sizeof(OnebpHeader) == 256, "OnebpHeader must be exactly 256 bytes");

// ─── Tensor index entry (on-disk) ─────────────────────────────────
// Variable-length: [name_len:u32][name:str][ndim:u32][shape:u32[]][offset:u64][bytes:u64]
// Not a fixed struct — written element by element in the converter.

// ─── Compute tiled size for a weight matrix ──────────────────────
// Returns bytes needed after tiling rows×cols to tile_rows×tile_cols
static inline uint64_t onebp_tiled_size(
    uint32_t rows, uint32_t cols,
    uint32_t tile_rows, uint32_t tile_cols,
    uint32_t group_size, OnebpQuant quant
) {
    uint32_t num_tile_rows = (rows + tile_rows - 1) / tile_rows;
    uint32_t num_tile_cols = (cols + tile_cols - 1) / tile_cols;
    uint32_t groups_per_row = tile_cols / group_size;
    
    uint64_t tile_bytes;
    switch (quant) {
        case ONEBP_Q4NX:
            // scales (bf16 × groups × rows) + zero_points + packed indices
            tile_bytes = (uint64_t)tile_rows * groups_per_row * 2  // scales
                       + (uint64_t)tile_rows * groups_per_row * 2  // zero_points
                       + (uint64_t)tile_rows * tile_cols / 2;      // 4-bit packed
            break;
        case ONEBP_I8:
            tile_bytes = (uint64_t)tile_rows * tile_cols;  // 1 byte per element
            break;
        case ONEBP_F16:
            tile_bytes = (uint64_t)tile_rows * tile_cols * 2;
            break;
        default:
            tile_bytes = (uint64_t)tile_rows * tile_cols * 4;  // F32 fallback
    }
    
    return (uint64_t)num_tile_rows * num_tile_cols * tile_bytes;
}

#endif // ONEBP_FORMAT_H
