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
 *  Tile layout — Q4NX (header.quant == ONEBP_Q4NX), 32×256:
 *    Each tile = 32 rows × 256 cols of quantized data
 *    Per tile row:
 *      [0..511]:   256 BF16 scales (8 groups × 32 rows)
 *      [512..1023]: 256 BF16 zero_points (same layout)
 *      [1024..5119]: 4096 bytes packed INT4 (2 per byte)
 *    Total per tile: 5120 bytes
 *
 *  Tile layout — TQ2 (header.quant == ONEBP_TQ2), 32×256:
 *    Symmetric ternary: every value is exactly -scale, 0, or +scale — no
 *    zero_point needed (unlike Q4NX's asymmetric min/scale). One scale per
 *    32-element group.
 *    Per tile row:
 *      [0..15]:    8 BF16 scales (one per 32-element group)
 *      [16..79]:   64 bytes packed 2-bit codes (4 per byte, LSB-first:
 *                  code = byte & 3, (byte>>2) & 3, (byte>>4) & 3, (byte>>6) & 3)
 *                  code 0 = -scale, 1 = 0, 2 = +scale, 3 = unused (encoder
 *                  never emits it; a decoder should treat it as 0)
 *    Total per tile row: 80 bytes → 2560 bytes per tile (exactly half of
 *    Q4NX's 5120 — the real "1-bit-ish" storage win Q4NX doesn't give you).
 *    This is a generic symmetric-ternary quantizer (round to nearest of
 *    {-scale,0,scale} per group) — lossless when the source is already
 *    ternary-valued within each 32-group (as with BitNet/TriLM/Bonsai-style
 *    checkpoints), lossy-but-functional otherwise, same as Q4NX is lossy
 *    for any source that isn't already 4-bit-quantizable.
 *
 *  TQ1 (1.58-bit, base-3 packing) is defined in the OnebpQuant enum but
 *  not yet implemented by the converter or loader.
 *
 *  Tensor index entry (variable-length):
 *    [name_len:u32][name:str][ndim:u32][dims:u32 × ndim][offset:u64][bytes:u64]
 *
 *    ndim=2: dims=[rows, cols] — a plain weight matrix. `bytes` is the
 *      tiled size of that one matrix.
 *    ndim=3: dims=[num_experts, rows, cols] — a stack of `num_experts`
 *      independently-tiled matrices (MoE expert weights: ffn_gate_up_exps,
 *      ffn_down_exps, etc.), laid out back-to-back with no padding between
 *      experts. `bytes` = num_experts × tiled_size(rows, cols). Each
 *      expert's slice uses the exact same 32×256 tile scheme as a regular
 *      2D tensor — there's no cross-expert structure to the quantization,
 *      it's just N matrices concatenated.
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
    ONEBP_MOE    = 1,  // Mixture of Experts (generic)
    ONEBP_VISION = 2,  // Vision-language
    ONEBP_AUDIO  = 3,  // Audio/speech (Whisper)
    ONEBP_TERNARY= 4,  // Ternary/1-bit
    ONEBP_MAMBA  = 5,  // Mamba-style SSM
    ONEBP_LAGUNA = 6,  // Poolside Laguna — sigmoid-routed MoE, hybrid SWA/global attn, softplus gate
};

// ─── Expert gating function types (Laguna/afmoe) ────────────────
enum OnebpExpertGating : uint32_t {
    ONEBP_EXPERT_GATING_SIGMOID  = 0,
    ONEBP_EXPERT_GATING_SOFTMAX  = 1,
    ONEBP_EXPERT_GATING_NONE     = 2,
};

// ─── Attention gate types (Laguna) ───────────────────────────────
enum OnebpAttnGate : uint32_t {
    ONEBP_ATTN_GATE_PER_HEAD    = 0,  // one scalar per head (XS.2 style)
    ONEBP_ATTN_GATE_PER_ELEMENT = 1,  // full per-element gate (M.1 style)
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
    uint32_t has_q_norm;            // bool
    uint32_t has_k_norm;            // bool
    uint32_t has_bias;              // bool
    uint32_t rope_theta_f;          // rope_theta * 1000 (stored as fixed-point)
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t tensor_count;
    // ── Laguna-specific fields (reserved[0..51]) ────────────────
    uint32_t num_experts;           // total routed experts (Laguna S 2.1: 256)
    uint32_t n_expert_used;         // experts used per token (top-k, S 2.1: 10)
    uint32_t n_ff_exp;              // expert FFN intermediate size
    uint32_t n_ff_shexp;            // shared expert FFN intermediate size
    uint32_t n_layer_dense_lead;    // leading dense layers before MoE begins (M.1: 3)
    uint32_t sliding_window;        // SWA window size (0 = no SWA, M.1: 0, S 2.1: 512)
    uint32_t swa_period;            // SWA pattern period (1:3 -> 4, FULL at il%period==0)
    uint32_t expert_gating_func;    // OnebpExpertGating
    uint32_t expert_weights_norm;   // bool: sum-normalize expert weights after top-k
    uint32_t expert_weights_scale_f; // routed scaling factor * 1000 (fixed-point)
    uint32_t attn_gate_type;        // OnebpAttnGate (per-head vs per-element)
    uint32_t rope_freq_base_swa_f;  // SWA layer RoPE freq base * 1000
    uint32_t n_rot_swa;             // SWA layer RoPE dimension count
    uint32_t n_rot_full;            // FULL attention layer RoPE dim count (0 = use head_dim)
    uint8_t  reserved[44];          // remaining pad to 256 bytes
    char     model_tag[64];         // model identifier string
    
    // validity: core dims always required; attention heads are optional
    // (Mamba/MoE architectures have no attention, set NH=0, HD=0).
    bool valid() const {
        if (magic != ONEBP_MAGIC || version != ONEBP_VERSION) return false;
        if (hidden_size <= 0 || num_layers <= 0 || vocab_size <= 0) return false;
        if (num_attention_heads > 0 && head_dim <= 0) return false;
        if (head_dim > 0 && num_attention_heads <= 0) return false;
        if (intermediate_size <= 0 || tile_rows <= 0 || tile_cols <= 0 || group_size <= 0) return false;
        return true;
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
    float expert_weights_scale() const { return (float)expert_weights_scale_f / 1000.0f; }
    void set_expert_weights_scale(float v) { expert_weights_scale_f = (uint32_t)(v * 1000.0f); }
    float rope_freq_base_swa() const { return (float)rope_freq_base_swa_f / 1000.0f; }
    void set_rope_freq_base_swa(float v) { rope_freq_base_swa_f = (uint32_t)(v * 1000.0f); }
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
        case ONEBP_TQ2:
            // scales (bf16 x groups x rows) + 2-bit packed codes (4/byte)
            tile_bytes = (uint64_t)tile_rows * groups_per_row * 2   // scales
                       + (uint64_t)tile_rows * tile_cols / 4;       // 2-bit packed
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
