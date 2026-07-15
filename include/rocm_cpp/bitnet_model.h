// BitNet model loader — pure C ABI.
// Reads .h1b files (halo-1bit format) and returns device pointers
// suitable for direct use with rocm_cpp kernels (rcpp_ternary_gemv,
// rcpp_rmsnorm_fp16, etc).

#ifndef ROCM_CPP_BITNET_MODEL_H
#define ROCM_CPP_BITNET_MODEL_H

#include "rocm_cpp/ck_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

// .h1b flag bits (live in cfg[8] — formerly "reserved"). Compose via bit-or.
//   0x1 : H-BitLinear Hadamard-rotation bake-in (activations must be pre-rotated).
//   0x2 : Sherry 1.25-bpw weights routed through the fp16-in/fp16-out kernel
//         (sherry_ternary_gemv_launch) instead of the int8-act sherry decoder.
//   0x4 : Bonsai Q1_0_g128 weights (1.0 bpw info / 1.125 bpw on-disk; 18 B / block).
//   0x8 : Bonsai TQ2_0_g128 weights (~1.585 bpw info / 2.125 bpw on-disk; 34 B / block).
//         Both Bonsai flags imply Qwen3 architecture with per-head q/k norms
//         and plain SwiGLU (no attn_sub_norm / ffn_sub_norm). See
//         docs/wiki/Bonsai-Kernel-Spec.md for the byte-exact layout.
#define H1B_FLAG_HADAMARD_ROTATED 0x1u
#define H1B_FLAG_SHERRY_FP16      0x2u
#define H1B_FLAG_BONSAI_Q1        0x4u
#define H1B_FLAG_BONSAI_TQ2       0x8u

// Dispatch tag for the per-layer ternary GEMV. Driven by file version + flags
// at load time; inference loop reads this instead of re-parsing the header.
//   HALO_V2    : halo v2 packing (2 bpw, uint8 [rows, (cols+3)/4])
//   SHERRY_I8  : halo-1bit v3 Sherry (1.25 bpw) with int8 acts + per-row scales
//   TQ1        : halo-1bit v4 base-3 TQ1 (1.6 bpw, lossless)
//   SHERRY_FP16: same pack as SHERRY_I8 but dispatched to the pure fp16 kernel
//                (sherry_ternary_gemv_launch) — no int8 quant, post-row-scale
//                folded in by the caller.
//   BONSAI_Q1  : Bonsai 1-bit g128 block-interleaved (oxibonsai layout).
//                Dispatched to bonsai_q1_gemv_launch, fp16-in/fp16-out, inline
//                per-block fp16 scales, no row_scales tensor on the layer.
//   BONSAI_TQ2 : Bonsai ternary 2-bit g128 block-interleaved. Dispatched to
//                bonsai_tq2_gemv_launch. Same fp16-in/fp16-out, inline scales.
typedef enum {
    RCPP_WEIGHT_FORMAT_HALO_V2    = 0,
    RCPP_WEIGHT_FORMAT_SHERRY_I8  = 1,
    RCPP_WEIGHT_FORMAT_TQ1        = 2,
    RCPP_WEIGHT_FORMAT_SHERRY_FP16 = 3,
    RCPP_WEIGHT_FORMAT_BONSAI_Q1  = 4,
    RCPP_WEIGHT_FORMAT_BONSAI_TQ2 = 5,
    RCPP_WEIGHT_FORMAT_WMMA_I8    = 6,  // Hadamard-rotated INT8 weights, wmma_i32_iu8 compute
} rcpp_weight_format_t;

// Model-level architecture tag. Decoupled from `weight_format` because the
// Bonsai Q1 / TQ2 kernel path is arch-agnostic — a `.h1b` emitted by
// `tools/bitnet-to-tq2/` carries Microsoft BitNet-b1.58 norms (attn_sub_norm
// + ffn_sub_norm + squared-ReLU GLU) in the ternary-GEMV-compatible TQ2
// payload, while oxibonsai emits Qwen3 norms (per-head attn_q/k_norm +
// SwiGLU) in the same TQ2 payload.
//
// The loader resolves this from the `.h1b` flags plus the (optional) sidecar
// GGUF: if `H1B_FLAG_BONSAI_*` is set AND a sidecar GGUF with
// `general.architecture = "qwen3"` is found next to the `.h1b`, the model is
// Qwen3; otherwise it falls back to BitNet (uses the sub-norms already
// written to the `.h1b` by the converter).
typedef enum {
    RCPP_ARCH_BITNET = 0,
    RCPP_ARCH_QWEN3  = 1,
} rcpp_arch_t;

typedef struct {
    // RMSNorm weights (FP16). BitNet-b1.58 has four per layer:
    //   input_norm    [hs] — pre Q/K/V
    //   attn_sub_norm [hs] — on attn output, before O proj
    //   post_attn_norm[hs] — pre gate/up
    //   ffn_sub_norm  [is] — on silu(gate)*up, before down proj
    //
    // Qwen3 / Bonsai carries two additional per-head RMSNorms that BitNet
    // doesn't have, both shape [head_dim]:
    //   attn_q_norm   [head_dim] — applied per head on Q before RoPE
    //   attn_k_norm   [head_dim] — applied per head on K before RoPE
    // These are nullptr on BitNet models; the Bonsai GGUF sidecar loader
    // fills them when the H1B_FLAG_BONSAI_* bits are set.
    void* input_norm_dev;
    void* post_attn_norm_dev;
    void* attn_sub_norm_dev;
    void* ffn_sub_norm_dev;
    void* attn_q_norm_dev;
    void* attn_k_norm_dev;

    // Ternary linear layers — halo-encoded uint8 packed (reinterpret as uint32
    // bytewise for rcpp_ternary_gemv_halo) + per-row FP32 scales.
    void* q_packed_dev;     float* q_scales_dev;      // [nh*hd, hs]
    void* k_packed_dev;     float* k_scales_dev;      // [nkv*hd, hs]
    void* v_packed_dev;     float* v_scales_dev;      // [nkv*hd, hs]
    void* o_packed_dev;     float* o_scales_dev;      // [hs, nh*hd]
    void* gate_packed_dev;  float* gate_scales_dev;   // [is, hs]
    void* up_packed_dev;    float* up_scales_dev;     // [is, hs]
    void* down_packed_dev;  float* down_scales_dev;   // [hs, is]

    // WMMA_I8 path: Hadamard-rotated INT8 weights + per-row fp32 scales.
    // When weight_format == WMMA_I8, these hold the INT8 data instead of
    // the ternary-packed buffers above. Same shapes, different element type.
    void* q_i8_dev;          float* q_i8_scales_dev;
    void* k_i8_dev;          float* k_i8_scales_dev;
    void* v_i8_dev;          float* v_i8_scales_dev;
    void* o_i8_dev;          float* o_i8_scales_dev;
    void* gate_i8_dev;       float* gate_i8_scales_dev;
    void* up_i8_dev;         float* up_i8_scales_dev;
    void* down_i8_dev;       float* down_i8_scales_dev;
} rcpp_bitnet_layer_t;

typedef struct {
    int hidden_size;         // hs
    int intermediate_size;   // is
    int num_layers;
    int num_heads;           // nh
    int num_kv_heads;        // nkv  (GQA)
    int vocab_size;
    int max_seq_len;
    int tie_embeddings;      // non-zero => LM head = embedding.T

    float rope_theta;
    float rms_norm_eps;

    int format_version;      // 1/2 = halo v2 (2 bpw); 3 = Sherry v3 (1.25 bpw); 4 = TQ1 (1.6 bpw)

    // .h1b flag bits (see H1B_FLAG_* above). Parsed from cfg[8] at load time.
    unsigned int flags;

    // Resolved dispatch tag for the per-layer ternary GEMV. Inference code
    // branches on this instead of re-deriving from (version, flags).
    rcpp_weight_format_t weight_format;

    // Non-zero when the resolved arch is RCPP_ARCH_QWEN3. Mirrors `arch`
    // below for backwards compat with call sites that predate the enum
    // (test_bonsai_e2e.cpp, older halo-1bit mirrors). New code should
    // branch on `arch` directly.
    int is_qwen3;

    // Resolved model architecture — drives the attention preamble + FFN
    // activation in the forward pass. Orthogonal to `weight_format`:
    // BONSAI_TQ2 weights can land under either BITNET (MS repack, no
    // sidecar GGUF) or QWEN3 (oxibonsai, sidecar GGUF arch=qwen3).
    rcpp_arch_t arch;

    void* embedding_dev;            // FP16 [vocab, hidden]
    void* embedding_packed_dev;     // TQ2_0_g128 [vocab, hidden] — nullptr unless loaded from Bonsai sidecar
    void* final_norm_weight_dev;    // FP16 [hidden]
    rcpp_bitnet_layer_t* layers;    // [num_layers]
} rcpp_bitnet_model_t;

// Load a .h1b model file and upload all weights to the GPU.
// Caller eventually frees with rcpp_bitnet_free.
rcpp_status_t
rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model);

// Load a GGUF model file and upload all weights to the GPU.
// Supports F32, F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M, Q6_K quantization.
// Other quantization types are dequantized on CPU to FP16.
rcpp_status_t
rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model);

// Release all device buffers + layer array.
void
rcpp_bitnet_free(rcpp_bitnet_model_t* model);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_BITNET_MODEL_H
