// rocm-cpp — Medusa speculative-decoding kernel skeleton (C API).
//
// Why this header exists
// ----------------------
// Medusa-1 (arXiv 2401.10774) drafts K candidate next-tokens with K small heads
// during decode, then verifies a TREE of candidate prefixes against the same
// backbone in one extra pass; the longest greedy-matching prefix is committed.
// Expected wall-clock on gfx1151 batch=1 BitNet-2B is 1.4–1.8x; long-context
// gain is bandwidth-capped (Sherry attacks the same ceiling directly).
//
// Two new HIP kernels are needed beyond the existing decode/prefill set:
//
//   1. rcpp_medusa_tree_attn_decode_fd  — tree-parallel split-KV Flash-Decoding.
//      Generalization of rcpp_kv_cache_attn_decode_fd from M=1 query to
//      M=tree_size queries, all writing at the SAME logical decode position
//      (the cache append happens once for the committed prefix only). Each
//      candidate carries its own per-row causal mask describing which tree
//      ancestors it inherits from. Outputs per-candidate fp16 attention out.
//
//   2. rcpp_medusa_small_m_gemv  — small-M ternary GEMV for the head-dispatch
//      and verify path. M = tree_size (typically 16–64); the production GEMV
//      at kernels/ternary_gemv_phase5_halo.hip is M=1 tuned and re-streams
//      weights M times if you call it in a loop. This launcher reuses the
//      halo-1bit packed-ternary format + per-row scale layout 1:1 (so the
//      same weight tensors flow through both M=1 decode and M=tree verify).
//
// Status (2026-04-24, scope-pass)
// -------------------------------
// THIS PASS lands ONLY the C entry-point declarations + compile-clean stub
// .hip implementations that return RCPP_NOT_IMPLEMENTED. No working kernel
// math, no tree-attention online-softmax loop, no LDS staging. The next
// chunk (~3-5 day kernel-author job, see project_medusa_plan.md) fills in
// the bodies and adds the differential test harness.
//
// Both launchers obey the rest of the rocm-cpp ABI: stream-based sync only,
// no hipBLAS, no Python at runtime, wave32 / gfx1151 / RDNA 3.5 first-class.

#ifndef ROCM_CPP_MEDUSA_H
#define ROCM_CPP_MEDUSA_H

#include <stddef.h>
#include <stdint.h>

#include "rocm_cpp/bitnet_model.h"   // rcpp_bitnet_model_t, rcpp_weight_format_t
#include "rocm_cpp/ck_gemm.h"        // rcpp_status_t

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel returned by every entry point in this header in the scope-pass
// build. Distinct from RCPP_UNSUPPORTED (which means "shape rejected") —
// this one means "the kernel exists at the ABI level but the body is not
// authored yet." Once the kernels land this define moves into a follow-up
// commit that drops the early-return.
#ifndef RCPP_NOT_IMPLEMENTED
#define RCPP_NOT_IMPLEMENTED ((rcpp_status_t)100)
#endif

// Maximum tree size the launchers will accept once authored. Picked from
// the upstream Medusa-1 default (top-1 root + 4 heads × 10 expansions ≈ 41
// candidates rounded up to 64 for power-of-two LDS staging). Callers that
// exceed this get RCPP_INVALID_ARG in the future; in this scope-pass build
// the value is informational only.
#define RCPP_MEDUSA_MAX_TREE_SIZE 64

// -----------------------------------------------------------------------------
// Tree-parallel split-KV Flash-Decoding attention (decode-time verify).
//
// Kernel intent: same online-softmax shape as rcpp_kv_cache_attn_decode_fd,
// but Q now carries M=tree_size candidate query vectors that all attend to
// the SAME K/V cache (the cache hasn't yet been extended for any candidate —
// commit happens in the host after the verify decision). Each candidate has
// its own causal mask describing which previous tree positions it inherits
// from; positions NOT in a candidate's ancestor set are forced to -inf
// before the softmax.
//
// Layouts (kernel will accept once authored)
// ------------------------------------------
//   Q          : FP16  [tree_size, num_q_heads, head_dim]
//   K_cache    : FP16  [seq_len + tree_size, num_kv_heads, head_dim]
//                (caller stages all candidate K's at positions
//                 [seq_len .. seq_len+tree_size); committed prefix length is
//                 seq_len, the rest are speculative.)
//   V_cache    : FP16  [seq_len + tree_size, num_kv_heads, head_dim]
//   tree_mask  : uint8 [tree_size, tree_size]
//                row m has a 1 at column j iff candidate j is an ancestor
//                (including self) of candidate m. Off-tree (committed
//                prefix) positions are always attended by all candidates;
//                this mask only gates the speculative tail.
//   out        : FP16  [tree_size, num_q_heads, head_dim]
//
// Stream-based sync only; caller synchronizes on its own event/stream.
// Wave32 / gfx1151 first-class. Sherry / int8-KV variants will arrive in
// follow-up entry points (rcpp_medusa_tree_attn_decode_fd_i8 etc).
rcpp_status_t
rcpp_medusa_tree_attn_decode_fd(const void* Q_dev,
                                const void* K_dev,
                                const void* V_dev,
                                const void* tree_mask_dev,
                                void*       out_dev,
                                int         tree_size,
                                int         num_q_heads,
                                int         num_kv_heads,
                                int         head_dim,
                                int         seq_len,
                                float       scale,
                                void*       stream);

// -----------------------------------------------------------------------------
// Small-M ternary GEMV — Medusa head dispatch + verify GEMM front.
//
// Kernel intent: M activations × ternary weights → FP16 output, with
// halo-1bit packed weights (uint8 [N, (K+3)/4], 2 bits/value, codes
// 0→-1, 1→0, 2→+1, 3→0 reserved) and per-row FP32 scales — i.e. the SAME
// on-device buffers the M=1 path uses. The M=1 GEMV is bandwidth-saturating
// (92% of LPDDR5 peak) but reuses each weight u8 only once; calling it M
// times in a loop re-streams weights M times. This kernel fuses the M
// queries via LDS staging so each weight byte is loaded once and FMA'd
// against M activation lanes.
//
// Layouts (kernel will accept once authored)
// ------------------------------------------
//   x_i8           : INT8   [M, K]              row-major activations
//   x_scale        : FP32   scalar              per-tensor activation scale
//   w_packed_halo  : uint8  [N, (K + 3) / 4]    halo-1bit packed ternary
//   w_row_scales   : FP32   [N]                 per-output-row weight scale
//   y_fp16_out     : FP16   [M, N]              row-major output
//   M, N, K        : 1 ≤ M ≤ RCPP_MEDUSA_MAX_TREE_SIZE; K % 16 == 0;
//                    N % 64 == 0 (tile width — caller pads).
//
// This is a NEW launcher distinct from onebit_ternary_gemm_smallm_launch
// in src/ternary_gemm_smallm.hip — that one consumes the rocm-cpp halo
// 2-bit-per-u32 K-major pack used by the prefill path; this one consumes
// the halo-1bit byte-packed (4 ternaries / byte) layout used by the M=1
// decode GEMV path. Sharing the M=1 layout means the same weight memory
// region serves both decode and Medusa verify.
//
// Stream-based sync only.
rcpp_status_t
rcpp_medusa_small_m_gemv(const void* x_i8_dev,
                         float       x_scale,
                         const void* w_packed_halo_dev,
                         const void* w_row_scales_dev,
                         void*       y_fp16_out_dev,
                         int         M,
                         int         N,
                         int         K,
                         void*       stream);

// -----------------------------------------------------------------------------
// .h1b-medusa sidecar loader. See docs/h1b-medusa-format.md for the wire
// format. Two on-disk variants share the "H1BM" magic:
//
//   v1 (variant=0)  — per-head (vocab, hidden) ternary projection. Each head
//                     directly emits logits without sharing the base lm_head.
//                     Synthetic-zero writer used for kernel smoke tests.
//   v2 (variant=1)  — residual-MLP topology:
//                     `next_h = h + w_out @ SiLU(w_in @ h)`
//                     followed by the SHARED base.lm_head to emit logits.
//                     Mirrors parrishcorcoran/MedusaBitNet-2B-4T (4 heads,
//                     hidden=2560, w_in/w_out are dense [hidden, hidden]
//                     in fp16).
//
// The loader cross-checks `hidden_size` against the base model on both
// variants. v1 also cross-checks `vocab_size`; v2 ignores vocab_size (no
// per-head vocab tensor) — the engine reuses base->embedding_dev as the
// shared lm_head.

#define RCPP_MEDUSA_MAX_HEADS 16

// On-disk variant tag (matches docs/h1b-medusa-format.md §"variant").
typedef enum {
    RCPP_MEDUSA_VARIANT_VOCAB         = 0,  // v1: per-head (vocab, hidden) ternary
    RCPP_MEDUSA_VARIANT_RESIDUAL_MLP  = 1,  // v2: w_in + w_out + shared lm_head
} rcpp_medusa_variant_t;

// v2 weight dtype tag (matches docs/h1b-medusa-format.md §"weight_dtype").
typedef enum {
    RCPP_MEDUSA_DTYPE_BF16            = 0,
    RCPP_MEDUSA_DTYPE_FP16            = 1,
    RCPP_MEDUSA_DTYPE_HALO_V2_TERNARY = 2,
    RCPP_MEDUSA_DTYPE_SHERRY_I8       = 3,
} rcpp_medusa_dtype_t;

typedef struct {
    // -- v1 (RCPP_MEDUSA_VARIANT_VOCAB) --
    void*  packed_dev;       // weight bytes — same layout as the base
                             // ternary path identified by weight_format.
    float* row_scales_dev;   // [vocab_size] fp32 (nullptr for Bonsai inline-
                             // scale formats — Medusa v1 doesn't ship those
                             // but we leave the slot null for forward compat).

    // -- v2 (RCPP_MEDUSA_VARIANT_RESIDUAL_MLP) --
    // Both fp16 [hidden, hidden] dense matrices, row-major. The on-disk
    // dtype may be bf16/fp16/halo-v2/sherry; the loader either uploads
    // fp16 directly or casts host-side (bf16→fp16) before upload. Today
    // only bf16+fp16 dispatch lands on the runtime path; halo-v2 / sherry
    // tags are accepted by the parser but reserved for a follow-up.
    void*  w_in_dev;         // fp16 [hidden, hidden] when dtype<=FP16
    void*  w_out_dev;        // fp16 [hidden, hidden] when dtype<=FP16
} rcpp_medusa_head_t;

typedef struct {
    uint32_t              num_heads;
    uint32_t              hidden_size;
    uint32_t              vocab_size;       // v1 only; 0 on v2 sidecars
    rcpp_weight_format_t  weight_format;    // v1 only
    rcpp_medusa_variant_t variant;          // dispatch tag
    rcpp_medusa_dtype_t   v2_dtype;         // v2 only (bf16/fp16/halo/sherry)
    rcpp_medusa_head_t    heads[RCPP_MEDUSA_MAX_HEADS];
} rcpp_medusa_heads_t;

// Load a `.h1b-medusa` sidecar (v1 or v2). `base` must already be loaded —
// this routine cross-checks `base->hidden_size` against the header (and
// `base->vocab_size` for v1 only), and refuses on mismatch / wrong magic /
// unsupported version / truncated file. On success `*out` is populated
// with device pointers and `out->variant` set to the on-disk tag; on
// failure `*out` is left zero-initialized.
rcpp_status_t
rcpp_medusa_load_h1b_sidecar(const char* path,
                             const rcpp_bitnet_model_t* base,
                             rcpp_medusa_heads_t* out);

// Release device buffers allocated by rcpp_medusa_load_h1b_sidecar.
// Idempotent — safe to call on a zero-initialized heads struct.
void
rcpp_medusa_free_heads(rcpp_medusa_heads_t* heads);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_MEDUSA_H
