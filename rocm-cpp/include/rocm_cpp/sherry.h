// rocm-cpp — Sherry 1.25-bit ternary GEMV C API (clean-room fp16 path).
//
// This is a standalone entry point distinct from the halo-1bit v3 format used
// by bitnet_decode (see ck_gemm.h → rcpp_ternary_gemv_sherry_f16). That path
// takes INT8 activations + per-row FP32 scales and is tuned for the live
// decode loop. The API exposed here is the minimal fp16-in/fp16-out kernel
// that matches the Sherry paper's 1.25-bpw format 1:1, for use by:
//   - the Rust wrapping layer (1bit-hip) that wants a no-scale bare kernel
//   - the differential test harness in tests/test_sherry_gemv.cpp
//   - offline tooling / microbench that doesn't need per-row scales
//
// Packing spec (matches project_sherry_spike.md):
//   Every 4 consecutive weights along K form one group. One position in the
//   group is forced to zero; the other three are ±1 signs. 5 bits encode the
//   group: [zero_pos : 2][signs : 3]. signs are MSB→LSB in positional order,
//   skipping zero_pos — so signs_bit[i] is the sign of the i-th surviving
//   (non-zero) lane.
//
//   Per row: K_in / 4 groups × 5 bits = K_in * 5 / 8 bits = K_in * 5 / 32
//   bytes (K_in must be a multiple of 32 for byte-alignment; enforced at the
//   launcher — short K returns early with no work). Total weight buffer is
//   N_out * K_in * 5 / 32 bytes, which the task spec writes as
//   (N_out * K_in / 4) * 5 / 8 bytes (same value).
//
// No per-row scale is baked into this API — the caller multiplies at its own
// level if needed. This keeps the kernel a pure signed-sum and matches the
// "(1.25 bpw weights) × (fp16 acts) → fp16 out" model that the differential
// test validates.
//
// Build: symbols land in librocm_cpp.so. No hipBLAS, no CK, no Python.

#ifndef ROCM_CPP_SHERRY_H
#define ROCM_CPP_SHERRY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// HIP-accelerated Sherry ternary GEMV.
//
//   packed_weights : const uint8_t*, device
//                    Row-major weights, row stride = K_in * 5 / 32 bytes.
//                    Each row contains (K_in / 4) 5-bit groups, packed LSB-
//                    first across consecutive bytes (group g's bits begin at
//                    bit index g*5 within row bytes).
//   act_fp16       : const uint16_t*, device, length K_in. __half bit pattern.
//   out_fp16       : uint16_t*, device, length N_out. __half bit pattern.
//   N_out, K_in    : row count and input dimension. K_in must be a multiple
//                    of 4 (group size); launcher early-returns otherwise.
//                    For byte-aligned rows the launcher additionally requires
//                    K_in % 32 == 0.
//   stream         : hipStream_t (as void* to keep the header HIP-free).
//
// Stream-based sync only; launcher issues the kernel and returns. Caller
// synchronizes on its own event / stream.
void sherry_ternary_gemv_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Scalar reference launcher — one HIP thread per row. Bit-identical fp32
// accumulation order as the fast kernel (both walk K_in groups in ascending
// order, both accumulate `sign * fp32(act)` into a single fp32 register).
// Exists for differential testing; do NOT benchmark.
void sherry_ternary_gemv_scalar_ref_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Same packing + activation contract as `sherry_ternary_gemv_launch`, but
// folds in the halo-1bit v3 per-row fp32 scale before the fp16 clamp +
// cast. Required when the caller has the `[N_out] f32` scales tensor
// emitted by `tools/h1b-sherry/` (Rust requantizer) or
// `tools/h1b_repack_sherry.cpp` (C++ requantizer); without this fold-in
// the output magnitudes are off by the absmean factor and downstream
// RMSNorm / softmax silently produce garbage.
//
//   row_scales_fp32 : const float*, device, length N_out (one per output
//                     row). Pass `nullptr` to opt out — the kernel then
//                     reduces to the pure signed-sum behavior of
//                     `sherry_ternary_gemv_launch`.
//
// All other parameters and constraints match `sherry_ternary_gemv_launch`
// exactly. Single kernel under the hood; the scale fold-in is one fmul
// on lane 0 of the writeback, no extra LDS / register pressure.
void sherry_ternary_gemv_with_scales_launch(
    const uint8_t*  packed_weights,
    const uint16_t* act_fp16,
    const float*    row_scales_fp32,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream);

// Host-side packer. Input is a contiguous int8 array of `count` ternary
// values in {-1, 0, +1}, laid out as consecutive groups of 4. For each group,
// the caller is responsible for ensuring exactly one lane is 0 (Sherry's
// structured-sparsity contract); if a group has 0 or ≥2 zeros this function
// packs deterministically by picking the LOWEST-INDEX zero as zero_pos and
// treating any other zeros as +1 (their weight contribution is then wrong —
// this is a caller bug, not a packer bug; the zero-choice heuristic lives in
// the requantizer).
//
//   ternary : const int8_t*, host, length = count
//   out     : uint8_t*, host, length = count * 5 / 32 (count must be mult. of
//             32 for byte-aligned output; short input writes the whole-byte
//             prefix and leaves the trailing partial byte untouched).
//   count   : number of ternary weights to pack. Must be a multiple of 4.
//
// Pure C / C++, no HIP. Safe to call from any thread.
void rcpp_sherry_pack(const int8_t* ternary, uint8_t* out, int count);

#ifdef __cplusplus
}
#endif
#endif  // ROCM_CPP_SHERRY_H
