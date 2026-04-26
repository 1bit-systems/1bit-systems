//! GGUF tensor bit-unpack → halo v2 packed ternary.
//!
//! This sprint plugs the gap between [`crate::gguf`] (parse-only GGUF mmap)
//! and [`crate::h1b`]'s 2-bit ternary layout (`uint8[rows, (cols+3)/4]`
// (deleted: was 1bit-router orphan crate)
//!   `unimplemented!("IQ2_S …")`.
//!
//! ## IQ2_S block layout (for future grep)
//!
//! From `ggml/src/ggml-common.h` (llama.cpp, MIT / Apache-2.0):
//!
//! ```c
//! typedef struct {
//!     ggml_half d;                 // fp16 super-block scale                 ( 2 B)
//!     uint8_t qs[QK_K/4];          // 64 B: qs[0..32] = grid idx low byte,
//!                                  //       qs[32..64] = sign bitmaps (1 bit/weight)
//!     uint8_t qh[QK_K/32];         //  8 B: qh[ib32] holds 4×2 high bits of
//!                                  //       the grid index for the 4 lanes in
//!                                  //       sub-block `ib32`
//!     uint8_t scales[QK_K/32];     //  8 B: two 4-bit packed scales per byte
//!                                  //       (covering 16 weights each, together
//!                                  //       forming the 32-weight sub-block)
//! } block_iq2_s;                   // sizeof == 2 + 64 + 8 + 8 = 82 bytes
//! ```
//!
//! `QK_K = 256`. One super-block carries 256 weights organised as
//! 8 × 32-weight "ib32" sub-blocks, each further split into two 16-weight
//! half-blocks that share a 4-bit scale. Each 16-weight half-block is
//! itself split into **two** grid lanes of 8 weights, where each lane is
//! one 10-bit index into the 1024-entry [`iq2s_grid::IQ2S_GRID`] codebook.
//!
//! Dequant formula (per weight `j` in `[0..8)` within lane `l`):
//!
//! ```text
//!   dl       = d * (0.5 + ((scales[ib32] >> (4*(l/2))) & 0xf)) * 0.25
//!   grid_idx = qs[ib32*4 + l] | (((qh[ib32] >> (2*l)) & 0x3) << 8)
//!   grid     = IQ2S_GRID[grid_idx]                               // 8 unsigned bytes
//!   sign     = (signs[ib32*4 + l] >> j) & 1 ? -1.0 : +1.0
//!   y[j]     = dl * grid[j] * sign
//! ```
//!
//! Grid bytes are ∈ {0x08, 0x19, 0x2b} — three magnitudes, **never zero**.
//! That's why IQ2_S → halo v2 ternary requires a re-thresholding step:
//! IQ2_S can't express "exact zero" directly, so a BitNet model quantized
//! to IQ2_S encodes its zeros as the smallest magnitude (0x08) and lets
//! the quantizer place them near the rounding floor. We undo that here by
//! computing a per-super-block absmean and snapping any |w| < absmean/2 to
//! the ternary-zero code.
//!
//! ## halo v2 packed format (receiving end)
//!
//! `uint8[rows, (cols + 3)/4]` — 4 ternaries per byte, K-contiguous inside
//! byte. 2-bit code: `0 → -1`, `1 → 0`, `2 → +1` (3 is unused).
//! Per-row fp16 scale. See `rocm-cpp/kernels/ternary_gemv_phase5_halo.hip`
//! for the matching GPU-side unpack.
//!
//! ## Block-granularity "rows"
//!
//! `iq2_s_to_halo_v2` treats each IQ2_S super-block (256 weights) as one
//! halo "row" for scale purposes. The caller knows the real matrix row
//! dimension and can re-aggregate adjacent super-block scales into a
//! per-matrix-row scale before handing the tensor to a GEMV kernel. This
//! keeps the unpacker decoupled from matrix shape — it doesn't need
//! `(rows, cols)`, just `n_weights`.

use half::f16;

use super::iq2s_grid::IQ2S_GRID;
use crate::error::HaloError;

// ---------------------------------------------------------------------------
// Layout constants — match ggml-common.h for QK_K=256.
// ---------------------------------------------------------------------------

/// Super-block size in weights. Matches llama.cpp `QK_K`.
pub const IQ2_S_BLOCK_WEIGHTS: usize = 256;
/// Bytes per IQ2_S super-block on disk. `sizeof(ggml_half) + QK_K/4 + QK_K/16`.
pub const IQ2_S_BLOCK_BYTES: usize = 2 + 64 + 16;

/// halo v2 2-bit codes. Kept as module-level constants so the packer and
/// the GPU kernel share one ground truth.
const HALO_V2_NEG1: u8 = 0b00;
const HALO_V2_ZERO: u8 = 0b01;
const HALO_V2_POS1: u8 = 0b10;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decode `n_weights` ternary weights from an IQ2_S tensor into halo v2
/// 2-bit packing, producing one fp16 absmean scale per super-block.
///
/// * `raw` — contiguous IQ2_S blocks. `raw.len()` must be at least
///   `ceil(n_weights / 256) * 82`.
/// * `out` — destination buffer. Must be at least
///   `(n_weights + 3) / 4` bytes. Any tail bits inside the last byte
///   (because `n_weights % 4 != 0`) are zeroed to the ternary-zero code.
/// * `n_weights` — logical weight count. May be less than a full
///   super-block boundary; trailing weights within the last block are
///   **dropped** (not included in the absmean) and their output codes
///   are set to ternary-zero.
///
/// Returns one fp16 scale per super-block (length =
/// `ceil(n_weights / 256)`). The scale is the per-block absmean of the
/// dequantized fp32 values — the canonical BitNet quantization scale.
pub fn iq2_s_to_halo_v2(
    raw: &[u8],
    out: &mut [u8],
    n_weights: usize,
) -> Result<Vec<f16>, HaloError> {
    let n_blocks = n_weights.div_ceil(IQ2_S_BLOCK_WEIGHTS);
    let need_raw = n_blocks * IQ2_S_BLOCK_BYTES;
    if raw.len() < need_raw {
        return Err(HaloError::Truncated {
            offset: 0,
            needed: need_raw,
            have: raw.len(),
        });
    }
    let need_out = n_weights.div_ceil(4);
    if out.len() < need_out {
        return Err(HaloError::InvalidConfig(
            "halo v2 output buffer too small for n_weights",
        ));
    }

    // Always zero the output slice we will write into, so any tail bits
    // we skip (because n_weights isn't a multiple of 4) default to the
    // ternary-zero code (0b01) rather than -1 (0b00) — matching
    // "zero-pad" semantics.
    for b in out[..need_out].iter_mut() {
        // 0b01 01 01 01 = 0x55 — four ternary-zeros.
        *b = 0x55;
    }

    let mut scales = Vec::with_capacity(n_blocks);
    // Scratch buffer for the 256 fp32 dequantized values of one super-block.
    let mut y = [0f32; IQ2_S_BLOCK_WEIGHTS];

    for ib in 0..n_blocks {
        let block = &raw[ib * IQ2_S_BLOCK_BYTES..(ib + 1) * IQ2_S_BLOCK_BYTES];
        dequant_one_iq2_s_block(block, &mut y);

        // Per-block absmean.
        let mut sum_abs = 0f64;
        for &v in y.iter() {
            sum_abs += v.abs() as f64;
        }
        let absmean = (sum_abs / IQ2_S_BLOCK_WEIGHTS as f64) as f32;
        scales.push(f16::from_f32(absmean));

        // Threshold half the absmean — standard BitNet absmean ternarization.
        let thr = absmean * 0.5;

        // Only the first (n_weights - ib*256) values of this block are
        // "real"; the rest are padding and stay at ternary-zero.
        let base = ib * IQ2_S_BLOCK_WEIGHTS;
        let block_end = (base + IQ2_S_BLOCK_WEIGHTS).min(n_weights);
        for i in base..block_end {
            let w = y[i - base];
            let code = if thr > 0.0 && w.abs() >= thr {
                if w > 0.0 { HALO_V2_POS1 } else { HALO_V2_NEG1 }
            } else {
                HALO_V2_ZERO
            };
            set_halo_v2_code(out, i, code);
        }
    }

    Ok(scales)
}

/// IQ1_S unpack stub. **Not yet implemented** — the IQ1_S codebook is a
/// separate 2048-entry grid with its own delta/scale math, and would
/// roughly double the bundled codebook size (~16 KB of `const` table).
/// Defer until we have a BitNet model shipping as IQ1_S in the wild.
///
/// Callers should fall back to `.h1b` or IQ2_S for now.
pub fn iq1_s_to_halo_v2(
    _raw: &[u8],
    _out: &mut [u8],
    _n_weights: usize,
) -> Result<Vec<f16>, HaloError> {
    // TODO(1bit-core/unpack): port ggml-common.h `iq1s_grid` (2048 u64
    // entries, int8 values in {-1,0,+1}) + IQ1S_DELTA (0.125) + the
    // `d * (2*((qh>>12)&7) + 1)` scale formula. Most of the math is
    // *simpler* than IQ2_S (no absmean thresholding step — IQ1_S's grid
    // is already ternary), but the table bloat is the real blocker.
    Err(HaloError::InvalidConfig(
        "IQ1_S → halo v2 unpack not yet implemented (codebook not bundled)",
    ))
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Dequantize a single IQ2_S super-block (82 input bytes) into 256 fp32
/// weights. Faithful port of `dequantize_row_iq2_s` (ggml-quants.c).
fn dequant_one_iq2_s_block(block: &[u8], y: &mut [f32; 256]) {
    debug_assert_eq!(block.len(), IQ2_S_BLOCK_BYTES);

    // fp16 super-block scale (little-endian u16 → f16 → f32).
    let d_raw = u16::from_le_bytes([block[0], block[1]]);
    let d = f16::from_bits(d_raw).to_f32();

    // qs[0..32] = grid idx low bytes; qs[32..64] = sign bitmaps.
    let qs = &block[2..2 + 64];
    let qh = &block[66..66 + 8];
    let scales = &block[74..74 + 8];
    let signs = &qs[32..64];
    let qs_idx = &qs[0..32];

    for ib32 in 0..8 {
        // Two 4-bit scales for the two 16-weight halves of this ib32.
        let s_lo = (scales[ib32] & 0x0f) as u32;
        let s_hi = (scales[ib32] >> 4) as u32;
        let db_0 = d * (0.5 + s_lo as f32) * 0.25;
        let db_1 = d * (0.5 + s_hi as f32) * 0.25;

        for l in 0..4usize {
            // First two lanes use db_0, last two use db_1.
            let dl = if l < 2 { db_0 } else { db_1 };
            let qs_byte = qs_idx[ib32 * 4 + l] as u32;
            // High 2 bits of the grid index live in qh[ib32] at bit 2*l.
            let qh_bits = ((qh[ib32] as u32) >> (2 * l)) & 0x3;
            let grid_idx = qs_byte | (qh_bits << 8);
            let grid_u64 = IQ2S_GRID[grid_idx as usize];
            let grid_bytes = grid_u64.to_le_bytes();
            let sign_byte = signs[ib32 * 4 + l];

            let out_base = ib32 * 32 + l * 8;
            for j in 0..8usize {
                let mag = grid_bytes[j] as f32;
                let neg = (sign_byte >> j) & 1 != 0;
                let v = dl * mag;
                y[out_base + j] = if neg { -v } else { v };
            }
        }
    }
}

/// Write a single 2-bit halo v2 code at logical weight index `i`.
/// `out` is `uint8[n_weights.div_ceil(4)]`, 4 codes per byte,
/// K-contiguous — weight `i` lives at `out[i/4]`, bits `(i%4)*2 .. (i%4)*2+2`.
/// Caller is responsible for clearing the byte (to 0x55) before first write
/// — we write with a mask-and-or rather than assuming a zeroed slot.
#[inline]
fn set_halo_v2_code(out: &mut [u8], i: usize, code: u8) {
    let byte_idx = i >> 2;
    let shift = (i & 3) * 2;
    let mask = 0b11 << shift;
    out[byte_idx] = (out[byte_idx] & !mask) | ((code & 0b11) << shift);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one IQ2_S super-block whose dequantized output is exactly
    /// `y[j] = d * grid_byte * (-1)^sign_bit(j)` for every weight, with a
    /// caller-controlled grid index and sign byte. Scale nibbles are set
    /// so `db` ≡ `d` (i.e. `(0.5 + 1.5) * 0.25 = 0.5` — *not* 1.0; see
    /// note below).
    ///
    /// Math: `db = d * (0.5 + s4bit) * 0.25`. For `db = d`, we need
    /// `0.5 + s4bit = 4`, i.e. `s4bit = 3.5` — not representable. The
    /// closest integer (`s4bit = 4`) gives `db = 1.125 * d`. Good enough
    /// for proportional-magnitude tests.
    fn build_block(d: f16, grid_idx_low: u8, qh_bits: u8, sign_byte: u8) -> Vec<u8> {
        // Layout: d(2) + qs(64) + qh(8) + scales(8) = 82.
        let mut b = vec![0u8; IQ2_S_BLOCK_BYTES];
        b[0..2].copy_from_slice(&d.to_bits().to_le_bytes());
        // Every ib32 lane gets the same grid idx and sign byte.
        for ib32 in 0..8 {
            for l in 0..4 {
                b[2 + ib32 * 4 + l] = grid_idx_low;
            }
            for l in 0..4 {
                b[2 + 32 + ib32 * 4 + l] = sign_byte;
            }
            // qh: pack the same 2-bit `qh_bits` at each of the 4 lane slots.
            let qh = (qh_bits & 0x3)
                | ((qh_bits & 0x3) << 2)
                | ((qh_bits & 0x3) << 4)
                | ((qh_bits & 0x3) << 6);
            b[66 + ib32] = qh;
            // Both 4-bit scale nibbles = 4 → db ≈ 1.125*d (see note above).
            b[74 + ib32] = 0x44;
        }
        b
    }

    #[test]
    fn iq2s_dequant_matches_closed_form_and_ternarizes() {
        // Grid index 0 → 0x0808080808080808. 8 unsigned bytes all equal 0x08
        // (= 8 decimal). d = 1.0; sign_byte = 0x00 → all positive.
        // qh_bits = 0 → grid_idx = 0.
        let block = build_block(f16::from_f32(1.0), 0, 0, 0x00);

        // Dequant to scratch.
        let mut y = [0f32; 256];
        dequant_one_iq2_s_block(&block, &mut y);

        // Every weight should be db * 8 where db = 1.0 * (0.5 + 4) * 0.25 = 1.125.
        // So y[*] = 1.125 * 8.0 = 9.0.
        let expected = 1.125 * 8.0;
        for (i, &v) in y.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "y[{i}] = {v}, expected {expected}"
            );
        }

        // Now re-run via the public API. All weights are +expected, so
        // absmean = expected and thr = expected/2. Every weight is ≥ thr
        // and positive → every ternary code should be +1 (0b10).
        let mut out = vec![0u8; 64]; // 256 weights / 4 = 64 bytes
        let scales = iq2_s_to_halo_v2(&block, &mut out, 256).unwrap();
        assert_eq!(scales.len(), 1);
        assert!(
            (scales[0].to_f32() - expected).abs() < 1e-2,
            "scale = {}, expected ≈ {expected}",
            scales[0].to_f32()
        );
        // Every 2-bit code = 0b10 → byte = 0b10_10_10_10 = 0xAA.
        assert!(
            out.iter().all(|&b| b == 0xAA),
            "expected all 0xAA, got {:x?}",
            &out[..8]
        );
    }

    #[test]
    fn iq2s_mixed_signs_produce_signed_codes() {
        // sign_byte = 0b01010101 → alternating negate / keep within each
        // 8-weight lane. With all magnitudes equal, absmean = expected and
        // thr = expected/2, so every weight crosses the threshold and
        // becomes ±1 according to the sign bit.
        let block = build_block(f16::from_f32(1.0), 0, 0, 0b0101_0101);

        let mut out = vec![0u8; 64];
        let _ = iq2_s_to_halo_v2(&block, &mut out, 256).unwrap();

        // For sign byte 0b01010101, bits 0,2,4,6 are set (negate) and
        // bits 1,3,5,7 are clear (keep). So within each 8-weight lane:
        //   j=0: neg (code 0b00), j=1: pos (code 0b10),
        //   j=2: neg (0b00), j=3: pos (0b10),
        //   j=4: neg (0b00), j=5: pos (0b10),
        //   j=6: neg (0b00), j=7: pos (0b10).
        // Packed 4-per-byte: bytes alternate
        //   byte 0 = [j=0,1,2,3] = 0b10_00_10_00 = 0x88
        //   byte 1 = [j=4,5,6,7] = 0b10_00_10_00 = 0x88
        // ...and so on, so every byte should be 0x88.
        assert!(
            out.iter().all(|&b| b == 0x88),
            "expected 0x88 pattern from alternating-sign lane, got {:x?}",
            &out[..8]
        );
    }

    #[test]
    fn iq2s_zero_padding_for_non_aligned_n_weights() {
        // Grid idx 0 (all-0x08 bytes), d = 1.0, no signs → every dequant
        // value is +9.0. Run with n_weights = 256*1 + 1 = 257, which needs
        // 2 super-blocks but only 257 "real" weights.
        //
        // Expected output layout:
        //   * out[0..64]   — block 0 fully packed as 0xAA (all +1).
        //   * out[64]      — one +1 code in slot 0, the other three slots
        //                    zero-padded → 0b01_01_01_10 = 0x56.
        //   * scales.len() == 2 (one per super-block).
        //
        // Only the first block's 257-th weight (== block 1, lane 0, j=0)
        // is "real"; the remaining 255 weights of block 1 get the
        // ternary-zero fallback code.
        let block0 = build_block(f16::from_f32(1.0), 0, 0, 0x00);
        let mut raw = Vec::with_capacity(2 * IQ2_S_BLOCK_BYTES);
        raw.extend_from_slice(&block0);
        raw.extend_from_slice(&block0);

        let mut out = vec![0u8; 257_usize.div_ceil(4)]; // = 65 bytes
        let scales = iq2_s_to_halo_v2(&raw, &mut out, 257).unwrap();

        assert_eq!(scales.len(), 2, "one scale per super-block");
        // First 64 bytes are fully packed +1 → 0xAA.
        assert!(
            out[..64].iter().all(|&b| b == 0xAA),
            "block-0 payload should be all 0xAA"
        );
        // Byte 64: slot 0 = +1 (0b10), slots 1..3 zero-pad (0b01 each).
        // Packing: bit0-1 slot0, bit2-3 slot1, bit4-5 slot2, bit6-7 slot3.
        //   0b01_01_01_10 = 0x56.
        assert_eq!(
            out[64], 0x56,
            "tail byte should be 0x56 (one +1 + three zero-pads)"
        );
    }

    #[test]
    fn iq2s_rejects_short_raw_input() {
        // 1 block of weights requested but raw has only 40 bytes (half a block).
        let raw = vec![0u8; 40];
        let mut out = vec![0u8; 64];
        let err = iq2_s_to_halo_v2(&raw, &mut out, 256).unwrap_err();
        assert!(
            matches!(err, HaloError::Truncated { .. }),
            "expected Truncated, got {err:?}"
        );
    }

    #[test]
    fn iq2s_rejects_short_output_buffer() {
        let block = build_block(f16::from_f32(1.0), 0, 0, 0x00);
        let mut out = vec![0u8; 10]; // way too small for 256 weights
        let err = iq2_s_to_halo_v2(&block, &mut out, 256).unwrap_err();
        assert!(matches!(err, HaloError::InvalidConfig(_)));
    }

    #[test]
    fn iq1s_is_still_todo() {
        // Surface guarantee: the function exists and fails loudly.
        let err = iq1_s_to_halo_v2(&[], &mut [], 0).unwrap_err();
        assert!(
            matches!(err, HaloError::InvalidConfig(m) if m.contains("IQ1_S")),
            "expected InvalidConfig(\"IQ1_S …\"), got {err:?}"
        );
    }

    #[test]
    fn halo_v2_code_packing_is_k_contiguous() {
        // Smoke-test the low-level bit packer directly. 8 weights:
        //   i=0 +1, i=1 0, i=2 -1, i=3 +1, i=4 -1, i=5 0, i=6 +1, i=7 -1.
        let mut out = vec![0x55u8; 2]; // both bytes start all-zero codes
        set_halo_v2_code(&mut out, 0, HALO_V2_POS1);
        set_halo_v2_code(&mut out, 1, HALO_V2_ZERO);
        set_halo_v2_code(&mut out, 2, HALO_V2_NEG1);
        set_halo_v2_code(&mut out, 3, HALO_V2_POS1);
        set_halo_v2_code(&mut out, 4, HALO_V2_NEG1);
        set_halo_v2_code(&mut out, 5, HALO_V2_ZERO);
        set_halo_v2_code(&mut out, 6, HALO_V2_POS1);
        set_halo_v2_code(&mut out, 7, HALO_V2_NEG1);
        // byte 0 = slots 0..3 low→high: 0b10, 0b01, 0b00, 0b10
        //        = 0b10_00_01_10 = 0x86
        // byte 1 = slots 4..7 low→high: 0b00, 0b01, 0b10, 0b00
        //        = 0b00_10_01_00 = 0x24
        assert_eq!(out, vec![0x86, 0x24]);
    }
}
