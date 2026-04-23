//! Sherry 1.25-bit row packer.
//!
//! Matches the layout documented in `rocm-cpp/include/rocm_cpp/sherry.h` and
//! implemented in `rocm-cpp/tools/bench_sherry.cpp::pack_sherry_v3`.
//!
//! Bit layout of the 5-bit code for one group of 4 weights:
//!
//! ```text
//!   code = (zero_pos << 3) | signs_field
//!   signs_field has bit[i] = 1 if the i-th non-zero lane (positional
//!   order, skipping zero_pos) is +1, else 0.
//! ```
//!
//! Groups are stored LSB-first across row bytes. Group `g`'s code occupies
//! bit indices `[5*g, 5*g + 5)` of the packed row; row length is
//! `cols * 5 / 32` bytes, `cols` a multiple of 32 (enforced by caller,
//! matches `H1bWeightFormat::SherryV3.row_bytes`).

use std::convert::TryInto;

/// Pick the "zero position" lane within a group of 4 ternary weights.
///
/// Rule:
///
///   * Pick the lane with **smallest sign-magnitude** (ternary ∈ {-1,0,+1}
///     so 0 always wins, ±1 tie at magnitude 1).
///   * Ties broken by **lower index**.
///
/// This matches the task spec and — when fed any ternary group including
/// groups with 0 zeros or ≥ 2 zeros — produces a deterministic zero
/// choice. The caller is expected to force that lane to zero in the
/// emitted group (`pack_sherry_row` does this); the bench kernel's
/// fallback described in `sherry.h` won't trip on our output because we
/// always emit exactly one zero.
#[inline]
pub fn pick_zero_pos(group: &[i8; 4]) -> u8 {
    let mut best = 0u8;
    let mut best_mag = group[0].unsigned_abs();
    for (i, &v) in group.iter().enumerate().skip(1) {
        let mag = v.unsigned_abs();
        if mag < best_mag {
            best = i as u8;
            best_mag = mag;
        }
        // ties: keep `best` (lower index wins)
    }
    best
}

/// Encode one group of 4 into its 5-bit code (`zero_pos` in the top 2
/// bits, `signs_field` in the low 3 bits).
///
/// The chosen `zero_pos` lane is **treated as a zero** regardless of its
/// input value — this is the lossy step (at most one ±1 lane per group
/// can be force-zeroed, capping sign-change rate at 25%).
///
/// Other lanes contribute their sign as-is: ternary `+1` → sign bit `1`,
/// ternary `-1` → sign bit `0`, ternary `0` on a non-zero-pos lane → sign
/// bit `1` ("treated as +1", matching `pack_sherry_v3`'s spec). Callers
/// that want accuracy-safe behaviour should avoid emitting groups with
/// multiple zeros; the zero-pos picker above minimises this by always
/// choosing an existing zero if present.
#[inline]
pub fn encode_group(group: &[i8; 4], zero_pos: u8) -> u8 {
    debug_assert!(zero_pos < 4);
    let mut signs_field: u8 = 0;
    let mut sign_idx: u8 = 0;
    for (p, &v) in group.iter().enumerate() {
        if p as u8 == zero_pos {
            continue;
        }
        // v == 1 → bit 1; v == -1 → bit 0; v == 0 (shouldn't happen for a
        // valid 3:4-sparse group with the chosen zero_pos, but if it does
        // we follow pack_sherry_v3's convention and treat it as +1).
        let bit: u8 = if v == 1 || v == 0 { 1 } else { 0 };
        signs_field |= bit << sign_idx;
        sign_idx += 1;
    }
    (zero_pos << 3) | (signs_field & 0b111)
}

/// Number of Sherry-packed bytes for a row of `cols` ternary weights.
/// Mirrors `H1bWeightFormat::SherryV3.row_bytes`; `cols % 32 == 0` required.
#[inline]
pub fn sherry_row_bytes(cols: usize) -> usize {
    assert_eq!(cols % 32, 0, "Sherry row requires cols % 32 == 0");
    cols * 5 / 32
}

/// Counter returned by [`pack_sherry_row`] so the caller (and round-trip
/// tests) can gate on the structured-sparsity budget.
#[derive(Debug, Default, Clone, Copy)]
pub struct PackRowStats {
    /// How many groups in the row were emitted with `zero_pos` pointing
    /// at a lane that was originally ±1 (i.e. we flipped a ±1 to zero).
    /// Each such group changes at most one position's sign; upper bound
    /// is `groups = cols / 4`. Division by cols gives the "fraction of
    /// positions changed sign", which the task spec caps at 12%.
    pub forced_zero_flips: u32,
}

/// Pack one row of `cols` ternary weights into Sherry's 5-bit-per-group
/// layout. `cols` must be a multiple of 4 (group size); the byte layout
/// additionally requires `cols % 32 == 0` so every group boundary is
/// byte-aligned modulo the row stride (same as the kernel's
/// early-return guard).
///
/// `packed` must be `sherry_row_bytes(cols)` bytes long. It is written
/// in full (`memset` + per-group `or`), so callers can reuse a buffer.
pub fn pack_sherry_row(ternary: &[i8], packed: &mut [u8], cols: usize) -> PackRowStats {
    assert_eq!(ternary.len(), cols, "pack_sherry_row: ternary.len mismatch");
    assert_eq!(
        packed.len(),
        sherry_row_bytes(cols),
        "pack_sherry_row: packed.len mismatch",
    );
    assert_eq!(cols % 4, 0, "pack_sherry_row: cols must be mult of 4");

    packed.fill(0);
    let groups = cols / 4;
    let mut stats = PackRowStats::default();

    for g in 0..groups {
        let start = g * 4;
        let group: &[i8; 4] = ternary[start..start + 4]
            .try_into()
            .expect("slice is 4 elements by construction");

        let zero_pos = pick_zero_pos(group);
        if group[zero_pos as usize] != 0 {
            // We're flipping a ±1 lane to zero — lossy step.
            stats.forced_zero_flips += 1;
        }
        let code = encode_group(group, zero_pos);

        let bit_pos = 5 * g;
        let byte_idx = bit_pos >> 3;
        let shift = (bit_pos & 7) as u32;
        packed[byte_idx] |= code.checked_shl(shift).unwrap_or(0);
        // Overflow into next byte when a 5-bit code straddles a byte boundary.
        if shift + 5 > 8 {
            packed[byte_idx + 1] |= code >> (8 - shift);
        }
    }

    stats
}

/// Scalar reference decoder — extracts one group's 5-bit code given a
/// row's packed bytes. Used by round-trip tests to reconstruct ternary
/// values from a Sherry row without going through the GPU kernel.
#[inline]
pub fn decode_group(packed: &[u8], g: usize) -> (u8, u8) {
    let bit_pos = 5 * g;
    let byte_idx = bit_pos >> 3;
    let shift = (bit_pos & 7) as u32;
    let lo = (packed[byte_idx] >> shift) & ((1u8 << (8 - shift).min(5)) - 1);
    let code = if shift + 5 > 8 {
        let hi_bits = (shift + 5) - 8;
        let hi = packed[byte_idx + 1] & ((1u8 << hi_bits) - 1);
        lo | (hi << (8 - shift))
    } else {
        lo & 0b11111
    };
    let zero_pos = (code >> 3) & 0b11;
    let signs_field = code & 0b111;
    (zero_pos, signs_field)
}

/// Scalar reference for a single row: walk each group, decode zero_pos
/// and signs, rebuild the 4-lane ternary vector. Used by the round-trip
/// unit test.
pub fn unpack_sherry_row(packed: &[u8], out: &mut [i8], cols: usize) {
    assert_eq!(out.len(), cols);
    assert_eq!(packed.len(), sherry_row_bytes(cols));
    let groups = cols / 4;
    for g in 0..groups {
        let (zero_pos, signs_field) = decode_group(packed, g);
        let mut sign_idx: u8 = 0;
        for p in 0..4u8 {
            let dst = g * 4 + p as usize;
            if p == zero_pos {
                out[dst] = 0;
            } else {
                let bit = (signs_field >> sign_idx) & 1;
                out[dst] = if bit == 1 { 1 } else { -1 };
                sign_idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_zero_pos_prefers_zero() {
        assert_eq!(pick_zero_pos(&[1, -1, 0, 1]), 2);
        assert_eq!(pick_zero_pos(&[0, -1, 0, 1]), 0, "ties → lowest index");
        assert_eq!(
            pick_zero_pos(&[1, -1, 1, -1]),
            0,
            "all ±1 → lowest index wins",
        );
    }

    #[test]
    fn encode_decode_group_basic() {
        // Group [+1, 0, -1, +1], zero_pos=1 → surviving lanes [+1, -1, +1]
        // sign bits LSB-first: [1, 0, 1] = 0b101 = 5
        // code = (1 << 3) | 5 = 0b01101 = 13
        let code = encode_group(&[1, 0, -1, 1], 1);
        assert_eq!(code, 0b01_101);
    }

    #[test]
    fn pack_unpack_row_round_trip() {
        // 32 cols = 8 groups. Build a row where every group has one zero —
        // then unpack MUST match exactly (zero forced-flips).
        let mut ternary = vec![0i8; 32];
        for g in 0..8 {
            let base = g * 4;
            // Deterministic but varied: rotate the zero through lanes, and
            // vary ±1 signs on the remaining lanes.
            ternary[base + (g % 4)] = 0;
            let nonzero_vals = [1i8, -1, 1];
            let mut idx = 0;
            for p in 0..4 {
                if p != g % 4 {
                    ternary[base + p] = nonzero_vals[idx % 3];
                    idx += 1;
                }
            }
        }
        let mut packed = vec![0u8; sherry_row_bytes(32)];
        let stats = pack_sherry_row(&ternary, &mut packed, 32);
        assert_eq!(stats.forced_zero_flips, 0, "every group already has a zero");

        let mut unpacked = vec![0i8; 32];
        unpack_sherry_row(&packed, &mut unpacked, 32);
        assert_eq!(
            unpacked, ternary,
            "round-trip must be exact for valid 3:4 rows"
        );
    }
}
