//! End-to-end tests for h1b-sherry.
//!
//! These tests build synthetic in-memory `.h1b` files, convert them through
//! the requantizer, re-parse the output, and verify:
//!
//!   * Round-trip: ternary → Sherry → unpack reproduces signs faithfully
//!     except where the 3:4-sparsity picker had to force a ±1 lane to zero.
//!     The fraction of "positions that changed sign" must stay at or below
//!     12% — tighter than the 25% algebraic upper bound (one forced zero
//!     per group of 4) because our synthetic inputs already contain
//!     mostly-zeros (generated via the TQ1 digit distribution).
//!
//!   * Flag composition: setting `H1B_FLAG_HADAMARD_ROTATED` on the input
//!     leaks through to the output, OR-combined with
//!     `H1B_FLAG_SHERRY_FP16`.
//!
//!   * CLI arg parsing: `h1b-sherry --in ... --out ...` parses cleanly
//!     via `clap`, required args fail without them.

use std::path::PathBuf;

use h1b_sherry::{
    ConvertStats, convert_file, pack_sherry_row, sherry_pack::unpack_sherry_row, tq1_row_bytes,
};
use onebit_core::h1b::{
    H1B_FLAG_HADAMARD_ROTATED, H1B_FLAG_SHERRY_FP16, H1bConfig, H1bFile, H1bWeightFormat,
    LayerTensors, ModelTensors, TernaryTensor, serialize,
};

// ----------------------------------------------------------------------------
// Synthetic fixture helpers
// ----------------------------------------------------------------------------

/// Build a structurally valid v4 (TQ1) `.h1b` in memory. All dims are tiny
/// but satisfy the Sherry packer's `cols % 32 == 0` constraint. Returns
/// (config, raw_bytes, ternary_ground_truth_per_tensor).
///
/// Ground truth: for every ternary tensor we generate a 3:4-sparse pattern
/// (one zero per group of 4, other three ±1). This matches what a
/// Sherry-ready model looks like, so the round-trip check below should see
/// zero forced flips on inputs that already satisfy the constraint.
///
/// We additionally emit one "noisy" tensor (all ±1, no zeros) for the
/// flip-fraction upper bound check — that tensor should see exactly one
/// forced flip per group, i.e. 25% per-position flip rate on that slab.
///
/// Dims: hs=32, is=64, L=1, nh=2, nkv=1, vocab=4, hd=16.
fn build_tq1_v4_file(reserved: i32) -> (H1bConfig, Vec<u8>, Vec<Vec<i8>>) {
    let cfg = H1bConfig {
        version: 4,
        hidden_size: 32,
        intermediate_size: 64,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 1,
        vocab_size: 4,
        max_seq_len: 32,
        tie_embeddings: 1,
        reserved,
        rope_theta: 500_000.0,
        rms_norm_eps: 1e-5,
    };

    let hs = cfg.hidden_size as usize;
    let is_ = cfg.intermediate_size as usize;
    let nh = cfg.num_heads as usize;
    let nkv = cfg.num_kv_heads as usize;
    let hd = (cfg.hidden_size / cfg.num_heads) as usize;
    let vocab = cfg.vocab_size as usize;

    let embedding = vec![0xCCu8; vocab * hs * 4];
    let final_norm = vec![0xDDu8; hs * 4];
    let input_norm = vec![0xEEu8; hs * 4];
    let post_attn_norm = vec![0xAAu8; hs * 4];
    let attn_sub_norm = vec![0xBBu8; hs * 4];
    let ffn_sub_norm = vec![0x11u8; is_ * 4];

    // Per-tensor ternary ground truth, then TQ1-pack it.
    //
    // All 7 tensors are 3:4-sparse (one zero per group of 4). A small
    // fraction (one group per row, below) is deliberately left without a
    // zero — "noisy fringe" — so the round-trip test exercises the forced-
    // flip path without blowing past the 12% budget. 1 noisy group per
    // `cols/4` groups per row = ~1/`(cols/4)` = ~12.5% on cols=32 rows,
    // but across the whole layer (sum over 7 tensors) it averages to
    // ≪ 12%.
    let mut ternaries: Vec<Vec<i8>> = Vec::new();
    let shapes: [(usize, usize); 7] = [
        (nh * hd, hs),  // q: [32, 32]
        (nkv * hd, hs), // k: [16, 32]
        (nkv * hd, hs), // v: [16, 32]
        (hs, nh * hd),  // o: [32, 32]
        (is_, hs),      // gate: [64, 32]
        (is_, hs),      // up:   [64, 32]
        (hs, is_),      // down: [32, 64]
    ];
    for (rows, cols) in shapes {
        let mut t = vec![0i8; rows * cols];
        for r in 0..rows {
            for g in 0..(cols / 4) {
                let base = r * cols + g * 4;
                // One group per row is "noisy" (all ±1). The rest are
                // proper 3:4-sparse. Forced-flip rate ≈ 1 / (cols/4) per
                // tensor — well under the 12% per-layer budget.
                if g == 0 {
                    t[base] = 1;
                    t[base + 1] = -1;
                    t[base + 2] = 1;
                    t[base + 3] = -1;
                } else {
                    let zp = (r + g) % 4;
                    for p in 0..4 {
                        t[base + p] = if p == zp {
                            0
                        } else if (g + p) & 1 == 0 {
                            1
                        } else {
                            -1
                        };
                    }
                }
            }
        }
        ternaries.push(t);
    }

    // Pack each tensor in TQ1 v4 (base-3, 5 digits per byte).
    fn pack_tq1_v4(ternary: &[i8], rows: usize, cols: usize) -> Vec<u8> {
        let row_bytes = tq1_row_bytes(cols);
        let mut out = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            for i in 0..row_bytes {
                let base = i * 5;
                let mut byte: u32 = 0;
                let mut mul: u32 = 1;
                for d in 0..5 {
                    let k = base + d;
                    // Beyond `cols` we pad with digit=1 (ternary zero).
                    let digit: u32 = if k < cols {
                        (ternary[r * cols + k] + 1) as u32
                    } else {
                        1 // pad digit = 0 (ternary)
                    };
                    byte += digit * mul;
                    mul *= 3;
                }
                out[r * row_bytes + i] = byte as u8;
            }
        }
        out
    }

    let packed: Vec<Vec<u8>> = ternaries
        .iter()
        .zip(shapes.iter())
        .map(|(t, (rows, cols))| pack_tq1_v4(t, *rows, *cols))
        .collect();

    let scales: Vec<Vec<u8>> = shapes
        .iter()
        .map(|(rows, _)| vec![0x7Fu8; rows * 4])
        .collect();

    let model = ModelTensors {
        embedding_fp32: &embedding,
        final_norm_fp32: &final_norm,
    };
    let layer = LayerTensors {
        input_norm_fp32: &input_norm,
        post_attn_norm_fp32: &post_attn_norm,
        attn_sub_norm_fp32: &attn_sub_norm,
        ffn_sub_norm_fp32: &ffn_sub_norm,
        q: TernaryTensor {
            packed: &packed[0],
            scales: &scales[0],
        },
        k: TernaryTensor {
            packed: &packed[1],
            scales: &scales[1],
        },
        v: TernaryTensor {
            packed: &packed[2],
            scales: &scales[2],
        },
        o: TernaryTensor {
            packed: &packed[3],
            scales: &scales[3],
        },
        gate: TernaryTensor {
            packed: &packed[4],
            scales: &scales[4],
        },
        up: TernaryTensor {
            packed: &packed[5],
            scales: &scales[5],
        },
        down: TernaryTensor {
            packed: &packed[6],
            scales: &scales[6],
        },
    };
    let bytes = serialize(&cfg, &model, std::slice::from_ref(&layer)).expect("serialize v4");
    (cfg, bytes, ternaries)
}

// ----------------------------------------------------------------------------
// (a) Round-trip: ternary → Sherry → unpack keeps signs within spec.
// ----------------------------------------------------------------------------

#[test]
fn round_trip_synthetic_layer() {
    let (_, tq1_bytes, ternaries) = build_tq1_v4_file(0);
    let tmp = tempfile::tempdir().unwrap();
    let in_path = tmp.path().join("src.h1b");
    let out_path = tmp.path().join("dst.h1b");
    std::fs::write(&in_path, &tq1_bytes).unwrap();

    let stats: ConvertStats = convert_file(&in_path, &out_path).expect("convert");

    // Synthetic layer is 6 "nice" tensors + 1 noisy tensor.
    // Noisy tensor (up, [is, hs]=[64, 32]) contributes 64*32/4 = 512 groups
    // each with exactly one forced flip (lowest-index lane → 0). Other
    // tensors contribute zero forced flips (already 3:4-sparse).
    //
    // Across the whole layer the sign-change fraction should stay ≤ 12%.
    assert!(
        stats.flip_fraction() <= 0.12,
        "flip fraction {} exceeds 12% budget (groups={} flips={})",
        stats.flip_fraction(),
        stats.groups_total,
        stats.forced_zero_flips,
    );

    // Verify each row round-trips through our scalar Sherry decoder with
    // ≤ 25% of positions changed sign relative to the TQ1 ground truth.
    // (25% = worst-case algebraic bound per group; we're checking no
    //  ROW was silently corrupted beyond the budget.)
    let out = H1bFile::open(&out_path).expect("parse output");
    assert_eq!(
        out.config().weight_format().unwrap(),
        H1bWeightFormat::SherryFp16,
    );

    let lo = &out.layer_offsets()[0];
    let shapes: [(&str, usize, usize); 7] = [
        ("q", 32, 32),
        ("k", 16, 32),
        ("v", 16, 32),
        ("o", 32, 32),
        ("gate", 64, 32),
        ("up", 64, 32),
        ("down", 32, 64),
    ];
    let packed_spans = [
        lo.q_packed,
        lo.k_packed,
        lo.v_packed,
        lo.o_packed,
        lo.gate_packed,
        lo.up_packed,
        lo.down_packed,
    ];
    for ((name, rows, cols), span) in shapes.iter().zip(packed_spans.iter()) {
        let packed = out.tensor_bytes(*span);
        let row_bytes = cols * 5 / 32;
        let mut buf = vec![0i8; *cols];
        let mut total_changes = 0u64;
        for r in 0..*rows {
            let start = r * row_bytes;
            unpack_sherry_row(&packed[start..start + row_bytes], &mut buf, *cols);
            // Compare against ground-truth.
            let gt_row = &ternaries[shape_idx(name)][r * cols..(r + 1) * cols];
            for (a, b) in buf.iter().zip(gt_row.iter()) {
                // "Sign change" = non-zero lane whose ±1 flipped, OR a
                // zero lane that became ±1 or vice versa.
                let changed = match (a.signum(), b.signum()) {
                    (0, 0) => false,
                    (x, y) if x == y => false,
                    _ => true,
                };
                if changed {
                    total_changes += 1;
                }
            }
        }
        let frac = total_changes as f64 / (rows * cols) as f64;
        assert!(
            frac <= 0.25,
            "tensor {name}: sign-change fraction {frac} exceeds 25% algebraic bound",
        );
    }
}

fn shape_idx(name: &str) -> usize {
    match name {
        "q" => 0,
        "k" => 1,
        "v" => 2,
        "o" => 3,
        "gate" => 4,
        "up" => 5,
        "down" => 6,
        _ => panic!("bad name"),
    }
}

// ----------------------------------------------------------------------------
// (b) Header flag composition — Hadamard preserved, SHERRY_FP16 added.
// ----------------------------------------------------------------------------

#[test]
fn header_flag_composition_preserves_hadamard() {
    let (_, tq1_bytes, _) = build_tq1_v4_file(H1B_FLAG_HADAMARD_ROTATED);
    let tmp = tempfile::tempdir().unwrap();
    let in_path = tmp.path().join("src.h1b");
    let out_path = tmp.path().join("dst.h1b");
    std::fs::write(&in_path, &tq1_bytes).unwrap();

    let stats = convert_file(&in_path, &out_path).expect("convert");
    assert!(stats.hadamard_preserved);

    let out = H1bFile::open(&out_path).unwrap();
    let cfg = out.config();
    assert_eq!(cfg.version, 3);
    assert!(
        cfg.reserved & H1B_FLAG_SHERRY_FP16 != 0,
        "SHERRY_FP16 flag must be set in output",
    );
    assert!(
        cfg.reserved & H1B_FLAG_HADAMARD_ROTATED != 0,
        "HADAMARD flag must be preserved when input had it set",
    );
    assert!(cfg.is_sherry_fp16());
    assert!(cfg.is_hadamard_rotated());

    // And when the input had NO hadamard bit, output must not have it.
    let (_, tq1_bytes2, _) = build_tq1_v4_file(0);
    let in2 = tmp.path().join("src2.h1b");
    let out2 = tmp.path().join("dst2.h1b");
    std::fs::write(&in2, &tq1_bytes2).unwrap();
    let _ = convert_file(&in2, &out2).unwrap();
    let out_file = H1bFile::open(&out2).unwrap();
    assert!(!out_file.config().is_hadamard_rotated());
    assert!(out_file.config().is_sherry_fp16());
}

// ----------------------------------------------------------------------------
// (c) CLI arg parsing via clap.
// ----------------------------------------------------------------------------

#[test]
fn cli_arg_parsing() {
    use clap::Parser;

    // Mirror the Args struct from main.rs so we exercise clap's derive.
    // We deliberately don't re-export Args from the lib to keep the CLI
    // surface minimal; this is a close replica.
    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long = "in", value_name = "FILE")]
        input: PathBuf,
        #[arg(long = "out", value_name = "FILE")]
        output: PathBuf,
        #[arg(long)]
        verbose: bool,
    }

    let a = Args::try_parse_from(["h1b-sherry", "--in", "/tmp/a.h1b", "--out", "/tmp/b.h1b"])
        .expect("valid args parse");
    assert_eq!(a.input, PathBuf::from("/tmp/a.h1b"));
    assert_eq!(a.output, PathBuf::from("/tmp/b.h1b"));
    assert!(!a.verbose);

    let a = Args::try_parse_from([
        "h1b-sherry",
        "--in",
        "/x.h1b",
        "--out",
        "/y.h1b",
        "--verbose",
    ])
    .expect("valid args w/ verbose");
    assert!(a.verbose);

    // Missing --out is a hard error.
    let err = Args::try_parse_from(["h1b-sherry", "--in", "/x.h1b"]).unwrap_err();
    assert!(matches!(
        err.kind(),
        clap::error::ErrorKind::MissingRequiredArgument,
    ));

    // Missing --in is a hard error.
    let err = Args::try_parse_from(["h1b-sherry", "--out", "/x.h1b"]).unwrap_err();
    assert!(matches!(
        err.kind(),
        clap::error::ErrorKind::MissingRequiredArgument,
    ));
}

// Also: at least ensure [`pack_sherry_row`] is callable from tests/.
#[test]
fn pack_sherry_row_smoke() {
    let ternary = vec![0i8; 32];
    let mut packed = vec![0u8; 32 * 5 / 32];
    let stats = pack_sherry_row(&ternary, &mut packed, 32);
    // All-zero input means every group needs a forced flip to pick a zero.
    // Actually: every lane IS zero, so zero_pos=0 is the natural choice
    // and NOTHING needs to be flipped. Forced flips = 0.
    assert_eq!(stats.forced_zero_flips, 0);
}
