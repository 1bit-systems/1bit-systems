//! `.h1b` — halo-1bit ternary BitNet model format.
//!
//! This is a faithful Rust port of the loader at
//! `/home/bcloud/repos/rocm-cpp/src/h1b_loader.cpp`. The file layout on disk
//! is (little-endian everywhere — matches `std::ifstream::read` on x86_64
//! Linux which is where the exporter runs):
//!
//! ```text
//!   offset  size  field
//!   0x00    4     magic      = b"H1B\0"
//!   0x04    4     version    (i32 : 1, 2, 3, 4)
//!   0x08    36    config     9 × i32
//!   0x2C    8     extras     2 × f32  (rope_theta, rms_norm_eps) — only if version >= 2
//!   ....    ....  weights    (per-tensor payload, see [`H1bLayerOffsets`])
//! ```
//!
//! The 9 config ints are, in order:
//! `hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads,
//! vocab_size, max_seq_len, tie_embeddings, reserved`.
//!
//! Weights, per layer, in write order (exactly what `h1b_loader.cpp` reads):
//!
//! ```text
//!   input_norm              [hs]   f32
//!   post_attn_norm          [hs]   f32
//!   attn_sub_norm           [hs]   f32   (exporter duplicates 4×; we keep first)
//!   attn_sub_norm_dup1      [hs]   f32   skipped
//!   attn_sub_norm_dup2      [hs]   f32   skipped
//!   attn_sub_norm_dup3      [hs]   f32   skipped
//!   ffn_sub_norm_hs_dup1    [hs]   f32   skipped (legacy truncated gate slot)
//!   ffn_sub_norm_hs_dup2    [hs]   f32   skipped (legacy truncated up slot)
//!   ffn_sub_norm            [is]   f32
//!   q, k, v, o, gate, up, down    — 7 ternary tensors (see H1bWeightFormat)
//! ```
//!
//! Each ternary tensor is packed bytes (format depends on version) followed
//! by `[rows] f32` scales. `1bit-core` **does not upload to GPU** — it hands
//! the caller raw `&[u8]` slices via the mmap. `1bit-hip` does the
//! device transfer.

use std::fs::File;
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian};

use crate::error::HaloError;
use crate::types::{
    DEFAULT_RMS_NORM_EPS, DEFAULT_ROPE_THETA, MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION,
};

pub const H1B_MAGIC: [u8; 4] = *b"H1B\0";

/// Bit flag (in `H1bConfig::reserved` / `cfg[8]`) marking a checkpoint
/// whose activation stream has been **Walsh-Hadamard-rotated offline**
/// per BitNet v2 (arXiv 2504.18415). When this bit is set the online
/// forward pass is expected to apply the inverse rotation's counterpart
/// (i.e. a fresh `H x / sqrt(B)` per activation quant site) before
/// dispatching the GEMV — the weights have already absorbed `W' = W @ H^T`.
///
/// **We reuse the `reserved` cfg slot rather than bumping the format
/// version** because:
///   * No on-disk layout changes — only the interpretation of a previously
///     unused integer.
///   * Existing v1/v2/v3/v4 loaders that predate this flag read `reserved`
///     as zero (no rotation) which is the correct fallback on untrained
///     weights. A version bump would reject those files outright.
///   * The offline requantizer (issuing `W' = W @ H^T` and rewriting the
///     `.h1b`) sets this bit; everyone else leaves it at zero.
///
/// Other bits of `reserved` remain unassigned — future flags compose.
pub const H1B_FLAG_HADAMARD_ROTATED: i32 = 0x1;

/// Bit flag (in `H1bConfig::reserved` / `cfg[8]`) marking a checkpoint whose
/// ternary weights are packed in the **clean-room fp16 Sherry 1.25-bit
/// layout** consumed by the new `sherry_ternary_gemv_launch` kernel
/// (`rocm-cpp/include/rocm_cpp/sherry.h`).
///
/// Byte layout on disk is identical to `H1bWeightFormat::SherryV3` —
/// `N * K * 5 / 32` bytes per tensor plus `[N] f32` scales — so the flag
/// only flips dispatch, not row-byte math. The distinction is runtime:
///   * `SherryV3` (halo-v3 path): INT8 activations + per-row FP32 scales
///     baked into the kernel (`rcpp_ternary_gemv_sherry_f16`).
///   * `SherryFp16` (this flag, fp16 path): FP16 activations, no scale
///     baked in — the caller multiplies externally if the model needs it
///     (`sherry_ternary_gemv_launch`).
///
/// **Flag bit, not a version bump** — same reasoning as
/// [`H1B_FLAG_HADAMARD_ROTATED`]:
///   * No on-disk layout changes; row bytes are identical to v3.
///   * Backward-compat: existing v3 loaders that predate this flag read
///     `reserved` as zero (no fp16 path) which is the correct fallback
///     on halo-v3 weights.
///   * The requantizer (separate follow-up) emits v3 files with this bit
///     set; everyone else leaves it at zero.
///
/// The per-file flag composes with [`H1B_FLAG_HADAMARD_ROTATED`]: a BitNet
/// v2 checkpoint can ship Hadamard-rotated activations AND request the
/// fp16 Sherry kernel in the same file (`reserved = 0x3`).
pub const H1B_FLAG_SHERRY_FP16: i32 = 0x2;

/// Bit flag (in `H1bConfig::reserved` / `cfg[8]`) marking a checkpoint whose
/// ternary weights are packed in PrismML's **`Q1_0_g128`** 1-bit format
/// (oxibonsai compat: `BlockQ1_0G128`, 18 B / 128-weight block). Each
/// block carries a 2-byte FP16 scale + 16 B of sign bits; weight reconstruction
/// is `w[i] = bit[i] ? +d : -d` (no zero code, strict 1 bit per weight).
///
/// Blocks are stored **row-major, interleaved on disk** as `[d : 2][qs : 16]`
/// × `(K / 128)` per row — the verbatim GGUF tensor payload from
/// `prism-ml/Bonsai-*-gguf` files (dtype tag 41 = 0x29 in Bonsai's flavour
/// of GGUF). The loader hands the bytes to the HIP kernel unchanged; any
/// SoA/AoS remixing is a kernel-side upload transform, not a file format
/// concern.
///
/// **Flag bit, not a version bump** — row-byte math differs fundamentally
/// from halo-v2/Sherry/TQ1 (block-interleaved, no per-row scale payload),
/// so we signal it via flag rather than bolting another integer onto the
/// version ladder. Mutually exclusive with [`H1B_FLAG_BONSAI_TQ2`]; both
/// set is a configuration error and the dispatcher rejects it.
///
/// This flag is **file-format only** as of 2026-04-20 — there is no HIP
/// kernel implementation yet. Setting it today produces an
/// `H1bWeightFormat::BonsaiQ1 { group_size: 128 }` format the loader
/// understands but no GEMV path consumes; the scaffold exists so the
/// subsequent HIP port agent can flip from "no kernel" to "kernel landed"
/// without touching the file format. See
/// `docs/wiki/Bonsai-Kernel-Spec.md`.
pub const H1B_FLAG_BONSAI_Q1: i32 = 0x4;

/// Bit flag (in `H1bConfig::reserved` / `cfg[8]`) marking a checkpoint whose
/// ternary weights are packed in PrismML's **`TQ2_0_g128`** ~1.585-bit
/// format (oxibonsai compat: `BlockTQ2_0_g128`, 34 B / 128-weight block).
/// Each block carries 32 B of 2-bit codes + a trailing 2-byte FP16 scale;
/// code map is `0b00→−1, 0b01→0, 0b10→+1, 0b11→0(reserved)` with 4 weights
/// packed per byte LSB-first.
///
/// Blocks are stored **row-major, interleaved on disk** as `[qs : 32][d : 2]`
/// × `(K / 128)` per row — the verbatim GGUF tensor payload from
/// `prism-ml/Ternary-Bonsai-*-gguf` files (dtype tag 42 = 0x2A in Bonsai's
/// flavour of GGUF; distinct from llama.cpp's canonical `TQ2_0` tag 35).
/// The loader hands the bytes to the HIP kernel unchanged; any SoA/AoS
/// remixing is a kernel-side upload transform (the oxibonsai Metal path
/// repacks to `[all d][all qs]` at upload), not a file format concern.
///
/// **Flag bit, not a version bump** — row-byte math differs fundamentally
/// from halo-v2/Sherry/TQ1 (block-interleaved with in-block scale bytes,
/// no separate per-row scale payload), so we signal it via flag rather
/// than bolting another integer onto the version ladder. Mutually
/// exclusive with [`H1B_FLAG_BONSAI_Q1`]; both set is a configuration
/// error and the dispatcher rejects it.
///
/// This flag is **file-format only** as of 2026-04-20 — there is no HIP
/// kernel implementation yet. Setting it today produces an
/// `H1bWeightFormat::BonsaiTQ2 { group_size: 128 }` format the loader
/// understands but no GEMV path consumes; the scaffold exists so the
/// subsequent HIP port agent can flip from "no kernel" to "kernel landed"
/// without touching the file format. See
/// `docs/wiki/Bonsai-Kernel-Spec.md`.
pub const H1B_FLAG_BONSAI_TQ2: i32 = 0x8;

/// Fixed group size for both Bonsai formats (oxibonsai `QK1_0_G128` /
/// `QK_TQ2_0_G128`). Carried as a field on the enum variants so future
/// variants (e.g. `g64`, `g256`) can extend the format family without
/// another flag bit.
pub const BONSAI_GROUP_SIZE: u32 = 128;

/// Fixed-size header sizes, for offset arithmetic.
const CFG_BYTES: usize = 9 * 4;
const EXTRAS_BYTES: usize = 2 * 4;
const HEADER_V1: usize = 4 + 4 + CFG_BYTES;
const HEADER_V2: usize = HEADER_V1 + EXTRAS_BYTES;

/// Which ternary packing scheme a given `.h1b` version uses.
///
/// All formats carry the same scale layout (`[rows] f32`) but differ in
/// how the sign values are packed and which kernel the dispatcher reaches
/// for at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H1bWeightFormat {
    /// halo v2: `uint8[rows, (cols + 3) / 4]`, 2 bpw.
    HaloV2,
    /// Sherry v3: `uint8[rows * cols * 5 / 32]`, 1.25 bpw. `cols % 32 == 0`.
    /// Dispatches through `rcpp_ternary_gemv_sherry_f16` (INT8 activations
    /// + per-row FP32 scale baked in).
    SherryV3,
    /// Sherry fp16: byte-identical packing to [`Self::SherryV3`], but the
    /// runtime dispatcher routes to the clean-room fp16 launcher
    /// (`sherry_ternary_gemv_launch`). Selected via the
    /// [`H1B_FLAG_SHERRY_FP16`] flag bit on a v3 file — the disk layout
    /// does not change, so a v3 loader that predates this flag reads the
    /// file as [`Self::SherryV3`] (correct halo-v3 fallback).
    SherryFp16,
    /// TQ1 v4: `uint8[rows * cols_padded / 5]`, 1.6 bpw. `cols` padded to mult. 20.
    TQ1V4,
    /// PrismML `Q1_0_g128`: 18-byte blocks of `{FP16 scale d, [u8; 16] sign bits}`,
    /// interleaved on disk. 1.125 bpw on-disk (`18 * 8 / 128`), 1.0 bpw information.
    /// `cols % group_size == 0` required. Signalled by
    /// [`H1B_FLAG_BONSAI_Q1`]; no per-row scale payload — scales live
    /// inside each block.
    BonsaiQ1 {
        /// Number of weights per block. Always 128 today; carried on the
        /// variant so a future `g64` / `g256` format can extend the family
        /// without a new flag bit.
        group_size: u32,
    },
    /// PrismML `TQ2_0_g128`: 34-byte blocks of `{[u8; 32] 2-bit codes, FP16 scale d}`,
    /// interleaved on disk. 2.125 bpw on-disk (`34 * 8 / 128`), ~1.585 bpw information.
    /// `cols % group_size == 0` required. Signalled by
    /// [`H1B_FLAG_BONSAI_TQ2`]; no per-row scale payload — scales live
    /// inside each block.
    BonsaiTQ2 {
        /// Number of weights per block. Always 128 today; carried on the
        /// variant so a future `g64` / `g256` format can extend the family
        /// without a new flag bit.
        group_size: u32,
    },
}

impl H1bWeightFormat {
    /// Bytes per packed weight row, given `cols` for the tensor.
    pub fn row_bytes(self, cols: usize) -> Result<usize, HaloError> {
        match self {
            H1bWeightFormat::HaloV2 => Ok(cols.div_ceil(4)),
            H1bWeightFormat::SherryV3 | H1bWeightFormat::SherryFp16 => {
                if cols % 32 != 0 {
                    return Err(HaloError::InvalidConfig(
                        "Sherry packing requires cols divisible by 32",
                    ));
                }
                Ok(cols * 5 / 32)
            }
            H1bWeightFormat::TQ1V4 => {
                let cols_padded = cols.div_ceil(20) * 20;
                Ok(cols_padded / 5)
            }
            H1bWeightFormat::BonsaiQ1 { group_size } => {
                let g = group_size as usize;
                if g == 0 || !g.is_power_of_two() {
                    return Err(HaloError::InvalidConfig(
                        "BonsaiQ1 group_size must be a positive power of two",
                    ));
                }
                if cols % g != 0 {
                    return Err(HaloError::InvalidConfig(
                        "BonsaiQ1 requires cols divisible by group_size",
                    ));
                }
                // 18 bytes per block (2 bytes FP16 scale + 16 bytes sign bits).
                Ok((cols / g) * 18)
            }
            H1bWeightFormat::BonsaiTQ2 { group_size } => {
                let g = group_size as usize;
                if g == 0 || !g.is_power_of_two() {
                    return Err(HaloError::InvalidConfig(
                        "BonsaiTQ2 group_size must be a positive power of two",
                    ));
                }
                if cols % g != 0 {
                    return Err(HaloError::InvalidConfig(
                        "BonsaiTQ2 requires cols divisible by group_size",
                    ));
                }
                // 34 bytes per block (32 bytes 2-bit codes + 2 bytes FP16 scale).
                Ok((cols / g) * 34)
            }
        }
    }

    /// Whether this format carries its per-group scale inside each packed
    /// block (Bonsai family) rather than as a separate `[rows] f32` scales
    /// tensor after the packed payload (halo / Sherry / TQ1 family).
    ///
    /// Loaders use this to decide whether to reserve bytes for the scales
    /// tensor when walking the layer directory: halo-style formats follow
    /// a packed tensor with `rows * 4` scale bytes; Bonsai formats write
    /// zero scale bytes after the packed tensor because the FP16 scales
    /// are already embedded in the 18 B / 34 B blocks.
    pub fn has_inline_block_scales(self) -> bool {
        matches!(
            self,
            H1bWeightFormat::BonsaiQ1 { .. } | H1bWeightFormat::BonsaiTQ2 { .. }
        )
    }

    /// Resolve the packing format given a file version AND the `reserved`
    /// flag word. The `H1B_FLAG_SHERRY_FP16` bit promotes a v3 file from
    /// [`Self::SherryV3`] (the halo-v3 INT8-activation kernel) to
    /// [`Self::SherryFp16`] (the clean-room fp16 kernel). Other flag bits
    /// (e.g. `H1B_FLAG_HADAMARD_ROTATED`) compose independently.
    fn from_version_and_flags(v: i32, flags: i32) -> Result<Self, HaloError> {
        // Bonsai flags take precedence across all versions — they carry their
        // own block-interleaved row-byte math independent of the version
        // ladder. Both set simultaneously is a configuration error: a
        // checkpoint is either 1-bit Q1 or ternary TQ2, never both.
        let bonsai_q1 = (flags & H1B_FLAG_BONSAI_Q1) != 0;
        let bonsai_tq2 = (flags & H1B_FLAG_BONSAI_TQ2) != 0;
        if bonsai_q1 && bonsai_tq2 {
            return Err(HaloError::InvalidConfig(
                "H1B_FLAG_BONSAI_Q1 and H1B_FLAG_BONSAI_TQ2 are mutually exclusive",
            ));
        }
        if bonsai_q1 {
            return Ok(H1bWeightFormat::BonsaiQ1 {
                group_size: BONSAI_GROUP_SIZE,
            });
        }
        if bonsai_tq2 {
            return Ok(H1bWeightFormat::BonsaiTQ2 {
                group_size: BONSAI_GROUP_SIZE,
            });
        }
        match v {
            1 | 2 => Ok(H1bWeightFormat::HaloV2),
            3 => {
                if (flags & H1B_FLAG_SHERRY_FP16) != 0 {
                    Ok(H1bWeightFormat::SherryFp16)
                } else {
                    Ok(H1bWeightFormat::SherryV3)
                }
            }
            4 => Ok(H1bWeightFormat::TQ1V4),
            _ => Err(HaloError::UnsupportedVersion {
                version: v,
                min: MIN_SUPPORTED_VERSION,
                max: MAX_SUPPORTED_VERSION,
            }),
        }
    }
}

/// Parsed `.h1b` config header. Mirrors `rcpp_bitnet_model_t`'s scalar fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct H1bConfig {
    pub version: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub vocab_size: i32,
    pub max_seq_len: i32,
    pub tie_embeddings: i32,
    /// `cfg[8]` — reserved in all current versions, preserved verbatim.
    pub reserved: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl H1bConfig {
    /// Disk-serialized size of the header, including magic + version.
    pub fn header_bytes(&self) -> usize {
        if self.version >= 2 {
            HEADER_V2
        } else {
            HEADER_V1
        }
    }

    /// Head dimension, `hidden_size / num_heads`. Matches C++ `hd`.
    pub fn head_dim(&self) -> Result<i32, HaloError> {
        if self.num_heads <= 0 {
            return Err(HaloError::InvalidConfig("num_heads must be > 0"));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(HaloError::InvalidConfig(
                "hidden_size not divisible by num_heads",
            ));
        }
        Ok(self.hidden_size / self.num_heads)
    }

    /// Resolve the packing format, honouring the flag bits in
    /// `reserved`. In particular, a v3 file with
    /// [`H1B_FLAG_SHERRY_FP16`] set returns
    /// [`H1bWeightFormat::SherryFp16`] (clean-room fp16 kernel) instead
    /// of [`H1bWeightFormat::SherryV3`] (INT8-activation halo-v3 kernel).
    pub fn weight_format(&self) -> Result<H1bWeightFormat, HaloError> {
        H1bWeightFormat::from_version_and_flags(self.version, self.reserved)
    }

    /// Whether this checkpoint ships Hadamard-rotated activations per
    /// BitNet v2. See [`H1B_FLAG_HADAMARD_ROTATED`] for the flag rationale.
    ///
    /// Today's 2B-4T base model (and every shipping checkpoint) returns
    /// `false` — the bit is cleared in all weights we distribute. The
    /// 1bit-router's `bitnet-v2` feature consults this accessor to decide
    /// whether to dispatch the online rotation kernel.
    pub fn is_hadamard_rotated(&self) -> bool {
        (self.reserved & H1B_FLAG_HADAMARD_ROTATED) != 0
    }

    /// Whether this checkpoint's ternary weights are packed for the
    /// clean-room fp16 Sherry kernel (`sherry_ternary_gemv_launch`)
    /// rather than the halo-v3 INT8-activation path. See
    /// [`H1B_FLAG_SHERRY_FP16`] for the flag rationale. Flag is only
    /// meaningful on v3 files; returns `false` otherwise even if the bit
    /// happens to be set.
    pub fn is_sherry_fp16(&self) -> bool {
        self.version == 3 && (self.reserved & H1B_FLAG_SHERRY_FP16) != 0
    }

    /// Whether this checkpoint's ternary weights are packed in PrismML's
    /// `Q1_0_g128` 1-bit format. See [`H1B_FLAG_BONSAI_Q1`] for the flag
    /// rationale. Mutually exclusive with [`Self::is_bonsai_tq2`].
    pub fn is_bonsai_q1(&self) -> bool {
        (self.reserved & H1B_FLAG_BONSAI_Q1) != 0 && (self.reserved & H1B_FLAG_BONSAI_TQ2) == 0
    }

    /// Whether this checkpoint's ternary weights are packed in PrismML's
    /// `TQ2_0_g128` ~1.585-bit format. See [`H1B_FLAG_BONSAI_TQ2`] for
    /// the flag rationale. Mutually exclusive with [`Self::is_bonsai_q1`].
    pub fn is_bonsai_tq2(&self) -> bool {
        (self.reserved & H1B_FLAG_BONSAI_TQ2) != 0 && (self.reserved & H1B_FLAG_BONSAI_Q1) == 0
    }
}

/// Byte offsets within the mmap for every tensor of a single transformer layer.
///
/// All offsets are absolute (from start of file). Lengths are in **bytes**,
/// so a caller can do `&mmap[ofs..ofs + len]` for zero-copy access.
#[derive(Debug, Clone, Copy)]
pub struct H1bLayerOffsets {
    pub input_norm: Span,
    pub post_attn_norm: Span,
    pub attn_sub_norm: Span,
    pub ffn_sub_norm: Span,

    pub q_packed: Span,
    pub q_scales: Span,
    pub k_packed: Span,
    pub k_scales: Span,
    pub v_packed: Span,
    pub v_scales: Span,
    pub o_packed: Span,
    pub o_scales: Span,
    pub gate_packed: Span,
    pub gate_scales: Span,
    pub up_packed: Span,
    pub up_scales: Span,
    pub down_packed: Span,
    pub down_scales: Span,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub offset: usize,
    pub len: usize,
}

impl Span {
    fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

/// Offsets for model-level tensors (embedding + final norm).
#[derive(Debug, Clone, Copy)]
pub struct H1bModelOffsets {
    pub embedding: Span,
    pub final_norm: Span,
}

/// The on-disk memory map + parsed header + tensor directory.
///
/// Construction `O(num_layers)`. The actual weight bytes are never read —
/// just walked to compute offsets, which is effectively free against the
/// kernel page cache.
pub struct H1bFile {
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Mapped,
    config: H1bConfig,
    model_offsets: H1bModelOffsets,
    layer_offsets: Vec<H1bLayerOffsets>,
}

impl std::fmt::Debug for H1bFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H1bFile")
            .field("path", &self.path)
            .field("bytes", &self.mmap.as_slice().len())
            .field("config", &self.config)
            .field("layers", &self.layer_offsets.len())
            .finish()
    }
}

impl H1bFile {
    /// Mmap the file and parse headers + layer offsets.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HaloError> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| HaloError::io_at(path_ref, e))?;
        let mmap = Mapped::from_file(&file).map_err(|e| HaloError::io_at(path_ref, e))?;
        Self::parse(path_ref.to_path_buf(), mmap)
    }

    /// Parse a mapped buffer (used by `open` and by round-trip tests that
    /// go through `Vec<u8>`).
    pub fn parse_bytes(path: impl Into<PathBuf>, bytes: Vec<u8>) -> Result<Self, HaloError> {
        Self::parse(path.into(), Mapped::Owned(bytes))
    }

    fn parse(path: PathBuf, mmap: Mapped) -> Result<Self, HaloError> {
        let buf = mmap.as_slice();
        if buf.len() < HEADER_V1 {
            return Err(HaloError::Truncated {
                offset: 0,
                needed: HEADER_V1,
                have: buf.len(),
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        // Match C++ behaviour: `strncmp(magic, "H1B", 3)` — only the first
        // three bytes are checked, the fourth is allowed to be anything.
        if magic[..3] != H1B_MAGIC[..3] {
            return Err(HaloError::BadMagic {
                expected: H1B_MAGIC,
                got: magic,
            });
        }

        let version = LittleEndian::read_i32(&buf[4..8]);
        if !(MIN_SUPPORTED_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
            return Err(HaloError::UnsupportedVersion {
                version,
                min: MIN_SUPPORTED_VERSION,
                max: MAX_SUPPORTED_VERSION,
            });
        }

        let cfg_start = 8usize;
        let mut cfg = [0i32; 9];
        for (i, slot) in cfg.iter_mut().enumerate() {
            *slot = LittleEndian::read_i32(&buf[cfg_start + i * 4..cfg_start + (i + 1) * 4]);
        }

        let (rope_theta, rms_norm_eps, header_end) = if version >= 2 {
            let extras_start = cfg_start + CFG_BYTES;
            if buf.len() < extras_start + EXTRAS_BYTES {
                return Err(HaloError::Truncated {
                    offset: extras_start,
                    needed: EXTRAS_BYTES,
                    have: buf.len() - extras_start.min(buf.len()),
                });
            }
            let rt = LittleEndian::read_f32(&buf[extras_start..extras_start + 4]);
            let eps = LittleEndian::read_f32(&buf[extras_start + 4..extras_start + 8]);
            // Apply the same "<=0 falls back to default" behaviour as C++.
            let rt = if rt > 0.0 { rt } else { DEFAULT_ROPE_THETA };
            let eps = if eps > 0.0 { eps } else { DEFAULT_RMS_NORM_EPS };
            (rt, eps, extras_start + EXTRAS_BYTES)
        } else {
            (
                DEFAULT_ROPE_THETA,
                DEFAULT_RMS_NORM_EPS,
                cfg_start + CFG_BYTES,
            )
        };

        let config = H1bConfig {
            version,
            hidden_size: cfg[0],
            intermediate_size: cfg[1],
            num_layers: cfg[2],
            num_heads: cfg[3],
            num_kv_heads: cfg[4],
            vocab_size: cfg[5],
            max_seq_len: cfg[6],
            tie_embeddings: cfg[7],
            reserved: cfg[8],
            rope_theta,
            rms_norm_eps,
        };

        if config.num_layers < 0
            || config.vocab_size <= 0
            || config.hidden_size <= 0
            || config.intermediate_size <= 0
        {
            return Err(HaloError::InvalidConfig("non-positive dim in header"));
        }

        let hs = config.hidden_size as usize;
        let is_ = config.intermediate_size as usize;
        let nh = config.num_heads as usize;
        let nkv = config.num_kv_heads as usize;
        let hd = config.head_dim()? as usize;
        let vocab = config.vocab_size as usize;
        let n_layers = config.num_layers as usize;
        let fmt = config.weight_format()?;

        let mut cursor = header_end;

        // Helper: reserve `n` bytes, return the span, advance cursor.
        let alloc = |cursor: &mut usize, n: usize| -> Result<Span, HaloError> {
            if *cursor + n > buf.len() {
                return Err(HaloError::Truncated {
                    offset: *cursor,
                    needed: n,
                    have: buf.len().saturating_sub(*cursor),
                });
            }
            let s = Span::new(*cursor, n);
            *cursor += n;
            Ok(s)
        };

        // Embeddings + final norm (both fp32 on disk).
        let embedding = alloc(&mut cursor, vocab * hs * 4)?;
        let final_norm = alloc(&mut cursor, hs * 4)?;
        let model_offsets = H1bModelOffsets {
            embedding,
            final_norm,
        };

        // Per-layer norms + 7 ternary tensors.
        let mut layer_offsets = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let input_norm = alloc(&mut cursor, hs * 4)?;
            let post_attn_norm = alloc(&mut cursor, hs * 4)?;
            let attn_sub_norm = alloc(&mut cursor, hs * 4)?;
            // Three duplicate attn_sub_norm slots — skipped in C++, skipped here.
            alloc(&mut cursor, hs * 4)?;
            alloc(&mut cursor, hs * 4)?;
            alloc(&mut cursor, hs * 4)?;
            // Two truncated ffn_sub slots ([hs]) — skipped.
            alloc(&mut cursor, hs * 4)?;
            alloc(&mut cursor, hs * 4)?;
            // The real ffn_sub_norm, length `is`.
            let ffn_sub_norm = alloc(&mut cursor, is_ * 4)?;

            let fmt_inline_scales = fmt.has_inline_block_scales();
            let mut ternary = |rows: usize, cols: usize| -> Result<(Span, Span), HaloError> {
                let row_bytes = fmt.row_bytes(cols)?;
                let packed = alloc(&mut cursor, rows * row_bytes)?;
                // Bonsai formats store per-group FP16 scales inline inside
                // each 18 B / 34 B block, so the trailing `[rows] f32`
                // scales tensor is zero-length. Halo / Sherry / TQ1 all
                // still write the per-row scale payload.
                let scales_bytes = if fmt_inline_scales { 0 } else { rows * 4 };
                let scales = alloc(&mut cursor, scales_bytes)?;
                Ok((packed, scales))
            };

            let (q_packed, q_scales) = ternary(nh * hd, hs)?;
            let (k_packed, k_scales) = ternary(nkv * hd, hs)?;
            let (v_packed, v_scales) = ternary(nkv * hd, hs)?;
            let (o_packed, o_scales) = ternary(hs, nh * hd)?;
            let (gate_packed, gate_scales) = ternary(is_, hs)?;
            let (up_packed, up_scales) = ternary(is_, hs)?;
            let (down_packed, down_scales) = ternary(hs, is_)?;

            layer_offsets.push(H1bLayerOffsets {
                input_norm,
                post_attn_norm,
                attn_sub_norm,
                ffn_sub_norm,
                q_packed,
                q_scales,
                k_packed,
                k_scales,
                v_packed,
                v_scales,
                o_packed,
                o_scales,
                gate_packed,
                gate_scales,
                up_packed,
                up_scales,
                down_packed,
                down_scales,
            });
        }

        // Trailing data (untied LM head) is allowed — we just don't parse it
        // in MVP, same as the C++ loader.

        Ok(Self {
            path,
            mmap,
            config,
            model_offsets,
            layer_offsets,
        })
    }

    pub fn config(&self) -> &H1bConfig {
        &self.config
    }

    pub fn model_offsets(&self) -> &H1bModelOffsets {
        &self.model_offsets
    }

    pub fn layer_offsets(&self) -> &[H1bLayerOffsets] {
        &self.layer_offsets
    }

    /// Raw byte view of the full mapped file. Weight accessors use this
    /// internally — prefer [`Self::tensor_bytes`] for individual tensors.
    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Zero-copy slice for a given span. Panics only if `span` is out of
    /// bounds, which cannot happen for spans produced during parsing.
    pub fn tensor_bytes(&self, span: Span) -> &[u8] {
        &self.bytes()[span.offset..span.offset + span.len]
    }
}

/// Serialize a `.h1b` file to bytes. Used by round-trip tests and by tools
/// that need to rewrite a model (e.g. requantizer). Matches the exporter
/// layout byte-for-byte.
pub fn serialize(
    config: &H1bConfig,
    model_tensors: &ModelTensors<'_>,
    layer_tensors: &[LayerTensors<'_>],
) -> Result<Vec<u8>, HaloError> {
    if layer_tensors.len() as i32 != config.num_layers {
        return Err(HaloError::InvalidConfig(
            "layer_tensors length must equal config.num_layers",
        ));
    }

    let hs = config.hidden_size as usize;
    let is_ = config.intermediate_size as usize;
    let nh = config.num_heads as usize;
    let nkv = config.num_kv_heads as usize;
    let hd = config.head_dim()? as usize;
    let vocab = config.vocab_size as usize;
    let fmt = config.weight_format()?;

    let mut out = Vec::new();
    out.extend_from_slice(&H1B_MAGIC);
    out.extend_from_slice(&config.version.to_le_bytes());
    for v in [
        config.hidden_size,
        config.intermediate_size,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads,
        config.vocab_size,
        config.max_seq_len,
        config.tie_embeddings,
        config.reserved,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    if config.version >= 2 {
        out.extend_from_slice(&config.rope_theta.to_le_bytes());
        out.extend_from_slice(&config.rms_norm_eps.to_le_bytes());
    }

    expect_len("embedding", model_tensors.embedding_fp32, vocab * hs * 4)?;
    expect_len("final_norm", model_tensors.final_norm_fp32, hs * 4)?;
    out.extend_from_slice(model_tensors.embedding_fp32);
    out.extend_from_slice(model_tensors.final_norm_fp32);

    for (i, layer) in layer_tensors.iter().enumerate() {
        let ctx = LayerCtx { idx: i };
        ctx.expect(out_len_is(layer.input_norm_fp32, hs * 4), "input_norm")?;
        ctx.expect(
            out_len_is(layer.post_attn_norm_fp32, hs * 4),
            "post_attn_norm",
        )?;
        ctx.expect(
            out_len_is(layer.attn_sub_norm_fp32, hs * 4),
            "attn_sub_norm",
        )?;
        ctx.expect(out_len_is(layer.ffn_sub_norm_fp32, is_ * 4), "ffn_sub_norm")?;

        out.extend_from_slice(layer.input_norm_fp32);
        out.extend_from_slice(layer.post_attn_norm_fp32);
        out.extend_from_slice(layer.attn_sub_norm_fp32);
        // Three duplicate attn_sub_norms (match the exporter).
        out.extend_from_slice(layer.attn_sub_norm_fp32);
        out.extend_from_slice(layer.attn_sub_norm_fp32);
        out.extend_from_slice(layer.attn_sub_norm_fp32);
        // Two truncated [hs] ffn_sub slots — exporter fills these with the
        // first hs floats of the real ffn_sub_norm. Keep the same convention.
        let trunc = &layer.ffn_sub_norm_fp32[..hs * 4.min(layer.ffn_sub_norm_fp32.len())];
        out.extend_from_slice(trunc);
        out.extend_from_slice(trunc);
        // The real one.
        out.extend_from_slice(layer.ffn_sub_norm_fp32);

        // Seven ternary tensors.
        let tensors: [(&str, &TernaryTensor<'_>, usize, usize); 7] = [
            ("q", &layer.q, nh * hd, hs),
            ("k", &layer.k, nkv * hd, hs),
            ("v", &layer.v, nkv * hd, hs),
            ("o", &layer.o, hs, nh * hd),
            ("gate", &layer.gate, is_, hs),
            ("up", &layer.up, is_, hs),
            ("down", &layer.down, hs, is_),
        ];
        let fmt_inline_scales = fmt.has_inline_block_scales();
        for (name, t, rows, cols) in tensors {
            let row_bytes = fmt.row_bytes(cols)?;
            if t.packed.len() != rows * row_bytes {
                return Err(HaloError::InvalidConfig(static_str_from_name(name)));
            }
            // Bonsai formats carry FP16 scales inside each block, so the
            // trailing `[rows] f32` scales buffer must be empty. Every
            // other format expects exactly `rows * 4` bytes.
            let expected_scales = if fmt_inline_scales { 0 } else { rows * 4 };
            if t.scales.len() != expected_scales {
                return Err(HaloError::InvalidConfig(static_str_from_name(name)));
            }
            out.extend_from_slice(t.packed);
            out.extend_from_slice(t.scales);
        }
    }

    Ok(out)
}

/// Model-level tensors (raw fp32 bytes) for serialization.
pub struct ModelTensors<'a> {
    pub embedding_fp32: &'a [u8],
    pub final_norm_fp32: &'a [u8],
}

/// Per-layer tensors (raw bytes) for serialization.
pub struct LayerTensors<'a> {
    pub input_norm_fp32: &'a [u8],
    pub post_attn_norm_fp32: &'a [u8],
    pub attn_sub_norm_fp32: &'a [u8],
    pub ffn_sub_norm_fp32: &'a [u8],

    pub q: TernaryTensor<'a>,
    pub k: TernaryTensor<'a>,
    pub v: TernaryTensor<'a>,
    pub o: TernaryTensor<'a>,
    pub gate: TernaryTensor<'a>,
    pub up: TernaryTensor<'a>,
    pub down: TernaryTensor<'a>,
}

pub struct TernaryTensor<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u8],
}

struct LayerCtx {
    idx: usize,
}

impl LayerCtx {
    fn expect(&self, ok: bool, _what: &'static str) -> Result<(), HaloError> {
        if ok {
            Ok(())
        } else {
            // Folding the layer index into the message would need allocation;
            // leak-free version: keep it static, caller already logs index.
            let _ = self.idx;
            Err(HaloError::InvalidConfig("layer tensor length mismatch"))
        }
    }
}

fn out_len_is(buf: &[u8], expected: usize) -> bool {
    buf.len() == expected
}

fn expect_len(_what: &'static str, buf: &[u8], expected: usize) -> Result<(), HaloError> {
    if buf.len() == expected {
        Ok(())
    } else {
        Err(HaloError::InvalidConfig("model tensor length mismatch"))
    }
}

fn static_str_from_name(_n: &str) -> &'static str {
    "ternary tensor length mismatch"
}

// --------------------------------------------------------------------------
// Mmap wrapper — the single unsafe surface in this crate.
// --------------------------------------------------------------------------

/// A backing buffer: either a real `mmap` or an owned `Vec<u8>` (used by
/// tests and round-trip helpers that serialize to memory).
pub(crate) enum Mapped {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Mapped {
    /// Build a read-only mmap from an open file.
    ///
    /// # Safety invariants
    /// * We hold no mutable aliases to the mapped range.
    /// * The mmap is read-only (`Mmap`, not `MmapMut`), so changes to the
    ///   underlying file by external processes are the only way to see a
    ///   mutation — that's explicitly a user-accepted risk for read-only
    ///   model files and matches what the C++ loader does (it just reads
    ///   the file via `std::ifstream`).
    fn from_file(file: &File) -> std::io::Result<Self> {
        // SAFETY: see invariants above. `Mmap::map` is `unsafe` purely
        // because the kernel could change bytes under us; we document
        // the trade-off and treat models as immutable.
        let mm = unsafe { memmap2::Mmap::map(file)? };
        Ok(Mapped::Mmap(mm))
    }

    /// Crate-public alias — other modules in `1bit-core` (e.g. `gguf`) reuse
    /// the same mmap machinery. Same safety rules as [`Self::from_file`].
    pub(crate) fn from_file_public(file: &File) -> std::io::Result<Self> {
        Self::from_file(file)
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Mapped::Mmap(m) => m.as_ref(),
            Mapped::Owned(v) => v.as_slice(),
        }
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// (config, embedding_fp32, per-layer-norms, per-layer-tensors)
    type TinyV2 = (H1bConfig, Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>);

    /// Build a tiny but structurally valid v2 `.h1b` in memory, then parse
    /// it, re-serialize, and compare byte-for-byte.
    fn make_tiny_v2() -> TinyV2 {
        // hs=8, is=16, L=1, nh=2, nkv=1 → hd=4; vocab=32.
        // HaloV2 row_bytes(cols) = (cols+3)/4.
        let config = H1bConfig {
            version: 2,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 1,
            vocab_size: 32,
            max_seq_len: 128,
            tie_embeddings: 1,
            reserved: 0,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
        };

        // Per-layer fp32 norm bytes.
        let norms = vec![
            vec![0u8; 8 * 4],  // input_norm
            vec![1u8; 8 * 4],  // post_attn_norm
            vec![2u8; 8 * 4],  // attn_sub_norm
            vec![3u8; 16 * 4], // ffn_sub_norm (len is)
        ];

        // Tensor shapes (rows, cols); hd = hs/nh = 4.
        //   q   : [nh*hd, hs]    = [8, 8]
        //   k   : [nkv*hd, hs]   = [4, 8]
        //   v   : [nkv*hd, hs]   = [4, 8]
        //   o   : [hs, nh*hd]    = [8, 8]
        //   gate: [is, hs]       = [16, 8]
        //   up  : [is, hs]       = [16, 8]
        //   down: [hs, is]       = [8, 16]
        let rbytes = |cols: usize| cols.div_ceil(4);
        let packed_for = |rows: usize, cols: usize| vec![0xAAu8; rows * rbytes(cols)];
        let scales_for = |rows: usize| vec![0xBBu8; rows * 4];

        let tensors = vec![
            packed_for(8, 8), // q        (rows=nh*hd=8)
            scales_for(8),    // q_scales
            packed_for(4, 8), // k        (rows=nkv*hd=4)
            scales_for(4),
            packed_for(4, 8), // v
            scales_for(4),
            packed_for(8, 8), // o        (rows=hs=8)
            scales_for(8),
            packed_for(16, 8), // gate
            scales_for(16),
            packed_for(16, 8), // up
            scales_for(16),
            packed_for(8, 16), // down
            scales_for(8),
        ];

        (
            config,
            vec![0xCCu8; 32 * 8 * 4], /*embedding*/
            norms,
            tensors,
        )
    }

    fn build_file_bytes() -> (H1bConfig, Vec<u8>) {
        let (config, embedding, norms, tensors) = make_tiny_v2();
        let final_norm = vec![0xDDu8; 8 * 4];

        let model = ModelTensors {
            embedding_fp32: &embedding,
            final_norm_fp32: &final_norm,
        };
        let layer = LayerTensors {
            input_norm_fp32: &norms[0],
            post_attn_norm_fp32: &norms[1],
            attn_sub_norm_fp32: &norms[2],
            ffn_sub_norm_fp32: &norms[3],
            q: TernaryTensor {
                packed: &tensors[0],
                scales: &tensors[1],
            },
            k: TernaryTensor {
                packed: &tensors[2],
                scales: &tensors[3],
            },
            v: TernaryTensor {
                packed: &tensors[4],
                scales: &tensors[5],
            },
            o: TernaryTensor {
                packed: &tensors[6],
                scales: &tensors[7],
            },
            gate: TernaryTensor {
                packed: &tensors[8],
                scales: &tensors[9],
            },
            up: TernaryTensor {
                packed: &tensors[10],
                scales: &tensors[11],
            },
            down: TernaryTensor {
                packed: &tensors[12],
                scales: &tensors[13],
            },
        };

        let bytes = serialize(&config, &model, std::slice::from_ref(&layer)).unwrap();
        (config, bytes)
    }

    #[test]
    fn magic_mismatch() {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(b"XXXX");
        let err = H1bFile::parse_bytes("x", bytes).unwrap_err();
        matches!(err, HaloError::BadMagic { .. });
    }

    #[test]
    fn parse_header_v2() {
        let (config, bytes) = build_file_bytes();
        let f = H1bFile::parse_bytes("t", bytes).unwrap();
        assert_eq!(f.config().version, 2);
        assert_eq!(f.config().hidden_size, config.hidden_size);
        assert_eq!(f.config().intermediate_size, config.intermediate_size);
        assert_eq!(f.config().num_layers, config.num_layers);
        assert_eq!(f.config().num_heads, config.num_heads);
        assert_eq!(f.config().num_kv_heads, config.num_kv_heads);
        assert_eq!(f.config().vocab_size, config.vocab_size);
        assert_eq!(f.layer_offsets().len(), 1);
    }

    #[test]
    fn round_trip_v2() {
        // Build → parse → re-serialize → compare. This is the format
        // round-trip the spec asks for.
        let (config, bytes) = build_file_bytes();
        let f = H1bFile::parse_bytes("t", bytes.clone()).unwrap();

        // Re-serialize from the parsed tensor views.
        let m = f.model_offsets();
        let l0 = &f.layer_offsets()[0];

        let model = ModelTensors {
            embedding_fp32: f.tensor_bytes(m.embedding),
            final_norm_fp32: f.tensor_bytes(m.final_norm),
        };
        let layer = LayerTensors {
            input_norm_fp32: f.tensor_bytes(l0.input_norm),
            post_attn_norm_fp32: f.tensor_bytes(l0.post_attn_norm),
            attn_sub_norm_fp32: f.tensor_bytes(l0.attn_sub_norm),
            ffn_sub_norm_fp32: f.tensor_bytes(l0.ffn_sub_norm),
            q: TernaryTensor {
                packed: f.tensor_bytes(l0.q_packed),
                scales: f.tensor_bytes(l0.q_scales),
            },
            k: TernaryTensor {
                packed: f.tensor_bytes(l0.k_packed),
                scales: f.tensor_bytes(l0.k_scales),
            },
            v: TernaryTensor {
                packed: f.tensor_bytes(l0.v_packed),
                scales: f.tensor_bytes(l0.v_scales),
            },
            o: TernaryTensor {
                packed: f.tensor_bytes(l0.o_packed),
                scales: f.tensor_bytes(l0.o_scales),
            },
            gate: TernaryTensor {
                packed: f.tensor_bytes(l0.gate_packed),
                scales: f.tensor_bytes(l0.gate_scales),
            },
            up: TernaryTensor {
                packed: f.tensor_bytes(l0.up_packed),
                scales: f.tensor_bytes(l0.up_scales),
            },
            down: TernaryTensor {
                packed: f.tensor_bytes(l0.down_packed),
                scales: f.tensor_bytes(l0.down_scales),
            },
        };

        let re = serialize(&config, &model, std::slice::from_ref(&layer)).unwrap();
        assert_eq!(re, bytes, "round-trip produced different bytes");
    }

    #[test]
    fn v1_fallback_rope_and_eps() {
        // Build a v1 file: no extras block, should get DEFAULT_* back.
        let mut cfg = H1bConfig {
            version: 1,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 1,
            vocab_size: 4,
            max_seq_len: 32,
            tie_embeddings: 1,
            reserved: 0,
            rope_theta: 0.0,
            rms_norm_eps: 0.0,
        };
        let model = ModelTensors {
            embedding_fp32: &[0u8; 4 * 8 * 4],
            final_norm_fp32: &[0u8; 8 * 4],
        };
        let bytes = serialize(&cfg, &model, &[]).unwrap();
        let f = H1bFile::parse_bytes("t", bytes).unwrap();
        assert_eq!(f.config().rope_theta, DEFAULT_ROPE_THETA);
        assert_eq!(f.config().rms_norm_eps, DEFAULT_RMS_NORM_EPS);
        // And header should be exactly HEADER_V1 bytes long + model tensors.
        assert_eq!(f.config().header_bytes(), HEADER_V1);
        cfg.version = 1;
        let _ = cfg;
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut bytes = vec![0u8; HEADER_V1];
        bytes[0..4].copy_from_slice(&H1B_MAGIC);
        bytes[4..8].copy_from_slice(&99i32.to_le_bytes());
        let err = H1bFile::parse_bytes("t", bytes).unwrap_err();
        matches!(err, HaloError::UnsupportedVersion { version: 99, .. });
    }

    #[test]
    fn weight_format_row_bytes() {
        assert_eq!(H1bWeightFormat::HaloV2.row_bytes(8).unwrap(), 2);
        assert_eq!(H1bWeightFormat::HaloV2.row_bytes(9).unwrap(), 3);
        assert_eq!(H1bWeightFormat::SherryV3.row_bytes(64).unwrap(), 10);
        assert!(H1bWeightFormat::SherryV3.row_bytes(31).is_err());
        // TQ1: cols rounded up to mult of 20, then /5.
        assert_eq!(H1bWeightFormat::TQ1V4.row_bytes(20).unwrap(), 4);
        assert_eq!(H1bWeightFormat::TQ1V4.row_bytes(21).unwrap(), 8);
        // BonsaiQ1: 18 bytes per g128 block.
        let bq1 = H1bWeightFormat::BonsaiQ1 { group_size: 128 };
        assert_eq!(bq1.row_bytes(128).unwrap(), 18);
        assert_eq!(bq1.row_bytes(256).unwrap(), 36);
        assert!(bq1.row_bytes(127).is_err()); // not divisible by group
        // BonsaiTQ2: 34 bytes per g128 block.
        let btq2 = H1bWeightFormat::BonsaiTQ2 { group_size: 128 };
        assert_eq!(btq2.row_bytes(128).unwrap(), 34);
        assert_eq!(btq2.row_bytes(2048).unwrap(), 544);
        assert!(btq2.row_bytes(129).is_err());
        // Inline-scales marker: Bonsai formats yes, everyone else no.
        assert!(bq1.has_inline_block_scales());
        assert!(btq2.has_inline_block_scales());
        assert!(!H1bWeightFormat::HaloV2.has_inline_block_scales());
        assert!(!H1bWeightFormat::SherryV3.has_inline_block_scales());
        assert!(!H1bWeightFormat::SherryFp16.has_inline_block_scales());
        assert!(!H1bWeightFormat::TQ1V4.has_inline_block_scales());
    }

    /// `H1B_FLAG_BONSAI_Q1` / `H1B_FLAG_BONSAI_TQ2` take precedence across
    /// every supported version, carry their own block-interleaved row-byte
    /// math (no per-row scale tensor afterwards), and are mutually
    /// exclusive. This is what lets a Bonsai Qwen3 GGUF land in an `.h1b`
    /// framing without a version bump.
    #[test]
    fn bonsai_flags_roundtrip_via_from_version_and_flags() {
        // TQ2 bit set on v2 file → BonsaiTQ2 wins over the v2 default.
        let f = H1bWeightFormat::from_version_and_flags(2, H1B_FLAG_BONSAI_TQ2).unwrap();
        assert_eq!(f, H1bWeightFormat::BonsaiTQ2 { group_size: 128 });
        // Q1 bit set on v4 file → BonsaiQ1 wins over the v4/TQ1 default.
        let f = H1bWeightFormat::from_version_and_flags(4, H1B_FLAG_BONSAI_Q1).unwrap();
        assert_eq!(f, H1bWeightFormat::BonsaiQ1 { group_size: 128 });
        // Both set → error.
        assert!(
            H1bWeightFormat::from_version_and_flags(2, H1B_FLAG_BONSAI_Q1 | H1B_FLAG_BONSAI_TQ2,)
                .is_err()
        );
        // Zero → status quo per-version dispatch.
        assert_eq!(
            H1bWeightFormat::from_version_and_flags(2, 0).unwrap(),
            H1bWeightFormat::HaloV2
        );
        // Bonsai flag composes with Hadamard-rotated flag (unrelated bits
        // must not interfere with the dispatch).
        let f = H1bWeightFormat::from_version_and_flags(
            2,
            H1B_FLAG_BONSAI_TQ2 | H1B_FLAG_HADAMARD_ROTATED,
        )
        .unwrap();
        assert_eq!(f, H1bWeightFormat::BonsaiTQ2 { group_size: 128 });
    }

    /// Config accessors honour the mutual-exclusion rule: if both flags
    /// are set simultaneously (caller error), neither accessor returns
    /// `true`, matching the dispatcher's refusal to produce a variant.
    #[test]
    fn bonsai_accessors_follow_mutual_exclusion() {
        let mut cfg = H1bConfig {
            version: 2,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_layers: 1,
            num_heads: 16,
            num_kv_heads: 8,
            vocab_size: 151669,
            max_seq_len: 32768,
            tie_embeddings: 1,
            reserved: 0,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };
        assert!(!cfg.is_bonsai_q1());
        assert!(!cfg.is_bonsai_tq2());

        cfg.reserved = H1B_FLAG_BONSAI_TQ2;
        assert!(cfg.is_bonsai_tq2());
        assert!(!cfg.is_bonsai_q1());
        assert_eq!(
            cfg.weight_format().unwrap(),
            H1bWeightFormat::BonsaiTQ2 { group_size: 128 }
        );

        cfg.reserved = H1B_FLAG_BONSAI_Q1;
        assert!(cfg.is_bonsai_q1());
        assert!(!cfg.is_bonsai_tq2());
        assert_eq!(
            cfg.weight_format().unwrap(),
            H1bWeightFormat::BonsaiQ1 { group_size: 128 }
        );

        // Both set — weight_format() errors, accessors stay false.
        cfg.reserved = H1B_FLAG_BONSAI_Q1 | H1B_FLAG_BONSAI_TQ2;
        assert!(!cfg.is_bonsai_q1());
        assert!(!cfg.is_bonsai_tq2());
        assert!(cfg.weight_format().is_err());
    }

    /// `H1B_FLAG_SHERRY_FP16` lives in bit 1 of the `reserved` cfg slot
    /// and only takes effect on v3 files. Zero-reserved → halo-v3 kernel
    /// (status quo). Bit 1 set on a v3 file → fp16 kernel. On non-v3
    /// files the flag is inert and the accessor returns `false` even
    /// when the bit is set, so a requantizer can't accidentally flip a
    /// halo-v2 file into an unreachable code path.
    #[test]
    fn is_sherry_fp16_reads_reserved_bit1_on_v3_only() {
        let mut cfg = H1bConfig {
            version: 3,
            hidden_size: 128,
            intermediate_size: 128,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 4,
            max_seq_len: 8,
            tie_embeddings: 0,
            reserved: 0,
            rope_theta: DEFAULT_ROPE_THETA,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
        };

        // Default: halo-v3 kernel path.
        assert!(!cfg.is_sherry_fp16());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::SherryV3);

        // Bit 1 set on a v3 file → fp16 kernel path.
        cfg.reserved = H1B_FLAG_SHERRY_FP16;
        assert!(cfg.is_sherry_fp16());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::SherryFp16);

        // Composes with Hadamard flag.
        cfg.reserved = H1B_FLAG_SHERRY_FP16 | H1B_FLAG_HADAMARD_ROTATED;
        assert!(cfg.is_sherry_fp16());
        assert!(cfg.is_hadamard_rotated());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::SherryFp16);

        // Unrelated reserved bits don't trip the accessor.
        cfg.reserved = 0x1000_0000;
        assert!(!cfg.is_sherry_fp16());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::SherryV3);

        // Flag is inert on non-v3 files — v2 with the bit set stays HaloV2.
        cfg.version = 2;
        cfg.reserved = H1B_FLAG_SHERRY_FP16;
        assert!(!cfg.is_sherry_fp16());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::HaloV2);

        // And on v4 (TQ1).
        cfg.version = 4;
        cfg.reserved = H1B_FLAG_SHERRY_FP16;
        assert!(!cfg.is_sherry_fp16());
        assert_eq!(cfg.weight_format().unwrap(), H1bWeightFormat::TQ1V4);
    }

    /// `H1B_FLAG_HADAMARD_ROTATED` lives in bit 0 of the `reserved` cfg
    /// slot. Zero-reserved → not rotated (status quo on every ship-side
    /// checkpoint today). Bit 0 set → rotated. Other bits are reserved
    /// for future flags so they must not trip the accessor.
    #[test]
    fn is_hadamard_rotated_reads_reserved_bit0() {
        let mut cfg = H1bConfig {
            version: 2,
            hidden_size: 128,
            intermediate_size: 128,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 4,
            max_seq_len: 8,
            tie_embeddings: 0,
            reserved: 0,
            rope_theta: DEFAULT_ROPE_THETA,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
        };
        assert!(
            !cfg.is_hadamard_rotated(),
            "default reserved=0 → not rotated"
        );

        cfg.reserved = H1B_FLAG_HADAMARD_ROTATED;
        assert!(cfg.is_hadamard_rotated(), "bit0 set → rotated");

        // Foreign bits don't leak into the accessor.
        cfg.reserved = 0x1000_0000;
        assert!(
            !cfg.is_hadamard_rotated(),
            "unrelated reserved bits must not trip the rotated flag"
        );

        cfg.reserved = H1B_FLAG_HADAMARD_ROTATED | 0x1000_0000;
        assert!(
            cfg.is_hadamard_rotated(),
            "bit0 still reads through when other reserved bits are set"
        );
    }
}
