//! `.h1b-medusa` on-disk format + zero-copy mmap loader.
//!
//! This module is the **parse-only** half of the Medusa weight-loading
//! pass. It opens the file, validates magic + shape, and exposes four
//! zero-copy per-head tensor views over the mmap. The GPU upload path
//! (device buffers, `HipBackend::upload`, and the small-M ternary GEMM
//! dispatch) lives in a follow-up pass behind the same env gate — this
//! module only speaks bytes.
//!
//! # File layout (from the real `parrishcorcoran/MedusaBitNet-2B-4T`
//! `medusa_heads_step2000.pt` checkpoint)
//!
//! All integers are little-endian. Magic is an 8-byte ASCII tag carrying
//! an explicit nul so it round-trips as a `u64` read without losing the
//! zero byte — mirrors the main `.h1b` convention.
//!
//! The upstream checkpoint contains:
//!   heads.w_in:  bf16[1, 4, 2560, 2560]
//!   heads.w_out: bf16[1, 4, 2560, 2560]
//!
//! No biases. No per-head projection — the shared `lm_head` lives in the
//! backbone `.h1b`. `cfg.num_layers_per_head == 1`.
//!
//! Per-head forward (from the upstream README):
//!
//!   h_out  = h + W_out · SiLU(W_in · h)
//!   logits = backbone.lm_head(h_out)
//!
//! ```text
//! offset  size   field
//! ------  ----   -----
//!   0      8     MAGIC            "MEDUSA1\0"
//!   8      4     version          u32 (== MEDUSA_FORMAT_VERSION, currently 2)
//!  12      4     num_heads        u32 (must == NUM_MEDUSA_HEADS == 4)
//!  16      4     hidden_dim       u32 (must == MEDUSA_HIDDEN_DIM == 2560)
//!  20      4     residual_layers  u32 (== 1 per head today)
//!  24      4     dtype            u32 (1 = fp16)
//!  28     40     _reserved        [u32; 10]
//!  68    ...     per-head payload (see below)
//! ```
//!
//! Header is exactly 68 bytes. Per-head payload, repeated `num_heads`
//! times in head-index order:
//!
//! ```text
//!   w_in    fp16[hidden_dim * hidden_dim]   pre-SiLU projection
//!   w_out   fp16[hidden_dim * hidden_dim]   post-SiLU projection
//! ```
//!
//! Per-head byte size at the Microsoft backbone's shape (hd = 2560):
//!
//!   2 × (2560 × 2560) × 2 = 26_214_400 bytes ≈ 25 MiB
//!
//! Four heads + header ≈ 100 MiB on disk, matching the upstream `.pt`.
//!
//! # Zero-copy
//!
//! [`MedusaHeadsFile::open`] maps the file read-only and hands back
//! borrowed `&[u16]` views into the mmap. fp16 is carried as `u16` at
//! the parser boundary; the forward-pass crate (`1bit-hip`) takes the
//! `&[u16]` bytes and `memcpy`s them into a device buffer — no
//! allocation in this module, no host-side decode.
//!
//! # What this module does NOT do
//!
//! * No device upload. [`MedusaHeadsFile`] is a pure CPU-side handle.
//! * No weight download. A follow-up lane fetches from Hugging Face
//!   and converts to `.h1b-medusa`. See `tools/medusa-convert/`.
//! * No activation of the Medusa path. The env gate + wiring in
//!   [`super::MedusaState::from_config`] owns that decision.
//! * No `lm_head` ownership — the projection stays on the backbone.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use super::MedusaError;
use super::heads::{MEDUSA_HIDDEN_DIM, NUM_MEDUSA_HEADS};

/// Magic bytes at offset 0. ASCII `MEDUSA1` with an explicit nul so the
/// tag reads back byte-identically regardless of whether it was written
/// as `u64`-LE or `[u8; 8]`.
pub const MEDUSA_MAGIC: [u8; 8] = *b"MEDUSA1\0";

/// Current on-disk format version.
///
/// v1 was designed against an incorrect guess (owned `proj` + per-head
/// biases + two residual layers). v2 matches the real upstream artifact:
/// single residual layer per head, `W_in` + `W_out` only, no biases,
/// shared `lm_head` lives on the backbone.
pub const MEDUSA_FORMAT_VERSION: u32 = 2;

/// Exact size of the fixed-width header prefix. 8 (magic) + 4 × 5
/// (version, num_heads, hidden_dim, residual_layers, dtype) + 4 × 10
/// (reserved). Size is unchanged from v1 so the on-disk layout's
/// first-tensor offset stays at 68.
pub const MEDUSA_HEADER_BYTES: usize = 68;

/// Per-head residual-block count. Pinned to the upstream artifact today
/// (`num_layers_per_head = 1` in the checkpoint's `cfg`). Kept as a
/// named const so the loader has a single source of truth.
pub const MEDUSA_RESIDUAL_LAYERS: u32 = 1;

/// dtype tag carried in the header. `1` is fp16, the only dtype the
/// loader accepts today. Distinct name kept so a later bf16/i4 variant
/// can add new tags without a format bump.
pub const MEDUSA_DTYPE_FP16: u32 = 1;

/// Parsed, validated, little-endian header. Cheap to clone (all u32s).
#[derive(Debug, Clone, Copy)]
pub struct MedusaHeader {
    /// File-format version. Validated against [`MEDUSA_FORMAT_VERSION`]
    /// at open time — older/newer versions get a structured error, not
    /// a best-effort parse.
    pub version: u32,
    /// Number of heads declared by the file. Must equal
    /// [`NUM_MEDUSA_HEADS`]; anything else is an ABI break.
    pub num_heads: u32,
    /// Per-head hidden dim. Must equal [`MEDUSA_HIDDEN_DIM`] today —
    /// matches the backbone. Carried in the header anyway so future
    /// versions can relax this without breaking the format.
    pub hidden_dim: u32,
    /// Number of residual blocks per head. Currently 1 — matches the
    /// upstream `cfg.num_layers_per_head`.
    pub residual_layers: u32,
    /// Tensor dtype. Only [`MEDUSA_DTYPE_FP16`] is accepted today.
    pub dtype: u32,
}

impl MedusaHeader {
    /// Byte size of one head's payload at this header's shape.
    ///
    /// Layout: `residual_layers × (W_in + W_out)`, all fp16 (2 bytes each).
    ///   = residual_layers × 2 × hidden_dim × hidden_dim × 2
    ///
    /// Returns `None` on integer overflow — practically unreachable at
    /// the scaffold shape, but cheap protection against a corrupt
    /// header claiming absurd dims.
    pub fn per_head_bytes(&self) -> Option<usize> {
        let hd = self.hidden_dim as usize;
        let layers = self.residual_layers as usize;

        // W_in + W_out per residual layer, each hd × hd fp16.
        let per_layer_fp16 = hd.checked_mul(hd)?.checked_mul(2)?;
        let total_fp16 = per_layer_fp16.checked_mul(layers)?;
        total_fp16.checked_mul(2) // fp16 → bytes
    }

    /// Total on-disk size this header describes: header + all heads.
    pub fn expected_file_bytes(&self) -> Option<usize> {
        let per = self.per_head_bytes()?;
        per.checked_mul(self.num_heads as usize)?
            .checked_add(MEDUSA_HEADER_BYTES)
    }
}

/// Per-head borrowed tensor views over the mmap.
///
/// Both views are fp16 carried as `u16` — no host-side decode. The
/// forward-pass crate (`1bit-hip`) takes the raw bytes and uploads them
/// to device memory; nothing in the router decodes fp16 on the host.
///
/// Per-head compute (from the upstream README):
///
///   h_out  = h + W_out · SiLU(W_in · h)
///   logits = backbone.lm_head(h_out)
///
/// `w_in` projects the backbone hidden state through the SiLU branch,
/// `w_out` mixes it back in as a residual. Both are row-major fp16.
#[derive(Debug, Clone, Copy)]
pub struct MedusaHeadView<'a> {
    /// Pre-SiLU projection. `hidden_dim × hidden_dim` row-major fp16.
    pub w_in: &'a [u16],
    /// Post-SiLU projection. `hidden_dim × hidden_dim` row-major fp16.
    pub w_out: &'a [u16],
}

/// Owning mmap handle + parsed header + cached head offsets.
///
/// The mmap lives for the lifetime of the struct; [`Self::head`] returns
/// views borrowed from it. Move this into the [`super::MedusaHeads`]
/// scaffold struct to give it real tensor data.
///
/// NOT `Clone` — the mmap is ref-counted inside `memmap2::Mmap`, but we
/// want exactly one owner of the parsed-offsets cache per open file.
/// Wrap in `Arc` if multiple owners are needed later.
#[derive(Debug)]
pub struct MedusaHeadsFile {
    /// The read-only mmap. Kept here so views can borrow from it.
    mmap: Mmap,
    /// Parsed + validated file header.
    header: MedusaHeader,
    /// Path the file was opened from. Kept for error messages and
    /// `/metrics` reporting (same contract as
    /// [`super::heads::MedusaHeads::source_path`]).
    path: PathBuf,
    /// Per-head base offset in the mmap (byte index of the head's first
    /// weight tensor). Cached so [`Self::head`] is O(1) without
    /// recomputing the `residual_layers`-dependent stride.
    head_offsets: Vec<usize>,
}

impl MedusaHeadsFile {
    /// Open `path`, validate against [`NUM_MEDUSA_HEADS`] /
    /// [`MEDUSA_HIDDEN_DIM`] and the header's own self-consistency, and
    /// return the parsed handle.
    ///
    /// Errors:
    ///
    /// * [`MedusaError::WeightsNotFound`] — file missing.
    /// * [`MedusaError::LoaderError`] — anything else: short file, bad
    ///   magic, version mismatch, shape mismatch, file-size mismatch,
    ///   alignment issue on the fp16 cast. The message carries enough
    ///   context for an operator to act without attaching a debugger.
    pub fn open(path: &Path) -> Result<Self, MedusaError> {
        let file = File::open(path).map_err(|e| {
            // `NotFound` → structured error; anything else carries the
            // OS error string verbatim through `LoaderError` so the
            // operator can grep dmesg.
            if e.kind() == std::io::ErrorKind::NotFound {
                MedusaError::WeightsNotFound {
                    path: path.to_path_buf(),
                }
            } else {
                MedusaError::LoaderError(format!("open {}: {e}", path.display()))
            }
        })?;

        // SAFETY: read-only map of an immutable model file. Same contract
        // as `1bit-core::h1b::Mapped::from_file` — the kernel could change
        // bytes under us, but model files are treated as immutable by
        // deployment policy. If ops swaps the file live, the resulting
        // torn reads are on them.
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| MedusaError::LoaderError(format!("mmap {}: {e}", path.display())))?
        };

        let bytes = &mmap[..];
        if bytes.len() < MEDUSA_HEADER_BYTES {
            return Err(MedusaError::LoaderError(format!(
                "short file: {} bytes, need at least {} for header",
                bytes.len(),
                MEDUSA_HEADER_BYTES
            )));
        }

        // Magic check first — gives the best error message for the most
        // common operator mistake (pointing at the wrong file).
        if bytes[..8] != MEDUSA_MAGIC {
            return Err(MedusaError::LoaderError(format!(
                "bad magic: expected {:?}, got {:?} (is this really a .h1b-medusa?)",
                MEDUSA_MAGIC,
                &bytes[..8],
            )));
        }

        let header = MedusaHeader {
            version: u32_le(&bytes[8..12]),
            num_heads: u32_le(&bytes[12..16]),
            hidden_dim: u32_le(&bytes[16..20]),
            residual_layers: u32_le(&bytes[20..24]),
            dtype: u32_le(&bytes[24..28]),
            // bytes[28..68] is 10 × u32 reserved — read-through for
            // round-trip but not surfaced today.
        };

        if header.version != MEDUSA_FORMAT_VERSION {
            return Err(MedusaError::LoaderError(format!(
                "version mismatch: file says {}, loader expects {}",
                header.version, MEDUSA_FORMAT_VERSION
            )));
        }

        if header.num_heads as usize != NUM_MEDUSA_HEADS {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: num_heads={}, expected {} (ABI break — did the \
                 retrained artifact change head count?)",
                header.num_heads, NUM_MEDUSA_HEADS
            )));
        }

        if header.hidden_dim as usize != MEDUSA_HIDDEN_DIM {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: hidden_dim={}, expected {} (must match backbone)",
                header.hidden_dim, MEDUSA_HIDDEN_DIM
            )));
        }

        if header.residual_layers != MEDUSA_RESIDUAL_LAYERS {
            return Err(MedusaError::LoaderError(format!(
                "shape mismatch: residual_layers={}, expected {} — the scaffold's \
                 per-head view is fixed-size, bump it if the format grows",
                header.residual_layers, MEDUSA_RESIDUAL_LAYERS
            )));
        }

        if header.dtype != MEDUSA_DTYPE_FP16 {
            return Err(MedusaError::LoaderError(format!(
                "dtype mismatch: header dtype={}, expected {} (fp16) — no other \
                 dtype is wired into the device-upload path yet",
                header.dtype, MEDUSA_DTYPE_FP16
            )));
        }

        let expected_total = header.expected_file_bytes().ok_or_else(|| {
            MedusaError::LoaderError(format!(
                "shape overflow: header dims hd={} layers={} \
                 overflow usize when computing per-head bytes",
                header.hidden_dim, header.residual_layers
            ))
        })?;

        if bytes.len() != expected_total {
            return Err(MedusaError::LoaderError(format!(
                "file-size mismatch: got {} bytes, header implies {} \
                 ({} × {} heads + {} header)",
                bytes.len(),
                expected_total,
                header
                    .per_head_bytes()
                    .expect("already checked above via expected_file_bytes"),
                header.num_heads,
                MEDUSA_HEADER_BYTES,
            )));
        }

        // Per-head base offset cache. Filled after validation so the
        // offsets are guaranteed to fall inside the mapped range.
        let per_head = header
            .per_head_bytes()
            .expect("already checked above via expected_file_bytes");
        let head_offsets: Vec<usize> = (0..header.num_heads as usize)
            .map(|i| MEDUSA_HEADER_BYTES + i * per_head)
            .collect();

        Ok(Self {
            mmap,
            header,
            path: path.to_path_buf(),
            head_offsets,
        })
    }

    /// Parsed header — dims, version, counts.
    pub fn header(&self) -> &MedusaHeader {
        &self.header
    }

    /// Path the file was opened from. Useful for error messages and
    /// `/metrics`.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Byte length of the backing mmap. Lets callers sanity-check the
    /// mapped region without walking through the `Mmap` handle.
    pub fn mmap_len(&self) -> usize {
        self.mmap.len()
    }

    /// Borrowed view of head `i` (0..NUM_MEDUSA_HEADS).
    ///
    /// Returns `MedusaError::BadInput` if `i` is out of range — callers
    /// must iterate over `0..NUM_MEDUSA_HEADS`, not trust a raw usize.
    pub fn head(&self, i: usize) -> Result<MedusaHeadView<'_>, MedusaError> {
        if i >= NUM_MEDUSA_HEADS {
            return Err(MedusaError::BadInput(
                "medusa loader: head index out of range (must be 0..NUM_MEDUSA_HEADS)",
            ));
        }

        let hd = self.header.hidden_dim as usize;
        let bytes = &self.mmap[..];

        let base = self.head_offsets[i];

        // Per-head payload is w_in then w_out, each hd × hd fp16. Offsets
        // inside the head are deterministic by layout; the mmap length
        // was verified at open time so none of the slices can overrun.
        let w_elems = hd * hd;
        let w_bytes = w_elems * 2;

        let w_in = fp16_slice(&bytes[base..base + w_bytes], w_elems)?;
        let w_out = fp16_slice(&bytes[base + w_bytes..base + 2 * w_bytes], w_elems)?;

        Ok(MedusaHeadView { w_in, w_out })
    }
}

/// Read a little-endian `u32` from a 4-byte slice. Panics only if the
/// caller passes a non-4-byte slice — all call sites above are bounded
/// to exactly 4 bytes, so this is infallible in practice.
fn u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().expect("u32_le requires a 4-byte slice"))
}

/// Cast a `&[u8]` span of `nbytes = elems * 2` into a `&[u16]` view
/// over `elems` fp16 values. Uses `bytemuck::try_cast_slice` for the
/// alignment + length check — returns a `LoaderError` if either fails.
///
/// In practice alignment is not an issue: mmap regions are page-aligned
/// (4 KiB) and the header is 68 bytes so every per-head tensor starts
/// at an even offset inside the map. The check stays to catch a future
/// layout change that accidentally introduces a 1-byte field.
fn fp16_slice(bytes: &[u8], elems: usize) -> Result<&[u16], MedusaError> {
    if bytes.len() != elems * 2 {
        return Err(MedusaError::LoaderError(format!(
            "internal: fp16 slice length mismatch, got {} bytes for {} elems",
            bytes.len(),
            elems
        )));
    }
    bytemuck::try_cast_slice::<u8, u16>(bytes).map_err(|e| {
        MedusaError::LoaderError(format!(
            "fp16 alignment/length: {e} (len={}, elems={})",
            bytes.len(),
            elems
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a valid-shape header into `buf` with the given field values
    /// — lets individual tests mutate one field to exercise a failure
    /// mode without reproducing the header layout in every test.
    fn write_header(
        buf: &mut Vec<u8>,
        magic: &[u8; 8],
        version: u32,
        num_heads: u32,
        hidden_dim: u32,
        residual_layers: u32,
        dtype: u32,
    ) {
        buf.extend_from_slice(magic);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&num_heads.to_le_bytes());
        buf.extend_from_slice(&hidden_dim.to_le_bytes());
        buf.extend_from_slice(&residual_layers.to_le_bytes());
        buf.extend_from_slice(&dtype.to_le_bytes());
        // reserved: 10 × u32 of zero.
        buf.extend_from_slice(&[0u8; 40]);
        debug_assert_eq!(buf.len(), MEDUSA_HEADER_BYTES);
    }

    /// Create a file of the exact expected length for the given header,
    /// write the header bytes into the prefix, and return the temp-file
    /// handle + path.
    ///
    /// At v2 each head is 2 × 2560 × 2560 × 2 = 25 MiB, so the full
    /// canonical file is ~100 MiB. We write the full length with
    /// `set_len` — that's a sparse extent on Linux tmpfs/ext4, reads
    /// back as zeros without allocating pages. Cheap even on small
    /// runners.
    fn make_medusa_file(
        num_heads: u32,
        hidden_dim: u32,
        residual_layers: u32,
    ) -> (tempfile::NamedTempFile, PathBuf) {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let mut header_bytes = Vec::with_capacity(MEDUSA_HEADER_BYTES);
        write_header(
            &mut header_bytes,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            num_heads,
            hidden_dim,
            residual_layers,
            MEDUSA_DTYPE_FP16,
        );

        // Same math as `MedusaHeader::per_head_bytes`.
        let hd = hidden_dim as usize;
        let layers = residual_layers as usize;
        let per_head = layers * 2 * hd * hd * 2;
        let total = MEDUSA_HEADER_BYTES + per_head * num_heads as usize;

        tmp.write_all(&header_bytes).expect("write header");
        tmp.as_file_mut()
            .set_len(total as u64)
            .expect("sparse extend to total");
        let path = tmp.path().to_path_buf();
        (tmp, path)
    }

    /// Reject a file whose magic bytes don't match `MEDUSA1\0`.
    ///
    /// The loader must return `LoaderError` with a message that mentions
    /// both the expected and observed magic so an operator can tell at a
    /// glance that they pointed at the wrong file.
    #[test]
    fn rejects_bad_magic() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = Vec::new();
        write_header(
            &mut header,
            b"NOTMEDSA", // 8 bytes, wrong tag
            MEDUSA_FORMAT_VERSION,
            NUM_MEDUSA_HEADS as u32,
            MEDUSA_HIDDEN_DIM as u32,
            MEDUSA_RESIDUAL_LAYERS,
            MEDUSA_DTYPE_FP16,
        );
        tmp.write_all(&header).unwrap();
        // Extend to a plausible size so the error reason is definitely
        // "bad magic", not "short file".
        tmp.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 4096)
            .unwrap();

        let err = MedusaHeadsFile::open(tmp.path()).expect_err("bad magic must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(msg.contains("magic"), "message must mention 'magic': {msg}");
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }
    }

    /// Reject a file whose declared shape doesn't match the scaffold's
    /// pinned constants.
    ///
    /// Two variants exercised in one test:
    ///   * `num_heads` != NUM_MEDUSA_HEADS (ABI break)
    ///   * `hidden_dim` != MEDUSA_HIDDEN_DIM (backbone mismatch)
    ///
    /// Both must surface as `LoaderError` so the operator sees which
    /// invariant failed.
    #[test]
    fn rejects_shape_mismatch() {
        // Variant A: num_heads=5 (scaffold expects 4). Use a tiny
        // hidden_dim so the file stays minimal — the shape check
        // fires on `num_heads` before file-size math runs.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = Vec::new();
        write_header(
            &mut header,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            5, // bad: not NUM_MEDUSA_HEADS
            MEDUSA_HIDDEN_DIM as u32,
            MEDUSA_RESIDUAL_LAYERS,
            MEDUSA_DTYPE_FP16,
        );
        tmp.write_all(&header).unwrap();
        tmp.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 64)
            .unwrap();
        let err = MedusaHeadsFile::open(tmp.path()).expect_err("bad num_heads must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(
                    msg.contains("num_heads"),
                    "message must mention num_heads: {msg}"
                );
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }

        // Variant B: hidden_dim mismatch. num_heads is right so we
        // reach the second shape check.
        let mut tmp2 = tempfile::NamedTempFile::new().unwrap();
        let mut header2 = Vec::new();
        write_header(
            &mut header2,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            NUM_MEDUSA_HEADS as u32,
            1024, // bad: not MEDUSA_HIDDEN_DIM (2560)
            MEDUSA_RESIDUAL_LAYERS,
            MEDUSA_DTYPE_FP16,
        );
        tmp2.write_all(&header2).unwrap();
        tmp2.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 64)
            .unwrap();
        let err = MedusaHeadsFile::open(tmp2.path()).expect_err("bad hidden_dim must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(
                    msg.contains("hidden_dim"),
                    "message must mention hidden_dim: {msg}"
                );
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }
    }

    /// Reject a file whose dtype tag isn't fp16. Protects against a
    /// future bf16/i4 variant being silently accepted by a loader that
    /// only knows fp16.
    #[test]
    fn rejects_bad_dtype() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut header = Vec::new();
        write_header(
            &mut header,
            &MEDUSA_MAGIC,
            MEDUSA_FORMAT_VERSION,
            NUM_MEDUSA_HEADS as u32,
            MEDUSA_HIDDEN_DIM as u32,
            MEDUSA_RESIDUAL_LAYERS,
            42, // unrecognized dtype
        );
        tmp.write_all(&header).unwrap();
        tmp.as_file_mut()
            .set_len(MEDUSA_HEADER_BYTES as u64 + 64)
            .unwrap();
        let err = MedusaHeadsFile::open(tmp.path()).expect_err("bad dtype must be rejected");
        match err {
            MedusaError::LoaderError(msg) => {
                assert!(msg.contains("dtype"), "message must mention dtype: {msg}");
            }
            other => panic!("expected LoaderError, got {other:?}"),
        }
    }

    /// Synthesize a well-formed `.h1b-medusa` file at the canonical
    /// shape, reopen it, and walk every per-head view to confirm the
    /// lengths match spec. This is the successful-path smoke test — it
    /// does not validate tensor contents (payload is sparse zeros).
    #[test]
    fn parses_synthetic_file() {
        // Canonical shape: NUM_MEDUSA_HEADS × MEDUSA_HIDDEN_DIM × 1-layer.
        // The file is ~100 MiB on disk (sparse), so actual block
        // allocation on tmpfs/ext4 is a few KiB.
        let num_heads = NUM_MEDUSA_HEADS as u32;
        let hidden_dim = MEDUSA_HIDDEN_DIM as u32;
        let (_tmp, path) = make_medusa_file(num_heads, hidden_dim, MEDUSA_RESIDUAL_LAYERS);

        let file = MedusaHeadsFile::open(&path).expect("successful open");

        // Header round-trip.
        let hdr = file.header();
        assert_eq!(hdr.version, MEDUSA_FORMAT_VERSION);
        assert_eq!(hdr.num_heads, num_heads);
        assert_eq!(hdr.hidden_dim, hidden_dim);
        assert_eq!(hdr.residual_layers, MEDUSA_RESIDUAL_LAYERS);
        assert_eq!(hdr.dtype, MEDUSA_DTYPE_FP16);

        // File-size math matches header.
        assert_eq!(
            file.mmap_len(),
            hdr.expected_file_bytes()
                .expect("no overflow at canonical shape"),
        );

        // Per-head bytes: 2 × 2560 × 2560 × 2 = 26_214_400.
        assert_eq!(
            hdr.per_head_bytes().unwrap(),
            2 * MEDUSA_HIDDEN_DIM * MEDUSA_HIDDEN_DIM * 2
        );

        // Every head view is the right length.
        let hd = hidden_dim as usize;
        let expected_w_len = hd * hd;

        for i in 0..NUM_MEDUSA_HEADS {
            let view = file.head(i).expect("head within bounds");
            assert_eq!(view.w_in.len(), expected_w_len, "head {i} w_in length");
            assert_eq!(view.w_out.len(), expected_w_len, "head {i} w_out length");
        }

        // Out-of-range head index must be a structured error, not a panic.
        let err = file
            .head(NUM_MEDUSA_HEADS)
            .expect_err("out-of-range head must error");
        assert!(matches!(err, MedusaError::BadInput(_)));
    }
}
