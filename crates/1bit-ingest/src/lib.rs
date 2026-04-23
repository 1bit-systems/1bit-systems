//! `1bit-ingest` — source-side tooling for catalog curators.
//!
//! Curators live upstream of the `.1bl` reader implemented in
//! [`onebit_stream`]. Their job is three short hops:
//!
//! 1. **prepare** — walk a directory of FLAC files, sidecar-manifest the
//!    metadata, and tar up a corpus ready to `scp` to the RunPod pod
//!    that runs the actual ternary-LM training (Run 8). This crate
//!    never touches Mimi tokens — that's pod-side.
//! 2. **pack** — assemble a `.1bl` from a trained `.gguf`, a
//!    hand-written `catalog.toml`, optional `cover.webp`, and optional
//!    per-track lyrics. Writes the TLV layout per
//!    `docs/wiki/1bl-container-spec.md`, with a CBOR header at the
//!    front and a SHA-256 footer at the back.
//! 3. **validate** — read a `.1bl` back, verify the footer hash,
//!    pretty-print the manifest, list TLV section tags + sizes.
//!
//! A fourth verb, **add-residual**, is the "Premium upgrade" path: take
//! a lossy-only `.1bl`, append `RESIDUAL_BLOB` + `RESIDUAL_INDEX`, and
//! rewrite the trailing SHA-256 so the file stays valid.
//!
//! None of the verbs train, invoke, or upload anything. Training runs
//! on a RunPod H200 pod; upload is a plain `scp` the curator does by
//! hand. This crate is a pure local-file-tool.

pub mod prepare;
pub mod pack;
pub mod validate;
pub mod residual;

/// `.1bl` magic bytes: ASCII "1BL" followed by the version byte.
/// Version byte is `0x01` for spec v0.1.
pub const MAGIC: [u8; 4] = *b"1BL\x01";

/// Spec version byte. Bump to `0x02` on any incompatible layout change.
pub const VERSION: u8 = 0x01;

/// Section tag constants (see spec §"Section tags").
pub mod tag {
    pub const MODEL_GGUF: u8 = 0x01;
    pub const COVER: u8 = 0x02;
    pub const TRACK_LYRICS: u8 = 0x03;
    pub const ATTRIBUTION_TXT: u8 = 0x04;
    pub const LICENSE_TXT: u8 = 0x05;
    pub const RESIDUAL_BLOB: u8 = 0x10;
    pub const RESIDUAL_INDEX: u8 = 0x11;
}

/// Errors raised by the ingest pipeline.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("bad magic bytes (got {0:02x?})")]
    BadMagic([u8; 4]),
    #[error("unsupported version byte: {0:#04x}")]
    BadVersion(u8),
    #[error("footer hash mismatch (file corrupt or truncated)")]
    FooterHashMismatch,
    #[error("truncated file: {0}")]
    Truncated(&'static str),
    #[error("header too large: {0} bytes (cap is 16 MiB)")]
    HeaderTooLarge(u64),
    #[error("cbor decode: {0}")]
    Cbor(String),
}
