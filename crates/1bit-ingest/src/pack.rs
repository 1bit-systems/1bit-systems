//! `pack` — TLV writer. Produces a `.1bl` from:
//!
//! * a trained `.gguf` (required, goes into `MODEL_GGUF` / 0x01),
//! * a `catalog.toml` (required, drives the CBOR header + ATTRIBUTION /
//!   LICENSE text sections),
//! * optional cover art (`COVER` / 0x02),
//! * optional lyrics bundle (`TRACK_LYRICS` / 0x03).
//!
//! See `docs/wiki/1bl-container-spec.md` §"Section order" for why the
//! GGUF is always written first and why the residual (if any) always
//! goes at the tail.

use std::fs::{File, read};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use byteorder::{LittleEndian, WriteBytesExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{MAGIC, tag};

/// Strictly-typed mirror of the CBOR manifest the spec §"Header" requires.
/// Intentionally kept flat so a curator can read it straight out of their
/// `catalog.toml` without learning CBOR. We serialise the header via
/// `ciborium` on the way out and via `serde` on the way back in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub v: String,
    pub catalog: String,
    pub title: String,
    pub artist: String,
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub created: String,
    pub tier: String,
    pub codec: CodecMeta,
    pub model: ModelMeta,
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub residual_present: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecMeta {
    pub audio: String,
    pub sample_rate: u32,
    pub channels: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub arch: String,
    pub params: u64,
    pub bpw: f64,
    /// Set by [`pack`] after hashing the embedded GGUF — the curator
    /// leaves this blank in `catalog.toml`.
    #[serde(default)]
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub id: String,
    pub title: String,
    pub length_ms: u64,
    #[serde(default)]
    pub sha256: String,
}

/// Top-level `catalog.toml` schema. Lives in a separate struct from
/// [`Manifest`] so the curator's file doesn't have to include derived
/// fields (header version tag, model sha, residual_present flag).
#[derive(Debug, Clone, Deserialize)]
pub struct CatalogToml {
    pub catalog: String,
    pub title: String,
    pub artist: String,
    pub license: String,
    #[serde(default)]
    pub license_url: Option<String>,
    #[serde(default)]
    pub attribution: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    pub created: String,
    #[serde(default = "default_tier")]
    pub tier: String,
    pub codec: CodecMeta,
    pub model: ModelMeta,
    #[serde(default)]
    pub tracks: Vec<Track>,
    /// Inline license text. If absent, curator must provide
    /// `license_txt_path`.
    #[serde(default)]
    pub license_txt: Option<String>,
    #[serde(default)]
    pub license_txt_path: Option<String>,
    /// Inline attribution block. If absent and `attribution` is set,
    /// we synthesise a one-liner.
    #[serde(default)]
    pub attribution_txt: Option<String>,
}

fn default_tier() -> String {
    "lossy".into()
}

#[derive(Debug, Clone, Copy)]
pub struct PackSummary {
    pub section_count: usize,
    pub total_bytes: u64,
}

/// Pack a `.1bl`. The curator owns the `.toml` schema; this function owns
/// the byte layout.
pub fn pack(
    model_path: &Path,
    manifest_toml_path: &Path,
    cover_path: Option<&Path>,
    lyrics_path: Option<&Path>,
    out_path: &Path,
) -> Result<PackSummary> {
    // --- Load inputs ---------------------------------------------------
    let gguf_bytes = read(model_path)
        .with_context(|| format!("read model gguf: {}", model_path.display()))?;
    let toml_text = std::fs::read_to_string(manifest_toml_path)
        .with_context(|| format!("read catalog.toml: {}", manifest_toml_path.display()))?;
    let cat: CatalogToml = toml::from_str(&toml_text)
        .context("parse catalog.toml")?;

    // --- Derive fields -------------------------------------------------
    let gguf_sha = sha256_hex(&gguf_bytes);
    let license_txt = resolve_license_text(&cat, manifest_toml_path)?;
    let attribution_txt = cat.attribution_txt.clone().unwrap_or_else(|| {
        cat.attribution.clone().unwrap_or_else(|| cat.artist.clone())
    });

    let mut model = cat.model.clone();
    model.sha256 = gguf_sha;

    let manifest = Manifest {
        v: "0.1".into(),
        catalog: cat.catalog.clone(),
        title: cat.title.clone(),
        artist: cat.artist.clone(),
        license: cat.license.clone(),
        license_url: cat.license_url.clone(),
        attribution: cat.attribution.clone(),
        source: cat.source.clone(),
        created: cat.created.clone(),
        tier: cat.tier.clone(),
        codec: cat.codec.clone(),
        model,
        tracks: cat.tracks.clone(),
        residual_present: false,
        residual_sha256: None,
    };

    // --- Sections ------------------------------------------------------
    let cover_bytes = match cover_path {
        Some(p) => Some(read(p).with_context(|| format!("read cover: {}", p.display()))?),
        None => None,
    };
    let lyrics_bytes = match lyrics_path {
        Some(p) => Some(read(p).with_context(|| format!("read lyrics: {}", p.display()))?),
        None => None,
    };

    // --- Write out -----------------------------------------------------
    let out = File::create(out_path)
        .with_context(|| format!("create out: {}", out_path.display()))?;
    let mut w = HashingWriter::new(BufWriter::new(out));

    w.write_all(&MAGIC).context("write magic")?;

    let mut header_cbor = Vec::new();
    ciborium::into_writer(&manifest, &mut header_cbor)
        .map_err(|e| anyhow!("cbor encode header: {e}"))?;
    w.write_u32::<LittleEndian>(header_cbor.len() as u32)
        .context("write header length")?;
    w.write_all(&header_cbor).context("write header cbor")?;

    let mut section_count = 0usize;
    write_section(&mut w, tag::MODEL_GGUF, &gguf_bytes)?; section_count += 1;
    write_section(&mut w, tag::ATTRIBUTION_TXT, attribution_txt.as_bytes())?; section_count += 1;
    write_section(&mut w, tag::LICENSE_TXT, license_txt.as_bytes())?; section_count += 1;
    if let Some(b) = &cover_bytes { write_section(&mut w, tag::COVER, b)?; section_count += 1; }
    if let Some(b) = &lyrics_bytes { write_section(&mut w, tag::TRACK_LYRICS, b)?; section_count += 1; }

    // Footer = SHA-256 of everything written so far. Write the digest
    // directly (footer is NOT covered by itself).
    let digest = w.finalize_digest();
    let mut inner = w.into_inner();
    inner.write_all(&digest).context("write footer hash")?;
    inner.flush().context("flush output")?;

    let total_bytes = std::fs::metadata(out_path)
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(PackSummary { section_count, total_bytes })
}

fn resolve_license_text(cat: &CatalogToml, toml_path: &Path) -> Result<String> {
    if let Some(t) = &cat.license_txt {
        return Ok(t.clone());
    }
    if let Some(rel) = &cat.license_txt_path {
        let base = toml_path.parent().unwrap_or(Path::new("."));
        let p = base.join(rel);
        let s = std::fs::read_to_string(&p)
            .with_context(|| format!("read license_txt_path: {}", p.display()))?;
        return Ok(s);
    }
    Err(anyhow!(
        "catalog.toml must set either `license_txt` (inline) or `license_txt_path` (sidecar). \
         Spec §\"Section tags\" lists LICENSE_TXT (0x05) as required."
    ))
}

pub(crate) fn write_section<W: Write>(
    w: &mut W,
    section_tag: u8,
    payload: &[u8],
) -> Result<()> {
    w.write_u8(section_tag).context("write section tag")?;
    w.write_u64::<LittleEndian>(payload.len() as u64).context("write section length")?;
    w.write_all(payload).context("write section payload")?;
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in &d[..] {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Writer that both passes bytes through to the inner `Write` and feeds
/// them into a running SHA-256, so the caller can append the digest as
/// a footer without re-reading the file.
pub(crate) struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
}

impl<W: Write> HashingWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self { inner, hasher: Sha256::new() }
    }

    pub(crate) fn finalize_digest(&mut self) -> [u8; 32] {
        let taken = std::mem::replace(&mut self.hasher, Sha256::new());
        let out = taken.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&out);
        arr
    }

    pub(crate) fn into_inner(self) -> W { self.inner }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml(cat: &str) -> String {
        format!(r#"
catalog = "{cat}"
title = "Test Catalog"
artist = "Test Artist"
license = "CC0-1.0"
license_url = "https://creativecommons.org/publicdomain/zero/1.0/"
attribution = "Test Artist"
created = "2026-04-23T00:00:00Z"
tier = "lossy"
license_txt = "CC0 — public domain dedication."

[codec]
audio = "mimi-12hz"
sample_rate = 24000
channels = 2

[model]
arch = "bitnet-1p58"
params = 1048576
bpw = 1.58

[[tracks]]
id = "01"
title = "Track One"
length_ms = 60000
"#)
    }

    #[test]
    fn pack_writes_magic_header_and_sections() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"GGUF\0\0\0\0fake weights bytes here").unwrap();

        let tomlp = tmp.path().join("catalog.toml");
        std::fs::write(&tomlp, sample_toml("test-pack")).unwrap();

        let out = tmp.path().join("test.1bl");
        let summary = pack(&gguf, &tomlp, None, None, &out).unwrap();
        assert_eq!(summary.section_count, 3); // MODEL_GGUF + ATTRIBUTION + LICENSE
        assert!(summary.total_bytes > 32);

        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[..4], &MAGIC);
    }
}
