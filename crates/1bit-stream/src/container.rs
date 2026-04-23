//! `.1bl` container reader — spec v0.1.
//!
//! See `docs/wiki/1bl-container-spec.md`. This module only implements the
//! read side: magic + version, CBOR manifest header, streaming TLV section
//! iterator, footer SHA-256 verification. The arithmetic-coded residual in
//! sections `0x10` / `0x11` is opaque here — we hand the byte range to the
//! HTTP layer and let the client decode.
//!
//! The parser is deliberately resilient: unknown tags are skipped so
//! forward-compat v0.2+ files loaded by a v0.1 server still list cleanly
//! on `/v1/catalogs`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// 4-byte magic, little-endian.  `1BL\x01` — version byte is last.
pub const MAGIC: [u8; 4] = *b"1BL\x01";
/// Accepted format version. Bumping here requires a spec update.
pub const FORMAT_VERSION: u8 = 0x01;

// Section tags defined in the spec. Kept as associated consts rather than
// an enum so unknown tags (forward compat) don't need a separate variant.
pub mod tag {
    pub const MODEL_GGUF: u8 = 0x01;
    pub const COVER: u8 = 0x02;
    pub const TRACK_LYRICS: u8 = 0x03;
    pub const ATTRIBUTION_TXT: u8 = 0x04;
    pub const LICENSE_TXT: u8 = 0x05;
    pub const RESIDUAL_BLOB: u8 = 0x10;
    pub const RESIDUAL_INDEX: u8 = 0x11;
}

#[derive(Debug, Error)]
pub enum ContainerError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("bad magic: expected 1BL + version 0x01")]
    BadMagic,
    #[error("unsupported format version {0:#x}")]
    UnsupportedVersion(u8),
    #[error("truncated: {0}")]
    Truncated(&'static str),
    #[error("manifest cbor decode: {0}")]
    Cbor(String),
    #[error("footer sha256 mismatch")]
    FooterMismatch,
}

/// CBOR manifest decoded from the header block. Unknown fields are dropped
/// silently so a newer writer doesn't break older readers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub v: String,
    pub catalog: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub artist: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub license_url: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub created: String,
    /// `"lossy" | "premium" | "both"`.
    #[serde(default)]
    pub tier: String,
    #[serde(default)]
    pub codec: serde_json::Value,
    #[serde(default)]
    pub model: serde_json::Value,
    #[serde(default)]
    pub tracks: Vec<serde_json::Value>,
    #[serde(default)]
    pub residual_present: bool,
    #[serde(default)]
    pub residual_sha256: String,
}

/// A single TLV section entry discovered during scan. Byte offsets are
/// absolute into the file so the HTTP layer can stream a sub-range without
/// re-parsing.
#[derive(Debug, Clone)]
pub struct Section {
    pub tag: u8,
    /// Absolute offset of the first content byte (after the u8 tag +
    /// u64 length header).
    pub offset: u64,
    pub length: u64,
}

impl Section {
    /// Lossy-tier sections are everything except RESIDUAL_BLOB + INDEX.
    pub fn is_lossy_tier(&self) -> bool {
        !matches!(self.tag, tag::RESIDUAL_BLOB | tag::RESIDUAL_INDEX)
    }
}

/// Parsed catalog. Holds the path + the decoded manifest + the TLV section
/// index. Does NOT hold file bytes; the handlers re-open the file to stream.
#[derive(Debug, Clone)]
pub struct Catalog {
    pub path: PathBuf,
    pub manifest: Manifest,
    pub sections: Vec<Section>,
    /// Total file length in bytes (including footer).
    pub total_bytes: u64,
    /// Offset at which the 32-byte footer SHA-256 begins.
    pub footer_offset: u64,
}

impl Catalog {
    /// Slug derived from the manifest `catalog` field. Used as the URL
    /// path segment in `/v1/catalogs/:slug`.
    pub fn slug(&self) -> &str {
        &self.manifest.catalog
    }

    /// Scan a file at `path`, parse the header + index the sections, and
    /// verify the footer SHA-256. Expensive enough that the server only
    /// runs it at reindex time.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContainerError> {
        let path = path.as_ref().to_path_buf();
        let mut f = BufReader::new(File::open(&path)?);

        // --- magic + version ------------------------------------------------
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic[..3] != MAGIC[..3] {
            return Err(ContainerError::BadMagic);
        }
        if magic[3] != FORMAT_VERSION {
            return Err(ContainerError::UnsupportedVersion(magic[3]));
        }

        // --- CBOR manifest --------------------------------------------------
        let header_len = read_u32_le(&mut f)? as usize;
        let mut header_buf = vec![0u8; header_len];
        f.read_exact(&mut header_buf)?;
        let manifest: Manifest = ciborium::de::from_reader(&header_buf[..])
            .map_err(|e| ContainerError::Cbor(e.to_string()))?;

        // --- scan TLV sections ---------------------------------------------
        let total_bytes = std::fs::metadata(&path)?.len();
        let footer_offset = total_bytes
            .checked_sub(32)
            .ok_or(ContainerError::Truncated("file smaller than footer"))?;

        let mut sections = Vec::new();
        loop {
            let here = f.stream_position()?;
            if here >= footer_offset {
                break;
            }
            // Peek tag; if we're exactly at footer_offset we're done.
            let mut tagb = [0u8; 1];
            if f.read(&mut tagb)? == 0 {
                break;
            }
            let tag = tagb[0];
            let len = read_u64_le(&mut f)?;
            let content_offset = f.stream_position()?;
            // Bounds check.
            let end = content_offset
                .checked_add(len)
                .ok_or(ContainerError::Truncated("section length overflow"))?;
            if end > footer_offset {
                return Err(ContainerError::Truncated("section runs past footer"));
            }
            sections.push(Section {
                tag,
                offset: content_offset,
                length: len,
            });
            f.seek(SeekFrom::Start(end))?;
        }

        // --- footer ---------------------------------------------------------
        let mut stored_footer = [0u8; 32];
        f.seek(SeekFrom::Start(footer_offset))?;
        f.read_exact(&mut stored_footer)?;

        let computed = sha256_prefix(&path, footer_offset)?;
        if computed != stored_footer {
            return Err(ContainerError::FooterMismatch);
        }

        Ok(Self {
            path,
            manifest,
            sections,
            total_bytes,
            footer_offset,
        })
    }

    /// Iterator of sections kept when serving the lossy tier. Skips 0x10
    /// and 0x11; includes everything else (including unknown forward-compat
    /// tags, which the client may safely skip).
    pub fn lossy_sections(&self) -> impl Iterator<Item = &Section> {
        self.sections.iter().filter(|s| s.is_lossy_tier())
    }
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// SHA-256 over the first `len` bytes of `path`. Used for footer verify.
fn sha256_prefix(path: &Path, len: u64) -> io::Result<[u8; 32]> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut remaining = len;
    let mut buf = [0u8; 64 * 1024];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let got = f.read(&mut buf[..want])?;
        if got == 0 {
            break;
        }
        hasher.update(&buf[..got]);
        remaining -= got as u64;
    }
    Ok(hasher.finalize().into())
}

/// Test-only container builder. Writes a minimal valid `.1bl` to `path`
/// with the supplied manifest and sections. Returns the path for chaining.
#[cfg(test)]
pub fn write_test_container(
    path: &Path,
    manifest: &Manifest,
    sections: &[(u8, Vec<u8>)],
) -> Result<(), ContainerError> {
    use std::io::Write;
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(&MAGIC);
    let mut cbor = Vec::<u8>::new();
    ciborium::ser::into_writer(manifest, &mut cbor)
        .map_err(|e| ContainerError::Cbor(e.to_string()))?;
    buf.extend_from_slice(&(cbor.len() as u32).to_le_bytes());
    buf.extend_from_slice(&cbor);
    for (tag, content) in sections {
        buf.push(*tag);
        buf.extend_from_slice(&(content.len() as u64).to_le_bytes());
        buf.extend_from_slice(content);
    }
    let hash: [u8; 32] = Sha256::digest(&buf).into();
    buf.extend_from_slice(&hash);
    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_manifest() -> Manifest {
        Manifest {
            v: "0.1".into(),
            catalog: "test-cat".into(),
            title: "Test".into(),
            artist: "Tester".into(),
            license: "CC-BY-4.0".into(),
            license_url: "https://example.com/cc-by".into(),
            attribution: "Tester".into(),
            source: "https://example.com".into(),
            created: "2026-04-23T00:00:00Z".into(),
            tier: "both".into(),
            codec: serde_json::json!({ "audio": "mimi-12hz" }),
            model: serde_json::json!({ "arch": "bitnet-1p58" }),
            tracks: vec![],
            residual_present: true,
            residual_sha256: "deadbeef".into(),
        }
    }

    #[test]
    fn roundtrip_minimal() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.1bl");
        let m = sample_manifest();
        let sections = vec![
            (tag::MODEL_GGUF, b"fake-gguf-bytes".to_vec()),
            (tag::ATTRIBUTION_TXT, b"credit: tester".to_vec()),
            (tag::LICENSE_TXT, b"CC-BY-4.0".to_vec()),
            (tag::RESIDUAL_BLOB, b"opaque-residual".to_vec()),
            (tag::RESIDUAL_INDEX, b"\x00\x01".to_vec()),
        ];
        write_test_container(&path, &m, &sections).unwrap();

        let cat = Catalog::open(&path).unwrap();
        assert_eq!(cat.slug(), "test-cat");
        assert_eq!(cat.sections.len(), 5);
        let lossy: Vec<u8> = cat.lossy_sections().map(|s| s.tag).collect();
        assert_eq!(lossy, vec![tag::MODEL_GGUF, tag::ATTRIBUTION_TXT, tag::LICENSE_TXT]);
    }

    #[test]
    fn footer_tamper_detected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.1bl");
        write_test_container(&path, &sample_manifest(), &[]).unwrap();

        // Flip the first byte of the footer.
        let mut bytes = std::fs::read(&path).unwrap();
        let flen = bytes.len();
        bytes[flen - 32] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        match Catalog::open(&path) {
            Err(ContainerError::FooterMismatch) => {}
            other => panic!("expected FooterMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tag_is_kept_as_section_and_ignored_on_lossy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.1bl");
        write_test_container(
            &path,
            &sample_manifest(),
            &[
                (tag::MODEL_GGUF, b"a".to_vec()),
                (0x88u8, b"future-tag-content".to_vec()),
                (tag::RESIDUAL_BLOB, b"b".to_vec()),
            ],
        )
        .unwrap();
        let cat = Catalog::open(&path).unwrap();
        assert_eq!(cat.sections.len(), 3);
        // Unknown tag 0x88 is classified as lossy-tier (not 0x10/0x11).
        let lossy: Vec<u8> = cat.lossy_sections().map(|s| s.tag).collect();
        assert_eq!(lossy, vec![tag::MODEL_GGUF, 0x88]);
    }

    #[test]
    fn bad_magic_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.1bl");
        std::fs::write(&path, b"NOPE\x00\x00\x00\x00").unwrap();
        assert!(matches!(
            Catalog::open(&path),
            Err(ContainerError::BadMagic | ContainerError::Truncated(_) | ContainerError::Io(_))
        ));
    }
}
