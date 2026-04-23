//! `validate` — read a `.1bl`, verify the footer SHA-256, pretty-print
//! the manifest, list section tags + sizes.
//!
//! A fast pass: we mmap-read the entire file, hash everything except
//! the trailing 32 bytes, and compare against the footer. A mismatch
//! surfaces as [`crate::IngestError::FooterHashMismatch`].

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use sha2::{Digest, Sha256};

use crate::{IngestError, MAGIC, VERSION};
use crate::pack::Manifest;

/// Human-printable summary of a validated `.1bl`.
#[derive(Debug, Clone)]
pub struct ValidateReport {
    pub total_bytes: u64,
    pub version: u8,
    pub manifest: Manifest,
    pub sections: Vec<SectionRecord>,
    pub footer_ok: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SectionRecord {
    pub tag: u8,
    pub offset: u64,
    pub length: u64,
}

impl std::fmt::Display for ValidateReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "1bl container: {} bytes, version {:#04x}", self.total_bytes, self.version)?;
        writeln!(f, "  catalog  : {}", self.manifest.catalog)?;
        writeln!(f, "  title    : {}", self.manifest.title)?;
        writeln!(f, "  artist   : {}", self.manifest.artist)?;
        writeln!(f, "  license  : {}", self.manifest.license)?;
        writeln!(f, "  tier     : {}", self.manifest.tier)?;
        writeln!(f, "  tracks   : {}", self.manifest.tracks.len())?;
        writeln!(f, "  model    : {} ({} params, {:.2} bpw)",
                 self.manifest.model.arch, self.manifest.model.params, self.manifest.model.bpw)?;
        writeln!(f, "  footer   : {}", if self.footer_ok { "OK" } else { "MISMATCH" })?;
        writeln!(f, "  sections : {}", self.sections.len())?;
        for s in &self.sections {
            writeln!(f, "    tag {:#04x} @ offset {:>10}  len {:>12}", s.tag, s.offset, s.length)?;
        }
        Ok(())
    }
}

/// Read + validate a `.1bl`. Returns a [`ValidateReport`].
pub fn validate(path: &Path) -> Result<ValidateReport> {
    let mut f = File::open(path)
        .with_context(|| format!("open 1bl: {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).context("read 1bl into memory")?;

    parse_bytes(&buf)
}

/// Shared entry point used by both `validate` and `add-residual`. Parses
/// the full `.1bl` byte stream.
pub fn parse_bytes(buf: &[u8]) -> Result<ValidateReport> {
    if buf.len() < 4 + 4 + 32 {
        return Err(IngestError::Truncated("file too small for magic+header+footer").into());
    }

    // Magic (4 bytes, version is the last byte).
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&buf[..4]);
    if &magic[..3] != &MAGIC[..3] {
        return Err(IngestError::BadMagic(magic).into());
    }
    let version = magic[3];
    if version != VERSION {
        return Err(IngestError::BadVersion(version).into());
    }

    // CBOR header.
    let header_len = LittleEndian::read_u32(&buf[4..8]) as usize;
    if header_len > 16 * 1024 * 1024 {
        return Err(IngestError::HeaderTooLarge(header_len as u64).into());
    }
    let header_start: usize = 8;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or(IngestError::Truncated("header length overflow"))?;
    if header_end + 32 > buf.len() {
        return Err(IngestError::Truncated("header extends past EOF").into());
    }
    let header_bytes = &buf[header_start..header_end];
    let manifest: Manifest = ciborium::from_reader(header_bytes)
        .map_err(|e| IngestError::Cbor(e.to_string()))?;

    // Sections run from header_end to buf.len() - 32.
    let sections_end = buf.len() - 32;
    let mut sections = Vec::new();
    let mut cursor = header_end;
    while cursor < sections_end {
        if sections_end - cursor < 1 + 8 {
            return Err(IngestError::Truncated("partial section header").into());
        }
        let tag = buf[cursor];
        let len = LittleEndian::read_u64(&buf[cursor + 1 .. cursor + 9]);
        let payload_start = cursor + 9;
        let payload_end = payload_start
            .checked_add(len as usize)
            .ok_or(IngestError::Truncated("section length overflow"))?;
        if payload_end > sections_end {
            return Err(IngestError::Truncated("section payload past sections_end").into());
        }
        sections.push(SectionRecord {
            tag,
            offset: payload_start as u64,
            length: len,
        });
        cursor = payload_end;
    }
    if cursor != sections_end {
        return Err(IngestError::Truncated("leftover bytes before footer").into());
    }

    // Footer.
    let mut h = Sha256::new();
    h.update(&buf[..sections_end]);
    let expected = h.finalize();
    let footer = &buf[sections_end..];
    let footer_ok = footer == expected.as_slice();
    if !footer_ok {
        return Err(IngestError::FooterHashMismatch.into());
    }

    Ok(ValidateReport {
        total_bytes: buf.len() as u64,
        version,
        manifest,
        sections,
        footer_ok,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack;

    fn sample_toml(cat: &str) -> String {
        format!(r#"
catalog = "{cat}"
title = "Test"
artist = "T"
license = "CC0-1.0"
created = "2026-04-23T00:00:00Z"
tier = "lossy"
license_txt = "public domain"

[codec]
audio = "mimi-12hz"
sample_rate = 24000
channels = 2

[model]
arch = "bitnet-1p58"
params = 1000
bpw = 1.58

[[tracks]]
id = "01"
title = "A"
length_ms = 1000
"#)
    }

    #[test]
    fn pack_then_validate_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"GGUF fake weights").unwrap();
        let tomlp = tmp.path().join("catalog.toml");
        std::fs::write(&tomlp, sample_toml("roundtrip")).unwrap();
        let out = tmp.path().join("rt.1bl");

        pack::pack(&gguf, &tomlp, None, None, &out).unwrap();
        let report = validate(&out).unwrap();
        assert_eq!(report.version, VERSION);
        assert!(report.footer_ok);
        assert_eq!(report.manifest.catalog, "roundtrip");
        // Three sections: MODEL_GGUF, ATTRIBUTION_TXT, LICENSE_TXT.
        assert_eq!(report.sections.len(), 3);
        assert_eq!(report.sections[0].tag, crate::tag::MODEL_GGUF);
    }

    #[test]
    fn corrupted_footer_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"GGUF fake").unwrap();
        let tomlp = tmp.path().join("catalog.toml");
        std::fs::write(&tomlp, sample_toml("corrupt")).unwrap();
        let out = tmp.path().join("c.1bl");
        pack::pack(&gguf, &tomlp, None, None, &out).unwrap();

        // Flip a byte just before the footer. This lands inside a
        // section payload (not the CBOR header, not the footer itself),
        // so validate() should reach the footer-hash check and fail
        // there — not earlier on a CBOR parse error.
        let mut bytes = std::fs::read(&out).unwrap();
        let target = bytes.len() - 33;
        bytes[target] ^= 0xff;
        std::fs::write(&out, &bytes).unwrap();

        let err = validate(&out).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("footer hash mismatch"),
            "expected footer mismatch, got: {msg}"
        );
    }

    #[test]
    fn pack_includes_optional_cover_and_lyrics() {
        // A curator who supplies cover + lyrics should see five TLV
        // sections land in the output, in the order declared by spec
        // §"Section order": MODEL_GGUF, ATTRIBUTION_TXT, LICENSE_TXT,
        // COVER, TRACK_LYRICS.
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"GGUF fake").unwrap();
        let tomlp = tmp.path().join("catalog.toml");
        std::fs::write(&tomlp, sample_toml("optsec")).unwrap();
        let cover = tmp.path().join("cover.webp");
        std::fs::write(&cover, b"RIFF\0\0WEBPVP8 fake").unwrap();
        let lyrics = tmp.path().join("lyrics.txt");
        std::fs::write(&lyrics, b"[01] la la la\n").unwrap();
        let out = tmp.path().join("o.1bl");

        pack::pack(&gguf, &tomlp, Some(&cover), Some(&lyrics), &out).unwrap();
        let report = validate(&out).unwrap();
        assert!(report.footer_ok);
        assert_eq!(report.sections.len(), 5);
        let tags: Vec<u8> = report.sections.iter().map(|s| s.tag).collect();
        assert_eq!(
            tags,
            vec![
                crate::tag::MODEL_GGUF,
                crate::tag::ATTRIBUTION_TXT,
                crate::tag::LICENSE_TXT,
                crate::tag::COVER,
                crate::tag::TRACK_LYRICS,
            ]
        );
    }
}
