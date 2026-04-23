//! `add-residual` — take a lossy-tier `.1bl`, append `RESIDUAL_BLOB`
//! (`0x10`) + `RESIDUAL_INDEX` (`0x11`), and rewrite the trailing
//! SHA-256 footer.
//!
//! This is the Premium upgrade path. Per spec §"Section order" the
//! residual sections sit at the tail so the upgrade is strictly
//! append-only (we do rewrite the footer, but we don't rewrite any
//! earlier sections). The input file is not modified — we stream it
//! into a new output.

use std::fs::{File, read};
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::pack::{HashingWriter, sha256_hex, write_section};
use crate::{tag, validate};

/// Summary returned by [`add_residual`].
#[derive(Debug, Clone, Copy)]
pub struct ResidualSummary {
    pub residual_bytes: u64,
    pub index_bytes: u64,
}

/// Copy `input` to `out`, stripping its 32-byte footer, append
/// `RESIDUAL_BLOB` + `RESIDUAL_INDEX` sections, write a fresh SHA-256
/// footer. The CBOR manifest at the head is left untouched — readers
/// that care about `residual_present` should notice the new sections
/// during the TLV loop. (A v0.2 flavour may also rewrite the manifest;
/// the v0.1 spec intentionally doesn't require it so the upgrade stays
/// seekable / append-only.)
pub fn add_residual(
    input: &Path,
    residual_path: &Path,
    index_path: &Path,
    out: &Path,
) -> Result<ResidualSummary> {
    // Validate the input end-to-end first so we don't pile new bytes on
    // top of a broken file.
    let report = validate::validate(input)
        .with_context(|| format!("validate input 1bl: {}", input.display()))?;
    if !report.footer_ok {
        // validate() already errors on mismatch, but be explicit.
        anyhow::bail!("input footer hash did not verify; refusing to upgrade");
    }

    let mut input_bytes = read(input)
        .with_context(|| format!("read input 1bl: {}", input.display()))?;
    // Drop the old 32-byte footer — we'll write a new one.
    let trailing_footer_len = 32usize;
    if input_bytes.len() < trailing_footer_len {
        anyhow::bail!("input too small to have a footer");
    }
    input_bytes.truncate(input_bytes.len() - trailing_footer_len);

    let residual_bytes = read(residual_path)
        .with_context(|| format!("read residual: {}", residual_path.display()))?;
    let index_bytes = read(index_path)
        .with_context(|| format!("read residual index: {}", index_path.display()))?;

    // Record sha of the residual blob so the manifest's `residual_sha256`
    // field (if the curator wants to stamp v0.2-style) can cite it. Not
    // written into the container in v0.1; we expose it via the return
    // value for the CLI log.
    let _residual_sha = sha256_hex(&residual_bytes);

    let out_f = File::create(out)
        .with_context(|| format!("create upgraded out: {}", out.display()))?;
    let mut w = HashingWriter::new(BufWriter::new(out_f));
    w.write_all(&input_bytes).context("copy base 1bl bytes")?;
    write_section(&mut w, tag::RESIDUAL_BLOB, &residual_bytes)?;
    write_section(&mut w, tag::RESIDUAL_INDEX, &index_bytes)?;
    let digest = w.finalize_digest();
    let mut inner = w.into_inner();
    inner.write_all(&digest).context("write new footer")?;
    inner.flush().context("flush upgraded out")?;

    Ok(ResidualSummary {
        residual_bytes: residual_bytes.len() as u64,
        index_bytes: index_bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack;

    fn sample_toml(cat: &str) -> String {
        format!(r#"
catalog = "{cat}"
title = "T"
artist = "T"
license = "CC0-1.0"
created = "2026-04-23T00:00:00Z"
tier = "lossy"
license_txt = "pd"

[codec]
audio = "mimi-12hz"
sample_rate = 24000
channels = 2

[model]
arch = "bitnet-1p58"
params = 1
bpw = 1.58
"#)
    }

    #[test]
    fn add_residual_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        std::fs::write(&gguf, b"GGUF fake").unwrap();
        let tomlp = tmp.path().join("catalog.toml");
        std::fs::write(&tomlp, sample_toml("premium")).unwrap();
        let lossy = tmp.path().join("lossy.1bl");
        pack::pack(&gguf, &tomlp, None, None, &lossy).unwrap();

        // Fake residual + index blobs. In real use the pod produces
        // these; for the test we just need non-empty bytes.
        let residual = tmp.path().join("residual.bin");
        std::fs::write(&residual, vec![0xAAu8; 1024]).unwrap();
        let index = tmp.path().join("index.cbor");
        std::fs::write(&index, b"\xA0").unwrap(); // CBOR empty-map

        let premium = tmp.path().join("premium.1bl");
        let s = add_residual(&lossy, &residual, &index, &premium).unwrap();
        assert_eq!(s.residual_bytes, 1024);
        assert_eq!(s.index_bytes, 1);

        // Upgraded file must validate end-to-end.
        let report = validate::validate(&premium).unwrap();
        assert!(report.footer_ok);
        // Two extra sections on top of the lossy-tier's three.
        assert_eq!(report.sections.len(), 5);
        let tags: Vec<u8> = report.sections.iter().map(|s| s.tag).collect();
        assert!(tags.contains(&tag::RESIDUAL_BLOB));
        assert!(tags.contains(&tag::RESIDUAL_INDEX));
    }
}
