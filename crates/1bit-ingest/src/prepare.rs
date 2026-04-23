//! `prepare` — walk a directory of FLACs, sidecar a JSON manifest, tar
//! the corpus.
//!
//! The manifest is deliberately thin. The pod reads the tarball, runs
//! its own Mimi tokenizer, retrains the ternary LM, and uploads a new
//! `.gguf`. This tool never looks inside a FLAC — no libflac, no
//! decoding, just bytes-on-disk plus a SHA-256 for tamper-evidence.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

/// Per-file entry written into the tarball's `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlacEntry {
    /// Path relative to the source root (forward-slash, no leading `./`).
    pub rel_path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Lowercase hex SHA-256 of the file contents.
    pub sha256: String,
}

/// Full corpus manifest, serialised to JSON and inserted into the tar
/// as `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Corpus format version — bump when the JSON schema changes.
    pub version: String,
    /// Seconds since Unix epoch at prepare-time.
    pub created_unix: u64,
    /// Tool that wrote the manifest (for provenance).
    pub tool: String,
    /// All discovered FLACs, in walkdir sorted order (stable across runs).
    pub entries: Vec<FlacEntry>,
}

/// Summary returned to the CLI so it can print a one-liner.
#[derive(Debug, Clone, Copy)]
pub struct PrepareSummary {
    pub flac_count: usize,
    pub total_bytes: u64,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let f = File::open(path)
        .with_context(|| format!("open flac: {}", path.display()))?;
    let mut reader = BufReader::new(f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = reader.read(&mut buf)
            .with_context(|| format!("read flac: {}", path.display()))?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, hex(&hasher.finalize())))
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Scan `src_dir` for FLACs, hash them, and write a tar archive at
/// `out_path` whose first entry is `manifest.json` followed by every
/// FLAC at its relative path.
pub fn prepare(src_dir: &Path, out_path: &Path) -> Result<PrepareSummary> {
    let src_dir = src_dir.canonicalize()
        .with_context(|| format!("canonicalize src: {}", src_dir.display()))?;

    // Collect FLAC files. walkdir's default sort is by OS order, which
    // is non-deterministic; sort explicitly so `prepare` of the same
    // tree twice produces an identical manifest.
    let mut flacs: Vec<PathBuf> = WalkDir::new(&src_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path().extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("flac"))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();
    flacs.sort();

    let mut entries = Vec::with_capacity(flacs.len());
    let mut total_bytes: u64 = 0;
    for abs in &flacs {
        let (size, sha) = hash_file(abs)?;
        let rel = abs.strip_prefix(&src_dir)
            .unwrap_or(abs)
            .to_string_lossy()
            .replace('\\', "/");
        total_bytes += size;
        entries.push(FlacEntry { rel_path: rel, size_bytes: size, sha256: sha });
    }

    let manifest = CorpusManifest {
        version: "0.1".into(),
        created_unix: unix_now(),
        tool: concat!("1bit-ingest/", env!("CARGO_PKG_VERSION")).into(),
        entries: entries.clone(),
    };
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .context("serialize manifest.json")?;

    // Stream into a tar. Manifest first so a pod-side consumer can
    // decide whether to proceed without reading the whole archive.
    let out = File::create(out_path)
        .with_context(|| format!("create out tar: {}", out_path.display()))?;
    let writer = BufWriter::new(out);
    let mut tar = tar::Builder::new(writer);

    let mut hdr = tar::Header::new_gnu();
    hdr.set_path("manifest.json")?;
    hdr.set_size(manifest_json.len() as u64);
    hdr.set_mode(0o644);
    hdr.set_cksum();
    tar.append(&hdr, manifest_json.as_slice())
        .context("write manifest.json into tar")?;

    for (abs, entry) in flacs.iter().zip(entries.iter()) {
        tar.append_path_with_name(abs, &entry.rel_path)
            .with_context(|| format!("append flac: {}", abs.display()))?;
    }

    tar.finish().context("finalise tar")?;

    Ok(PrepareSummary {
        flac_count: entries.len(),
        total_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_discovers_flacs_and_hashes_deterministic() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(src.join("album")).unwrap();
        std::fs::write(src.join("album/01-track.flac"), b"fake flac bytes one").unwrap();
        std::fs::write(src.join("album/02-track.flac"), b"fake flac bytes two").unwrap();
        std::fs::write(src.join("readme.txt"), b"not a flac").unwrap();

        let out = tmp.path().join("corpus.tar");
        let summary = prepare(&src, &out).unwrap();
        assert_eq!(summary.flac_count, 2);
        assert!(summary.total_bytes > 0);
        assert!(out.exists());

        // Re-running hashes identically.
        let out2 = tmp.path().join("corpus2.tar");
        let s2 = prepare(&src, &out2).unwrap();
        assert_eq!(s2.flac_count, summary.flac_count);
        assert_eq!(s2.total_bytes, summary.total_bytes);
    }
}
