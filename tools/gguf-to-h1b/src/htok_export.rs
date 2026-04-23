//! Extract tokenizer metadata from a GGUF file and write a sidecar
//! `.htok`.
//!
//! Why this lives here (and not next to `htok.rs` in `1bit-core`): the
//! conversion crosses two formats (`gguf::BitnetHeader` ↔
//! `htok::HtokFile`), and `1bit-core` deliberately keeps its two
//! parsers independent of each other — neither module imports from the
//! other. The `tools/gguf-to-h1b` binary already sits at the
//! intersection of both formats, so bolting the tokenizer extraction on
//! here costs one module, not one more crate dependency edge.
//!
//! # Scope
//!
//! - Input: the `tokenizer.ggml.{model,tokens,merges,bos_token_id,
//!   eos_token_id}` fields that [`onebit_core::BitnetHeader`] already
//!   pulls out of the GGUF header.
//! - Output: a `.htok` on disk whose byte layout is defined in
//!   `crates/1bit-core/src/htok.rs`.
//!
//! # What we do NOT do
//!
//! - Do not re-tokenize anything. The GGUF's `tokens` array is already
//!   the surface form (GPT-2 byte-mapped UTF-8) we want in `.htok`.
//! - Do not regex-parse anything beyond splitting a merge on its sole
//!   space separator.
//! - Do not invent synthetic merges when `merges` is empty. A
//!   tokenizer with no merges is valid (unigram / word-piece); we
//!   just write `num_merges = 0`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use onebit_core::gguf::{BitnetHeader, GgufFile};
use onebit_core::htok::{HtokFile, Merge};

/// Errors surfaced by the `.htok` exporter. Mirrors the `ConvertError`
/// shape in `lib.rs` so the CLI can `?`-chain them.
#[derive(Debug, thiserror::Error)]
pub enum HtokExportError {
    #[error("GGUF parsing failed: {0}")]
    Gguf(#[from] onebit_core::HaloError),

    #[error("GGUF contains no tokenizer.ggml.tokens — cannot emit .htok")]
    NoTokens,

    #[error(
        "merge {rank} `{merge}`: missing `{side}` token in vocab (split pieces: `{a}` + `{b}`)"
    )]
    MergeTokenMissing {
        rank: usize,
        merge: String,
        side: &'static str,
        a: String,
        b: String,
    },

    #[error("merge {rank} `{merge}` does not contain exactly one space separator")]
    MergeBadShape { rank: usize, merge: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Summary of a `.htok` export, printed by the CLI.
#[derive(Debug, Clone)]
pub struct HtokStats {
    pub vocab_size: u32,
    pub num_merges: u32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub output_bytes: u64,
    pub output_path: PathBuf,
    /// How many merge strings we dropped because one of the two sides
    /// was unresolvable. Expected to be zero for well-formed GGUFs;
    /// nonzero values go to the CLI so the operator can investigate.
    pub dropped_merges: u32,
}

/// Build an `HtokFile` in memory from a GGUF's tokenizer metadata.
///
/// The output is a pure struct — callers can inspect it in tests
/// before writing to disk.
pub fn build_htok_from_gguf(header: &BitnetHeader) -> Result<(HtokFile, u32), HtokExportError> {
    if header.tokens.is_empty() {
        return Err(HtokExportError::NoTokens);
    }

    // 1) Vocab: each token string is already GPT-2-byte-mapped UTF-8
    //    (Qwen3 + LLaMA-3 + Bonsai all use GPT-2 byte-level BPE); we
    //    take the UTF-8 bytes verbatim.
    let id_to_bytes: Vec<Vec<u8>> = header
        .tokens
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();

    // 2) Merges: GGUF stores each merge as "A B" (space-separated
    //    surface forms). The `.htok` writer wants the numeric
    //    (a_id, b_id, merged_id) triple. We build a surface→id map once
    //    and do three lookups per merge.
    let mut surface_to_id: HashMap<&str, i32> = HashMap::with_capacity(header.tokens.len());
    for (i, tok) in header.tokens.iter().enumerate() {
        // `insert` silently overwrites duplicates — Qwen3's vocab is
        // unique by construction, but guarding explicitly would add a
        // third error variant for no operational benefit.
        surface_to_id.insert(tok.as_str(), i as i32);
    }

    let mut merges: Vec<Merge> = Vec::with_capacity(header.merges.len());
    let mut dropped = 0u32;
    for (rank, m) in header.merges.iter().enumerate() {
        // GGUF merge strings contain exactly one separator space. Use
        // `splitn(2, ' ')` so an `A ` with trailing space still splits
        // (first side gets `A`, second gets empty — we catch that as
        // MergeBadShape via the `is_empty` check below).
        let mut parts = m.splitn(2, ' ');
        let a = parts.next().unwrap_or("");
        let b = parts.next().unwrap_or("");
        if a.is_empty() || b.is_empty() || parts.next().is_some() {
            return Err(HtokExportError::MergeBadShape {
                rank,
                merge: m.clone(),
            });
        }
        let merged = format!("{a}{b}");
        let a_id = match surface_to_id.get(a) {
            Some(&v) => v,
            None => {
                dropped += 1;
                continue;
            }
        };
        let b_id = match surface_to_id.get(b) {
            Some(&v) => v,
            None => {
                dropped += 1;
                continue;
            }
        };
        // `merged` may legitimately not exist in the vocab if the
        // tokenizer was trained with a vocab cap that dropped the
        // merged token. llama.cpp's tokenizer does the same lookup and
        // skips the merge in that case; we match.
        let merged_id = match surface_to_id.get(merged.as_str()) {
            Some(&v) => v,
            None => {
                dropped += 1;
                continue;
            }
        };
        merges.push(Merge {
            a: a_id,
            b: b_id,
            merged: merged_id,
            rank: merges.len() as u32,
        });
    }

    let bos = header.bos_token_id.unwrap_or(0) as i32;
    let eos = header.eos_token_id.unwrap_or(0) as i32;
    let htok = HtokFile::from_parts(bos, eos, id_to_bytes, merges);
    Ok((htok, dropped))
}

/// Read the GGUF at `input`, build a `.htok`, write it to `output`.
/// Returns the summary shape for the CLI.
pub fn export_htok_sidecar(input: &Path, output: &Path) -> Result<HtokStats, HtokExportError> {
    let gguf = GgufFile::open(input)?;
    let header = gguf.read_bitnet_metadata()?;
    let (htok, dropped) = build_htok_from_gguf(&header)?;
    let bytes = htok.to_bytes()?;
    let mut f = File::create(output)?;
    f.write_all(&bytes)?;
    f.flush()?;
    f.sync_all()?;
    Ok(HtokStats {
        vocab_size: htok.vocab_size() as u32,
        num_merges: htok.num_merges() as u32,
        bos_id: htok.bos_id,
        eos_id: htok.eos_id,
        output_bytes: bytes.len() as u64,
        output_path: output.to_path_buf(),
        dropped_merges: dropped,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use onebit_core::htok::HTOK_MAGIC;

    /// Fabricate a `BitnetHeader` with a tiny hand-written tokenizer.
    /// This isolates the export logic from the GGUF parser entirely.
    fn tiny_header() -> BitnetHeader {
        // 5 tokens: "<bos>", "<eos>", "a", "b", "ab".
        // One merge: "a b" → "ab" (ids 2+3 → 4).
        BitnetHeader {
            architecture: "qwen3".to_string(),
            block_count: 0,
            embedding_length: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            attention_head_count_kv: 0,
            rope_freq_base: 1e4,
            rms_norm_eps: 1e-6,
            tokenizer_model: "gpt2".to_string(),
            tokens: vec![
                "<bos>".to_string(),
                "<eos>".to_string(),
                "a".to_string(),
                "b".to_string(),
                "ab".to_string(),
            ],
            merges: vec!["a b".to_string()],
            bos_token_id: Some(0),
            eos_token_id: Some(1),
        }
    }

    /// The on-disk bytes produced by `export_htok_sidecar` start with
    /// the `HTOK` magic defined in `1bit-core::htok`. If this breaks,
    /// the loader in `rocm-cpp/src/tokenizer.cpp` will reject the file
    /// with `bad .htok magic`.
    #[test]
    fn export_bytes_have_htok_magic() {
        let h = tiny_header();
        let (htok, dropped) = build_htok_from_gguf(&h).expect("build ok");
        assert_eq!(dropped, 0);
        let bytes = htok.to_bytes().unwrap();
        assert_eq!(&bytes[0..4], &HTOK_MAGIC);
    }

    /// Token count in the exported `.htok` equals the token count in the
    /// GGUF. Merge count equals GGUF merges minus any dropped for
    /// missing-side — for this tiny vocab that's 1 → 1 with zero drops.
    /// Also round-trip through `HtokFile::parse_bytes` to prove the
    /// exported bytes are parseable by the same code that reads
    /// `halo-1bit-2b.htok`.
    #[test]
    fn roundtrip_preserves_vocab_and_merge_counts() {
        let h = tiny_header();
        let (htok, dropped) = build_htok_from_gguf(&h).expect("build ok");
        assert_eq!(dropped, 0);
        assert_eq!(htok.vocab_size(), 5);
        assert_eq!(htok.num_merges(), 1);
        let bytes = htok.to_bytes().unwrap();
        let parsed = HtokFile::parse_bytes("roundtrip", &bytes).expect("parse ok");
        assert_eq!(parsed.vocab_size(), 5);
        assert_eq!(parsed.num_merges(), 1);
        assert_eq!(parsed.bos_id, 0);
        assert_eq!(parsed.eos_id, 1);
        // The single merge is `a (id=2) b (id=3) → ab (id=4)`.
        assert_eq!(parsed.merges[0].a, 2);
        assert_eq!(parsed.merges[0].b, 3);
        assert_eq!(parsed.merges[0].merged, 4);
    }
}
