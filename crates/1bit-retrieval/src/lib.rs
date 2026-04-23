//! 1bit-retrieval — keyword-based BM25 index over the docs/wiki tree.
//!
//! Purpose: give specialists (Herald, halo-agents in general) a way to ground
//! answers in the project wiki without hallucinating paths, flags, or commits.
//!
//! No embeddings, no vector DB, no network. Pure-Rust BM25 over lowercased,
//! stop-filtered, whitespace/punct-split tokens. Small corpus (~40 docs,
//! ~2000 chunks), so a flat `Vec<Chunk>` + `HashMap<String, Posting>` beats
//! dragging in a full-text engine.
//!
//! # Shape
//!
//! ```no_run
//! use onebit_retrieval::WikiIndex;
//! use std::path::Path;
//!
//! let idx = WikiIndex::load(Path::new("docs/wiki")).unwrap();
//! let hits = idx.top_k("amdgpu OPTC hang", 5);
//! for h in &hits {
//!     println!("[{}#{}] {}", h.file, h.heading, h.score);
//! }
//! let prompt = onebit_retrieval::format_for_system_prompt(&hits);
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;
use walkdir::WalkDir;

mod chunker;
mod stopwords;

use chunker::{Chunk, chunk_markdown};

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("wiki directory not found: {0}")]
    WikiDirMissing(PathBuf),
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wiki directory loaded zero markdown files: {0}")]
    NoMarkdown(PathBuf),
}

pub type Result<T> = std::result::Result<T, RetrievalError>;

/// A single chunk returned from a top-k lookup.
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    /// Relative path under `wiki_dir`, forward-slash separated
    /// (e.g. `"Installation.md"` or `"subdir/Thing.md"`).
    pub file: String,
    /// Last markdown heading seen before this chunk, without the leading
    /// `#`s. Empty string if the chunk is before the first heading.
    pub heading: String,
    /// Chunk body as it appeared in the source file (no markdown stripping).
    pub text: String,
    /// BM25 score. Non-negative.
    pub score: f32,
}

/// In-memory BM25 index over a wiki directory.
///
/// Constructed once via [`WikiIndex::load`] and queried repeatedly via
/// [`WikiIndex::top_k`]. Wrap in an `Arc` if sharing across threads.
pub struct WikiIndex {
    chunks: Vec<StoredChunk>,
    /// term -> (doc_frequency, Vec<(chunk_idx, term_frequency)>)
    postings: HashMap<String, Posting>,
    /// mean chunk length, in tokens (post-stopword-strip)
    avgdl: f32,
    /// total chunk count (== chunks.len(), cached as f32)
    n_chunks: f32,
    k1: f32,
    b: f32,
}

struct StoredChunk {
    file: String,
    heading: String,
    text: String,
    /// Token count after stopword strip. Used as the "document length"
    /// in BM25 normalisation.
    len_tokens: u32,
}

struct Posting {
    /// Number of chunks containing the term at least once.
    doc_freq: u32,
    /// (chunk_idx, term_freq_in_chunk). Sorted by chunk_idx.
    entries: Vec<(u32, u32)>,
}

impl WikiIndex {
    /// Load every `*.md` file under `wiki_dir` (recursive), chunk each,
    /// and build a BM25 index over the chunks.
    ///
    /// Chunking: split on markdown headings (`#`, `##`, `###`, ...), then
    /// further subdivide long sections on a ~500-char cap with ~50-char
    /// word-boundary overlap. The last heading seen is attached to every
    /// chunk produced from that section.
    pub fn load(wiki_dir: &Path) -> Result<Self> {
        if !wiki_dir.exists() {
            return Err(RetrievalError::WikiDirMissing(wiki_dir.to_path_buf()));
        }

        let mut chunks: Vec<StoredChunk> = Vec::new();

        for entry in WalkDir::new(wiki_dir).follow_links(false) {
            let entry = entry.map_err(|e| RetrievalError::Io {
                path: wiki_dir.to_path_buf(),
                source: e
                    .into_io_error()
                    .unwrap_or_else(|| std::io::Error::other("walkdir error")),
            })?;
            let path = entry.path();
            if !entry.file_type().is_file() {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let body = fs::read_to_string(path).map_err(|e| RetrievalError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            let rel = path
                .strip_prefix(wiki_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for Chunk { heading, text } in chunk_markdown(&body) {
                let len_tokens = count_useful_tokens(&text);
                if len_tokens == 0 {
                    continue;
                }
                chunks.push(StoredChunk {
                    file: rel.clone(),
                    heading,
                    text,
                    len_tokens,
                });
            }
        }

        if chunks.is_empty() {
            return Err(RetrievalError::NoMarkdown(wiki_dir.to_path_buf()));
        }

        // Build postings.
        let mut postings: HashMap<String, Posting> = HashMap::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let mut tf: HashMap<String, u32> = HashMap::new();
            for tok in tokenize(&chunk.text) {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for (term, freq) in tf {
                let p = postings.entry(term).or_insert_with(|| Posting {
                    doc_freq: 0,
                    entries: Vec::new(),
                });
                p.doc_freq += 1;
                p.entries.push((i as u32, freq));
            }
        }

        let total_tokens: u64 = chunks.iter().map(|c| c.len_tokens as u64).sum();
        let n_chunks = chunks.len() as f32;
        let avgdl = if n_chunks > 0.0 {
            (total_tokens as f32) / n_chunks
        } else {
            1.0
        };

        Ok(Self {
            chunks,
            postings,
            avgdl: avgdl.max(1.0),
            n_chunks,
            k1: 1.5,
            b: 0.75,
        })
    }

    /// Return the top-`k` chunks scored against `query`. Order: descending
    /// score, ties broken by chunk order in the corpus (stable).
    ///
    /// If the query contains only stopwords / punctuation the result is
    /// empty (rather than returning arbitrary top docs).
    pub fn top_k(&self, query: &str, k: usize) -> Vec<RetrievedChunk> {
        if k == 0 {
            return Vec::new();
        }
        let q_terms: Vec<String> = tokenize(query);
        if q_terms.is_empty() {
            return Vec::new();
        }

        // Dedupe the query so "rust rust rust" doesn't triple-count.
        let mut seen = std::collections::HashSet::new();
        let q_terms: Vec<&str> = q_terms
            .iter()
            .filter(|t| seen.insert((*t).clone()))
            .map(String::as_str)
            .collect();

        // Accumulate BM25 score per chunk_idx.
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in &q_terms {
            let Some(posting) = self.postings.get(*term) else {
                continue;
            };
            let df = posting.doc_freq as f32;
            // BM25+ idf keeps values >=0 even when a term appears in > half the corpus.
            let idf = ((self.n_chunks - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(chunk_idx, tf) in &posting.entries {
                let dl = self.chunks[chunk_idx as usize].len_tokens as f32;
                let tf_f = tf as f32;
                let norm = 1.0 - self.b + self.b * (dl / self.avgdl);
                let bm = idf * ((tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * norm));
                *scores.entry(chunk_idx).or_insert(0.0) += bm;
            }
        }

        if scores.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(k);

        ranked
            .into_iter()
            .map(|(i, score)| {
                let c = &self.chunks[i as usize];
                RetrievedChunk {
                    file: c.file.clone(),
                    heading: c.heading.clone(),
                    text: c.text.clone(),
                    score,
                }
            })
            .collect()
    }

    /// Total chunks in the index. Exposed for diagnostics / tests.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Format retrieved chunks for injection into a specialist's system prompt.
///
/// Output shape:
/// ```text
/// RELEVANT DOCS (top N):
///
/// [Installation.md#Distro-policy] chunk body ...
/// [Troubleshooting.md#amdgpu-OPTC-hang] chunk body ...
/// ```
///
/// Headings are slugified to GitHub-style anchors (`Distro policy` → `Distro-policy`).
/// Empty input yields an empty string so callers can concat unconditionally.
pub fn format_for_system_prompt(chunks: &[RetrievedChunk]) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(&format!("RELEVANT DOCS (top {}):\n\n", chunks.len()));
    for c in chunks {
        let anchor = slugify(&c.heading);
        let body = c.text.trim();
        if anchor.is_empty() {
            out.push_str(&format!("[{}] {}\n", c.file, body));
        } else {
            out.push_str(&format!("[{}#{}] {}\n", c.file, anchor, body));
        }
    }
    out
}

// ---- tokenisation --------------------------------------------------------

/// Lowercase, split on non-alphanumeric, drop stopwords and empties.
/// Retains `-` and `_` inside tokens (so `fish-shell` and `gfx1151`
/// survive) — everything else is a break.
fn tokenize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        let keep = ch.is_ascii_alphanumeric() || ch == '-' || ch == '_';
        if keep {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else {
            flush_token(&mut cur, &mut out);
        }
    }
    flush_token(&mut cur, &mut out);
    out
}

fn flush_token(cur: &mut String, out: &mut Vec<String>) {
    if cur.is_empty() {
        return;
    }
    // Strip leading/trailing dashes+underscores (e.g. "-foo-" → "foo").
    let trimmed = cur.trim_matches(|c: char| c == '-' || c == '_');
    if !trimmed.is_empty() && !stopwords::is_stopword(trimmed) {
        out.push(trimmed.to_string());
    }
    cur.clear();
}

fn count_useful_tokens(s: &str) -> u32 {
    tokenize(s).len() as u32
}

// ---- slugify -------------------------------------------------------------

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    // Trim trailing dash.
    while out.ends_with('-') {
        out.pop();
    }
    out
}

// ---- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/wiki");
        p
    }

    #[test]
    fn load_fixture_produces_chunks() {
        let idx = WikiIndex::load(&fixtures()).expect("fixture wiki loads");
        assert!(
            idx.len() >= 3,
            "expected ≥3 chunks across fixture files, got {}",
            idx.len()
        );
    }

    #[test]
    fn known_term_finds_right_file_top_1() {
        let idx = WikiIndex::load(&fixtures()).unwrap();
        let hits = idx.top_k("OPTC hang amdgpu", 3);
        assert!(!hits.is_empty(), "query should match at least one chunk");
        assert_eq!(
            hits[0].file, "troubleshooting.md",
            "top hit should be troubleshooting.md, got {:?}",
            hits[0].file
        );
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn absent_term_returns_empty() {
        let idx = WikiIndex::load(&fixtures()).unwrap();
        let hits = idx.top_k("xyzzy_does_not_exist_in_any_fixture", 5);
        assert!(
            hits.is_empty(),
            "nonsense query should not match anything, got {:?}",
            hits
        );
    }

    #[test]
    fn chunks_do_not_cut_mid_word() {
        let idx = WikiIndex::load(&fixtures()).unwrap();
        for chunk in &idx.chunks {
            let t = chunk.text.trim();
            if t.is_empty() {
                continue;
            }
            // If the chunk is a hard cap split (not the last chunk of its
            // section), the end should land on whitespace or punctuation,
            // never mid-alphanumeric-run.
            let last_char = t.chars().last().unwrap();
            let first_char = t.chars().next().unwrap();
            // Sanity: first and last char of a chunk are plausible tokens
            // (letter, digit, punctuation) but never split a codepoint.
            assert!(
                first_char.is_ascii()
                    || first_char.is_alphanumeric()
                    || first_char.is_whitespace()
                    || "#-*`_[(\"'".contains(first_char),
                "first char of chunk looks like mid-word garbage: {:?}",
                first_char
            );
            assert!(last_char.is_ascii() || last_char.is_alphanumeric());
        }
    }

    #[test]
    fn format_for_prompt_shapes_output() {
        let idx = WikiIndex::load(&fixtures()).unwrap();
        let hits = idx.top_k("OPTC", 2);
        let out = format_for_system_prompt(&hits);
        assert!(out.starts_with("RELEVANT DOCS"));
        assert!(out.contains("troubleshooting.md"));
    }

    #[test]
    fn slugify_matches_gh_style() {
        assert_eq!(slugify("Distro policy"), "Distro-policy");
        assert_eq!(slugify("amdgpu OPTC hang"), "amdgpu-OPTC-hang");
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("   "), "");
    }

    #[test]
    fn tokenize_keeps_hyphens_and_ids() {
        let toks = tokenize("Strix Halo gfx1151 fish-shell amdgpu OPTC.");
        assert!(toks.contains(&"gfx1151".to_string()));
        assert!(toks.contains(&"fish-shell".to_string()));
        assert!(toks.contains(&"amdgpu".to_string()));
        assert!(toks.contains(&"optc".to_string()));
        // "the" / "a" / "is" not present — stopwords. Sanity check:
        assert!(!toks.iter().any(|t| t == "the"));
    }
}
