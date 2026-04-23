//! Artifact discovery and `genai_config.json` parsing.
//!
//! An OGA Model Builder output directory has a fixed shape:
//!
//! ```text
//! <artifact-dir>/
//!   model.onnx              # thin graph (no weights inlined; ~220 KB for TriLM 3.9B)
//!   model.onnx.data         # external-data weight blob (~2.96 GB for TriLM 3.9B)
//!   genai_config.json       # tokenizer + decode config (see `GenAiConfig`)
//!   tokenizer.json          # HF tokenizers config
//!   tokenizer_config.json   # HF AutoTokenizer config (we ignore; tokenizer.json is enough)
//! ```
//!
//! The graph and the external-data blob **must** live in the same directory —
//! ORT resolves the `external_data_location` attribute relative to the .onnx
//! file path, not the caller's CWD.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::OnnxError;

/// Resolved paths for every file in an OGA output directory.
///
/// Construct via [`ArtifactPaths::discover`]; fields are publicly readable
/// because downstream crates (router, server) want to log them without
/// reaching back through accessors.
#[derive(Debug, Clone)]
pub struct ArtifactPaths {
    /// Root directory the caller pointed us at.
    pub root: PathBuf,
    /// `model.onnx` — the thin graph.
    pub model: PathBuf,
    /// `model.onnx.data` — the external weight blob.
    pub weights: PathBuf,
    /// `genai_config.json` — decode config.
    pub genai_config: PathBuf,
    /// `tokenizer.json` — HF tokenizers config.
    pub tokenizer: PathBuf,
}

impl ArtifactPaths {
    /// Probe `root` for the five files above. Returns
    /// [`OnnxError::NotAnArtifactDir`] if `model.onnx` is missing and
    /// [`OnnxError::MissingArtifact`] for any other absent file, so callers
    /// can distinguish "wrong directory" from "partial sync".
    pub fn discover(root: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let root = root.as_ref().to_path_buf();
        let model = root.join("model.onnx");
        if !model.exists() {
            return Err(OnnxError::NotAnArtifactDir(root));
        }

        let weights = root.join("model.onnx.data");
        let genai_config = root.join("genai_config.json");
        let tokenizer = root.join("tokenizer.json");

        for p in [&weights, &genai_config, &tokenizer] {
            if !p.exists() {
                return Err(OnnxError::MissingArtifact(p.clone()));
            }
        }

        Ok(Self {
            root,
            model,
            weights,
            genai_config,
            tokenizer,
        })
    }
}

/// Decoded subset of `genai_config.json` that we actually use.
///
/// We deserialize loosely — the full OGA config has dozens of sampler knobs
/// we do not touch yet. Keeping the struct minimal means bumps in OGA
/// don't break us so long as the core fields stay put.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiConfig {
    /// Container matching `{ "model": { ... } }` in the file.
    pub model: GenAiModel,
    /// Container matching `{ "search": { ... } }` in the file.
    #[serde(default)]
    pub search: GenAiSearch,
}

/// The `model` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiModel {
    /// Beginning-of-sequence token. TriLM uses 0.
    pub bos_token_id: u32,
    /// End-of-sequence token. TriLM uses 0 (shared with BOS).
    pub eos_token_id: u32,
    /// Max context the graph was exported with (TriLM: 2048).
    pub context_length: usize,
    /// Vocab size (TriLM: 50688).
    pub vocab_size: u32,
    /// Arch string, e.g. `"llama"`.
    #[serde(rename = "type")]
    pub arch: String,
    /// The decoder sub-section.
    pub decoder: GenAiDecoder,
}

/// The `model.decoder` section.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiDecoder {
    /// Head dimension.
    pub head_size: usize,
    /// Hidden size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads. For MHA == num_attention_heads.
    pub num_key_value_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
}

/// The `search` section of `genai_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GenAiSearch {
    /// Default `do_sample` from the exporter.
    #[serde(default)]
    pub do_sample: bool,
    /// Default temperature from the exporter.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Default top-k from the exporter.
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

impl Default for GenAiSearch {
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: default_temperature(),
            top_k: default_top_k(),
        }
    }
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_k() -> u32 {
    50
}

impl GenAiConfig {
    /// Parse `genai_config.json` at `path`. Errors surface as typed
    /// [`OnnxError::InvalidGenAiConfig`] so callers can handle schema drift
    /// specifically (e.g. "bump the parser, don't panic on startup").
    pub fn load(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let bytes = std::fs::read(path.as_ref())?;
        let cfg: Self = serde_json::from_slice(&bytes)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TriLM 3.9B artifact is our one well-known fixture. If it isn't
    /// present on this box we skip rather than fail — CI boxes don't carry
    /// the 2.9 GB blob.
    const ARTIFACT: &str = "/home/bcloud/1bit-halo-core/artifacts/trilm-3.9b-int4-n4";

    fn artifact_present() -> bool {
        std::path::Path::new(ARTIFACT).join("model.onnx").exists()
    }

    #[test]
    fn discover_matches_well_known_layout() {
        if !artifact_present() {
            eprintln!("skipping: {ARTIFACT} not present");
            return;
        }
        let p = ArtifactPaths::discover(ARTIFACT).expect("discover");
        assert!(p.model.ends_with("model.onnx"));
        assert!(p.weights.ends_with("model.onnx.data"));
        assert!(p.tokenizer.ends_with("tokenizer.json"));
    }

    #[test]
    fn discover_rejects_non_artifact_dir() {
        let td = tempfile::tempdir().unwrap();
        match ArtifactPaths::discover(td.path()) {
            Err(OnnxError::NotAnArtifactDir(_)) => {}
            other => panic!("expected NotAnArtifactDir, got {other:?}"),
        }
    }

    #[test]
    fn discover_flags_partial_sync() {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("model.onnx"), b"").unwrap();
        match ArtifactPaths::discover(td.path()) {
            Err(OnnxError::MissingArtifact(p)) => {
                assert!(p.ends_with("model.onnx.data"));
            }
            other => panic!("expected MissingArtifact, got {other:?}"),
        }
    }

    #[test]
    fn genai_config_parses_trilm_shape() {
        if !artifact_present() {
            eprintln!("skipping: {ARTIFACT} not present");
            return;
        }
        let cfg = GenAiConfig::load(format!("{ARTIFACT}/genai_config.json")).expect("load");
        assert_eq!(cfg.model.arch, "llama");
        assert_eq!(cfg.model.vocab_size, 50688);
        assert_eq!(cfg.model.context_length, 2048);
        assert_eq!(cfg.model.decoder.num_hidden_layers, 30);
        assert_eq!(cfg.model.decoder.num_attention_heads, 24);
        assert_eq!(cfg.model.decoder.hidden_size, 3072);
    }
}
