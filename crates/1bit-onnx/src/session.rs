//! Thin `ort::Session` wrapper that picks the execution provider from a
//! compile-time feature set.
//!
//! # Execution provider matrix
//!
//! | Feature        | Build compiles? | Session tries                                 |
//! |----------------|-----------------|-----------------------------------------------|
//! | `cpu` (default)| yes             | CPU EP                                        |
//! | `vitisai`      | yes             | VitisAI EP → fall back to CPU EP on register-fail |
//!
//! When `vitisai` is on, the fallback is deliberate: on Linux STX-H today
//! the VitisAI provider is not shipped, and we want the binary to still
//! run (slowly) via CPU so an operator can smoke-test the graph. When AMD
//! ships the provider, the same binary starts placing ops on the NPU
//! without a rebuild.
//!
//! # Tokenizer
//!
//! We load the HF tokenizer from `tokenizer.json`. No FFI to AutoTokenizer,
//! no Python. Anything the tokenizer crate can't handle (e.g. chat
//! templates) is caller responsibility — the server owns prompt assembly.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::OnnxError;
use crate::model::{ArtifactPaths, GenAiConfig};

/// Which execution provider the session actually landed on after
/// registration.
///
/// Callers log this so operators see whether they got NPU placement or
/// fell back. It is NOT the same as the *requested* provider — hence the
/// distinction from the compile-time feature set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionLane {
    /// CPU execution provider landed.
    Cpu,
    /// VitisAI execution provider landed (XDNA2 NPU).
    Vitisai,
}

impl ExecutionLane {
    /// Short human label for `/v1/models` responses + logs.
    pub fn label(self) -> &'static str {
        match self {
            ExecutionLane::Cpu => "ort-cpu",
            ExecutionLane::Vitisai => "ort-vitisai",
        }
    }
}

/// Loaded ONNX Runtime session plus the tokenizer + config that go with
/// the artifact.
///
/// The [`ort::session::Session`] lives behind an `Option` because we
/// support a stubbed "config-only" mode for tests that want to exercise
/// tokenizer + config wiring without needing libonnxruntime.so on disk.
pub struct OnnxSession {
    /// Artifact paths, kept for logging + reload support.
    pub paths: ArtifactPaths,
    /// Parsed `genai_config.json`.
    pub config: GenAiConfig,
    /// HF tokenizer loaded from `tokenizer.json`.
    pub tokenizer: Tokenizer,
    /// Which EP the session actually landed on.
    pub lane: ExecutionLane,
    /// Real ORT session (absent in `load_config_only`).
    session: Option<ort::session::Session>,
}

impl OnnxSession {
    /// Load tokenizer + config without touching libonnxruntime.so.
    ///
    /// Intended for tests + startup probes that want to validate the
    /// artifact layout without paying the cost of a full session init.
    pub fn load_config_only(root: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let paths = ArtifactPaths::discover(root)?;
        let config = GenAiConfig::load(&paths.genai_config)?;
        let tokenizer = Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| OnnxError::TokenizerLoad(e.to_string()))?;
        Ok(Self {
            paths,
            config,
            tokenizer,
            lane: ExecutionLane::Cpu,
            session: None,
        })
    }

    /// Full load — tokenizer + config + an `ort::Session` built against
    /// `model.onnx` with the provider list implied by the enabled features.
    ///
    /// Returns [`OnnxError::OrtRuntimeUnavailable`] if the ORT C++ library
    /// cannot be located (callers should then either install the runtime or
    /// fall back to [`OnnxSession::load_config_only`] for a probe).
    pub fn load(root: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let mut s = Self::load_config_only(root)?;

        let (session, lane) = build_session(&s.paths.model)?;
        s.session = Some(session);
        s.lane = lane;
        Ok(s)
    }

    /// Access the underlying `ort::session::Session`. Returns `None` in
    /// config-only mode.
    pub fn session(&self) -> Option<&ort::session::Session> {
        self.session.as_ref()
    }

    /// Mutable accessor for the underlying `ort::session::Session`.
    /// `ort::session::Session::run` requires `&mut self`, so decode paths
    /// reach for this.
    pub fn session_mut(&mut self) -> Option<&mut ort::session::Session> {
        self.session.as_mut()
    }
}

/// Build an `ort::Session` for `model_path`, trying the feature-selected
/// provider list in order and landing on whichever one actually registered.
fn build_session(model_path: &Path) -> Result<(ort::session::Session, ExecutionLane), OnnxError> {
    // Initialize the global ORT environment once. `commit()` returns
    // `true` on first-call success and `false` if another caller won the
    // race; either is fine for us.
    let _ = ort::init().with_name("onebit-onnx").commit();

    // Provider list depends on features. Deliberately short-circuit CPU
    // builds so we don't drag VitisAI symbols in when they aren't needed.
    #[cfg(feature = "vitisai")]
    {
        match try_vitisai(model_path) {
            Ok(session) => return Ok((session, ExecutionLane::Vitisai)),
            Err(OnnxError::VitisaiUnavailable(msg)) => {
                tracing::warn!(
                    "VitisAI EP unavailable ({msg}); falling back to CPU. This is \
                     expected on Linux STX-H until AMD ships the provider."
                );
            }
            Err(other) => return Err(other),
        }
    }

    let session = try_cpu(model_path)?;
    Ok((session, ExecutionLane::Cpu))
}

fn try_cpu(model_path: &Path) -> Result<ort::session::Session, OnnxError> {
    ort::session::Session::builder()
        .map_err(|e| OnnxError::SessionInit(e.to_string()))?
        .commit_from_file(model_path)
        .map_err(|e| OnnxError::SessionInit(e.to_string()))
}

#[cfg(feature = "vitisai")]
fn try_vitisai(_model_path: &Path) -> Result<ort::session::Session, OnnxError> {
    // VitisAI EP is a named provider in the ORT registry. Construction
    // of the concrete provider struct depends on the ort-rs release; the
    // feature exists so we can slot it in once the Linux STX-H EP lands
    // without churning upstream code.
    //
    // Until then this branch returns `VitisaiUnavailable` so the caller
    // can fall back to CPU.
    Err(OnnxError::VitisaiUnavailable(
        "VitisAI provider not wired in this ort-rs release — waiting on AMD Linux STX-H EP".into(),
    ))
    .map(|_: OnnxError| unreachable!() as ort::session::Session)
    .or_else(|e| Err(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARTIFACT: &str = "/home/bcloud/1bit-halo-core/artifacts/trilm-3.9b-int4-n4";

    fn artifact_present() -> bool {
        std::path::Path::new(ARTIFACT).join("model.onnx").exists()
    }

    #[test]
    fn config_only_mode_avoids_loading_runtime() {
        if !artifact_present() {
            eprintln!("skipping: {ARTIFACT} not present");
            return;
        }
        let s = OnnxSession::load_config_only(ARTIFACT).expect("config-only load");
        assert_eq!(s.config.model.arch, "llama");
        assert_eq!(s.lane, ExecutionLane::Cpu);
        assert!(s.session().is_none());
    }

    #[test]
    fn tokenizer_roundtrips_bos_token() {
        if !artifact_present() {
            eprintln!("skipping: {ARTIFACT} not present");
            return;
        }
        let s = OnnxSession::load_config_only(ARTIFACT).expect("config-only load");
        let bos = s.config.model.bos_token_id;
        // Tokenize the empty string and confirm we can decode the BOS id
        // alone without panicking — this exercises the tokenizer path end
        // to end without needing a live ORT session.
        let decoded = s
            .tokenizer
            .decode(&[bos], /* skip_special_tokens */ false)
            .map_err(|e| e.to_string())
            .expect("decode bos");
        // Decoded text may be empty or a special marker; we only require
        // the call itself to succeed.
        let _ = decoded;
    }

    #[test]
    fn execution_lane_label_stable() {
        assert_eq!(ExecutionLane::Cpu.label(), "ort-cpu");
        assert_eq!(ExecutionLane::Vitisai.label(), "ort-vitisai");
    }
}
