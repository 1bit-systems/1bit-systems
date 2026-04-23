//! 1bit-kokoro — safe Rust surface over kokoro.cpp (single-header C++ TTS
//! wrapper around onnxruntime).
//!
//! The crate is structured as two swappable implementations behind the
//! same public API:
//!
//! - **`stub` feature (default)**: `KokoroEngine` is backed by
//!   [`stub::StubEngine`]; every method returns
//!   [`KokoroError::UnsupportedStub`]. No native dependency, no link
//!   step, safe for CI hosts without kokoro.cpp / onnxruntime.
//! - **`real-kokoro` feature**: `KokoroEngine` is backed by
//!   [`real::RealEngine`], which owns a `KokoroCtx*` allocated by the
//!   C++ shim in `cpp/shim.cpp`. The shim wraps kokoro.cpp's single-
//!   header API; Rust feeds text + voice + speed and receives mono
//!   22 050 Hz s16le PCM.
//!
//! The two features are mutually exclusive from a behavioural standpoint
//! — if both are enabled `real-kokoro` wins (see [`EngineImpl`]). Cargo
//! does not allow us to mark them as mutually exclusive declaratively,
//! so downstream crates should pick one via
//! `default-features = false, features = ["real-kokoro"]`.
//!
//! Mirrors the layout of `1bit-whisper` — same `ffi` / `stub` / `real`
//! split, same error surface pattern, same build.rs prefix env var.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod ffi;

#[cfg(feature = "stub")]
pub mod stub;

#[cfg(feature = "real-kokoro")]
pub mod real;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// Metadata describing a successful synthesis call.
///
/// Returned alongside the PCM samples so callers can correctly frame
/// the resulting audio (sample rate, channel count, duration).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesisInfo {
    /// Output sample rate in Hz. Kokoro v1 is fixed at 22 050 Hz.
    pub sample_rate: u32,
    /// Number of audio channels. Always 1 (mono) for kokoro.
    pub channels: u16,
    /// Total number of samples (per channel) returned.
    pub samples: u64,
    /// Approximate duration of the returned audio, in milliseconds.
    pub duration_ms: u64,
}

/// Error surface for the kokoro engine.
#[derive(Debug)]
pub enum KokoroError {
    /// The `stub` feature is active and the caller tried to do real work.
    /// Build with `--features real-kokoro --no-default-features` to enable
    /// the libkokoro-backed path.
    UnsupportedStub,
    /// The model file failed to load (bad path, corrupt weights, OOM).
    ModelLoadFailed {
        /// Path passed to the engine.
        path: String,
    },
    /// The requested voice id was not found in the voice pack on disk.
    VoiceNotFound {
        /// Voice id the caller asked for.
        voice: String,
    },
    /// Caller supplied an empty or obviously invalid text payload.
    InvalidText,
    /// Caller supplied an empty or unknown voice id.
    InvalidVoice,
    /// Caller supplied a speed outside the supported `(0, 4]` range.
    /// NaN / Inf / <=0 / >4 all land here.
    InvalidSpeed {
        /// The offending speed value.
        speed: f32,
    },
    /// The underlying shim returned a non-zero status code.
    ShimError {
        /// Name of the shim entry point that failed.
        op: &'static str,
        /// Raw status returned.
        code: i32,
    },
    /// Could not convert the caller-supplied model path / text / voice id
    /// to a C string (interior NUL byte).
    InvalidPath,
}

impl fmt::Display for KokoroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStub => f.write_str(
                "1bit-kokoro built with `stub` feature; rebuild with \
                 --features real-kokoro --no-default-features",
            ),
            Self::ModelLoadFailed { path } => {
                write!(f, "failed to load kokoro model at {path:?}")
            }
            Self::VoiceNotFound { voice } => {
                write!(f, "voice {voice:?} not present in kokoro voice pack")
            }
            Self::InvalidText => f.write_str("kokoro synth called with empty text payload"),
            Self::InvalidVoice => f.write_str("kokoro synth called with empty voice id"),
            Self::InvalidSpeed { speed } => {
                write!(
                    f,
                    "kokoro synth speed {speed} out of supported range (0, 4]"
                )
            }
            Self::ShimError { op, code } => {
                write!(f, "shim call {op} failed with code {code}")
            }
            Self::InvalidPath => {
                f.write_str("model/text/voice string contained an interior NUL byte")
            }
        }
    }
}

impl std::error::Error for KokoroError {}

// -----------------------------------------------------------------------------
// Input validation
// -----------------------------------------------------------------------------

/// Validate the `speed` argument passed to [`KokoroEngine::synthesize`].
///
/// Accepted range is the half-open interval `(0, 4]` — zero, negative
/// values, NaN, and +inf all reject. The ceiling matches kokoro.cpp's
/// upstream clamp; we mirror it on the Rust side so bad inputs never
/// cross the FFI boundary.
fn validate_speed(speed: f32) -> Result<(), KokoroError> {
    if !speed.is_finite() || speed <= 0.0 || speed > 4.0 {
        return Err(KokoroError::InvalidSpeed { speed });
    }
    Ok(())
}

/// Validate the `text` argument. Empty / whitespace-only rejects before
/// we pay the FFI cost.
fn validate_text(text: &str) -> Result<(), KokoroError> {
    if text.trim().is_empty() {
        return Err(KokoroError::InvalidText);
    }
    Ok(())
}

/// Validate the `voice` argument. Empty rejects before we pay the FFI
/// cost. (The shim does its own unknown-voice check against the voice
/// pack; we only cheaply guard the empty-string case here.)
fn validate_voice(voice: &str) -> Result<(), KokoroError> {
    if voice.is_empty() {
        return Err(KokoroError::InvalidVoice);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Public engine — dispatch to stub or real based on active feature
// -----------------------------------------------------------------------------

/// Internal enum kept private; `KokoroEngine` forwards to it.
enum EngineImpl {
    #[cfg(feature = "real-kokoro")]
    Real(real::RealEngine),
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    Stub(stub::StubEngine),
}

/// Safe Rust handle for a kokoro TTS session.
///
/// Create with [`KokoroEngine::new`], then call
/// [`KokoroEngine::synthesize`] as many times as you need — the shim is
/// stateless between synth calls (each one runs a full ONNX forward
/// pass and returns the resulting PCM).
pub struct KokoroEngine {
    inner: EngineImpl,
}

impl KokoroEngine {
    /// Load a kokoro ONNX model from disk.
    ///
    /// Under the default `stub` feature this always returns
    /// [`KokoroError::UnsupportedStub`] — it does not touch the filesystem.
    pub fn new<P: AsRef<Path>>(model: P) -> Result<Self, KokoroError> {
        #[cfg(feature = "real-kokoro")]
        {
            let eng = real::RealEngine::new(model)?;
            Ok(Self {
                inner: EngineImpl::Real(eng),
            })
        }
        #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
        {
            let eng = stub::StubEngine::new(model)?;
            Ok(Self {
                inner: EngineImpl::Stub(eng),
            })
        }
        #[cfg(not(any(feature = "stub", feature = "real-kokoro")))]
        {
            let _ = model;
            Err(KokoroError::UnsupportedStub)
        }
    }

    /// Synthesize `text` with `voice` at `speed`, returning mono 22 050 Hz
    /// s16le PCM samples.
    ///
    /// Validates inputs before hitting the FFI boundary:
    ///
    /// - `text` must be non-empty after trimming ASCII whitespace,
    /// - `voice` must be non-empty,
    /// - `speed` must be finite and lie in `(0, 4]`.
    ///
    /// Under the default `stub` feature this always returns
    /// [`KokoroError::UnsupportedStub`] *after* input validation passes,
    /// so bad-input tests still see the validation error they expect.
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<i16>, KokoroError> {
        validate_text(text)?;
        validate_voice(voice)?;
        validate_speed(speed)?;

        match &mut self.inner {
            #[cfg(feature = "real-kokoro")]
            EngineImpl::Real(e) => e.synthesize(text, voice, speed),
            #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
            EngineImpl::Stub(e) => e.synthesize(text, voice, speed),
        }
    }
}

// -----------------------------------------------------------------------------
// In-crate tests (>=7 required per the scaffold spec)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub mode: `new` on a missing file returns `UnsupportedStub`, not
    /// a panic, and not a filesystem-layer error.
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    #[test]
    fn stub_new_on_missing_file_returns_unsupported() {
        match KokoroEngine::new("/does/not/exist.onnx") {
            Ok(_) => panic!("stub engine should never return Ok"),
            Err(KokoroError::UnsupportedStub) => { /* expected */ }
            Err(other) => panic!("expected UnsupportedStub, got {other:?}"),
        }
    }

    /// Stub mode: even if we could construct an engine (we can't, `new`
    /// errors), the underlying `StubEngine::synthesize` method also
    /// returns `UnsupportedStub`. Exercise it directly so the check is
    /// not dead code.
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    #[test]
    fn stub_synthesize_returns_unsupported() {
        let mut eng = stub::StubEngine;
        match eng.synthesize("hello", "af_bella", 1.0) {
            Ok(_) => panic!("stub synthesize should never return Ok"),
            Err(KokoroError::UnsupportedStub) => { /* expected */ }
            Err(other) => panic!("expected UnsupportedStub, got {other:?}"),
        }
    }

    /// Reject empty text *before* hitting the FFI. We construct a
    /// `KokoroEngine` wrapping a `StubEngine` directly so we can drive
    /// the public `synthesize` path without libkokoro.
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    #[test]
    fn reject_empty_text() {
        let mut eng = KokoroEngine {
            inner: EngineImpl::Stub(stub::StubEngine),
        };
        match eng.synthesize("", "af_bella", 1.0) {
            Err(KokoroError::InvalidText) => { /* expected */ }
            other => panic!("expected InvalidText, got {other:?}"),
        }
        match eng.synthesize("   \t\n  ", "af_bella", 1.0) {
            Err(KokoroError::InvalidText) => { /* expected */ }
            other => panic!("expected InvalidText (whitespace), got {other:?}"),
        }
    }

    /// Reject empty voice id before hitting the FFI.
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    #[test]
    fn reject_empty_voice() {
        let mut eng = KokoroEngine {
            inner: EngineImpl::Stub(stub::StubEngine),
        };
        match eng.synthesize("hello world", "", 1.0) {
            Err(KokoroError::InvalidVoice) => { /* expected */ }
            other => panic!("expected InvalidVoice, got {other:?}"),
        }
    }

    /// Reject out-of-range speeds: zero, negative, >4, NaN, +inf all
    /// produce `InvalidSpeed`.
    #[cfg(all(feature = "stub", not(feature = "real-kokoro")))]
    #[test]
    fn reject_bad_speed() {
        let mut eng = KokoroEngine {
            inner: EngineImpl::Stub(stub::StubEngine),
        };
        let bad = [0.0_f32, -1.0, 4.5, f32::NAN, f32::INFINITY];
        for s in bad {
            match eng.synthesize("hello", "af_bella", s) {
                Err(KokoroError::InvalidSpeed { speed }) => {
                    // NaN != NaN, so compare bit patterns for NaN case,
                    // direct equality otherwise.
                    if s.is_nan() {
                        assert!(
                            speed.is_nan(),
                            "expected NaN speed echoed back, got {speed}"
                        );
                    } else {
                        assert_eq!(speed, s, "expected speed echoed back");
                    }
                }
                other => panic!("expected InvalidSpeed for {s}, got {other:?}"),
            }
        }
    }

    /// `SynthesisInfo` serde roundtrip — proves the serde derives compile
    /// and preserve field values.
    #[test]
    fn synthesis_info_serde_roundtrip() {
        let info = SynthesisInfo {
            sample_rate: 22_050,
            channels: 1,
            samples: 44_100,
            duration_ms: 2_000,
        };
        let j = serde_json::to_string(&info).expect("serialize");
        let info2: SynthesisInfo = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(info, info2);
        assert!(j.contains("\"sample_rate\":22050"));
        assert!(j.contains("\"channels\":1"));
        assert!(j.contains("\"samples\":44100"));
        assert!(j.contains("\"duration_ms\":2000"));
    }

    /// Error `Display` renders without panic and mentions the relevant
    /// context — a cheap guard that error messages stay useful.
    #[test]
    fn error_display_is_informative() {
        let e = KokoroError::UnsupportedStub;
        let s = format!("{e}");
        assert!(s.contains("stub"));

        let e = KokoroError::ShimError {
            op: "onebit_kokoro_synthesize",
            code: -1,
        };
        let s = format!("{e}");
        assert!(s.contains("onebit_kokoro_synthesize"));
        assert!(s.contains("-1"));

        let e = KokoroError::VoiceNotFound {
            voice: "af_bella".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("af_bella"));

        let e = KokoroError::InvalidSpeed { speed: 9.0 };
        let s = format!("{e}");
        assert!(s.contains("9"));
    }
}
