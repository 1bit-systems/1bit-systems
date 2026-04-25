//! 1bit-voice — sentence-boundary-streaming orchestrator.
//!
//! Point of this crate: cut mouth-to-ear latency for Echo.
//!
//! Today's naive voice loop serializes three stages:
//!
//! ```text
//!   LLM finishes full reply    ──────────────▶ 1.0 s
//!   then TTS starts synthesis  ──────────────▶ 0.5 s
//!   then audio plays           ──────────────▶ 0.02 s
//!                              total mouth-to-ear: ~1.5 s
//! ```
//!
//! 1bit-voice interleaves: as soon as the LLM emits a sentence-boundary
//! (`.`, `!`, `?`, `\n`), that sentence is handed to TTS while the LLM
//! keeps generating the next one. First-audio-out arrives 3-5× sooner:
//!
//! ```text
//!   LLM emits "Paris is the capital"                ──▶ 0.3 s
//!     ↳ TTS starts on the first clause              ──▶ 0.3 + 0.2 = 0.5 s FIRST AUDIO OUT
//!   LLM keeps streaming "of France..."              ──▶ 0.7 s
//!     ↳ TTS overlaps with the LLM                   ──▶ no extra wall-clock
//! ```
//!
//! Two public types:
//!
//! * [`SentenceSplitter`] — stateful chunker that eats streamed text and
//!   yields complete sentences one at a time.
//! * [`VoicePipeline`] — top-level orchestrator. Takes a prompt, streams
//!   the reply from 1bit-server's SSE, drives the splitter, calls kokoro
//!   per sentence, and exposes the resulting WAV chunks as a `Stream`.
//!
//! No whisper on the input side yet — that stage is still a separate HTTP
//! POST in the live system. STT is now served by lemond's `whispercpp:vulkan`
//! backend at /home/bcloud/repos/lemonade/; when streaming whisper lands
//! upstream we add it here as the front of the pipeline.

pub mod pipeline;
pub mod splitter;

pub use pipeline::{VoiceChunk, VoiceConfig, VoicePipeline};
pub use splitter::SentenceSplitter;
