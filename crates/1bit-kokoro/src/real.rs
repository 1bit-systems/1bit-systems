//! Real engine — libkokoro-backed, compiled only with the `real-kokoro`
//! feature.
//!
//! The heavy lifting (text tokenization, phoneme expansion, ONNX forward
//! pass, waveform assembly) lives on the C++ side in `cpp/shim.cpp`.
//! This module is a thin `unsafe`-wrapping layer that converts Rust
//! strings + slices into pointers the shim can consume and marshals the
//! i16 PCM buffer back out.
//!
//! First-pass skeleton: `synthesize` is `unimplemented!()` so we don't
//! ship half-baked logic under the real feature. The shim side is
//! declared but returns -1 until the kokoro.cpp wiring lands in a
//! follow-up.

use crate::KokoroError;
use crate::ffi;
use std::path::Path;

/// Owns the opaque shim context and releases it on drop.
pub struct RealEngine {
    ctx: *mut ffi::KokoroCtx,
}

// SAFETY: the underlying CtxImpl (defined in cpp/shim.cpp) owns its
// onnxruntime session plus any per-session state. We expose `&mut self`
// on every Rust method, so the Rust type system enforces single-threaded
// access from the Rust side; the shim is expected to guard any internal
// mutable state with its own mutex (mirroring the 1bit-whisper CtxImpl).
unsafe impl Send for RealEngine {}

impl RealEngine {
    /// Load a kokoro ONNX model from disk via the shim.
    pub fn new<P: AsRef<Path>>(_model: P) -> Result<Self, KokoroError> {
        // Real shim wiring lands in a follow-up once kokoro.cpp builds
        // clean on strixhalo (gfx1151) + sliger (Xe2) CI. Keeping this
        // as an explicit `unimplemented!` makes it impossible to ship a
        // half-initialized engine under the `real-kokoro` feature.
        unimplemented!("real-kokoro path not yet wired — see project_halo_kokoro.md")
    }

    /// Synthesize speech via the shim. See [`crate::KokoroEngine::synthesize`]
    /// for input validation rules — by the time this method runs, the
    /// caller's `text` / `voice` / `speed` have already been vetted.
    pub fn synthesize(
        &mut self,
        _text: &str,
        _voice: &str,
        _speed: f32,
    ) -> Result<Vec<i16>, KokoroError> {
        unimplemented!("real-kokoro synthesize not yet wired — see project_halo_kokoro.md")
    }
}

impl Drop for RealEngine {
    fn drop(&mut self) {
        // SAFETY: `self.ctx` was either never populated (unimplemented
        // construction path) or would have been produced by
        // `onebit_kokoro_init`. The shim `_free` entry point tolerates
        // NULL. Once the real wiring lands, this will become a single
        // `_free` call exactly once.
        if !self.ctx.is_null() {
            unsafe { ffi::onebit_kokoro_free(self.ctx) };
            self.ctx = core::ptr::null_mut();
        }
    }
}
