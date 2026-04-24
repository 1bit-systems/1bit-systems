//! 1bit-aie — skeleton NPU backend for AIE2P (XDNA2 / Strix Halo `npu5`).
//!
//! This crate is the Rust side of the BitNet-1.58 NPU kernel lane scoped for
//! v0.1.2 (`project_ship_gate_npu.md`). The native side lives in
//! `rocm-cpp/aie/` — tile-local C++ kernels compiled via Peano, assembled
//! into an `.xclbin` via IRON + MLIR-AIE, and loaded at runtime through
//! `libxrt`. See `docs/wiki/NPU-Kernel-Handoff.md` for the compile /
//! dispatch pipeline and `docs/wiki/NPU-AIE2P.md` for the toolchain.
//!
//! ## Status — 2026-04-24
//!
//! **Skeleton only.** The trait + data types are pinned so the router and
//! server layers can grow an `AieBackend` call site without waiting for the
//! real kernel. Every method body is `unimplemented!()` or returns
//! [`AieError::NotYetWired`]. Turn on `--features real-npu` to unlock the
//! `#[ignore]`-gated hardware tests once the xclbin lands.
//!
//! ## Scope of what's wired / not wired
//!
//! | Layer                      | This crate             | Status                          |
//! |----------------------------|------------------------|---------------------------------|
//! | Trait + types              | `AieBackend`, `AieError`, `AieBuffer`, `AieKernelHandle`, `AieDeviceInfo` | Pinned                     |
//! | xclbin load                | [`AieBackend::load_xclbin`] | stub — returns `NotYetWired` |
//! | BitNet ternary gemv        | [`AieBackend::bitnet_gemv`] | stub — returns `NotYetWired` |
//! | Device introspection       | [`AieBackend::device_info`] | stub — returns placeholder  |
//! | libxrt FFI                 | out-of-tree           | TODO — see `Cargo.toml` note      |
//! | Peano tile compile         | `rocm-cpp/aie/`        | README + placeholder kernel only |
//!
//! ## Why a trait, not a free function
//!
//! The router (`crates/1bit-router/src/backend_impl.rs`) already carries a
//! concrete [`HipBackend`] and a `Backend::Cpu` stub. When the NPU lane
//! lights up the router grows a `Backend::Aie` variant that holds a
//! `Box<dyn AieBackend>`. A trait lets us:
//!
//! 1. Swap the xclbin behind the scenes (e.g. short-K tile vs long-K tile)
//!    without re-plumbing the router.
//! 2. Mock in unit tests — see `MockAieBackend` below.
//! 3. Keep libxrt out of the workspace default build. `real-npu` is the
//!    link-gate, same pattern as `onebit-hip::link-rocm`.

#![deny(missing_docs)]

use std::path::Path;

/// Opaque handle for a loaded .xclbin kernel, returned by
/// [`AieBackend::load_xclbin`] and threaded back into dispatch calls.
///
/// The `u32` payload is a backend-local index into a kernel table; it is
/// **not** a pointer and does not alias any host or device memory. When
/// libxrt bindings land the real layout will wrap `xrt::kernel` + a
/// `xrt::hw_context` handle — the public API stays the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AieKernelHandle(pub u32);

/// A device-resident buffer managed by the AIE backend.
///
/// Today this is an opaque tag — the skeleton crate does not allocate.
/// When `real-npu` lights up, this will wrap `xrt::bo` (a buffer object
/// sync'd between host + tile memory) with `len_bytes` and the element
/// dtype tracked host-side for safety.
///
/// The `dtype` discriminant mirrors what the BitNet ternary gemv pipeline
/// consumes (`unpack_ternary_2bit_to_int8` → `matmul_vectorized_8x8x8_i8_i32`
/// → `scale_i32_fp16`; see `npu-kernels/bitnet/README.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AieBuffer {
    /// Backend-local id (real impl: `xrt::bo` cookie).
    pub id: u32,
    /// Buffer size in **bytes**, not elements.
    pub len_bytes: usize,
    /// Element dtype the kernel expects on this buffer's port.
    pub dtype: AieDtype,
}

/// Element types the BitNet NPU pipeline understands today.
///
/// Deliberately narrow. AIE2P does int8 / int16 / int32 / bf16 natively;
/// IEEE fp16 is emulated and not competitive (see
/// `docs/wiki/NPU-AIE2P.md` — "What does NOT land"). Halo `.h1b` fp16
/// scales are converted to bf16 at load time for this reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieDtype {
    /// 2-bit packed ternary — halo v2 weight layout (4 codes per byte).
    PackedT2,
    /// 8-bit signed integer — activations + unpacked ternary weights.
    I8,
    /// 32-bit signed integer — accumulator output of the stock matmul tile.
    I32,
    /// bfloat16 — per-row scales and the final post-scale tile. See
    /// `scale_i32_fp16.cc`; loader converts fp16→bf16 up-front.
    Bf16,
}

/// Reported at [`AieBackend::device_info`]; covers only what the router
/// needs for log lines + telemetry today. If it grows to include live
/// temperature / tile-memory-free / queue depth, wrap it with a method
/// rather than breaking this struct.
#[derive(Debug, Clone)]
pub struct AieDeviceInfo {
    /// Linux device node, e.g. `RyzenAI-npu5` on Strix Halo.
    pub device_name: String,
    /// `amdxdna` firmware version string (e.g. `1.0.0.166`).
    pub firmware_version: String,
    /// Number of compute columns reported by xrt-smi.
    pub columns: u8,
    /// Class of AIE silicon (`AIE2P` for Strix Halo; the Phoenix NPU
    /// predecessor is `AIE2`). IRON's `device_utils.py` terminology.
    pub tile_class: &'static str,
}

/// Error surface for the AIE backend. Kept narrow and `thiserror`-based so
/// the router layer can lift these into [`crate::BackendError`] without
/// regex on a `String`.
#[derive(Debug, thiserror::Error)]
pub enum AieError {
    /// The xclbin path does not exist or is not a regular file.
    #[error("xclbin not found: {0}")]
    XclbinNotFound(String),
    /// The kernel binary loaded, but its entry symbol doesn't match
    /// what this backend expects (e.g. wrong tile size).
    #[error("xclbin shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    /// Buffer dtype / port dtype disagreement. See [`AieDtype`].
    #[error("dtype mismatch on port {port}: expected {expected:?}, got {got:?}")]
    DtypeMismatch {
        /// Kernel port name (e.g. `weights`, `activations`, `scales`).
        port: &'static str,
        /// Dtype the kernel expects.
        expected: AieDtype,
        /// Dtype the caller supplied.
        got: AieDtype,
    },
    /// libxrt reported a runtime error while launching or syncing a BO.
    /// Carries the backend-local error number; the real impl will
    /// translate to a string via `xrt::error::what()`.
    #[error("xrt runtime error: code {0}")]
    Xrt(i32),
    /// The skeleton crate is doing its job: the selected code path is
    /// scaffolded but not yet wired to real hardware. This is what every
    /// method returns today.
    #[error("aie backend: {0} is scaffolded but not yet wired (build with --features real-npu)")]
    NotYetWired(&'static str),
}

/// Public trait for the NPU backend.
///
/// Implementations:
///
/// - [`StubAieBackend`] (this crate, always compiled) — every method
///   returns [`AieError::NotYetWired`]. Keeps the router's
///   `Backend::Aie` variant compilable on CI without NPU hardware.
/// - `XrtAieBackend` (gated on `--features real-npu`, future) — the real
///   dispatch path through libxrt.
///
/// See `docs/wiki/NPU-Kernel-Handoff.md` §"FFI" for the planned
/// C++-side entry points and how Rust will call into them.
pub trait AieBackend {
    /// Load a compiled `.xclbin` from disk and prepare it for dispatch.
    ///
    /// The returned [`AieKernelHandle`] is what you feed into
    /// [`AieBackend::bitnet_gemv`]. Calling this twice with the same path
    /// is allowed and must be idempotent at the hardware level (libxrt
    /// handles the hw_context caching).
    fn load_xclbin(&mut self, path: &Path) -> Result<AieKernelHandle, AieError>;

    /// Dispatch a BitNet-1.58 ternary GEMV onto the NPU.
    ///
    /// Kernel pipeline (see `npu-kernels/bitnet/README.md` for the MLIR
    /// emitter and the three stock tile objects):
    ///
    /// ```text
    ///   weights (PackedT2) ─┐
    ///                       │
    ///     unpack_ternary_2bit_to_int8  →  i8 tile
    ///                                       │
    ///   x (I8)      ──────────────────────┐ │
    ///                                     ▼ ▼
    ///          matmul_vectorized_8x8x8_i8_i32  →  i32 tile
    ///                                                │
    ///   scales (Bf16)  ──────────────────────────────┤
    ///                                                ▼
    ///                                          scale_i32_bf16  →  out (Bf16)
    /// ```
    ///
    /// Invariants enforced on entry:
    ///
    /// * `weights.dtype == PackedT2`, `x.dtype == I8`,
    ///   `out.dtype == Bf16`, `scales.dtype == Bf16`.
    /// * `weights.len_bytes * 4 == M*K` (4 ternary codes / byte).
    /// * `x.len_bytes == K * N` (i8 activations).
    /// * `scales.len_bytes == M * 2` (one bf16 per output row).
    /// * `out.len_bytes == M * N * 2`.
    ///
    /// `M`, `K`, `N` come from the xclbin metadata — not the trait — so the
    /// dispatch is shape-checked against the compiled tile geometry, not
    /// re-guessed at call time.
    fn bitnet_gemv(
        &mut self,
        k: AieKernelHandle,
        weights: AieBuffer,
        x: AieBuffer,
        out: &mut AieBuffer,
        scales: AieBuffer,
    ) -> Result<(), AieError>;

    /// Cheap, call-at-startup device introspection. Used by
    /// `/v1/models`-style probes + log lines. Never fails in the stub.
    fn device_info(&self) -> AieDeviceInfo;
}

// ---------------------------------------------------------------------------
// Stub implementation — always compiled, always returns NotYetWired.
// ---------------------------------------------------------------------------

/// The null NPU backend. Builds without libxrt. Every method returns
/// [`AieError::NotYetWired`] so the router can hold a `Box<dyn
/// AieBackend>` today and flip to the real impl when the xclbin lands.
#[derive(Debug, Default)]
pub struct StubAieBackend;

impl StubAieBackend {
    /// Constructs the stub. Infallible — no device handle to acquire.
    pub fn new() -> Self {
        Self
    }
}

impl AieBackend for StubAieBackend {
    fn load_xclbin(&mut self, _path: &Path) -> Result<AieKernelHandle, AieError> {
        Err(AieError::NotYetWired("load_xclbin"))
    }

    fn bitnet_gemv(
        &mut self,
        _k: AieKernelHandle,
        _weights: AieBuffer,
        _x: AieBuffer,
        _out: &mut AieBuffer,
        _scales: AieBuffer,
    ) -> Result<(), AieError> {
        Err(AieError::NotYetWired("bitnet_gemv"))
    }

    fn device_info(&self) -> AieDeviceInfo {
        AieDeviceInfo {
            device_name: "stub".to_string(),
            firmware_version: "0.0.0-stub".to_string(),
            columns: 0,
            tile_class: "AIE2P",
        }
    }
}

// ---------------------------------------------------------------------------
// Real backend — gated on `real-npu` so CI doesn't need libxrt.
// ---------------------------------------------------------------------------

/// Stand-in for the future `XrtAieBackend`. Currently only builds under
/// `--features real-npu` and its body is still `unimplemented!()`. Kept
/// behind the feature so the crate doesn't pretend to have a real path.
///
/// When libxrt bindings land, replace this block with the real impl and
/// drop the `unimplemented!` bodies.
#[cfg(feature = "real-npu")]
pub mod real {
    use super::*;

    /// NPU backend that dispatches through libxrt.
    #[derive(Debug, Default)]
    pub struct XrtAieBackend {
        // Future: xrt::device, xrt::hw_context, kernel table.
    }

    impl XrtAieBackend {
        /// Open the NPU and claim a hw_context. Fails if `amdxdna` is
        /// not loaded or no npu5 device is present.
        pub fn new() -> Result<Self, AieError> {
            unimplemented!("XrtAieBackend::new — requires libxrt bindings + xclbin load")
        }
    }

    impl AieBackend for XrtAieBackend {
        fn load_xclbin(&mut self, _path: &Path) -> Result<AieKernelHandle, AieError> {
            unimplemented!("XrtAieBackend::load_xclbin — requires libxrt bindings")
        }

        fn bitnet_gemv(
            &mut self,
            _k: AieKernelHandle,
            _weights: AieBuffer,
            _x: AieBuffer,
            _out: &mut AieBuffer,
            _scales: AieBuffer,
        ) -> Result<(), AieError> {
            unimplemented!("XrtAieBackend::bitnet_gemv — requires libxrt bindings + xclbin")
        }

        fn device_info(&self) -> AieDeviceInfo {
            unimplemented!("XrtAieBackend::device_info — requires libxrt bindings")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The stub backend must compile, construct, and refuse every real
    /// call with `NotYetWired`. This is the canary that tells us the
    /// router's `Backend::Aie` variant is link-safe even when no xclbin
    /// is present.
    #[test]
    fn stub_backend_refuses_every_op() {
        let mut be = StubAieBackend::new();
        let err = be
            .load_xclbin(Path::new("/nonexistent/halo_bitnet.xclbin"))
            .unwrap_err();
        assert!(matches!(err, AieError::NotYetWired("load_xclbin")));

        let buf = AieBuffer {
            id: 0,
            len_bytes: 0,
            dtype: AieDtype::I8,
        };
        let mut out = AieBuffer {
            id: 1,
            len_bytes: 0,
            dtype: AieDtype::Bf16,
        };
        let err = be
            .bitnet_gemv(AieKernelHandle(0), buf, buf, &mut out, buf)
            .unwrap_err();
        assert!(matches!(err, AieError::NotYetWired("bitnet_gemv")));
    }

    /// Device info on the stub returns a deterministic "stub" payload;
    /// the server /metrics probe uses this to decide whether to render
    /// the NPU lane badge as "off" or "live".
    #[test]
    fn stub_backend_device_info_is_deterministic() {
        let be = StubAieBackend::new();
        let info = be.device_info();
        assert_eq!(info.device_name, "stub");
        assert_eq!(info.columns, 0);
        assert_eq!(info.tile_class, "AIE2P");
    }

    /// `AieDtype` ordering + equality is load-bearing — it's what guards
    /// the port dtype-mismatch in `bitnet_gemv`. Locking the discriminants
    /// keeps a rogue refactor from silently swapping Bf16 and I32 past the
    /// shape check.
    #[test]
    fn dtype_equality_is_stable() {
        assert_ne!(AieDtype::I8, AieDtype::I32);
        assert_ne!(AieDtype::I32, AieDtype::Bf16);
        assert_ne!(AieDtype::PackedT2, AieDtype::I8);
        // Same discriminant compares equal.
        let a = AieDtype::I32;
        let b = AieDtype::I32;
        assert_eq!(a, b);
    }

    /// Sanity: the real backend type is only reachable with `--features
    /// real-npu`. This test exists so `cargo test --workspace` confirms
    /// the cfg-gate continues to exclude the unimplemented bodies from
    /// the default build. Only runs when the feature is on; marked
    /// `#[ignore]` so even then it's opt-in per the CLAUDE.md convention
    /// for GPU / real-hardware tests.
    #[cfg(feature = "real-npu")]
    #[test]
    #[ignore = "requires npu5 + xclbin on disk"]
    fn real_backend_is_reachable_under_feature() {
        // Deliberately does not construct — `XrtAieBackend::new` is
        // `unimplemented!` today. The test's presence is the signal.
        let _t: Option<real::XrtAieBackend> = None;
    }
}
