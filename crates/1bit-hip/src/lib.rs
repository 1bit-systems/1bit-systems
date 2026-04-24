//! 1bit-hip — safe Rust surface over the rocm-cpp C API.
//!
//! The native side (`librocm_cpp.so`) owns every HIP-device kernel. This
//! crate is only a bridge: it translates Rust slices + scalars into raw
//! device pointers and forwards them to the underlying `rcpp_*` calls.
//!
//! # Safety model
//!
//! Every wrapper here takes **opaque device pointers** as `DevicePtr<T>` /
//! `DeviceMutPtr<T>` — thin newtypes around `*const T` / `*mut T`. The
//! caller is responsible for having allocated them on the correct HIP
//! device (e.g. via `hipMalloc`, or via a higher-level allocator crate).
//! *We do not validate device-residency from Rust*; a host pointer masquerading
//! as a device pointer is UB on the native side.
//!
//! Slice-taking entry points (the host-side packer) are bounds-checked in
//! Rust before touching the FFI boundary.
//!
//! # Feature flags
//!
//! - `link-rocm` (default): link `librocm_cpp.so` at build time. Required
//!   on any machine that actually runs kernels.
//! - No `link-rocm`: every call returns `RcppStatus::Unsupported`. The
//!   crate still type-checks and builds, so CI hosts without AMD silicon
//!   can run `cargo check` / `cargo build` without a ROCm install.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod ffi;

use core::ffi::{c_int, c_void};
use core::ptr;

// -----------------------------------------------------------------------------
// Error / status types
// -----------------------------------------------------------------------------

/// Mirror of `rcpp_status_t` from `include/rocm_cpp/ck_gemm.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RcppStatus {
    /// Success.
    Ok = 0,
    /// Caller-side argument validation failed (shape, null pointer, etc.).
    InvalidArg = 1,
    /// The native side does not implement this kernel on this target /
    /// shape, or the `link-rocm` feature is off.
    Unsupported = 2,
    /// A HIP runtime call failed inside the kernel launch.
    HipError = 3,
    /// Internal invariant broken inside rocm-cpp. Report upstream.
    Internal = 4,
}

impl RcppStatus {
    /// Convert from the raw C integer as returned by `rcpp_*` functions.
    /// Unknown values are folded into `Internal` to stay safe rather than
    /// panicking across an FFI boundary.
    #[inline]
    pub fn from_raw(raw: c_int) -> Self {
        match raw {
            0 => Self::Ok,
            1 => Self::InvalidArg,
            2 => Self::Unsupported,
            3 => Self::HipError,
            4 => Self::Internal,
            _ => Self::Internal,
        }
    }

    /// `true` iff the native call returned `RCPP_OK`.
    #[inline]
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Turn this status into a `Result`, discarding the `Ok` payload.
    #[inline]
    pub fn into_result(self) -> Result<(), RcppError> {
        match self {
            Self::Ok => Ok(()),
            other => Err(RcppError::from(other)),
        }
    }
}

/// Structured error with a `thiserror`-derived `Display` impl, for use at
/// the application boundary. Keep `RcppStatus` around for hot-path code
/// that wants to branch without a heap allocation.
#[derive(Debug, thiserror::Error)]
pub enum RcppError {
    /// Caller supplied invalid arguments (null, wrong shape, etc).
    #[error("rocm-cpp: invalid argument")]
    InvalidArg,
    /// The native side does not support this operation on this hardware.
    #[error("rocm-cpp: unsupported operation or shape")]
    Unsupported,
    /// A HIP runtime API call failed inside the kernel launcher.
    #[error("rocm-cpp: HIP runtime error")]
    HipError,
    /// Internal invariant broken inside rocm-cpp.
    #[error("rocm-cpp: internal error")]
    Internal,
    /// A Rust-side precondition (slice length, alignment) was violated
    /// before the FFI call.
    #[error("rocm-cpp: precondition failed: {0}")]
    Precondition(&'static str),
}

impl From<RcppStatus> for RcppError {
    fn from(s: RcppStatus) -> Self {
        match s {
            RcppStatus::Ok => Self::Internal, // should never happen, but don't panic
            RcppStatus::InvalidArg => Self::InvalidArg,
            RcppStatus::Unsupported => Self::Unsupported,
            RcppStatus::HipError => Self::HipError,
            RcppStatus::Internal => Self::Internal,
        }
    }
}

// -----------------------------------------------------------------------------
// Device pointer newtypes
// -----------------------------------------------------------------------------

/// Read-only device pointer. Opaque to the caller; the constructor is
/// `unsafe` because we cannot verify device residency from Rust.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct DevicePtr<T>(pub *const T);

/// Writable device pointer. Same caveats as [`DevicePtr`].
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct DeviceMutPtr<T>(pub *mut T);

impl<T> DevicePtr<T> {
    /// Wrap a raw device pointer. Caller asserts the pointer refers to an
    /// allocation on the current HIP device with at least the expected
    /// number of elements of type `T`, and that the allocation outlives
    /// any kernel call that reads it.
    ///
    /// # Safety
    /// See struct docs.
    #[inline]
    pub const unsafe fn new(ptr: *const T) -> Self {
        Self(ptr)
    }

    /// A null device pointer. Valid only where the native API explicitly
    /// accepts null.
    pub const fn null() -> Self {
        Self(ptr::null())
    }

    /// Erase the element type. Used at the FFI boundary where the C API
    /// takes `const void*`.
    #[inline]
    fn as_void(self) -> *const c_void {
        self.0 as *const c_void
    }
}

impl<T> DeviceMutPtr<T> {
    /// Wrap a raw device pointer. See [`DevicePtr::new`].
    ///
    /// # Safety
    /// See struct docs.
    #[inline]
    pub const unsafe fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// A null device pointer.
    pub const fn null() -> Self {
        Self(ptr::null_mut())
    }

    /// Erase the element type.
    #[inline]
    fn as_void(self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

/// Opaque HIP stream handle. `None` = default (null) stream.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct HipStream(pub *mut c_void);

impl Default for HipStream {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// SAFETY: `HipStream` is a pointer-sized opaque handle whose value is
// only interpreted by the HIP runtime. Sending it between threads is
// legal provided the runtime is thread-safe (which HIP's default
// streams are). We do not provide `Sync` because concurrent use of the
// same stream from multiple threads requires external synchronization.
unsafe impl Send for HipStream {}

impl HipStream {
    /// The default / null stream. Kernels submitted here serialize with
    /// every other stream on the device.
    pub const DEFAULT: Self = Self(ptr::null_mut());

    #[inline]
    fn as_raw(self) -> *mut c_void {
        self.0
    }
}

// -----------------------------------------------------------------------------
// Public API — ternary GEMV family
// -----------------------------------------------------------------------------

/// `y[m] = sum_k W[m,k] * x[k]` with halo-1bit-packed ternary weights and
/// INT8 activations. Output is FP32 on device.
///
/// - `packed_weights`: `uint8 [M, (K+3)/4]` on device, halo code.
/// - `activations_i8`: `int8 [K]` on device.
/// - `activation_scale`: scalar, so that `real_a[k] = i8[k] * scale`.
/// - `row_scales`: `float [M]` on device — per-row weight scale.
/// - `output`: `float [M]` on device.
/// - `stream`: HIP stream (pass `HipStream::DEFAULT` for serial).
pub fn ternary_gemv_halo(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output: DeviceMutPtr<f32>,
    m: i32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: caller has upheld the `DevicePtr` contract (valid device
    // allocations of appropriate length). All pointers are forwarded
    // unchanged and the C side writes only to `output`. No Rust references
    // are constructed, so aliasing rules do not apply.
    let raw = unsafe {
        ffi::rcpp_ternary_gemv_halo(
            packed_weights.as_void(),
            activations_i8.as_void(),
            activation_scale,
            row_scales.as_void(),
            output.as_void(),
            m as c_int,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// FP16-output variant of [`ternary_gemv_halo`]: writes `__half` directly,
/// eliminating the FP32→FP16 follow-up in the decode loop.
pub fn ternary_gemv_halo_f16(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output_f16: DeviceMutPtr<u16>,
    m: i32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: same contract as `ternary_gemv_halo`. `u16` is used as the
    // FP16 container (IEEE-754 binary16 bit pattern); the kernel writes
    // exactly M * sizeof(__half) bytes starting at `output_f16`.
    let raw = unsafe {
        ffi::rcpp_ternary_gemv_halo_f16(
            packed_weights.as_void(),
            activations_i8.as_void(),
            activation_scale,
            row_scales.as_void(),
            output_f16.as_void(),
            m as c_int,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// Sherry 1.25-bit (halo-1bit v3) ternary GEMV with FP16 output.
///
/// Row bytes = `K * 5 / 32`; `K` must be a multiple of 32. The kernel
/// rejects bad shapes with `RCPP_INVALID_ARG`.
pub fn ternary_gemv_sherry_f16(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output_f16: DeviceMutPtr<u16>,
    m: i32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    if k % 32 != 0 {
        return RcppStatus::InvalidArg;
    }
    // SAFETY: device-pointer contract upheld by caller; K divisibility
    // pre-checked above so the C side sees only valid shapes.
    let raw = unsafe {
        ffi::rcpp_ternary_gemv_sherry_f16(
            packed_weights.as_void(),
            activations_i8.as_void(),
            activation_scale,
            row_scales.as_void(),
            output_f16.as_void(),
            m as c_int,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// TQ1 base-3 (halo-1bit v4) ternary GEMV with FP16 output. **`k_padded`
/// is already rounded up by the caller to a multiple of 20** (one
/// 4-byte macro-group = 20 ternaries).
pub fn ternary_gemv_tq1_halo_f16(
    packed_weights: DevicePtr<u8>,
    activations_i8: DevicePtr<i8>,
    activation_scale: f32,
    row_scales: DevicePtr<f32>,
    output_f16: DeviceMutPtr<u16>,
    m: i32,
    k_padded: i32,
    stream: HipStream,
) -> RcppStatus {
    if k_padded % 20 != 0 {
        return RcppStatus::InvalidArg;
    }
    // SAFETY: device-pointer contract + K padding pre-check above.
    let raw = unsafe {
        ffi::rcpp_ternary_gemv_tq1_halo_f16(
            packed_weights.as_void(),
            activations_i8.as_void(),
            activation_scale,
            row_scales.as_void(),
            output_f16.as_void(),
            m as c_int,
            k_padded as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// FP16 × FP16 GEMV (LM head)
// -----------------------------------------------------------------------------

/// `y[m] = sum_k W[m,k] * x[k]` with FP16 weights and FP16 inputs,
/// producing FP32 output. Used for the tied-embedding LM head.
pub fn fp16_gemv(
    w: DevicePtr<u16>,
    x: DevicePtr<u16>,
    y: DeviceMutPtr<f32>,
    m: i32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pure forwarding of caller-supplied device pointers to the
    // native kernel. No Rust-side dereference occurs.
    let raw = unsafe {
        ffi::rcpp_fp16_gemv(
            w.as_void(),
            x.as_void(),
            y.as_void(),
            m as c_int,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// KV cache attention
// -----------------------------------------------------------------------------

/// FP16 flash-style decode attention, single-pass (no seq-tile split).
///
/// Shapes:
/// - Q: `[num_q_heads, head_dim]`
/// - K / V: `[seq_len, num_kv_heads, head_dim]`
/// - out: `[num_q_heads, head_dim]`
pub fn kv_cache_attn_decode(
    q: DevicePtr<u16>,
    k: DevicePtr<u16>,
    v: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    scale: f32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only; kernel owns all reads/writes.
    let raw = unsafe {
        ffi::rcpp_kv_cache_attn_decode(
            q.as_void(),
            k.as_void(),
            v.as_void(),
            out.as_void(),
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            seq_len as c_int,
            scale,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// Split-KV ("flash-decoding") FP16 decode attention. Same math as
/// [`kv_cache_attn_decode`] but splits the seq axis into TILE=128 chunks
/// to recover occupancy on gfx1151.
pub fn kv_cache_attn_decode_fd(
    q: DevicePtr<u16>,
    k: DevicePtr<u16>,
    v: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    scale: f32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: same as `kv_cache_attn_decode`.
    let raw = unsafe {
        ffi::rcpp_kv_cache_attn_decode_fd(
            q.as_void(),
            k.as_void(),
            v.as_void(),
            out.as_void(),
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            seq_len as c_int,
            scale,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// INT8 KV-cache decode attention. Q is FP16, K/V are INT8 with per-
/// `(pos, kv_head)` FP16 scales; dequant is fused inside the kernel.
pub fn kv_cache_attn_decode_i8(
    q: DevicePtr<u16>,
    k_i8: DevicePtr<i8>,
    v_i8: DevicePtr<i8>,
    k_scales_f16: DevicePtr<u16>,
    v_scales_f16: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    scale: f32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only; the kernel dequantizes K/V
    // internally using the supplied scale buffers.
    let raw = unsafe {
        ffi::rcpp_kv_cache_attn_decode_i8(
            q.as_void(),
            k_i8.as_void(),
            v_i8.as_void(),
            k_scales_f16.as_void(),
            v_scales_f16.as_void(),
            out.as_void(),
            num_q_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            seq_len as c_int,
            scale,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// RMSNorm variants
// -----------------------------------------------------------------------------

/// In/out FP16 RMSNorm: `y = (x / sqrt(mean(x^2) + eps)) * weight`.
pub fn rmsnorm_fp16(
    x: DevicePtr<u16>,
    weight: DevicePtr<u16>,
    y: DeviceMutPtr<u16>,
    eps: f32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_rmsnorm_fp16(
            x.as_void(),
            weight.as_void(),
            y.as_void(),
            eps,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// RMSNorm reading FP32 input, writing FP16 output with FP16 weight.
/// Pairs with [`residual_add_fp32_from_fp16`] for FP32 residual streams.
pub fn rmsnorm_fp32_in_fp16_out(
    x_fp32: DevicePtr<f32>,
    weight: DevicePtr<u16>,
    y_fp16: DeviceMutPtr<u16>,
    eps: f32,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_rmsnorm_fp32_in_fp16_out(
            x_fp32.as_void(),
            weight.as_void(),
            y_fp16.as_void(),
            eps,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// FP32 residual accumulator: `x_fp32[i] += (float)src_fp16[i]`.
pub fn residual_add_fp32_from_fp16(
    x_fp32: DeviceMutPtr<f32>,
    src_fp16: DevicePtr<u16>,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_residual_add_fp32_from_fp16(
            x_fp32.as_void(),
            src_fp16.as_void(),
            n as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// RoPE
// -----------------------------------------------------------------------------

/// In-place rotary position embedding on `[num_heads, head_dim]` at the
/// given position. `head_dim` must be even (checked on the native side).
pub fn rope_fp16(
    x: DeviceMutPtr<u16>,
    pos: i32,
    theta: f32,
    num_heads: i32,
    head_dim: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only; the kernel modifies x in place
    // and takes no aliasing inputs.
    let raw = unsafe {
        ffi::rcpp_rope_fp16(
            x.as_void(),
            pos as c_int,
            theta,
            num_heads as c_int,
            head_dim as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// FP16 -> INT8 quantizer
// -----------------------------------------------------------------------------

/// Per-vector FP16 → INT8 quantizer. Writes `K` int8 values and a single
/// FP32 scale at `scale_dev`.
pub fn quantize_fp16_to_i8(
    x_fp16: DevicePtr<u16>,
    x_i8_out: DeviceMutPtr<i8>,
    scale_out: DeviceMutPtr<f32>,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only; kernel writes to x_i8_out and
    // scale_out, reads x_fp16.
    let raw = unsafe {
        ffi::rcpp_quantize_fp16_to_i8(
            x_fp16.as_void(),
            x_i8_out.as_void(),
            scale_out.0,
            k as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// Argmax
// -----------------------------------------------------------------------------

/// Argmax over FP32 logits — writes max-index to `*out_idx_dev`.
/// Caller pre-allocates one `i32` on device for `out_idx_dev`.
pub fn argmax_fp32(
    logits: DevicePtr<f32>,
    out_idx: DeviceMutPtr<i32>,
    v: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding; out_idx is a single int32 on device.
    let raw = unsafe {
        ffi::rcpp_argmax_fp32(
            logits.as_void(),
            out_idx.as_void(),
            v as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// Top-1 argmax over FP16 logits — writes max-index to `*out_idx_dev`.
/// Paired with `forward_token_greedy`'s D→H-skip fast path for a future
/// fp16-output lm_head variant. Caller pre-allocates one `i32` on device
/// for `out_idx_dev`.
pub fn argmax_fp16_top1(
    logits: DevicePtr<u16>,
    out_idx: DeviceMutPtr<i32>,
    v: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding; out_idx is a single int32 on device.
    // `u16` is the Rust-side stand-in for `__half` (same layout on gfx1151).
    let raw = unsafe {
        ffi::rcpp_argmax_fp16_top1(
            logits.as_void(),
            out_idx.as_void(),
            v as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// Embedding lookup
// -----------------------------------------------------------------------------

/// `y[k] = embedding[token_id, k]` for `k in 0..hidden-1`.
pub fn embedding_lookup_fp16(
    embedding: DevicePtr<u16>,
    token_id: i32,
    y: DeviceMutPtr<u16>,
    hidden: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding; the kernel reads one row of `embedding`.
    let raw = unsafe {
        ffi::rcpp_embedding_lookup_fp16(
            embedding.as_void(),
            token_id as c_int,
            y.as_void(),
            hidden as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// Additional elementwise + FFN kernels
// -----------------------------------------------------------------------------

/// SiLU-GLU: `y[i] = silu(up[i]) * gate[i]`.
pub fn silu_glu_fp16(
    up: DevicePtr<u16>,
    gate: DevicePtr<u16>,
    y: DeviceMutPtr<u16>,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_silu_glu_fp16(
            up.as_void(),
            gate.as_void(),
            y.as_void(),
            n as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// ReLU²-GLU (BitNet-b1.58 `hidden_act="relu2"`): `y[i] = relu(gate[i])² * up[i]`.
pub fn relu2_glu_fp16(
    gate: DevicePtr<u16>,
    up: DevicePtr<u16>,
    y: DeviceMutPtr<u16>,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_relu2_glu_fp16(
            gate.as_void(),
            up.as_void(),
            y.as_void(),
            n as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// Fused ReLU²-GLU + FFN sub-norm. Keeps the raw `relu²(gate) * up`
/// product in FP32 long enough to avoid FP16 overflow on real BitNet
/// weights before applying the ffn sub-norm and casting to FP16.
pub fn relu2_glu_rmsnorm_fp16(
    gate: DevicePtr<u16>,
    up: DevicePtr<u16>,
    ffn_sub_norm: DevicePtr<u16>,
    y: DeviceMutPtr<u16>,
    eps: f32,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_relu2_glu_rmsnorm_fp16(
            gate.as_void(),
            up.as_void(),
            ffn_sub_norm.as_void(),
            y.as_void(),
            eps,
            n as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

/// FP16 residual add: `y[i] += src[i]`.
pub fn residual_add_fp16(
    y: DeviceMutPtr<u16>,
    src: DevicePtr<u16>,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_residual_add_fp16(y.as_void(), src.as_void(), n as c_int, stream.as_raw())
    };
    RcppStatus::from_raw(raw)
}

/// In-place FP32 → FP16 cast.
pub fn fp32_to_fp16(
    x_fp32: DevicePtr<f32>,
    y_fp16: DeviceMutPtr<u16>,
    n: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_fp32_to_fp16(
            x_fp32.as_void(),
            y_fp16.as_void(),
            n as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// Medusa — small-M ternary GEMM (M ∈ [1, 16]).
// -----------------------------------------------------------------------------

/// Medusa-plan shape bounds. Native kernel silently no-ops out of these.
///
/// Public so the router-side scaffolding (and its tests) can use the same
/// constants without re-hardcoding them and drifting from the kernel.
pub const MEDUSA_GEMM_MAX_M: i32 = 16;
/// N must be a multiple of the kernel's BLOCK_N tile.
pub const MEDUSA_GEMM_BLOCK_N: i32 = 64;
/// K must be a multiple of the kernel's BLOCK_K tile.
pub const MEDUSA_GEMM_BLOCK_K: i32 = 64;

/// Scaffold wrapper around the `onebit_ternary_gemm_smallm_launch` kernel.
///
/// This is **boundary-validation scaffolding**, not the production dispatch
/// path. Medusa speculative decoding needs a small-M ternary GEMM to verify
/// tree-expanded candidate tokens in one backbone pass (see
/// `docs/wiki/Medusa-Integration-Plan.md`). The live decode loop will call
/// the kernel through `DevicePtr<T>` / `DeviceMutPtr<T>` buffers owned by
/// the router's scratch, same as every other GEMV in this crate — this
/// host-slice wrapper exists so the router scaffold + tests can exercise
/// the arg-validation boundary without standing up a HipBackend.
///
/// Arguments:
///   * `codes` — packed ternary weights, treated as `u32 [K/16, N]` bytes.
///     We accept `&[i32]` so the caller can source them from a type-erased
///     loader without a byte-reinterpret cast on every call site; the
///     memory representation of `i32` and `u32` are identical on every
///     supported platform.
///   * `act` — activations, treated as `u16 [M, K]` on the host. Accepted
///     as `&[u16]` for symmetry with the fp16/bf16 output buffer; the
///     scaffold treats the byte length as authoritative, not the element
///     interpretation.
///   * `out` — bf16 `[M, N]` output buffer, as `&mut [u16]`. Same
///     reasoning as `act`.
///   * `m`, `n`, `k` — GEMM dimensions, with the architect-pinned shape
///     contract: `m ∈ [1, 16]`, `n % 64 == 0`, `k % 64 == 0`.
///   * `stream` — optional HIP stream. `None` = default (null) stream.
///
/// Behaviour:
///   * Shape validation runs before anything else; invalid shapes return
///     `Err(RcppError::InvalidArg)` without touching the FFI boundary.
///     This is the property the unit tests cover on CI hosts without ROCm.
///   * On a real ROCm build (feature `link-rocm`), the wrapper currently
///     returns `Err(RcppError::Unsupported)` — the host-slice path is
///     *scaffolding*. The retrained-weights follow-up pass will replace
///     this body with a device-pointer form that takes pre-allocated
///     `DeviceBuffer<T>` arguments (matching every other GEMV wrapper in
///     this file). Keeping the symbol here now locks the call site in so
///     `1bit-router::medusa` can be written against it without a future
///     churn.
///
/// The underlying FFI symbol is always defined — either as the real kernel
/// launcher (`link-rocm` on) or as a no-op stub (`link-rocm` off). The
/// `Unsupported` return on real builds is a deliberate scaffolding choice,
/// not a linkage gap.
pub fn ternary_gemm_smallm(
    codes: &[i32],
    act: &[u16],
    out: &mut [u16],
    m: i32,
    n: i32,
    k: i32,
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    // Shape contract — architect-pinned; `ternary_gemm_smallm.hip` silently
    // no-ops if these are violated, so we validate in Rust and surface a
    // structured error instead.
    if m < 1 || m > MEDUSA_GEMM_MAX_M {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: M must be in [1, 16]",
        ));
    }
    if n <= 0 || n % MEDUSA_GEMM_BLOCK_N != 0 {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: N must be a positive multiple of 64",
        ));
    }
    if k <= 0 || k % MEDUSA_GEMM_BLOCK_K != 0 {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: K must be a positive multiple of 64",
        ));
    }

    // Length cross-check against the declared shape. `codes` is treated as
    // u32 [K/16, N] — one u32 per 16 codes per output column. `act` is
    // M × K int8 elements re-sliced as u16 for a 2:1 element ratio. `out`
    // is M × N bf16 elements (= u16).
    let expected_codes_words = ((k as usize) / 16) * (n as usize);
    if codes.len() != expected_codes_words {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: codes.len() must equal K/16 * N",
        ));
    }
    let expected_act_u16 = ((m as usize) * (k as usize)).div_ceil(2);
    if act.len() != expected_act_u16 {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: act.len() must equal ceil(M*K / 2) u16 (i.e. M*K int8 bytes)",
        ));
    }
    let expected_out = (m as usize) * (n as usize);
    if out.len() != expected_out {
        return Err(RcppError::Precondition(
            "ternary_gemm_smallm: out.len() must equal M * N (bf16 elements)",
        ));
    }

    let _stream = stream.unwrap_or(HipStream::DEFAULT);

    // Scaffold path: shape is valid, but the host-slice body isn't wired
    // through H2D copies yet. The retrained-weights follow-up pass lands
    // the DeviceBuffer-form dispatch and removes this early return.
    //
    // Touch the FFI symbol so link errors surface here (rather than lazily
    // inside `1bit-router::medusa`) on real builds. We pass every argument
    // as a null / zero so no kernel launch fires — the launcher's own
    // bounds check (`M <= 0 || M > 16 || N % 64 != 0 || K % 64 != 0`)
    // filters it into a no-op.
    #[cfg(feature = "link-rocm")]
    {
        // SAFETY: all pointers are null; the native launcher early-outs on
        // the M/N/K = 0 guard before touching any of them. This is only a
        // symbol-resolution touch to force a link error if the kernel is
        // missing from librocm_cpp.so, not a real dispatch.
        unsafe {
            ffi::onebit_ternary_gemm_smallm_launch(
                core::ptr::null(),
                0.0,
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null_mut(),
                0,
                0,
                0,
                core::ptr::null_mut(),
            );
        }
    }

    Err(RcppError::Unsupported)
}

// -----------------------------------------------------------------------------
// Sherry 1.25-bit ternary GEMV — clean-room fp16-in/fp16-out path.
// -----------------------------------------------------------------------------

/// Sherry group size along K: every 4 consecutive weights → one 5-bit group
/// (2 bits `zero_pos` + 3 bits `signs`). Must match the C side contract in
/// `rocm-cpp/include/rocm_cpp/sherry.h`.
pub const SHERRY_GROUP_SIZE: usize = 4;

/// Sherry bits per group — 5 (2 `zero_pos` + 3 `signs`).
pub const SHERRY_BITS_PER_GROUP: usize = 5;

/// Byte-alignment constraint on `K_in` (= 32). A multiple of 32 weights
/// encodes to `32 * 5 / 8 = 20` whole bytes; short K aligns to bit boundaries
/// within a partial byte and the launcher rejects it.
pub const SHERRY_K_ALIGNMENT: usize = 32;

/// Packed byte count for `count` ternary weights.
///
/// The byte total is `count * 5 / 32`; `count` must be a multiple of 32 so
/// this is an exact division.
#[inline]
pub const fn sherry_packed_bytes(count: usize) -> usize {
    count * SHERRY_BITS_PER_GROUP / (8 * SHERRY_GROUP_SIZE)
}

/// Host-side Sherry 1.25-bit packer.
///
/// Input is a contiguous `i8` array of ternary values in `{-1, 0, +1}`, laid
/// out as consecutive groups of 4. For each group, exactly one lane is zero
/// (Sherry's structured-sparsity contract); the other three carry ±1 signs.
/// The packer emits 5 bits per group: `[zero_pos : 2][signs : 3]`, LSB-first
/// across output bytes.
///
/// Preconditions:
/// - `ternary.len() % 4 == 0` (group size)
/// - `out.len() == ternary.len() * 5 / 32` (byte-aligned output)
///
/// The native packer is pure host C code, thread-safe, and never touches the
/// GPU. Behaviour on a group with 0 or ≥2 zeros is deterministic but produces
/// a WRONG encoding — it's a caller bug (the requantizer's zero-choice
/// heuristic is responsible for enforcing the one-zero-per-group invariant).
///
/// Returns [`RcppError::Unsupported`] when the `link-rocm` feature is off
/// (symbol is a no-op stub in that build mode).
pub fn sherry_pack(ternary: &[i8], out: &mut [u8]) -> Result<(), RcppError> {
    if ternary.len() % SHERRY_GROUP_SIZE != 0 {
        return Err(RcppError::Precondition(
            "sherry_pack: ternary.len() must be a multiple of 4",
        ));
    }
    let expected = sherry_packed_bytes(ternary.len());
    if out.len() != expected {
        return Err(RcppError::Precondition(
            "sherry_pack: out.len() must equal ternary.len() * 5 / 32",
        ));
    }

    // No `link-rocm` → we don't have the native packer on this build.
    // Return `Unsupported` so callers that accidentally wire the packer
    // into a CI host observe a structured error rather than silent
    // no-op output.
    #[cfg(not(feature = "link-rocm"))]
    {
        return Err(RcppError::Unsupported);
    }

    // SAFETY: slices are borrowed disjointly (`&` vs `&mut`), lengths are
    // validated above to match what the C packer reads (`count` i8 bytes)
    // and writes (`count * 5 / 32` u8 bytes). The native packer does no
    // HIP calls — pure CPU.
    #[cfg(feature = "link-rocm")]
    unsafe {
        ffi::rcpp_sherry_pack(ternary.as_ptr(), out.as_mut_ptr(), ternary.len() as c_int);
    }
    #[cfg_attr(not(feature = "link-rocm"), allow(unreachable_code))]
    Ok(())
}

/// Sherry 1.25-bit ternary GEMV with FP16 activations and FP16 output.
///
/// Computes `out_fp16[n] = sum_{k=0..K_in} sign(W[n,k]) * act_fp16[k]` where
/// `sign(W[n,k]) ∈ {-1, 0, +1}` comes from the 1.25-bit packed weight
/// buffer (`N_out * K_in * 5 / 32` bytes). No per-row scale is baked in —
/// caller multiplies externally if the model needs one.
///
/// Preconditions:
/// - `k_in % 32 == 0` (byte-aligned row stride; launcher rejects otherwise)
/// - `packed.0` points to at least `n_out * k_in * 5 / 32` device bytes
/// - `act.0` points to at least `k_in` fp16 elements on device
/// - `out.0` points to at least `n_out` fp16 writable elements on device
///
/// The launcher returns `void`; kernel-launch failures surface at the next
/// synchronization boundary (via `hipGetLastError`). Shape violations
/// pre-validate in Rust and return [`RcppError::Precondition`] without
/// dispatching.
///
/// Returns [`RcppError::Unsupported`] when `link-rocm` is off.
pub fn sherry_gemv_fp16(
    packed: DevicePtr<u8>,
    act: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    n_out: i32,
    k_in: i32,
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    if n_out <= 0 {
        return Err(RcppError::Precondition(
            "sherry_gemv_fp16: n_out must be positive",
        ));
    }
    if k_in <= 0 || (k_in as usize) % SHERRY_K_ALIGNMENT != 0 {
        return Err(RcppError::Precondition(
            "sherry_gemv_fp16: k_in must be a positive multiple of 32",
        ));
    }

    let stream = stream.unwrap_or(HipStream::DEFAULT);

    #[cfg(not(feature = "link-rocm"))]
    {
        let _ = (packed, act, out, stream);
        return Err(RcppError::Unsupported);
    }

    // SAFETY: device-pointer contract upheld by caller (see `DevicePtr`
    // docs); shape divisibility pre-checked above; the launcher only
    // reads `packed`/`act` and writes `out`.
    #[cfg(feature = "link-rocm")]
    unsafe {
        ffi::sherry_ternary_gemv_launch(
            packed.0,
            act.0,
            out.0,
            n_out as c_int,
            k_in as c_int,
            stream.as_raw(),
        );
    }
    #[cfg_attr(not(feature = "link-rocm"), allow(unreachable_code))]
    Ok(())
}

/// Scalar-reference variant of [`sherry_gemv_fp16`] — one HIP thread per
/// output row, bit-identical fp32 accumulation order as the fast kernel.
/// Exists for differential testing; **do not benchmark**.
///
/// Same preconditions and error handling as [`sherry_gemv_fp16`].
pub fn sherry_gemv_fp16_scalar_ref(
    packed: DevicePtr<u8>,
    act: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    n_out: i32,
    k_in: i32,
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    if n_out <= 0 {
        return Err(RcppError::Precondition(
            "sherry_gemv_fp16_scalar_ref: n_out must be positive",
        ));
    }
    if k_in <= 0 || (k_in as usize) % SHERRY_K_ALIGNMENT != 0 {
        return Err(RcppError::Precondition(
            "sherry_gemv_fp16_scalar_ref: k_in must be a positive multiple of 32",
        ));
    }

    let stream = stream.unwrap_or(HipStream::DEFAULT);

    #[cfg(not(feature = "link-rocm"))]
    {
        let _ = (packed, act, out, stream);
        return Err(RcppError::Unsupported);
    }

    // SAFETY: same as `sherry_gemv_fp16`.
    #[cfg(feature = "link-rocm")]
    unsafe {
        ffi::sherry_ternary_gemv_scalar_ref_launch(
            packed.0,
            act.0,
            out.0,
            n_out as c_int,
            k_in as c_int,
            stream.as_raw(),
        );
    }
    #[cfg_attr(not(feature = "link-rocm"), allow(unreachable_code))]
    Ok(())
}

// -----------------------------------------------------------------------------
// Bonsai Q1_0_g128 / TQ2_0_g128 — fp16-in / fp16-out GEMV
//
// HIP kernels live in `rocm-cpp/src/bonsai_{q1,tq2}_gemv.hip`. Both
// consume the AoS on-disk layout directly (no host-side SoA repack) and
// embed the per-block FP16 scale inline per oxibonsai's format. Status
// 2026-04-21: kernels correctness-gated by
// `rocm-cpp/tests/test_bonsai_gemv.cpp` (differential against scalar
// reference, 50 seeds × {2048, 6912}² × both formats, ≤ 3 fp16 ULP).
// -----------------------------------------------------------------------------

/// Fixed group size shared by both Bonsai formats (oxibonsai
/// `QK1_0_G128` / `QK_TQ2_0_G128`). `K_in` must be divisible by this.
pub const BONSAI_GROUP_SIZE: usize = 128;

/// Bytes per Q1_0_g128 block (2 bytes FP16 scale + 16 bytes sign bits).
pub const BONSAI_Q1_BLOCK_BYTES: usize = 18;

/// Bytes per TQ2_0_g128 block (32 bytes 2-bit codes + 2 bytes FP16 scale).
pub const BONSAI_TQ2_BLOCK_BYTES: usize = 34;

/// Packed byte count for `count` Bonsai Q1_0_g128 weights.
/// `count` must be a multiple of [`BONSAI_GROUP_SIZE`].
#[inline]
pub const fn bonsai_q1_packed_bytes(count: usize) -> usize {
    (count / BONSAI_GROUP_SIZE) * BONSAI_Q1_BLOCK_BYTES
}

/// Packed byte count for `count` Bonsai TQ2_0_g128 weights.
/// `count` must be a multiple of [`BONSAI_GROUP_SIZE`].
#[inline]
pub const fn bonsai_tq2_packed_bytes(count: usize) -> usize {
    (count / BONSAI_GROUP_SIZE) * BONSAI_TQ2_BLOCK_BYTES
}

/// Bonsai Q1_0_g128 1-bit GEMV with FP16 activations and FP16 output.
///
/// Computes `out_fp16[n] = sum_{k=0..K_in} w[n,k] * act_fp16[k]` where
/// `w[n,k] = sign_bit ? +d : -d` comes from the 18-byte-per-block packed
/// weight buffer (`N_out * K_in / 128 * 18` bytes). Per-block FP16 scale
/// `d` is embedded in each block (no external scales tensor).
///
/// Preconditions:
/// - `k_in > 0` and `k_in % 128 == 0`
/// - `n_out > 0`
/// - `packed.0` points to at least `n_out * k_in / 128 * 18` device bytes
/// - `act.0` points to at least `k_in` fp16 elements on device
/// - `out.0` points to at least `n_out` fp16 writable elements on device
///
/// Returns [`RcppError::Unsupported`] when `link-rocm` is off.
pub fn bonsai_q1_gemv_fp16(
    packed: DevicePtr<u8>,
    act: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    n_out: i32,
    k_in: i32,
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    if n_out <= 0 {
        return Err(RcppError::Precondition(
            "bonsai_q1_gemv_fp16: n_out must be positive",
        ));
    }
    if k_in <= 0 || (k_in as usize) % BONSAI_GROUP_SIZE != 0 {
        return Err(RcppError::Precondition(
            "bonsai_q1_gemv_fp16: k_in must be a positive multiple of 128",
        ));
    }

    let stream = stream.unwrap_or(HipStream::DEFAULT);

    #[cfg(not(feature = "link-rocm"))]
    {
        let _ = (packed, act, out, stream);
        return Err(RcppError::Unsupported);
    }

    // SAFETY: device-pointer contract upheld by caller (see `DevicePtr`
    // docs); shape divisibility pre-checked above; the launcher only
    // reads `packed`/`act` and writes `out`.
    #[cfg(feature = "link-rocm")]
    unsafe {
        ffi::bonsai_q1_gemv_launch(
            packed.0,
            act.0,
            out.0,
            n_out as c_int,
            k_in as c_int,
            stream.as_raw(),
        );
    }
    #[cfg_attr(not(feature = "link-rocm"), allow(unreachable_code))]
    Ok(())
}

/// Bonsai TQ2_0_g128 ternary GEMV with FP16 activations and FP16 output.
///
/// Computes `out_fp16[n] = sum_{k=0..K_in} w[n,k] * act_fp16[k]` where
/// `w[n,k] = code ∈ {-1, 0, +1}` times the block's FP16 scale `d`, sourced
/// from the 34-byte-per-block packed weight buffer
/// (`N_out * K_in / 128 * 34` bytes). The 2-bit codes map
/// `00→-1, 01→0, 10→+1, 11→0(reserved)`, 4 per byte LSB-first, matching
/// oxibonsai's `BlockTQ2_0_g128` and the Metal reference kernel.
///
/// Preconditions:
/// - `k_in > 0` and `k_in % 128 == 0`
/// - `n_out > 0`
/// - `packed.0` points to at least `n_out * k_in / 128 * 34` device bytes
/// - `act.0` points to at least `k_in` fp16 elements on device
/// - `out.0` points to at least `n_out` fp16 writable elements on device
///
/// Returns [`RcppError::Unsupported`] when `link-rocm` is off.
pub fn bonsai_tq2_gemv_fp16(
    packed: DevicePtr<u8>,
    act: DevicePtr<u16>,
    out: DeviceMutPtr<u16>,
    n_out: i32,
    k_in: i32,
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    if n_out <= 0 {
        return Err(RcppError::Precondition(
            "bonsai_tq2_gemv_fp16: n_out must be positive",
        ));
    }
    if k_in <= 0 || (k_in as usize) % BONSAI_GROUP_SIZE != 0 {
        return Err(RcppError::Precondition(
            "bonsai_tq2_gemv_fp16: k_in must be a positive multiple of 128",
        ));
    }

    let stream = stream.unwrap_or(HipStream::DEFAULT);

    #[cfg(not(feature = "link-rocm"))]
    {
        let _ = (packed, act, out, stream);
        return Err(RcppError::Unsupported);
    }

    // SAFETY: same as `bonsai_q1_gemv_fp16`.
    #[cfg(feature = "link-rocm")]
    unsafe {
        ffi::bonsai_tq2_gemv_launch(
            packed.0,
            act.0,
            out.0,
            n_out as c_int,
            k_in as c_int,
            stream.as_raw(),
        );
    }
    #[cfg_attr(not(feature = "link-rocm"), allow(unreachable_code))]
    Ok(())
}

// -----------------------------------------------------------------------------
// BitNet v2 — Hadamard rotation (H-BitLinear)
// -----------------------------------------------------------------------------

/// Block size used by the device-side Walsh-Hadamard butterfly kernel.
/// Must match `HADAMARD_BLOCK` in `rocm-cpp/kernels/hadamard_rotate_butterfly.hip`.
/// Any change on the device side must be reflected here (and vice versa) or
/// the pre-dispatch divisibility check in [`hadamard_rotate_fp16_device`]
/// will silently accept a bad shape.
pub const HADAMARD_BLOCK: usize = 128;

/// Apply a block-diagonal Walsh-Hadamard rotation to FP16 activations
/// **in device memory**. Writes `y = H_128 x / sqrt(128)` block-by-block;
/// `H_128` is orthogonal so the L2 norm is preserved.
///
/// This is the low-level, device-pointer entry point matching every other
/// kernel wrapper in this crate. The higher-level host-side helper
/// [`hadamard_rotate_fp16`] allocates scratch + copies for test harnesses.
///
/// `k` is the total element count; the kernel hardcodes `block_size = 128`,
/// so `k` must be divisible by 128. Aliasing `x == y` (in-place) is
/// permitted — the device kernel finishes reading its source element before
/// issuing the store.
///
/// Returns [`RcppStatus::InvalidArg`] without dispatching if `k` is not a
/// multiple of [`HADAMARD_BLOCK`]. The native launcher has no return code
/// (see `ck_gemm.h`), so kernel-launch failures surface at the next
/// HIP synchronization point rather than here.
pub fn hadamard_rotate_fp16_device(
    x: DevicePtr<u16>,
    y: DeviceMutPtr<u16>,
    k: i32,
    stream: HipStream,
) -> RcppStatus {
    if k <= 0 || (k as usize) % HADAMARD_BLOCK != 0 {
        return RcppStatus::InvalidArg;
    }
    // SAFETY: device-pointer contract upheld by caller; the native launcher
    // does no host-side deref of the buffers. Shape divisibility is
    // pre-checked above, so the kernel's `K / 128` division is exact.
    unsafe {
        ffi::rcpp_hadamard_rotate_fp16_butterfly_launch(
            x.as_void(),
            y.as_void(),
            k as c_int,
            stream.as_raw(),
        );
    }
    RcppStatus::Ok
}

/// Host-side convenience wrapper around [`hadamard_rotate_fp16_device`]:
/// takes two host FP16 slices (as the raw `u16` IEEE-754 half bit pattern),
/// allocates a pair of matching `DeviceBuffer<u16>` scratch tensors, H2D's
/// the input, runs the kernel on the supplied stream (default = null
/// stream), and D2H's the output.
///
/// Not suitable for the decode hot path — every call performs two PCIe-
/// equivalent round-trips (we're on a unified-memory iGPU so "PCIe" = LPDDR5
/// bandwidth, but the copies still serialize). Used by
/// * the test suite, for bit-exact round-trip checks against the scalar
///   oracle without writing a standalone HIP harness in Rust;
/// * any future tooling that wants to verify a model's Hadamard rotation
///   pipeline end-to-end without standing up a full HipBackend.
///
/// Length must be a positive multiple of [`HADAMARD_BLOCK`] (= 128); the
/// device kernel hardcodes block_size=128 as a constexpr, so anything else
/// is rejected up front with [`RcppError::InvalidArg`]. Input and output
/// slices must have equal length.
///
/// `stream = None` dispatches on the default (null) HIP stream.
///
/// Returns [`RcppError::InvalidArg`] on bad length / zero-length / length
/// mismatch; [`RcppError::HipError`] on any HIP runtime failure.
pub fn hadamard_rotate_fp16(
    x: &[u16],
    y: &mut [u16],
    stream: Option<HipStream>,
) -> Result<(), RcppError> {
    if x.len() != y.len() {
        return Err(RcppError::InvalidArg);
    }
    let n = x.len();
    if n == 0 || n % HADAMARD_BLOCK != 0 {
        return Err(RcppError::InvalidArg);
    }

    let stream = stream.unwrap_or(HipStream::DEFAULT);

    let mut d_in: DeviceBuffer<u16> = DeviceBuffer::alloc(n)?;
    let mut d_out: DeviceBuffer<u16> = DeviceBuffer::alloc(n)?;
    d_in.copy_from_slice(x)?;

    let status = hadamard_rotate_fp16_device(
        d_in.as_device_ptr(),
        d_out.as_device_mut_ptr(),
        n as i32,
        stream,
    );
    status.into_result()?;
    device_synchronize()?;
    d_out.copy_to_slice(y)?;
    Ok(())
}

/// Per-row FP16 → INT8 quantizer with per-row FP16 scale. Used to write
/// INT8 KV cache entries with their scale in one launch.
pub fn quantize_fp16_to_i8_rowscale(
    x_fp16: DevicePtr<u16>,
    out_i8: DeviceMutPtr<i8>,
    scale_fp16_out: DeviceMutPtr<u16>,
    num_rows: i32,
    row_len: i32,
    stream: HipStream,
) -> RcppStatus {
    // SAFETY: pointer forwarding only.
    let raw = unsafe {
        ffi::rcpp_quantize_fp16_to_i8_rowscale(
            x_fp16.as_void(),
            out_i8.as_void(),
            scale_fp16_out.as_void(),
            num_rows as c_int,
            row_len as c_int,
            stream.as_raw(),
        )
    };
    RcppStatus::from_raw(raw)
}

// -----------------------------------------------------------------------------
// HIP device memory — safe wrappers over libamdhip64's raw C entry points.
//
// Every allocation is returned as a `DeviceBuffer<T>` smart pointer that
// owns the device allocation and frees it on drop. Copy / synchronize / set
// operations are plain functions that take already-owned device pointers.
//
// The HIP runtime is thread-safe so these wrappers do NOT require external
// synchronization; concurrent `alloc`s are fine. `Send` + `Sync` bounds on
// `DeviceBuffer<T>` make it legal to move a buffer between threads or share
// it behind an `Arc` as long as the holder respects the kernel's own
// read/write semantics.
// -----------------------------------------------------------------------------

/// Owned device allocation. Drops via `hipFree`.
///
/// The element type `T` is informational — HIP allocations are untyped at
/// the API level, but we carry `T` through the `DevicePtr<T>` /
/// `DeviceMutPtr<T>` accessors so the borrow checker can help downstream
/// crates avoid mismatched-element bugs.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    // Element count, NOT byte count. `byte_len()` folds sizeof in.
    len: usize,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: `DeviceBuffer<T>` owns a device allocation whose backing store
// lives on the GPU. Transferring ownership between threads or sharing
// behind a `&` (Sync) is legal — HIP's runtime is thread-safe for
// `hipMalloc` / `hipFree` / `hipMemcpy`, and we never dereference the
// pointer from Rust (only pass it through FFI to kernels).
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    /// Allocate `count` elements of `T` on the current HIP device.
    pub fn alloc(count: usize) -> Result<Self, RcppError> {
        if count == 0 {
            // HIP accepts zero-byte allocations but they're a footgun
            // (null pointer vs valid-but-empty); just return a well-formed
            // empty buffer.
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                len: 0,
                _marker: core::marker::PhantomData,
            });
        }
        let bytes = count
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(RcppError::Precondition("alloc size overflow"))?;
        let mut raw: *mut c_void = core::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; `bytes` is positive.
        let status = unsafe { ffi::hipMalloc(&mut raw as *mut *mut c_void, bytes) };
        if status != ffi::HIP_SUCCESS || raw.is_null() {
            return Err(RcppError::HipError);
        }
        Ok(Self {
            ptr: raw as *mut T,
            len: count,
            _marker: core::marker::PhantomData,
        })
    }

    /// Allocate and zero-fill.
    pub fn alloc_zeroed(count: usize) -> Result<Self, RcppError> {
        let buf = Self::alloc(count)?;
        if count > 0 {
            // SAFETY: buf.ptr is a valid device allocation of `byte_len`.
            let status = unsafe { ffi::hipMemset(buf.ptr as *mut c_void, 0, buf.byte_len()) };
            if status != ffi::HIP_SUCCESS {
                return Err(RcppError::HipError);
            }
        }
        Ok(buf)
    }

    /// Element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if this buffer owns no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocation size in bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }

    /// Read-only device pointer. The invariant is upheld by construction:
    /// we got `ptr` back from `hipMalloc`, so it refers to a valid device
    /// allocation that stays alive until `self` drops.
    #[inline]
    pub fn as_device_ptr(&self) -> DevicePtr<T> {
        // SAFETY: self.ptr came from hipMalloc and is valid for self.len
        // elements for the lifetime of `self`.
        unsafe { DevicePtr::new(self.ptr as *const T) }
    }

    /// Writable device pointer — same invariants as `as_device_ptr`.
    #[inline]
    pub fn as_device_mut_ptr(&mut self) -> DeviceMutPtr<T> {
        // SAFETY: self.ptr came from hipMalloc; `&mut self` proves exclusive access.
        unsafe { DeviceMutPtr::new(self.ptr) }
    }

    /// Copy `src` (host) into this buffer. Lengths must match.
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<(), RcppError>
    where
        T: Copy,
    {
        if src.len() != self.len {
            return Err(RcppError::Precondition("copy_from_slice length mismatch"));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: src has `self.len` elements; self.ptr owns self.len slots.
        let status = unsafe {
            ffi::hipMemcpy(
                self.ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                self.byte_len(),
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if status != ffi::HIP_SUCCESS {
            return Err(RcppError::HipError);
        }
        Ok(())
    }

    /// Copy this buffer's contents into `dst` (host). Lengths must match.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<(), RcppError>
    where
        T: Copy,
    {
        if dst.len() != self.len {
            return Err(RcppError::Precondition("copy_to_slice length mismatch"));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: dst has `self.len` slots; self.ptr owns self.len elements.
        let status = unsafe {
            ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.byte_len(),
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if status != ffi::HIP_SUCCESS {
            return Err(RcppError::HipError);
        }
        Ok(())
    }

    /// Copy `src` (device) into this buffer. Lengths must match.
    pub fn copy_from_device(&mut self, src: &Self) -> Result<(), RcppError> {
        if src.len != self.len {
            return Err(RcppError::Precondition("copy_from_device length mismatch"));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: both pointers are valid device allocations of matching length.
        let status = unsafe {
            ffi::hipMemcpy(
                self.ptr as *mut c_void,
                src.ptr as *const c_void,
                self.byte_len(),
                ffi::HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        if status != ffi::HIP_SUCCESS {
            return Err(RcppError::HipError);
        }
        Ok(())
    }

    /// Read a single element back to the host (small-value fast path —
    /// avoids a full slice allocation for scalars like `x_scale_dev`).
    pub fn copy_to_host_scalar(&self) -> Result<T, RcppError>
    where
        T: Copy + Default,
    {
        if self.len == 0 {
            return Err(RcppError::Precondition(
                "copy_to_host_scalar on empty buffer",
            ));
        }
        let mut out: T = T::default();
        // SAFETY: self.ptr owns at least one element of type T.
        let status = unsafe {
            ffi::hipMemcpy(
                &mut out as *mut T as *mut c_void,
                self.ptr as *const c_void,
                core::mem::size_of::<T>(),
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if status != ffi::HIP_SUCCESS {
            return Err(RcppError::HipError);
        }
        Ok(out)
    }

    /// Pointer arithmetic helper — produces a `DeviceMutPtr<T>` offset
    /// by `elem_offset` elements from the base. No bounds check (kernel
    /// is assumed to respect caller-supplied bounds).
    ///
    /// # Safety
    /// Caller must ensure `elem_offset <= self.len`.
    #[inline]
    pub unsafe fn offset_mut(&mut self, elem_offset: usize) -> DeviceMutPtr<T> {
        // SAFETY: forwarded to the caller via `unsafe fn` contract.
        unsafe { DeviceMutPtr::new(self.ptr.add(elem_offset)) }
    }

    /// Read-only variant of [`Self::offset_mut`].
    ///
    /// # Safety
    /// Caller must ensure `elem_offset <= self.len`.
    #[inline]
    pub unsafe fn offset(&self, elem_offset: usize) -> DevicePtr<T> {
        // SAFETY: forwarded to the caller via `unsafe fn` contract.
        unsafe { DevicePtr::new(self.ptr.add(elem_offset) as *const T) }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: self.ptr came from hipMalloc and is freed exactly once
            // here. After this, self is consumed.
            let _ = unsafe { ffi::hipFree(self.ptr as *mut c_void) };
            self.ptr = core::ptr::null_mut();
        }
    }
}

impl<T> core::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.len)
            .field("bytes", &self.byte_len())
            .field("elem_size", &core::mem::size_of::<T>())
            .finish()
    }
}

/// Block the calling thread until every submitted HIP task on the current
/// device has completed. Equivalent to `hipDeviceSynchronize`.
pub fn device_synchronize() -> Result<(), RcppError> {
    // SAFETY: no arguments; HIP runtime is thread-safe.
    let status = unsafe { ffi::hipDeviceSynchronize() };
    if status != ffi::HIP_SUCCESS {
        return Err(RcppError::HipError);
    }
    Ok(())
}

/// Number of HIP-visible devices on this host. Returns `0` (no error) when
/// the runtime reports no devices — callers use this to distinguish
/// "no GPU" from "HIP runtime broken".
pub fn device_count() -> Result<i32, RcppError> {
    let mut count: c_int = 0;
    // SAFETY: `count` is a valid out-pointer.
    let status = unsafe { ffi::hipGetDeviceCount(&mut count as *mut c_int) };
    if status != ffi::HIP_SUCCESS {
        return Err(RcppError::HipError);
    }
    Ok(count as i32)
}

/// Make `device_id` the current device for subsequent allocs / kernels.
pub fn set_device(device_id: i32) -> Result<(), RcppError> {
    // SAFETY: no pointer arguments.
    let status = unsafe { ffi::hipSetDevice(device_id as c_int) };
    if status != ffi::HIP_SUCCESS {
        return Err(RcppError::HipError);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Host-side weight packer (no device touch)
// -----------------------------------------------------------------------------

/// Offline pack of ternary `{-1, 0, +1}` weights to the pk_i4 WMMA-
/// permuted layout consumed by [`DevicePtr`]-form weights.
///
/// `ternary_host` is **col-major `[K, N]`** as the C header specifies.
/// `packed_host_out` must be exactly `K * N / 2` bytes. `K` must be a
/// multiple of 32 (the header says "K % 32 == 0 and K % 8 == 0").
pub fn ternary_pack_pk_i4(
    ternary_host: &[i8],
    packed_host_out: &mut [i8],
    k: i32,
    n: i32,
) -> Result<(), RcppError> {
    if k <= 0 || n <= 0 {
        return Err(RcppError::Precondition("K and N must be positive"));
    }
    if k % 32 != 0 {
        return Err(RcppError::Precondition("K must be a multiple of 32"));
    }
    let expected_in = (k as usize)
        .checked_mul(n as usize)
        .ok_or(RcppError::Precondition("K*N overflow"))?;
    let expected_out = expected_in / 2;
    if ternary_host.len() != expected_in {
        return Err(RcppError::Precondition(
            "ternary_host.len() must equal K * N",
        ));
    }
    if packed_host_out.len() != expected_out {
        return Err(RcppError::Precondition(
            "packed_host_out.len() must equal K * N / 2",
        ));
    }

    // SAFETY: lengths are validated above to exactly match what the C
    // kernel reads (K*N i8 bytes) and writes (K*N/2 i8 bytes). The two
    // slices are borrowed disjointly (one `&`, one `&mut`), so Rust's
    // aliasing model already guarantees non-overlap.
    let raw = unsafe {
        ffi::rcpp_ternary_pack_pk_i4(
            ternary_host.as_ptr(),
            packed_host_out.as_mut_ptr(),
            k as c_int,
            n as c_int,
        )
    };
    RcppStatus::from_raw(raw).into_result()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trip() {
        assert_eq!(RcppStatus::from_raw(0), RcppStatus::Ok);
        assert_eq!(RcppStatus::from_raw(1), RcppStatus::InvalidArg);
        assert_eq!(RcppStatus::from_raw(2), RcppStatus::Unsupported);
        assert_eq!(RcppStatus::from_raw(3), RcppStatus::HipError);
        assert_eq!(RcppStatus::from_raw(4), RcppStatus::Internal);
        // Unknown code folds into Internal.
        assert_eq!(RcppStatus::from_raw(42), RcppStatus::Internal);
        assert!(RcppStatus::Ok.is_ok());
        assert!(!RcppStatus::HipError.is_ok());
    }

    #[test]
    fn precondition_checks() {
        let mut out = [0i8; 16];
        let w = [0i8; 16];
        // K not a multiple of 32.
        let err = ternary_pack_pk_i4(&w, &mut out, 16, 1).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
    }

    /// Hadamard wrapper rejects bad lengths without touching the GPU.
    /// Executes on every CI host regardless of ROCm presence.
    #[test]
    fn hadamard_rotate_fp16_preconditions() {
        let inp = vec![0u16; 128];

        // Length mismatch between input and output.
        let mut short = vec![0u16; 64];
        let err = hadamard_rotate_fp16(&inp, &mut short, None).unwrap_err();
        assert!(matches!(err, RcppError::InvalidArg));

        // Length not a multiple of HADAMARD_BLOCK (=128).
        let inp_bad = vec![0u16; 100];
        let mut out_bad = vec![0u16; 100];
        let err = hadamard_rotate_fp16(&inp_bad, &mut out_bad, None).unwrap_err();
        assert!(matches!(err, RcppError::InvalidArg));

        // Zero length is rejected (the kernel does nothing meaningful on 0).
        let err = hadamard_rotate_fp16(&[], &mut Vec::<u16>::new(), None).unwrap_err();
        assert!(matches!(err, RcppError::InvalidArg));

        // Explicit-stream form uses the same guards.
        let err =
            hadamard_rotate_fp16(&inp_bad, &mut out_bad, Some(HipStream::DEFAULT)).unwrap_err();
        assert!(matches!(err, RcppError::InvalidArg));
    }

    /// Round-trip smoke on the device-side Walsh-Hadamard butterfly at the
    /// two block-count grains the router actually uses: K=128 (one block,
    /// minimal path) and K=256 (two blocks, covers block-index dispatch).
    ///
    /// `H_128` is self-inverse up to a 1/sqrt(128) scale (orthogonal), so
    /// applying the rotation twice should return the original input within
    /// FP16 round-trip precision.
    ///
    /// Ignored by default — requires a GPU + `--features link-rocm`.
    #[test]
    #[ignore = "requires GPU + link-rocm feature; run with --ignored"]
    fn hadamard_rotate_roundtrip_k128_k256() {
        use half::f16;

        for n in [128usize, 256usize] {
            let mut inp_bits: Vec<u16> = Vec::with_capacity(n);
            for i in 0..n {
                let phase = (i as f32) * 0.173 + 0.25;
                let v = phase.cos() * 2.5;
                inp_bits.push(f16::from_f32(v).to_bits());
            }
            let original: Vec<f32> = inp_bits
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect();

            let mut once = vec![0u16; n];
            hadamard_rotate_fp16(&inp_bits, &mut once, None)
                .unwrap_or_else(|e| panic!("first rotation (n={n}): {e:?}"));

            let mut twice = vec![0u16; n];
            hadamard_rotate_fp16(&once, &mut twice, None)
                .unwrap_or_else(|e| panic!("second rotation (n={n}): {e:?}"));

            let mut max_abs = 0.0f32;
            for i in 0..n {
                let rec = f16::from_bits(twice[i]).to_f32();
                let delta = (rec - original[i]).abs();
                if delta > max_abs {
                    max_abs = delta;
                }
            }
            assert!(
                max_abs < 0.1,
                "n={n}: Hadamard round-trip exceeds fp16 tolerance: max_abs={max_abs}"
            );
        }
    }

    /// End-to-end identity: applying the rotation twice to the same vector
    /// returns the original within fp16 round-trip precision, because
    /// H·H = B·I so `(H · (H x / sqrt(B))) / sqrt(B) = x`.
    ///
    /// Ignored by default — requires a GPU + `--features link-rocm`. Run
    /// with `cargo test -p onebit-hip --features link-rocm --ignored
    /// hadamard_rotate_inverse_identity`.
    #[test]
    #[ignore = "requires GPU + link-rocm feature; run with --ignored"]
    fn hadamard_rotate_inverse_identity() {
        use half::f16;

        const N: usize = 128 * 8; // 8 blocks
        // Deterministic pseudo-random fp16 input in a moderate range.
        // Magnitude kept under ~4 so 7 butterfly adds stay far from fp16
        // overflow (max fp16 ≈ 65504; worst-case after 7 doublings ≈ 512).
        let mut inp_bits: Vec<u16> = Vec::with_capacity(N);
        for i in 0..N {
            let phase = (i as f32) * 0.137 + 0.5;
            let v = phase.sin() * 3.0;
            inp_bits.push(f16::from_f32(v).to_bits());
        }
        let original: Vec<f32> = inp_bits
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect();

        let mut once = vec![0u16; N];
        hadamard_rotate_fp16(&inp_bits, &mut once, None).expect("first rotation");

        let mut twice = vec![0u16; N];
        hadamard_rotate_fp16(&once, &mut twice, None).expect("second rotation (inverse)");

        // Expected error budget: 7 stages of ±1 adds in fp32 accumulator
        // then a single fp16 store per stage, repeated. Empirically ≤
        // ~1.5% relative error per element on this input scale.
        let mut max_abs = 0.0f32;
        for i in 0..N {
            let rec = f16::from_bits(twice[i]).to_f32();
            let delta = (rec - original[i]).abs();
            if delta > max_abs {
                max_abs = delta;
            }
        }
        assert!(
            max_abs < 0.1,
            "Hadamard round-trip exceeds fp16 tolerance: max_abs={max_abs}"
        );
    }

    /// Smoke test: dynamically open librocm_cpp.so and resolve a known
    /// host-only symbol. Verifies build-path + linkage without a device.
    ///
    /// Ignored by default — run with `cargo test --ignored` on a host
    /// that actually has ROCm + the rocm-cpp build output.
    #[test]
    #[ignore = "requires librocm_cpp.so on disk; run with --ignored"]
    fn smoke_dlopen_rocm_cpp() {
        use libloading::{Library, Symbol};
        use std::ffi::c_char;
        use std::path::PathBuf;

        let lib_dir = std::env::var("ROCM_CPP_LIB_DIR").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/repos/rocm-cpp/build"))
                .unwrap_or_else(|_| "rocm-cpp/build".into())
        });
        let so_path = PathBuf::from(lib_dir).join("librocm_cpp.so");
        if !so_path.exists() {
            eprintln!("skipping: {} not present", so_path.display());
            return;
        }

        // SAFETY: we load a library we trust (our own build output) and
        // only look up a symbol — we do NOT invoke it. No device
        // interaction, so this is safe even on a CPU-only host.
        unsafe {
            let lib = Library::new(&so_path).expect("failed to dlopen librocm_cpp.so");

            // rcpp_ternary_pack_pk_i4 is host-only (pure permutation).
            // Just resolving the symbol proves linkage.
            let _sym: Symbol<unsafe extern "C" fn(*const i8, *mut i8, c_int, c_int) -> c_int> = lib
                .get(b"rcpp_ternary_pack_pk_i4\0")
                .expect("symbol rcpp_ternary_pack_pk_i4 not exported");

            // Optional: also look up one of the device kernels to confirm
            // the full kernel surface is present, without calling it.
            let _sym2: Symbol<
                unsafe extern "C" fn(
                    *const core::ffi::c_void,
                    *const core::ffi::c_void,
                    f32,
                    *const core::ffi::c_void,
                    *mut core::ffi::c_void,
                    c_int,
                    c_int,
                    *mut core::ffi::c_void,
                ) -> c_int,
            > = lib
                .get(b"rcpp_ternary_gemv_halo_f16\0")
                .expect("symbol rcpp_ternary_gemv_halo_f16 not exported");

            // Silence unused warning on c_char.
            let _: *const c_char = core::ptr::null();
        }
    }

    /// Bonsai Q1_0_g128 GEMV wrapper rejects bad shapes BEFORE dispatching.
    /// Runs on every host (no GPU required) because the default build has
    /// `link-rocm` off: the valid-shape path short-circuits to `Unsupported`
    /// without dereferencing the (null) device pointers. On `link-rocm`
    /// builds the valid-shape path actually dispatches the HIP kernel, so
    /// this test is gated off for that feature combination.
    #[cfg(not(feature = "link-rocm"))]
    #[test]
    fn bonsai_q1_gemv_fp16_preconditions() {
        // Dummy pointers — never dereferenced because validation fires first
        // on bad shapes, and on the valid-shape case the `link-rocm`-off
        // branch returns Unsupported before any FFI call.
        let packed: DevicePtr<u8> = DevicePtr(core::ptr::null());
        let act: DevicePtr<u16> = DevicePtr(core::ptr::null());
        let out: DeviceMutPtr<u16> = DeviceMutPtr(core::ptr::null_mut());

        // K must be a positive multiple of 128.
        let err = bonsai_q1_gemv_fp16(packed, act, out, 16, 64, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        let err = bonsai_q1_gemv_fp16(packed, act, out, 16, 0, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        // N_out must be positive.
        let err = bonsai_q1_gemv_fp16(packed, act, out, 0, 128, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        // Valid shape — on the non-link-rocm build the wrapper returns
        // Unsupported after validation.
        let err = bonsai_q1_gemv_fp16(packed, act, out, 16, 128, None).unwrap_err();
        assert!(matches!(err, RcppError::Unsupported));

        // Packed-byte accounting math is exposed as a const fn.
        assert_eq!(bonsai_q1_packed_bytes(128), 18);
        assert_eq!(bonsai_q1_packed_bytes(2048), 16 * 18);
    }

    /// Shape-validation subset of [`bonsai_q1_gemv_fp16_preconditions`] that
    /// stays live under `link-rocm` (the valid-shape branch dispatches the
    /// kernel, so we don't cover it here).
    #[cfg(feature = "link-rocm")]
    #[test]
    fn bonsai_q1_gemv_fp16_precondition_errors() {
        let packed: DevicePtr<u8> = DevicePtr(core::ptr::null());
        let act: DevicePtr<u16> = DevicePtr(core::ptr::null());
        let out: DeviceMutPtr<u16> = DeviceMutPtr(core::ptr::null_mut());

        let err = bonsai_q1_gemv_fp16(packed, act, out, 16, 64, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        let err = bonsai_q1_gemv_fp16(packed, act, out, 0, 128, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));

        assert_eq!(bonsai_q1_packed_bytes(128), 18);
        assert_eq!(bonsai_q1_packed_bytes(2048), 16 * 18);
    }

    /// Bonsai TQ2_0_g128 GEMV wrapper rejects bad shapes BEFORE dispatching.
    /// Default-features (non `link-rocm`) path: the valid-shape branch
    /// returns Unsupported without calling FFI.
    #[cfg(not(feature = "link-rocm"))]
    #[test]
    fn bonsai_tq2_gemv_fp16_preconditions() {
        let packed: DevicePtr<u8> = DevicePtr(core::ptr::null());
        let act: DevicePtr<u16> = DevicePtr(core::ptr::null());
        let out: DeviceMutPtr<u16> = DeviceMutPtr(core::ptr::null_mut());

        // K must be a positive multiple of 128.
        let err = bonsai_tq2_gemv_fp16(packed, act, out, 16, 127, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        let err = bonsai_tq2_gemv_fp16(packed, act, out, 16, -1, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        // N_out must be positive.
        let err = bonsai_tq2_gemv_fp16(packed, act, out, -5, 128, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        // Valid shape on non-link-rocm: Unsupported after validation.
        let err = bonsai_tq2_gemv_fp16(packed, act, out, 32, 256, None).unwrap_err();
        assert!(matches!(err, RcppError::Unsupported));

        // Packed-byte accounting math is exposed as a const fn.
        assert_eq!(bonsai_tq2_packed_bytes(128), 34);
        assert_eq!(bonsai_tq2_packed_bytes(2048), 16 * 34);
    }

    /// `link-rocm` variant of the TQ2 precondition test — shape-validation
    /// only. The valid-shape branch dispatches the real kernel, which
    /// would dereference the null device pointers, so we skip it here.
    #[cfg(feature = "link-rocm")]
    #[test]
    fn bonsai_tq2_gemv_fp16_precondition_errors() {
        let packed: DevicePtr<u8> = DevicePtr(core::ptr::null());
        let act: DevicePtr<u16> = DevicePtr(core::ptr::null());
        let out: DeviceMutPtr<u16> = DeviceMutPtr(core::ptr::null_mut());

        let err = bonsai_tq2_gemv_fp16(packed, act, out, 16, 127, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));
        let err = bonsai_tq2_gemv_fp16(packed, act, out, -5, 128, None).unwrap_err();
        assert!(matches!(err, RcppError::Precondition(_)));

        assert_eq!(bonsai_tq2_packed_bytes(128), 34);
        assert_eq!(bonsai_tq2_packed_bytes(2048), 16 * 34);
    }
}
