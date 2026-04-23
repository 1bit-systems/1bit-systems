// HIP / rocm-cpp FFI names intentionally mirror the C API (camelCase for the
// HIP runtime, snake_case for rocm-cpp's rcpp_*). Clippy's snake_case lint
// is not useful here.
#![allow(non_snake_case)]

//! Raw `extern "C"` bindings for the rocm-cpp C API.
//!
//! Signatures are taken verbatim from
//! `/home/bcloud/repos/rocm-cpp/include/rocm_cpp/ck_gemm.h`.
//!
//! **Pointer conventions (matching the C header):**
//!   * `*const c_void` / `*mut c_void` are **device** pointers unless the
//!     comment says "host". Callers must have produced them via `hipMalloc`
//!     (or equivalent) on the same device that runs the kernel.
//!   * `*mut c_void` for a stream is a `hipStream_t`. Passing `null_mut()`
//!     means "default (null) stream" in HIP.
//!
//! **ABI:** every function here has C linkage. `rcpp_status_t` is a C enum
//! whose underlying type is `int` on every platform that builds rocm-cpp
//! (verified: the header uses a plain `typedef enum { ... } rcpp_status_t;`
//! with values 0..4 — sign does not matter for the 5 enumerators).
//!
//! This module is a leaf: no safety wrappers, no `Result`s, no logging.
//! The public API in `lib.rs` is the place for that.

#![allow(non_camel_case_types)]
#![allow(dead_code)]
// The canonical documentation for every `rcpp_*` symbol lives in
// `rocm-cpp/include/rocm_cpp/ck_gemm.h`; duplicating it here would go
// stale. The safe wrappers in `lib.rs` carry the Rust-facing docs.
#![allow(missing_docs)]

use core::ffi::c_uint;
use core::ffi::{c_char, c_float, c_int, c_void};

// -----------------------------------------------------------------------------
// HIP runtime bindings — a narrow subset of `hip/hip_runtime_api.h` sufficient
// for 1bit-router's device-memory lifecycle (alloc / copy / synchronize) and
// device enumeration. Symbols live in `libamdhip64.so`, which is already
// linked alongside `librocm_cpp.so` (see build.rs).
//
// `hipError_t` is an int-valued enum; 0 == hipSuccess. We fold any non-zero
// return into `RcppError::HipError` at the safe-wrapper boundary.
// `hipMemcpyKind` is likewise an int-valued enum with the familiar constants.
// -----------------------------------------------------------------------------
pub type hip_error_t = c_int;

/// Matches `hipMemcpyKind` from hip/hip_runtime_api.h.
pub const HIP_MEMCPY_HOST_TO_HOST: c_uint = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_uint = 3;
pub const HIP_MEMCPY_DEFAULT: c_uint = 4;

/// `hipSuccess = 0`.
pub const HIP_SUCCESS: hip_error_t = 0;

#[cfg(feature = "link-rocm")]
unsafe extern "C" {
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hip_error_t;
    pub fn hipFree(ptr: *mut c_void) -> hip_error_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
    ) -> hip_error_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_uint,
        stream: *mut c_void,
    ) -> hip_error_t;
    pub fn hipMemset(dst: *mut c_void, value: c_int, count: usize) -> hip_error_t;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        count: usize,
        stream: *mut c_void,
    ) -> hip_error_t;
    pub fn hipDeviceSynchronize() -> hip_error_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hip_error_t;
    pub fn hipSetDevice(device_id: c_int) -> hip_error_t;
    pub fn hipGetLastError() -> hip_error_t;
    pub fn hipGetErrorString(err: hip_error_t) -> *const c_char;
}

#[cfg(not(feature = "link-rocm"))]
mod hip_stub {
    use super::*;
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMalloc(_ptr: *mut *mut c_void, _size: usize) -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipFree(_ptr: *mut c_void) -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemcpy(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: usize,
        _kind: c_uint,
    ) -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemcpyAsync(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: usize,
        _kind: c_uint,
        _stream: *mut c_void,
    ) -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemset(
        _dst: *mut c_void,
        _value: c_int,
        _count: usize,
    ) -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipMemsetAsync(
        _dst: *mut c_void,
        _value: c_int,
        _count: usize,
        _stream: *mut c_void,
    ) -> hip_error_t {
        1
    }
    #[inline]
    pub unsafe extern "C" fn hipDeviceSynchronize() -> hip_error_t {
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipGetDeviceCount(count: *mut c_int) -> hip_error_t {
        // Safe to write zero into a valid pointer without a feature flag, but
        // we prefer to just report failure on CI hosts.
        1
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipSetDevice(_device_id: c_int) -> hip_error_t {
        1
    }
    #[inline]
    pub unsafe extern "C" fn hipGetLastError() -> hip_error_t {
        0
    }
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn hipGetErrorString(_err: hip_error_t) -> *const c_char {
        core::ptr::null()
    }
}

#[cfg(not(feature = "link-rocm"))]
pub use hip_stub::*;

/// Mirrors `rcpp_status_t` from `ck_gemm.h`. Kept as a `#[repr(C)] i32` so
/// the compiler uses the same integer width the C side emits for an enum.
pub type rcpp_status_t = c_int;

pub const RCPP_OK: rcpp_status_t = 0;
pub const RCPP_INVALID_ARG: rcpp_status_t = 1;
pub const RCPP_UNSUPPORTED: rcpp_status_t = 2;
pub const RCPP_HIP_ERROR: rcpp_status_t = 3;
pub const RCPP_INTERNAL: rcpp_status_t = 4;

/// Opaque CK-GEMM handle. Callers pass a pointer; they never deref it.
#[repr(C)]
pub struct rcpp_ck_gemm_handle_t {
    _private: [u8; 0],
}

// -----------------------------------------------------------------------------
// CK GEMM (prefill)
// -----------------------------------------------------------------------------
#[cfg(feature = "link-rocm")]
unsafe extern "C" {
    pub fn rcpp_ck_gemm_create(
        M: c_int,
        N: c_int,
        K: c_int,
        handle_out: *mut *mut rcpp_ck_gemm_handle_t,
    ) -> rcpp_status_t;

    pub fn rcpp_ck_gemm_destroy(handle: *mut rcpp_ck_gemm_handle_t);

    pub fn rcpp_ck_gemm_run(
        handle: *mut rcpp_ck_gemm_handle_t,
        A_dev: *const c_void,
        B_dev_packed: *const c_void,
        C_dev: *mut c_void,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_pack_pk_i4(
        ternary_host: *const i8,
        packed_host: *mut i8,
        K: c_int,
        N: c_int,
    ) -> rcpp_status_t;

    pub fn rcpp_ck_gemm_instance_string(handle: *const rcpp_ck_gemm_handle_t) -> *const c_char;

    // -------------------------------------------------------------------------
    // Phase 5 decode GEMV
    // -------------------------------------------------------------------------
    pub fn rcpp_ternary_gemv(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_halo(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_halo_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_sherry_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_ternary_gemv_tq1_halo_f16(
        packed_weights_dev: *const c_void,
        activations_i8_dev: *const c_void,
        activation_scale: c_float,
        row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void,
        M: c_int,
        K_padded: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Primitive kernels
    // -------------------------------------------------------------------------
    pub fn rcpp_quantize_fp16_to_i8(
        x_fp16_dev: *const c_void,
        x_i8_dev: *mut c_void,
        scale_dev: *mut c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rmsnorm_fp16(
        x_dev: *const c_void,
        weight_dev: *const c_void,
        y_dev: *mut c_void,
        eps: c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rope_fp16(
        x_dev: *mut c_void,
        pos: c_int,
        theta: c_float,
        num_heads: c_int,
        head_dim: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_silu_glu_fp16(
        up_dev: *const c_void,
        gate_dev: *const c_void,
        y_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_relu2_glu_fp16(
        gate_dev: *const c_void,
        up_dev: *const c_void,
        y_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_relu2_glu_rmsnorm_fp16(
        gate_dev: *const c_void,
        up_dev: *const c_void,
        ffn_sub_norm_dev: *const c_void,
        y_dev: *mut c_void,
        eps: c_float,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_embedding_lookup_fp16(
        embedding_dev: *const c_void,
        token_id: c_int,
        y_dev: *mut c_void,
        hidden: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_residual_add_fp16(
        y_dev: *mut c_void,
        src_dev: *const c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_residual_add_fp32_from_fp16(
        x_fp32_dev: *mut c_void,
        src_fp16_dev: *const c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_rmsnorm_fp32_in_fp16_out(
        x_fp32_dev: *const c_void,
        weight_dev: *const c_void,
        y_fp16_dev: *mut c_void,
        eps: c_float,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_argmax_fp32(
        logits_dev: *const c_void,
        out_idx_dev: *mut c_void,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_fp16_gemv(
        W_dev: *const c_void,
        x_dev: *const c_void,
        y_dev: *mut c_void,
        M: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_fp32_to_fp16(
        x_fp32_dev: *const c_void,
        y_fp16_dev: *mut c_void,
        N: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_top_k_fp32(
        logits_dev: *mut c_void,
        k: c_int,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_softmax_fp32(
        logits_dev: *mut c_void,
        V: c_int,
        temperature: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_sample_multinomial_fp32(
        probs_dev: *const c_void,
        r: c_float,
        out_idx_dev: *mut c_void,
        V: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Phase 6 — KV cache attention
    // -------------------------------------------------------------------------
    pub fn rcpp_kv_cache_attn_decode(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_decode_fd(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_prefill(
        Q_dev: *const c_void,
        K_dev: *const c_void,
        V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_decode_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void,
        V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void,
        V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_kv_cache_attn_prefill_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void,
        V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void,
        V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int,
        num_kv_heads: c_int,
        head_dim: c_int,
        seq_len: c_int,
        scale: c_float,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    pub fn rcpp_quantize_fp16_to_i8_rowscale(
        x_fp16_dev: *const c_void,
        out_i8_dev: *mut c_void,
        scale_fp16_out_dev: *mut c_void,
        num_rows: c_int,
        row_len: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Medusa speculative decoding — small-M ternary GEMM (M ∈ [1, 16]).
    //
    // Kernel source: `rocm-cpp/src/ternary_gemm_smallm.hip`.
    // ABI is pinned by the `onebit_ternary_gemm_smallm_launch` signature in
    // that file. Shapes:
    //   x_i8           : [M, K]       int8, row-major.
    //   x_scale        :               float32 per-tensor activation scale.
    //   w_packed       : [K/16, N]     uint32, 16 halo-codes packed per u32.
    //   w_row_scales   : [N]           float32, per-column weight scale.
    //   y_out          : [M, N]        hip_bfloat16, row-major.
    // Preconditions: M ∈ [1, 16], N % 64 == 0, K % 64 == 0. The native
    // launcher silently no-ops on shape violations; safe wrappers in
    // `lib.rs` pre-check the contract and return `InvalidArg`.
    //
    // Returns `void` — no rcpp_status_t (matches the Hadamard launcher and
    // every other direct-kernel entry point). Callers that need error
    // reporting must check `hipGetLastError()` at the next synchronization
    // boundary.
    // -------------------------------------------------------------------------
    pub fn onebit_ternary_gemm_smallm_launch(
        x_i8: *const c_void,
        x_scale: c_float,
        w_packed: *const c_void,
        w_row_scales: *const c_void,
        y_out: *mut c_void,
        M: c_int,
        N: c_int,
        K: c_int,
        stream: *mut c_void,
    );

    // -------------------------------------------------------------------------
    // BitNet v2 — H-BitLinear Hadamard activation rotation.
    //
    // `K` is the total fp16 element count (= n_blocks * 128). Device-side
    // kernel hardcodes the block size to HADAMARD_BLOCK=128; `K` must be a
    // multiple of 128. In-place aliasing (x_in == y_out) is permitted.
    //
    // Returns `void` — no rcpp_status_t. Callers that need error reporting
    // must check hipGetLastError() at the next synchronization boundary.
    // -------------------------------------------------------------------------
    pub fn rcpp_hadamard_rotate_fp16_butterfly_launch(
        x_in: *const c_void,
        y_out: *mut c_void,
        K: c_int,
        stream: *mut c_void,
    );

    // -------------------------------------------------------------------------
    // Standalone (CK-free) prefill
    // -------------------------------------------------------------------------
    pub fn rcpp_standalone_gemm(
        A_dev: *const c_void,
        B_dev_packed: *const c_void,
        C_dev: *mut c_void,
        M: c_int,
        N: c_int,
        K: c_int,
        stream: *mut c_void,
    ) -> rcpp_status_t;

    // -------------------------------------------------------------------------
    // Sherry 1.25-bit ternary GEMV — clean-room fp16-in/fp16-out path.
    //
    // Header: `include/rocm_cpp/sherry.h` (standalone, distinct from the
    // `rcpp_ternary_gemv_sherry_f16` entry point in ck_gemm.h, which takes
    // INT8 activations + per-row FP32 scales and targets the halo-v3
    // decode loop). This path takes FP16 activations directly and does
    // NOT bake a per-row scale — the caller multiplies externally if
    // needed.
    //
    // Packing (matches project_sherry_spike.md):
    //   Every 4 consecutive weights along K form one group; one lane is
    //   zero, the other three carry ±1 signs. 5 bits per group:
    //   [zero_pos : 2][signs : 3]. Row bytes = K_in * 5 / 32; K_in must
    //   be a multiple of 32 for byte alignment (launcher enforces).
    //
    // Returns `void` — launcher has no rcpp_status_t. Callers check
    // hipGetLastError() at the next synchronization boundary for launch
    // failures. Shape violations early-return as no-ops inside the
    // launcher, so the safe wrappers in `lib.rs` pre-validate.
    //
    // `rcpp_sherry_pack` is a pure host packer (no HIP) — safe from any
    // thread, no stream argument.
    // -------------------------------------------------------------------------
    pub fn sherry_ternary_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    );

    pub fn sherry_ternary_gemv_scalar_ref_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    );

    pub fn rcpp_sherry_pack(ternary: *const i8, out: *mut u8, count: c_int);

    // -------------------------------------------------------------------------
    // Bonsai Q1_0_g128 / TQ2_0_g128 ternary GEMV — FP16-in / FP16-out.
    //
    // Status 2026-04-20: **not yet implemented** in rocm-cpp. Forward
    // declarations + stubs land here so scaffolding + safe wrappers compile
    // while the HIP port agent writes the actual .hip kernels. Until the
    // kernel is in place the `link-rocm` build will fail to link these
    // symbols; the safe wrappers in `lib.rs` short-circuit before
    // dispatching (see the TODO gate there).
    //
    // On-disk packing (matches oxibonsai, block-interleaved):
    //   Q1_0_g128  : 18 B per g128 block = [FP16 d : 2][sign_bits : 16]
    //                  128 sign bits LSB-first; w[i] = bit ? +d : -d.
    //   TQ2_0_g128 : 34 B per g128 block = [codes : 32][FP16 d : 2]
    //                  2 bits per weight, 4 per byte LSB-first;
    //                  00→-1, 01→0, 10→+1, 11→0(reserved).
    //
    // Shapes:
    //   packed_weights : row-major, row_bytes = (K_in / 128) * {18, 34}
    //   act_fp16       : K_in fp16 elements on device
    //   out_fp16       : N_out fp16 writable elements on device
    //
    // Preconditions (enforced in safe wrappers, not in the launchers):
    //   N_out > 0; K_in > 0; K_in % 128 == 0.
    //
    // Returns `void` — launchers do not carry rcpp_status_t. Callers check
    // hipGetLastError() at the next synchronization boundary for kernel
    // launch failures. Shape violations pre-validate in Rust.
    // -------------------------------------------------------------------------
    pub fn bonsai_q1_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    );

    pub fn bonsai_tq2_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    );
}

// -----------------------------------------------------------------------------
// Stub shims (no `link-rocm` feature) — every symbol returns RCPP_UNSUPPORTED.
//
// These have identical signatures to the extern block above, so the rest of
// the crate can call them without cfg gating at every call site. The linker
// sees pure Rust symbols and never references the native .so.
// -----------------------------------------------------------------------------
#[cfg(not(feature = "link-rocm"))]
mod stub {
    use super::*;

    macro_rules! stub_fn {
        ($name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) -> $ret:ty) => {
            #[inline]
            #[allow(unused_variables, non_snake_case)]
            pub unsafe extern "C" fn $name( $( $arg : $ty ),* ) -> $ret {
                RCPP_UNSUPPORTED
            }
        };
        ($name:ident ( $( $arg:ident : $ty:ty ),* $(,)? )) => {
            #[inline]
            #[allow(unused_variables, non_snake_case)]
            pub unsafe extern "C" fn $name( $( $arg : $ty ),* ) {}
        };
    }

    stub_fn!(rcpp_ck_gemm_create(
        M: c_int, N: c_int, K: c_int,
        handle_out: *mut *mut rcpp_ck_gemm_handle_t
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ck_gemm_destroy(handle: *mut rcpp_ck_gemm_handle_t));

    stub_fn!(rcpp_ck_gemm_run(
        handle: *mut rcpp_ck_gemm_handle_t,
        A_dev: *const c_void, B_dev_packed: *const c_void, C_dev: *mut c_void,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_pack_pk_i4(
        ternary_host: *const i8, packed_host: *mut i8, K: c_int, N: c_int
    ) -> rcpp_status_t);

    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn rcpp_ck_gemm_instance_string(
        _handle: *const rcpp_ck_gemm_handle_t,
    ) -> *const c_char {
        core::ptr::null()
    }

    stub_fn!(rcpp_ternary_gemv(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_halo(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_halo_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_sherry_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_ternary_gemv_tq1_halo_f16(
        packed_weights_dev: *const c_void, activations_i8_dev: *const c_void,
        activation_scale: c_float, row_scales_dev: *const c_void,
        output_f16_dev: *mut c_void, M: c_int, K_padded: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_quantize_fp16_to_i8(
        x_fp16_dev: *const c_void, x_i8_dev: *mut c_void,
        scale_dev: *mut c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rmsnorm_fp16(
        x_dev: *const c_void, weight_dev: *const c_void, y_dev: *mut c_void,
        eps: c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rope_fp16(
        x_dev: *mut c_void, pos: c_int, theta: c_float,
        num_heads: c_int, head_dim: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_silu_glu_fp16(
        up_dev: *const c_void, gate_dev: *const c_void, y_dev: *mut c_void,
        N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_relu2_glu_fp16(
        gate_dev: *const c_void, up_dev: *const c_void, y_dev: *mut c_void,
        N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_relu2_glu_rmsnorm_fp16(
        gate_dev: *const c_void, up_dev: *const c_void,
        ffn_sub_norm_dev: *const c_void, y_dev: *mut c_void,
        eps: c_float, N: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_embedding_lookup_fp16(
        embedding_dev: *const c_void, token_id: c_int, y_dev: *mut c_void,
        hidden: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_residual_add_fp16(
        y_dev: *mut c_void, src_dev: *const c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_residual_add_fp32_from_fp16(
        x_fp32_dev: *mut c_void, src_fp16_dev: *const c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_rmsnorm_fp32_in_fp16_out(
        x_fp32_dev: *const c_void, weight_dev: *const c_void,
        y_fp16_dev: *mut c_void, eps: c_float, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_argmax_fp32(
        logits_dev: *const c_void, out_idx_dev: *mut c_void, V: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_fp16_gemv(
        W_dev: *const c_void, x_dev: *const c_void, y_dev: *mut c_void,
        M: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_fp32_to_fp16(
        x_fp32_dev: *const c_void, y_fp16_dev: *mut c_void, N: c_int,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_top_k_fp32(
        logits_dev: *mut c_void, k: c_int, V: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_softmax_fp32(
        logits_dev: *mut c_void, V: c_int, temperature: c_float,
        stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_sample_multinomial_fp32(
        probs_dev: *const c_void, r: c_float, out_idx_dev: *mut c_void,
        V: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode_fd(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_prefill(
        Q_dev: *const c_void, K_dev: *const c_void, V_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_decode_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void, V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void, V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_kv_cache_attn_prefill_i8(
        Q_dev: *const c_void,
        K_i8_dev: *const c_void, V_i8_dev: *const c_void,
        K_scales_fp16_dev: *const c_void, V_scales_fp16_dev: *const c_void,
        out_dev: *mut c_void,
        num_q_heads: c_int, num_kv_heads: c_int, head_dim: c_int,
        seq_len: c_int, scale: c_float, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_quantize_fp16_to_i8_rowscale(
        x_fp16_dev: *const c_void, out_i8_dev: *mut c_void,
        scale_fp16_out_dev: *mut c_void,
        num_rows: c_int, row_len: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    stub_fn!(rcpp_standalone_gemm(
        A_dev: *const c_void, B_dev_packed: *const c_void, C_dev: *mut c_void,
        M: c_int, N: c_int, K: c_int, stream: *mut c_void
    ) -> rcpp_status_t);

    // Hadamard rotate: void return — no-op in stub mode. Callers see
    // `Unsupported` at the safe-wrapper level (lib.rs checks the feature
    // before dispatching) so this shim is only for symbol completeness.
    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn rcpp_hadamard_rotate_fp16_butterfly_launch(
        x_in: *const c_void,
        y_out: *mut c_void,
        K: c_int,
        stream: *mut c_void,
    ) {
    }

    // Medusa small-M ternary GEMM: void return — no-op in stub mode. Safe
    // wrappers in `lib.rs` pre-check the shape contract and return
    // `RcppStatus::Unsupported` when `link-rocm` is off, so callers never
    // observe this no-op directly.
    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn onebit_ternary_gemm_smallm_launch(
        x_i8: *const c_void,
        x_scale: c_float,
        w_packed: *const c_void,
        w_row_scales: *const c_void,
        y_out: *mut c_void,
        M: c_int,
        N: c_int,
        K: c_int,
        stream: *mut c_void,
    ) {
    }

    // Sherry fp16-in/fp16-out GEMV: void return — no-op in stub mode. Safe
    // wrappers in `lib.rs` validate shape and return `Unsupported` when
    // `link-rocm` is off, so the no-op is never observed directly.
    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn sherry_ternary_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    ) {
    }

    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn sherry_ternary_gemv_scalar_ref_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    ) {
    }

    // Host-only packer: no-op in stub mode. Safe wrapper in `lib.rs`
    // returns `Unsupported` before reaching this shim.
    #[inline]
    #[allow(unused_variables)]
    pub unsafe extern "C" fn rcpp_sherry_pack(ternary: *const i8, out: *mut u8, count: c_int) {}

    // Bonsai Q1 / TQ2 GEMV launchers: void return — no-op in stub mode.
    // Safe wrappers in `lib.rs` short-circuit to `Unsupported` before
    // reaching these shims (see the TODO note above the FFI declarations).
    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn bonsai_q1_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    ) {
    }

    #[inline]
    #[allow(unused_variables, non_snake_case)]
    pub unsafe extern "C" fn bonsai_tq2_gemv_launch(
        packed_weights: *const u8,
        act_fp16: *const u16,
        out_fp16: *mut u16,
        N_out: c_int,
        K_in: c_int,
        stream: *mut c_void,
    ) {
    }
}

#[cfg(not(feature = "link-rocm"))]
pub use stub::*;
