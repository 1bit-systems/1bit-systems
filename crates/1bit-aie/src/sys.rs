//! Raw FFI declarations mirroring `native/xrt_c_shim.h`.
//!
//! Compile-time gated on `--features real-npu`. Default builds see empty
//! module so CI stays green on boxes without libxrt.
//!
//! Kernel-author fills in `native/xrt_c_shim.cpp` — this file only
//! declares the ABI. The [`super::xrt`] module wraps these with safe
//! RAII types + Rust-idiomatic error translation.

#![allow(non_camel_case_types, non_snake_case, missing_docs)]

use core::ffi::c_int;
#[cfg(feature = "real-npu")]
use core::ffi::{c_char, c_uint, c_void};

// ---------------------------------------------------------------------------
// Error codes — must match the XRT_SHIM_* constants in the C header.
// ---------------------------------------------------------------------------
pub const XRT_SHIM_OK: c_int = 0;
pub const XRT_SHIM_NOT_IMPLEMENTED: c_int = -1;
pub const XRT_SHIM_DEVICE_OPEN_FAILED: c_int = -2;
pub const XRT_SHIM_XCLBIN_LOAD_FAILED: c_int = -3;
pub const XRT_SHIM_HW_CONTEXT_FAILED: c_int = -4;
pub const XRT_SHIM_KERNEL_NOT_FOUND: c_int = -5;
pub const XRT_SHIM_BO_ALLOC_FAILED: c_int = -6;
pub const XRT_SHIM_BO_SYNC_FAILED: c_int = -7;
pub const XRT_SHIM_RUN_FAILED: c_int = -8;
pub const XRT_SHIM_BAD_HANDLE: c_int = -9;

// ---------------------------------------------------------------------------
// Opaque handles — C uses `struct xrt_shim_* *`; we use a zero-sized
// phantom struct + raw pointer alias. `*mut xrt_shim_device` etc. are
// what the extern fns take.
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct xrt_shim_device { _private: [u8; 0] }
#[repr(C)]
pub struct xrt_shim_xclbin { _private: [u8; 0] }
#[repr(C)]
pub struct xrt_shim_hw_context { _private: [u8; 0] }
#[repr(C)]
pub struct xrt_shim_kernel { _private: [u8; 0] }
#[repr(C)]
pub struct xrt_shim_bo { _private: [u8; 0] }
#[repr(C)]
pub struct xrt_shim_run { _private: [u8; 0] }

#[cfg(feature = "real-npu")]
#[link(name = "xrt_c_shim", kind = "static")]
unsafe extern "C" {
    // device
    pub fn xrt_shim_device_open(index: c_uint, out: *mut *mut xrt_shim_device) -> c_int;
    pub fn xrt_shim_device_close(dev: *mut xrt_shim_device) -> c_int;
    pub fn xrt_shim_device_name(dev: *mut xrt_shim_device, buf: *mut c_char, cap: usize) -> c_int;

    // xclbin
    pub fn xrt_shim_xclbin_load(
        dev: *mut xrt_shim_device,
        path: *const c_char,
        out: *mut *mut xrt_shim_xclbin,
    ) -> c_int;
    pub fn xrt_shim_xclbin_destroy(xb: *mut xrt_shim_xclbin) -> c_int;

    // hw_context
    pub fn xrt_shim_hw_context_create(
        dev: *mut xrt_shim_device,
        xb: *mut xrt_shim_xclbin,
        out: *mut *mut xrt_shim_hw_context,
    ) -> c_int;
    pub fn xrt_shim_hw_context_destroy(hw: *mut xrt_shim_hw_context) -> c_int;

    // kernel
    pub fn xrt_shim_kernel_open(
        hw: *mut xrt_shim_hw_context,
        name: *const c_char,
        out: *mut *mut xrt_shim_kernel,
    ) -> c_int;
    pub fn xrt_shim_kernel_close(k: *mut xrt_shim_kernel) -> c_int;

    // bo
    pub fn xrt_shim_bo_alloc(
        dev: *mut xrt_shim_device,
        bytes: usize,
        flags: c_uint,
        out: *mut *mut xrt_shim_bo,
    ) -> c_int;
    pub fn xrt_shim_bo_destroy(bo: *mut xrt_shim_bo) -> c_int;
    pub fn xrt_shim_bo_map(bo: *mut xrt_shim_bo, out_ptr: *mut *mut c_void) -> c_int;
    pub fn xrt_shim_bo_sync_to_device(bo: *mut xrt_shim_bo, offset: usize, bytes: usize) -> c_int;
    pub fn xrt_shim_bo_sync_from_device(bo: *mut xrt_shim_bo, offset: usize, bytes: usize) -> c_int;

    // run
    pub fn xrt_shim_run_create(k: *mut xrt_shim_kernel, out: *mut *mut xrt_shim_run) -> c_int;
    pub fn xrt_shim_run_destroy(r: *mut xrt_shim_run) -> c_int;
    pub fn xrt_shim_run_set_arg_bo(r: *mut xrt_shim_run, idx: c_uint, bo: *mut xrt_shim_bo) -> c_int;
    pub fn xrt_shim_run_set_arg_u32(r: *mut xrt_shim_run, idx: c_uint, val: u32) -> c_int;
    pub fn xrt_shim_run_set_arg_i32(r: *mut xrt_shim_run, idx: c_uint, val: i32) -> c_int;
    pub fn xrt_shim_run_set_arg_f32(r: *mut xrt_shim_run, idx: c_uint, val: f32) -> c_int;
    pub fn xrt_shim_run_start(r: *mut xrt_shim_run) -> c_int;
    pub fn xrt_shim_run_wait(r: *mut xrt_shim_run, timeout_ms: c_uint) -> c_int;

    // diagnostics
    pub fn xrt_shim_last_error_cstr() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_are_unique() {
        let codes = [
            XRT_SHIM_OK,
            XRT_SHIM_NOT_IMPLEMENTED,
            XRT_SHIM_DEVICE_OPEN_FAILED,
            XRT_SHIM_XCLBIN_LOAD_FAILED,
            XRT_SHIM_HW_CONTEXT_FAILED,
            XRT_SHIM_KERNEL_NOT_FOUND,
            XRT_SHIM_BO_ALLOC_FAILED,
            XRT_SHIM_BO_SYNC_FAILED,
            XRT_SHIM_RUN_FAILED,
            XRT_SHIM_BAD_HANDLE,
        ];
        let mut seen = std::collections::HashSet::new();
        for c in codes {
            assert!(seen.insert(c), "duplicate error code {c}");
        }
    }

    #[test]
    fn ok_is_zero() {
        // Every XRT success convention; the shim inherits it.
        assert_eq!(XRT_SHIM_OK, 0);
    }

    #[test]
    fn error_codes_are_negative() {
        // Sanity: non-zero errors stay negative so callers can test
        // `rc != XRT_SHIM_OK` without worrying about warning codes.
        for c in [
            XRT_SHIM_NOT_IMPLEMENTED,
            XRT_SHIM_DEVICE_OPEN_FAILED,
            XRT_SHIM_XCLBIN_LOAD_FAILED,
            XRT_SHIM_HW_CONTEXT_FAILED,
            XRT_SHIM_KERNEL_NOT_FOUND,
            XRT_SHIM_BO_ALLOC_FAILED,
            XRT_SHIM_BO_SYNC_FAILED,
            XRT_SHIM_RUN_FAILED,
            XRT_SHIM_BAD_HANDLE,
        ] {
            assert!(c < 0, "error code {c} must be negative");
        }
    }
}
