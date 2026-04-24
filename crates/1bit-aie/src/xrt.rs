//! Safe RAII wrappers over the [`super::sys`] C shim.
//!
//! Compile-time gated on `--features real-npu`. Default build exposes
//! the types but every method returns [`AieError::NotYetWired`]; call
//! sites can be written against the safe API today and light up when
//! the shim's C++ side is filled in.
//!
//! ## RAII ownership model
//!
//! Every wrapper owns one C handle (`*mut xrt_shim_*`) and calls the
//! matching `xrt_shim_*_destroy` / `_close` in `Drop`. The handles do
//! not implement `Clone`. Transfer happens via move; aliasing would
//! require `Arc` and the shim doesn't yet expose thread-safe refcounts.
//!
//! ## Error translation
//!
//! C return codes map to [`AieError`] via [`map_rc`]:
//!
//! - `XRT_SHIM_OK` → `Ok(())`
//! - `XRT_SHIM_NOT_IMPLEMENTED` → `AieError::NotYetWired(op)` with the
//!   operation name passed at the call site.
//! - Anything else → `AieError::Xrt(code)`. Kernel-author can enrich
//!   this to carry the shim's thread-local `xrt_shim_last_error_cstr()`
//!   message once the C++ side is producing meaningful strings.

use crate::AieError;

use core::ffi::c_int;

#[cfg(feature = "real-npu")]
use core::ffi::{c_uint, CStr};
#[cfg(feature = "real-npu")]
use std::ffi::CString;
#[cfg(feature = "real-npu")]
use std::path::Path;
#[cfg(feature = "real-npu")]
use std::ptr;

use super::sys;

// ---------------------------------------------------------------------------
// Error translation
// ---------------------------------------------------------------------------

/// Translate a C return code into a `Result`. `op` is a static operation
/// label used only for `NotYetWired` — callers pass something like
/// `"device_open"` so the error identifies which step failed.
#[cfg(feature = "real-npu")]
fn map_rc(rc: c_int, op: &'static str) -> Result<(), AieError> {
    match rc {
        sys::XRT_SHIM_OK => Ok(()),
        sys::XRT_SHIM_NOT_IMPLEMENTED => Err(AieError::NotYetWired(op)),
        _ => Err(AieError::Xrt(rc)),
    }
}

#[cfg(not(feature = "real-npu"))]
#[allow(dead_code)]
fn map_rc(_rc: c_int, op: &'static str) -> Result<(), AieError> {
    Err(AieError::NotYetWired(op))
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// Owned handle to an XRT device (npu0, npu1, …). RAII: closed on Drop.
#[derive(Debug)]
pub struct Device {
    #[allow(dead_code)] // real-npu feature reads `inner`; stub build keeps it for API shape.
    inner: *mut sys::xrt_shim_device,
}

impl Device {
    /// Open device by index (0 == primary npu). Returns
    /// [`AieError::NotYetWired`] in stub builds.
    pub fn open(#[allow(unused_variables)] index: u32) -> Result<Self, AieError> {
        #[cfg(feature = "real-npu")]
        {
            let mut out: *mut sys::xrt_shim_device = ptr::null_mut();
            let rc = unsafe { sys::xrt_shim_device_open(index as c_uint, &mut out) };
            map_rc(rc, "device_open")?;
            Ok(Self { inner: out })
        }
        #[cfg(not(feature = "real-npu"))]
        {
            let _ = index;
            Err(AieError::NotYetWired("device_open"))
        }
    }

    /// Device name (e.g. "npu0"). Truncates to 128 bytes.
    pub fn name(&self) -> Result<String, AieError> {
        #[cfg(feature = "real-npu")]
        {
            let mut buf = [0i8; 128];
            let rc = unsafe {
                sys::xrt_shim_device_name(self.inner, buf.as_mut_ptr(), buf.len())
            };
            map_rc(rc, "device_name")?;
            let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
            Ok(cstr.to_string_lossy().into_owned())
        }
        #[cfg(not(feature = "real-npu"))]
        {
            Err(AieError::NotYetWired("device_name"))
        }
    }

    #[cfg(feature = "real-npu")]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_device {
        self.inner
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_device_close(self.inner);
            }
        }
    }
}

// SAFETY: XRT devices are thread-safe per AMD docs; the shim serializes
// access internally. Revisit if kernel-author finds races.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

// ---------------------------------------------------------------------------
// Xclbin
// ---------------------------------------------------------------------------

/// Loaded .xclbin image. RAII.
#[derive(Debug)]
pub struct Xclbin {
    #[allow(dead_code)] // same pattern as `Device::inner`.
    inner: *mut sys::xrt_shim_xclbin,
}

impl Xclbin {
    /// Load an .xclbin from disk.
    #[cfg(feature = "real-npu")]
    pub fn load(dev: &Device, path: &Path) -> Result<Self, AieError> {
        let cpath = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .map_err(|_| AieError::XclbinNotFound(format!("{path:?} contains NUL byte")))?;
        let mut out: *mut sys::xrt_shim_xclbin = ptr::null_mut();
        let rc = unsafe { sys::xrt_shim_xclbin_load(dev.raw(), cpath.as_ptr(), &mut out) };
        map_rc(rc, "xclbin_load")?;
        Ok(Self { inner: out })
    }

    /// Stub-build variant: always returns `NotYetWired`.
    #[cfg(not(feature = "real-npu"))]
    pub fn load(_dev: &Device, _path: &std::path::Path) -> Result<Self, AieError> {
        Err(AieError::NotYetWired("xclbin_load"))
    }
}

impl Drop for Xclbin {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_xclbin_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Xclbin {}
unsafe impl Sync for Xclbin {}

// ---------------------------------------------------------------------------
// Compile-only unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "real-npu"))]
    fn device_open_stub_returns_not_yet_wired() {
        let err = Device::open(0).unwrap_err();
        assert!(matches!(err, AieError::NotYetWired("device_open")));
    }

    #[test]
    fn device_drop_is_idempotent_on_null() {
        // Null inner → Drop must not deref. Only legal because we
        // fabricate the null in-place; real code never does.
        let d = Device { inner: core::ptr::null_mut() };
        drop(d);
    }

    #[test]
    #[cfg(not(feature = "real-npu"))]
    fn xclbin_load_stub_returns_not_yet_wired() {
        let d = Device { inner: core::ptr::null_mut() };
        let err = Xclbin::load(&d, std::path::Path::new("/nonexistent.xclbin")).unwrap_err();
        assert!(matches!(err, AieError::NotYetWired("xclbin_load")));
    }
}
