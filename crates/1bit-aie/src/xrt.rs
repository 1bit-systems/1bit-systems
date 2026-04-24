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

// Placeholder accessors for stub build — keep API surface identical
// across feature configs so downstream code doesn't need cfg-gating.
#[cfg(not(feature = "real-npu"))]
impl Device {
    #[allow(dead_code)]
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
    #[allow(missing_docs)]
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

impl Xclbin {
    #[cfg(feature = "real-npu")]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_xclbin {
        self.inner
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_xclbin {
        self.inner
    }
}

unsafe impl Send for Xclbin {}
unsafe impl Sync for Xclbin {}

// ---------------------------------------------------------------------------
// HwContext — multi-xclbin isolation (mandatory on XDNA2)
// ---------------------------------------------------------------------------

/// Hardware context binding a device + xclbin. RAII: destroyed on Drop.
///
/// XDNA2 requires every kernel dispatch to go through a hw_context;
/// opening one claims the AIE tile array for this xclbin until drop.
#[derive(Debug)]
pub struct HwContext {
    #[allow(dead_code)]
    inner: *mut sys::xrt_shim_hw_context,
}

impl HwContext {
    /// Create a hw_context from a device + loaded xclbin.
    #[cfg(feature = "real-npu")]
    pub fn create(dev: &Device, xb: &Xclbin) -> Result<Self, AieError> {
        let mut out: *mut sys::xrt_shim_hw_context = ptr::null_mut();
        let rc = unsafe { sys::xrt_shim_hw_context_create(dev.raw(), xb.raw(), &mut out) };
        map_rc(rc, "hw_context_create")?;
        Ok(Self { inner: out })
    }

    /// Stub-build variant: always returns `NotYetWired`.
    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn create(_dev: &Device, _xb: &Xclbin) -> Result<Self, AieError> {
        Err(AieError::NotYetWired("hw_context_create"))
    }

    #[cfg(feature = "real-npu")]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_hw_context {
        self.inner
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_hw_context {
        self.inner
    }
}

impl Drop for HwContext {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_hw_context_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for HwContext {}
unsafe impl Sync for HwContext {}

// ---------------------------------------------------------------------------
// Kernel — named compute unit inside the xclbin
// ---------------------------------------------------------------------------

/// Opened kernel by name (e.g. `"bitnet_gemv"`). RAII.
#[derive(Debug)]
pub struct Kernel {
    #[allow(dead_code)]
    inner: *mut sys::xrt_shim_kernel,
}

impl Kernel {
    /// Open a kernel by its name inside the hw_context's xclbin.
    #[cfg(feature = "real-npu")]
    pub fn open(hw: &HwContext, name: &str) -> Result<Self, AieError> {
        let cname = CString::new(name)
            .map_err(|_| AieError::ShapeMismatch("kernel name contains NUL"))?;
        let mut out: *mut sys::xrt_shim_kernel = ptr::null_mut();
        let rc = unsafe { sys::xrt_shim_kernel_open(hw.raw(), cname.as_ptr(), &mut out) };
        map_rc(rc, "kernel_open")?;
        Ok(Self { inner: out })
    }

    /// Stub-build variant: always returns `NotYetWired`.
    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn open(_hw: &HwContext, _name: &str) -> Result<Self, AieError> {
        Err(AieError::NotYetWired("kernel_open"))
    }

    #[cfg(feature = "real-npu")]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_kernel {
        self.inner
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_kernel {
        self.inner
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_kernel_close(self.inner);
            }
        }
    }
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

// ---------------------------------------------------------------------------
// Bo — buffer object (host↔tile memory sync)
// ---------------------------------------------------------------------------

/// Buffer-object allocation flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BoFlags {
    /// Cacheable host-mapped. Host reads/writes via `map()` + explicit
    /// sync. Default for weight / activation buffers.
    HostMapped = 0,
    /// Device-only. No host mapping. Used for intermediate / scratch
    /// buffers the kernel writes but the host never reads.
    DeviceOnly = 1,
}

/// Buffer object. RAII; destroyed on Drop.
#[derive(Debug)]
pub struct Bo {
    #[allow(dead_code)]
    inner: *mut sys::xrt_shim_bo,
    #[allow(dead_code)]
    len_bytes: usize,
}

impl Bo {
    /// Allocate a BO of `bytes` on the device, with the given map flag.
    #[cfg(feature = "real-npu")]
    pub fn alloc(dev: &Device, bytes: usize, flags: BoFlags) -> Result<Self, AieError> {
        let mut out: *mut sys::xrt_shim_bo = ptr::null_mut();
        let rc = unsafe {
            sys::xrt_shim_bo_alloc(dev.raw(), bytes, flags as c_uint, &mut out)
        };
        map_rc(rc, "bo_alloc")?;
        Ok(Self { inner: out, len_bytes: bytes })
    }

    /// Stub-build variant: always returns `NotYetWired`.
    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn alloc(_dev: &Device, _bytes: usize, _flags: BoFlags) -> Result<Self, AieError> {
        Err(AieError::NotYetWired("bo_alloc"))
    }

    /// Host-mapped pointer. Lifetime tied to `self`; caller must not
    /// free or outlive the Bo.
    #[cfg(feature = "real-npu")]
    pub fn map(&self) -> Result<*mut u8, AieError> {
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        let rc = unsafe { sys::xrt_shim_bo_map(self.inner, &mut ptr) };
        map_rc(rc, "bo_map")?;
        Ok(ptr as *mut u8)
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn map(&self) -> Result<*mut u8, AieError> {
        Err(AieError::NotYetWired("bo_map"))
    }

    /// Host → device sync of bytes starting at `offset`.
    #[cfg(feature = "real-npu")]
    pub fn sync_to_device(&self, offset: usize, bytes: usize) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_bo_sync_to_device(self.inner, offset, bytes) };
        map_rc(rc, "bo_sync_to_device")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn sync_to_device(&self, _offset: usize, _bytes: usize) -> Result<(), AieError> {
        Err(AieError::NotYetWired("bo_sync_to_device"))
    }

    /// Device → host sync of bytes starting at `offset`.
    #[cfg(feature = "real-npu")]
    pub fn sync_from_device(&self, offset: usize, bytes: usize) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_bo_sync_from_device(self.inner, offset, bytes) };
        map_rc(rc, "bo_sync_from_device")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn sync_from_device(&self, _offset: usize, _bytes: usize) -> Result<(), AieError> {
        Err(AieError::NotYetWired("bo_sync_from_device"))
    }

    #[cfg(feature = "real-npu")]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_bo {
        self.inner
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(dead_code)]
    pub(crate) fn raw(&self) -> *mut sys::xrt_shim_bo {
        self.inner
    }

    /// Length in bytes of the allocation.
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }
}

impl Drop for Bo {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_bo_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Bo {}
unsafe impl Sync for Bo {}

// ---------------------------------------------------------------------------
// Run — dispatched kernel invocation
// ---------------------------------------------------------------------------

/// Dispatched kernel invocation. RAII: destroyed on Drop.
///
/// Build by calling [`Run::create`] from a [`Kernel`], bind args via
/// `set_arg_*`, then `start()` + `wait()`.
#[derive(Debug)]
pub struct Run {
    #[allow(dead_code)]
    inner: *mut sys::xrt_shim_run,
}

impl Run {
    /// Construct a new run from a kernel. Args bind via `set_arg_*`.
    #[cfg(feature = "real-npu")]
    pub fn create(k: &Kernel) -> Result<Self, AieError> {
        let mut out: *mut sys::xrt_shim_run = ptr::null_mut();
        let rc = unsafe { sys::xrt_shim_run_create(k.raw(), &mut out) };
        map_rc(rc, "run_create")?;
        Ok(Self { inner: out })
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn create(_k: &Kernel) -> Result<Self, AieError> {
        Err(AieError::NotYetWired("run_create"))
    }

    /// Bind a BO to arg slot `idx`.
    #[cfg(feature = "real-npu")]
    pub fn set_arg_bo(&self, idx: u32, bo: &Bo) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_set_arg_bo(self.inner, idx as c_uint, bo.raw()) };
        map_rc(rc, "run_set_arg_bo")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn set_arg_bo(&self, _idx: u32, _bo: &Bo) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_set_arg_bo"))
    }

    /// Bind a u32 scalar to arg slot `idx`.
    #[cfg(feature = "real-npu")]
    pub fn set_arg_u32(&self, idx: u32, val: u32) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_set_arg_u32(self.inner, idx as c_uint, val) };
        map_rc(rc, "run_set_arg_u32")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn set_arg_u32(&self, _idx: u32, _val: u32) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_set_arg_u32"))
    }

    /// Bind an i32 scalar to arg slot `idx`.
    #[cfg(feature = "real-npu")]
    pub fn set_arg_i32(&self, idx: u32, val: i32) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_set_arg_i32(self.inner, idx as c_uint, val) };
        map_rc(rc, "run_set_arg_i32")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn set_arg_i32(&self, _idx: u32, _val: i32) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_set_arg_i32"))
    }

    /// Bind an f32 scalar to arg slot `idx`.
    #[cfg(feature = "real-npu")]
    pub fn set_arg_f32(&self, idx: u32, val: f32) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_set_arg_f32(self.inner, idx as c_uint, val) };
        map_rc(rc, "run_set_arg_f32")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn set_arg_f32(&self, _idx: u32, _val: f32) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_set_arg_f32"))
    }

    /// Start the run asynchronously. Must `wait()` before reading
    /// output BOs.
    #[cfg(feature = "real-npu")]
    pub fn start(&self) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_start(self.inner) };
        map_rc(rc, "run_start")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn start(&self) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_start"))
    }

    /// Block until the run completes. `timeout_ms = 0` = infinite.
    #[cfg(feature = "real-npu")]
    pub fn wait(&self, timeout_ms: u32) -> Result<(), AieError> {
        let rc = unsafe { sys::xrt_shim_run_wait(self.inner, timeout_ms as c_uint) };
        map_rc(rc, "run_wait")
    }

    #[cfg(not(feature = "real-npu"))]
    #[allow(missing_docs)]
    pub fn wait(&self, _timeout_ms: u32) -> Result<(), AieError> {
        Err(AieError::NotYetWired("run_wait"))
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        #[cfg(feature = "real-npu")]
        if !self.inner.is_null() {
            unsafe {
                let _ = sys::xrt_shim_run_destroy(self.inner);
            }
        }
    }
}

unsafe impl Send for Run {}
unsafe impl Sync for Run {}

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
