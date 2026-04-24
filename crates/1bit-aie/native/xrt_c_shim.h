// xrt_c_shim.h
//
// Thin C ABI around Xilinx XRT's C++ API. Rust bindgen consumes this
// header to produce `crates/1bit-aie/src/sys.rs`. The C++ side is
// compiled into libxrt_c_shim.a via `build.rs` and linked when
// --features real-npu is active.
//
// WHY a shim instead of binding the C++ headers directly:
//   * XRT's public API is C++ (namespace xrt, template-heavy).
//   * Rust bindgen on C++ class templates is fragile; cxx / autocxx
//     adds a large dep tree for a handful of calls.
//   * This shim exposes only the ~dozen calls BitNet NPU dispatch
//     needs (device, xclbin, hw_context, kernel, bo, run).
//
// ABI: all handles are opaque pointers. Ownership:
//   * xrt_create_* returns a heap-allocated handle; caller frees with
//     the matching xrt_destroy_*.
//   * All functions return 0 on success, non-zero XRT error code on
//     failure. Error strings fetched via xrt_last_error_cstr().
//
// Scope gate: this header is STUB-ONLY as of 2026-04-24. The real
// bodies live in xrt_c_shim.cpp — each one currently returns
// XRT_SHIM_NOT_IMPLEMENTED. Kernel-author fills these in per
// `docs/wiki/NPU-Kernel-Handoff.md` §"FFI layer".

#ifndef XRT_C_SHIM_H
#define XRT_C_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------
#define XRT_SHIM_OK                      0
#define XRT_SHIM_NOT_IMPLEMENTED         -1
#define XRT_SHIM_DEVICE_OPEN_FAILED      -2
#define XRT_SHIM_XCLBIN_LOAD_FAILED      -3
#define XRT_SHIM_HW_CONTEXT_FAILED       -4
#define XRT_SHIM_KERNEL_NOT_FOUND        -5
#define XRT_SHIM_BO_ALLOC_FAILED         -6
#define XRT_SHIM_BO_SYNC_FAILED          -7
#define XRT_SHIM_RUN_FAILED              -8
#define XRT_SHIM_BAD_HANDLE              -9

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
typedef struct xrt_shim_device*       xrt_shim_device_t;
typedef struct xrt_shim_xclbin*       xrt_shim_xclbin_t;
typedef struct xrt_shim_hw_context*   xrt_shim_hw_context_t;
typedef struct xrt_shim_kernel*       xrt_shim_kernel_t;
typedef struct xrt_shim_bo*           xrt_shim_bo_t;
typedef struct xrt_shim_run*          xrt_shim_run_t;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
int xrt_shim_device_open(unsigned int index, xrt_shim_device_t* out);
int xrt_shim_device_close(xrt_shim_device_t dev);
int xrt_shim_device_name(xrt_shim_device_t dev, char* buf, size_t cap);

// ---------------------------------------------------------------------------
// xclbin
// ---------------------------------------------------------------------------
int xrt_shim_xclbin_load(xrt_shim_device_t dev, const char* path, xrt_shim_xclbin_t* out);
int xrt_shim_xclbin_destroy(xrt_shim_xclbin_t xb);

// ---------------------------------------------------------------------------
// hw_context (multi-xclbin isolation, required on XDNA2)
// ---------------------------------------------------------------------------
int xrt_shim_hw_context_create(xrt_shim_device_t dev, xrt_shim_xclbin_t xb, xrt_shim_hw_context_t* out);
int xrt_shim_hw_context_destroy(xrt_shim_hw_context_t hw);

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------
int xrt_shim_kernel_open(xrt_shim_hw_context_t hw, const char* name, xrt_shim_kernel_t* out);
int xrt_shim_kernel_close(xrt_shim_kernel_t k);

// ---------------------------------------------------------------------------
// bo — buffer object
// ---------------------------------------------------------------------------
// flags: 0 = cacheable host-mapped, 1 = device-only.
int xrt_shim_bo_alloc(xrt_shim_device_t dev, size_t bytes, unsigned int flags, xrt_shim_bo_t* out);
int xrt_shim_bo_destroy(xrt_shim_bo_t bo);

// Host-map. Returns mmap'd pointer for the lifetime of the bo.
int xrt_shim_bo_map(xrt_shim_bo_t bo, void** out_ptr);

// H→D sync. offset+bytes must fit inside the bo.
int xrt_shim_bo_sync_to_device(xrt_shim_bo_t bo, size_t offset, size_t bytes);
// D→H sync.
int xrt_shim_bo_sync_from_device(xrt_shim_bo_t bo, size_t offset, size_t bytes);

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------
// Build a run object from a kernel. Set args with xrt_shim_run_set_arg_*.
int xrt_shim_run_create(xrt_shim_kernel_t k, xrt_shim_run_t* out);
int xrt_shim_run_destroy(xrt_shim_run_t r);

int xrt_shim_run_set_arg_bo(xrt_shim_run_t r, unsigned int idx, xrt_shim_bo_t bo);
int xrt_shim_run_set_arg_u32(xrt_shim_run_t r, unsigned int idx, uint32_t val);
int xrt_shim_run_set_arg_i32(xrt_shim_run_t r, unsigned int idx, int32_t val);
int xrt_shim_run_set_arg_f32(xrt_shim_run_t r, unsigned int idx, float val);

int xrt_shim_run_start(xrt_shim_run_t r);
// Blocking wait with timeout in ms. 0 = infinite.
int xrt_shim_run_wait(xrt_shim_run_t r, unsigned int timeout_ms);

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
// NUL-terminated last error for the current thread. Returns empty string
// if no error. Pointer is valid until the next shim call on this thread.
const char* xrt_shim_last_error_cstr(void);

#ifdef __cplusplus
}
#endif

#endif // XRT_C_SHIM_H
