// xrt_c_shim.cpp — STUB implementation.
//
// Every function currently returns XRT_SHIM_NOT_IMPLEMENTED. The purpose
// of this file is to compile cleanly and be linked by Rust's build.rs
// so the shape of the library + ABI is exercised. Kernel-author fills
// in the real bodies per `docs/wiki/NPU-Kernel-Handoff.md` §"FFI layer"
// (target ~400 LOC of xrt:: calls).
//
// When filling in:
//   * Include <xrt/xrt_device.h>, <xrt/xrt_xclbin.h>, <xrt/xrt_hw_context.h>,
//     <xrt/xrt_kernel.h>, <xrt/xrt_bo.h>.
//   * Wrap xrt::device / xrt::xclbin / xrt::hw_context / xrt::kernel /
//     xrt::bo / xrt::run in std::unique_ptr or a pair of
//     new-into-handle / delete-on-destroy.
//   * xrt:: throws std::runtime_error on failure — catch in each shim
//     function, stash the what() string in thread-local storage, return
//     the matching XRT_SHIM_* error code.

#include "xrt_c_shim.h"

#include <cstring>
#include <string>

namespace {
// Thread-local last-error string. C callers fetch via xrt_shim_last_error_cstr().
thread_local std::string g_last_error;
}  // namespace

extern "C" {

int xrt_shim_device_open(unsigned int /*index*/, xrt_shim_device_t* /*out*/) {
  g_last_error = "xrt_shim_device_open: not implemented";
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_device_close(xrt_shim_device_t /*dev*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_device_name(xrt_shim_device_t /*dev*/, char* /*buf*/, size_t /*cap*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_xclbin_load(xrt_shim_device_t /*dev*/, const char* /*path*/, xrt_shim_xclbin_t* /*out*/) {
  g_last_error = "xrt_shim_xclbin_load: not implemented";
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_xclbin_destroy(xrt_shim_xclbin_t /*xb*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_hw_context_create(xrt_shim_device_t /*dev*/, xrt_shim_xclbin_t /*xb*/, xrt_shim_hw_context_t* /*out*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_hw_context_destroy(xrt_shim_hw_context_t /*hw*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_kernel_open(xrt_shim_hw_context_t /*hw*/, const char* /*name*/, xrt_shim_kernel_t* /*out*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_kernel_close(xrt_shim_kernel_t /*k*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_bo_alloc(xrt_shim_device_t /*dev*/, size_t /*bytes*/, unsigned int /*flags*/, xrt_shim_bo_t* /*out*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_bo_destroy(xrt_shim_bo_t /*bo*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_bo_map(xrt_shim_bo_t /*bo*/, void** /*out_ptr*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_bo_sync_to_device(xrt_shim_bo_t /*bo*/, size_t /*offset*/, size_t /*bytes*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_bo_sync_from_device(xrt_shim_bo_t /*bo*/, size_t /*offset*/, size_t /*bytes*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_create(xrt_shim_kernel_t /*k*/, xrt_shim_run_t* /*out*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_destroy(xrt_shim_run_t /*r*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_set_arg_bo(xrt_shim_run_t /*r*/, unsigned int /*idx*/, xrt_shim_bo_t /*bo*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_set_arg_u32(xrt_shim_run_t /*r*/, unsigned int /*idx*/, uint32_t /*val*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_set_arg_i32(xrt_shim_run_t /*r*/, unsigned int /*idx*/, int32_t /*val*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_set_arg_f32(xrt_shim_run_t /*r*/, unsigned int /*idx*/, float /*val*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_start(xrt_shim_run_t /*r*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

int xrt_shim_run_wait(xrt_shim_run_t /*r*/, unsigned int /*timeout_ms*/) {
  return XRT_SHIM_NOT_IMPLEMENTED;
}

const char* xrt_shim_last_error_cstr(void) {
  return g_last_error.c_str();
}

}  // extern "C"
