// medusa_small_m_gemv.hip — small-M ternary GEMV (halo-1bit pack) (SKELETON).
//
// See include/rocm_cpp/medusa.h for the contract this file will eventually
// implement. THIS PASS lands only:
//
//   * The C-linkage entry point declared in medusa.h, returning
//     RCPP_NOT_IMPLEMENTED so callers can link + smoke-test the dispatch
//     without device-side work.
//   * A compile-clean __global__ tile placeholder so the .hip TU lives
//     inside the rocm_cpp HIP source set; the kernel never launches in
//     this pass.
//
// What lands in the next chunk
// ----------------------------
//   * Tile shape: BLOCK_M = up to 16 (clamped to actual M), BLOCK_N = 64,
//     BLOCK_K = 64. 128 threads / block, 4 waves × 32 lanes (wave32).
//     Mirrors src/ternary_gemm_smallm.hip's pinned shape so the differential
//     test harness already in place can be extended rather than rewritten.
//   * Inner loop: v_dot4_i32_iu8 against unpacked halo-1bit codes. The 4
//     ternaries-per-byte unpack is the same routine prim_kernels uses for
//     the M=1 GEMV; promote it to a shared device __forceinline__ rather
//     than duplicating.
//   * Activation staging via LDS so each weight byte is read from HBM once
//     and reused across all M lanes — this is the only reason this kernel
//     is faster than M loop-calls into the M=1 launcher.
//   * Differential test against onebit_ternary_gemm_smallm_scalar_launch's
//     reference walker, after teaching that scalar TU the halo-1bit pack
//     (currently it consumes the K-major u32 pack used by the prefill
//     path; see src/ternary_gemm_smallm_scalar_ref.hip).
//
// gfx1151 / wave32 first-class. Halo-1bit packing matches the M=1 decode
// GEMV layout 1:1 so weight tensors stay shared. No hipBLAS, no Python.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "rocm_cpp/medusa.h"

namespace {

// Placeholder kernel — exists only so the compiler / CI verifies the TU is
// part of the rocm_cpp HIP source set. Never launched in this pass.
// __launch_bounds__(128) mirrors the planned 4-wave BLOCK_M=16 / BLOCK_N=64
// tile this kernel will grow into.
__global__ __launch_bounds__(128)
void medusa_small_m_gemv_stub_kernel(
    const int8_t*  __restrict__ /*x_i8*/,
    float                       /*x_scale*/,
    const uint8_t* __restrict__ /*w_packed_halo*/,
    const float*   __restrict__ /*w_row_scales*/,
    __half*        __restrict__ /*y_fp16_out*/,
    int /*M*/, int /*N*/, int /*K*/)
{
    // Intentionally empty — see file header for the staged plan.
}

}  // namespace

extern "C"
rcpp_status_t
rcpp_medusa_small_m_gemv(const void* /*x_i8_dev*/,
                         float       /*x_scale*/,
                         const void* /*w_packed_halo_dev*/,
                         const void* /*w_row_scales_dev*/,
                         void*       /*y_fp16_out_dev*/,
                         int         /*M*/,
                         int         /*N*/,
                         int         /*K*/,
                         void*       /*stream*/)
{
    (void)&medusa_small_m_gemv_stub_kernel;
    return RCPP_NOT_IMPLEMENTED;
}
