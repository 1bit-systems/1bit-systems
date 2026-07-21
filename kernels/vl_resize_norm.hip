// vl_resize_norm.hip — GPU-accelerated bilinear resize + normalization.
//
// Runs on the Radeon 8060S (gfx1151) via ROCm HIP.
// Keeps the image tensor on GPU — no CPU round-trip.
//
// Kernel: one thread per output pixel (3 channels per thread).
// Memory: coalesced reads from input via texture or global memory.
//
// Upstream tracking: additive file, no existing file modified.

#include <hip/hip_runtime.h>

// ── GPU kernel: bilinear resize + per-channel normalize ──
// Each thread handles one output pixel (all 3 RGB channels).
// Input:  src[sh][sw][3] float [0..1]
// Output: dst[out_h][out_w][3] float normalized
//
// Launch config:
//   dim3 block(16, 16)
//   dim3 grid((out_w + 15) / 16, (out_h + 15) / 16)
//
// Template parameter NCHW: if true, output is in NCHW layout
//   (channels-first: [3][out_h][out_w]). Default false = HWC
//   (channels-last: [out_h][out_w][3]).
template<bool NCHW = false>
__global__ void vl_resize_norm_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int sw, int sh,
    int out_w, int out_h,
    float mean_r, float mean_g, float mean_b,
    float std_r,  float std_g,  float std_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    // Map output pixel center to input coordinates
    float sy = (y + 0.5f) * sh / out_h - 0.5f;
    float sx = (x + 0.5f) * sw / out_w - 0.5f;

    int y0 = (int)__float2int_rd(sy);
    float fy = sy - y0;
    int y1 = min(max(y0 + 1, 0), sh - 1);
    y0 = min(max(y0, 0), sh - 1);

    int x0 = (int)__float2int_rd(sx);
    float fx = sx - x0;
    int x1 = min(max(x0 + 1, 0), sw - 1);
    x0 = min(max(x0, 0), sw - 1);

    const float means[3] = {mean_r, mean_g, mean_b};
    const float stds[3]  = {std_r,  std_g,  std_b};

    for (int c = 0; c < 3; c++) {
        float v00 = src[((size_t)y0 * sw + x0) * 3 + c];
        float v01 = src[((size_t)y0 * sw + x1) * 3 + c];
        float v10 = src[((size_t)y1 * sw + x0) * 3 + c];
        float v11 = src[((size_t)y1 * sw + x1) * 3 + c];

        float v0 = v00 * (1.0f - fx) + v01 * fx;
        float v1 = v10 * (1.0f - fx) + v11 * fx;
        float val = v0 * (1.0f - fy) + v1 * fy;

        val = (val - means[c]) / stds[c];

        if (NCHW) {
            // Channels-first: [c][out_h][out_w]
            dst[(size_t)c * out_h * out_w + (size_t)y * out_w + x] = val;
        } else {
            // Channels-last: [out_h][out_w][c]
            dst[((size_t)y * out_w + x) * 3 + c] = val;
        }
    }
}

// ── Explicit instantiations ──
template __global__ void vl_resize_norm_kernel<false>(
    const float*, float*, int, int, int, int,
    float, float, float, float, float, float);
template __global__ void vl_resize_norm_kernel<true>(
    const float*, float*, int, int, int, int,
    float, float, float, float, float, float);

// ── Host-side launcher ──
// Returns 0 on success, -1 on error.
extern "C" int vl_resize_norm_launch(
    const float* src,    // device pointer, [sh][sw][3] HWC
    float* dst,          // device pointer, [out_h][out_w][3] HWC or [3][out_h][out_w] NCHW
    int sw, int sh,
    int out_w, int out_h,
    const float mean[3],
    const float std[3],
    bool nchw_output,
    hipStream_t stream)
{
    if (!src || !dst || out_w <= 0 || out_h <= 0) return -1;

    dim3 block(16, 16);
    dim3 grid((out_w + 15) / 16, (out_h + 15) / 16);

    if (nchw_output) {
        hipLaunchKernelGGL(vl_resize_norm_kernel<true>,
            grid, block, 0, stream,
            src, dst, sw, sh, out_w, out_h,
            mean[0], mean[1], mean[2],
            std[0], std[1], std[2]);
    } else {
        hipLaunchKernelGGL(vl_resize_norm_kernel<false>,
            grid, block, 0, stream,
            src, dst, sw, sh, out_w, out_h,
            mean[0], mean[1], mean[2],
            std[0], std[1], std[2]);
    }

    return 0;
}
