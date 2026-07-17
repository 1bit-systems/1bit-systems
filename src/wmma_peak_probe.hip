// gfx1151 WMMA throughput ceiling probe — no global memory on hot path.
// Each wave loads A and B fragments from tiny global init once, then spins
// for N_INNER WMMA iterations using register-resident fragments. Timed.
// Gives the hardware ceiling for f32_16x16x16_f16_w32 on Strix Halo.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half16_t = __attribute__((ext_vector_type(16))) __fp16;
using float8_t = __attribute__((ext_vector_type(8))) float;
using int32x8_t = __attribute__((ext_vector_type(8))) int32_t;
using int8x16_t = __attribute__((ext_vector_type(16))) int8_t;

namespace {

// 16 accumulators (like Phase 4g/h), pure compute in inner loop.
__global__ __launch_bounds__(32)
void wmma_peak_inner(
    const __half* __restrict__ A_seed,
    const __half* __restrict__ B_seed,
    float* __restrict__ sink,
    int n_inner)
{
    half16_t A_frag[4];
    half16_t B_frag[4];
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            A_frag[i][e] = (__fp16)(float)A_seed[(threadIdx.x * 4 + i) * 16 + e];
            B_frag[i][e] = (__fp16)(float)B_seed[(threadIdx.x * 4 + i) * 16 + e];
        }
    }

    float8_t acc[4][4] = {};
    for(int it = 0; it < n_inner; ++it) {
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    A_frag[i], B_frag[j], acc[i][j]);
            }
        }
    }

    // Prevent DCE — write one value per lane from every acc
    float s = 0.0f;
    #pragma unroll
    for(int i = 0; i < 4; ++i)
        #pragma unroll
        for(int j = 0; j < 4; ++j)
            s += acc[i][j][0];
    sink[blockIdx.x * 32 + threadIdx.x] = s;
}

// INT8 WMMA peak — same geometry, i32_16x16x16_iu8_w32 builtin.
// This is the Phase 5 compute path (A quantized to INT8, B = ternary as INT8).
__global__ __launch_bounds__(32)
void wmma_peak_inner_i8(
    const int8_t* __restrict__ A_seed,
    const int8_t* __restrict__ B_seed,
    int32_t* __restrict__ sink,
    int n_inner)
{
    int8x16_t A_frag[4];
    int8x16_t B_frag[4];
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        #pragma unroll
        for(int e = 0; e < 16; ++e) {
            A_frag[i][e] = A_seed[(threadIdx.x * 4 + i) * 16 + e];
            B_frag[i][e] = B_seed[(threadIdx.x * 4 + i) * 16 + e];
        }
    }

    int32x8_t acc[4][4] = {};
    for(int it = 0; it < n_inner; ++it) {
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            #pragma unroll
            for(int j = 0; j < 4; ++j) {
                // signed x signed = int32, no clamp
                acc[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    /*neg_a=*/true, A_frag[i],
                    /*neg_b=*/true, B_frag[j],
                    acc[i][j], /*clamp=*/false);
            }
        }
    }

    int32_t s = 0;
    #pragma unroll
    for(int i = 0; i < 4; ++i)
        #pragma unroll
        for(int j = 0; j < 4; ++j)
            s += acc[i][j][0];
    sink[blockIdx.x * 32 + threadIdx.x] = s;
}

}  // namespace

extern "C" void rcpp_peak_probe_launch(
    const void* A_seed, const void* B_seed, void* sink,
    int n_blocks, int n_inner, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL(wmma_peak_inner,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(A_seed),
                       static_cast<const __half*>(B_seed),
                       static_cast<float*>(sink),
                       n_inner);
}

extern "C" void rcpp_peak_probe_launch_i8(
    const void* A_seed, const void* B_seed, void* sink,
    int n_blocks, int n_inner, void* stream)
{
    dim3 block(32, 1, 1);
    dim3 grid(n_blocks, 1, 1);
    hipLaunchKernelGGL(wmma_peak_inner_i8,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const int8_t*>(A_seed),
                       static_cast<const int8_t*>(B_seed),
                       static_cast<int32_t*>(sink),
                       n_inner);
}
