// bonsai_q1_gemv_soa1.hip — 1-row/CTA SoA Q1_0 GEMV for gfx1151.
// Same occupancy as original (128 threads/CTA, 1 row/CTA) but SoA layout
// gives coalesced 128-byte cache-line reads.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rocm_cpp/bonsai.h"

namespace {
constexpr int kBlockSize    = 128;
constexpr int kWaveSize     = 32;
constexpr int kWaves        = kBlockSize / kWaveSize;
constexpr int kWeightsPerB  = 128;
constexpr int kQ1BlockBytes = 18;
} // namespace

__global__
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
void bonsai_q1_gemv_soa1_kernel(
    const uint8_t* __restrict__ packed_soa,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int row  = blockIdx.x;
    if (row >= N_out) return;

    const int nb  = K_in >> 7;
    const size_t row_stride = (size_t)nb * (size_t)kQ1BlockBytes;  // same total as AoS

    // SoA pointers for this row
    const __half*  d_row = reinterpret_cast<const __half*>(packed_soa + (size_t)row * row_stride);
    const uint8_t* qs_row = packed_soa + (size_t)row * row_stride + (size_t)nb * 2;

    __shared__ __half x_lds[kWeightsPerB];
    __shared__ float  wave_sums[kWaves];

    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();

        // SoA: d[b] is contiguous (2 bytes each)
        const float d = __half2float(d_row[b]);

        // SoA: qs starts at offset nb*2 from row start
        // qs[b * 16 + byte_idx]
        const int byte_idx = tid >> 3;
        const int bit_idx  = tid & 7;
        const uint32_t qbyte = (uint32_t)qs_row[b * 16 + byte_idx];
        const uint32_t bit   = (qbyte >> bit_idx) & 1u;
        const float sign = bit ? 1.0f : -1.0f;

        const float a = __half2float(x_lds[tid]);
        acc = fmaf(sign * d, a, acc);
    }

    // Intra-wave reduction
    #pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1)
        acc += __shfl_xor(acc, off);

    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();

    if (wave == 0) {
        float v = (lane < kWaves) ? wave_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1)
            v += __shfl_xor(v, off);
        if (lane == 0) {
            constexpr float FP16_MAX = 65504.0f;
            if (v >  FP16_MAX) v =  FP16_MAX;
            if (v < -FP16_MAX) v = -FP16_MAX;
            out[row] = __float2half(v);
        }
    }
}

extern "C" void bonsai_q1_gemv_soa1_launch(
    const uint8_t*  packed_weights_soa,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0) return;
    if ((K_in & 127) != 0) return;

    dim3 grid(N_out, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(bonsai_q1_gemv_soa1_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights_soa,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}

// Same converter as before - AoS to SoA
extern "C" void bonsai_q1_convert_aos_to_soa(
    const uint8_t* packed_aos,
    uint8_t**      packed_soa,
    int N_out, int K_in)
{
    if (N_out <= 0 || K_in <= 0 || (K_in & 127) != 0) return;

    int nb = K_in >> 7;
    size_t row_bytes = (size_t)nb * kQ1BlockBytes;
    size_t total = (size_t)N_out * row_bytes;

    uint8_t* d_soa = nullptr;
    hipMalloc(&d_soa, total);
    *packed_soa = d_soa;

    std::vector<uint8_t> h_aos(total);
    std::vector<uint8_t> h_soa(total);
    hipMemcpy(h_aos.data(), packed_aos, total, hipMemcpyDeviceToHost);

    for (int r = 0; r < N_out; ++r) {
        const uint8_t* src = h_aos.data() + (size_t)r * row_bytes;
        uint8_t* dst = h_soa.data() + (size_t)r * row_bytes;
        // d values: first 2 bytes of each block
        for (int b = 0; b < nb; ++b) {
            dst[b*2] = src[b*kQ1BlockBytes];
            dst[b*2+1] = src[b*kQ1BlockBytes+1];
        }
        // qs values: bytes 2-17 of each block
        size_t d_off = (size_t)nb * 2;
        for (int b = 0; b < nb; ++b)
            for (int j = 0; j < 16; ++j)
                dst[d_off + b*16 + j] = src[b*kQ1BlockBytes + 2 + j];
    }
    hipMemcpy(d_soa, h_soa.data(), total, hipMemcpyHostToDevice);
}
