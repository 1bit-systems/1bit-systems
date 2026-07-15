#include <cstdio>
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// bonsai_q1_gemv_soa.hip — SoA Q1_0 GEMV, 4 rows per CTA, gfx1151 wave32.
//
// SoA layout per row (K/128 blocks × 128 weights):
//   [d[0..nb-1]]    : K/128 × fp16 = K/64 bytes  (contiguous)
//   [qs[0..nb*16-1]]: K/128 × 16B  = K/8 bytes   (contiguous)
// Total per row: K/64 + K/8 = 9K/64 bytes (vs 9K/64 in AoS — same total!)
// But layout is: ALL d values first, then ALL qs values, per row.
//
// For a matrix with N rows:
//   d_area:  N × (K/128) × 2B  (all rows' d values, row-major)
//   qs_area: N × (K/128) × 16B (all rows' qs values, row-major)
//
// 4 rows per CTA (512 threads), each row gets 128 threads.
// Memory reads are coalesced: each CTA reads d for all 4 rows (contiguous),
// then qs for all 4 rows (contiguous).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rocm_cpp/bonsai.h"

namespace {

constexpr int kRowsPerCTA   = 4;     // number of output rows per CTA
constexpr int kBlockSize    = 128;   // threads per row
constexpr int kCTAThreads   = kRowsPerCTA * kBlockSize;  // 512
constexpr int kWaveSize     = 32;
constexpr int kWavesPerRow  = kBlockSize / kWaveSize;    // 4
constexpr int kWeightsPerB  = 128;
constexpr int kQ1BlockBytes = 18;

__device__ __forceinline__
uint32_t load_qs_byte(const uint8_t* qs_ptr, int byte_idx) {
    return (uint32_t)qs_ptr[byte_idx];
}

} // namespace

// SoA weight pointer helper: returns pointer to d and qs for a given row.
// SoA layout per row: nb * 2B d values, then nb * 16B qs values.
// Row stride in SoA: nb * (2 + 16) = nb * 18B (same total as AoS!)
__device__ __forceinline__
void get_row_ptrs_soa(const uint8_t* base, int row, int num_blocks,
                      const __half*& d_row, const uint8_t*& qs_row) {
    size_t row_stride = (size_t)num_blocks * (size_t)kQ1BlockBytes;  // 18 * nb
    size_t off = (size_t)row * row_stride;
    d_row = reinterpret_cast<const __half*>(base + off);
    qs_row = base + off + (size_t)num_blocks * 2;  // skip d area
}

__attribute__((amdgpu_flat_work_group_size(kCTAThreads, kCTAThreads)))
__global__ void bonsai_q1_gemv_soa_kernel(
    const uint8_t* __restrict__ packed_soa,   // SoA layout
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    // CTA workflow
    const int cta_row_base = blockIdx.x * kRowsPerCTA;
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave_in_cta = tid >> 5;

    // Which row within this CTA (0..kRowsPerCTA-1) and which position within the row
    const int row_in_cta = tid / kBlockSize;       // 0..3
    const int tid_in_row = tid % kBlockSize;       // 0..127
    const int wave_in_row = wave_in_cta % kWavesPerRow;
    const int abs_row = cta_row_base + row_in_cta;
    if (abs_row >= N_out) return;

    const int num_blocks  = K_in >> 7;

    // Activation LDS: one 128-weight slab (reused across all 4 rows)
    __shared__ __half x_lds[kWeightsPerB];
    // Wave reduction scratch per row (4 waves × 1 float each)
    __shared__ float  wave_sums[kRowsPerCTA][kWavesPerRow];

    // Per-row accumulator
    float acc = 0.0f;

    // Get pointers to this row's d and qs in SoA layout
    const __half*  d_row;
    const uint8_t* qs_row;
    get_row_ptrs_soa(packed_soa, abs_row, num_blocks, d_row, qs_row);

    // Block loop
    for (int b = 0; b < num_blocks; ++b) {
        // Coop load 128 fp16 acts into LDS
        __syncthreads();
        x_lds[tid_in_row] = act[b * kWeightsPerB + tid_in_row];
        __syncthreads();

        // SoA: read d[b] — all threads in the row read the same value
        // With SoA, d values are contiguous: d_row[b] = next fp16
        const float d = __half2float(d_row[b]);

        // SoA: qs starts at d_row + num_blocks, qs[b*16 + byte_idx]
        const int byte_idx = tid_in_row >> 3;
        const int bit_idx  = tid_in_row & 7;
        const uint32_t qbyte = load_qs_byte(qs_row + b * 16, byte_idx);
        const uint32_t bit   = (qbyte >> bit_idx) & 1u;
        const float sign = bit ? 1.0f : -1.0f;

        const float a = __half2float(x_lds[tid_in_row]);
        acc = fmaf(sign * d, a, acc);
    }

    // ── Reduce within the row (4 waves) ──
    // Intra-wave butterfly
    #pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off);
    }

    // Inter-wave: lane 0 of each wave → LDS slot for this row
    if (lane == 0) wave_sums[row_in_cta][wave_in_row] = acc;
    __syncthreads();

    // Wave 0 of each row folds
    if (wave_in_row == 0) {
        float v = (lane < kWavesPerRow) ? wave_sums[row_in_cta][lane] : 0.0f;
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            constexpr float FP16_MAX = 65504.0f;
            if (v >  FP16_MAX) v =  FP16_MAX;
            if (v < -FP16_MAX) v = -FP16_MAX;
            out[abs_row] = __float2half(v);
        }
    }
}

// ── SoA transposition (host-side, GPU-mapped) ──
// Converts AoS packed (18B per block, block-interleaved) to SoA
// (all d values for a row, then all qs values for a row).
// Returns allocated device memory with SoA layout.
extern "C" void bonsai_q1_convert_aos_to_soa(
    const uint8_t* packed_aos,  // device, AoS layout
    uint8_t**       packed_soa, // device, SoA layout (allocated inside)
    int N_out, int K_in)
{
    if (N_out <= 0 || K_in <= 0 || (K_in & 127) != 0) return;

    int num_blocks = K_in >> 7;
    size_t row_bytes = (size_t)num_blocks * kQ1BlockBytes;  // same for AoS and SoA
    size_t total_bytes = (size_t)N_out * row_bytes;

    // Allocate SoA buffer (same size!)
    uint8_t* d_soa = nullptr;
    HIP_CHECK(hipMalloc(&d_soa, total_bytes));
    *packed_soa = d_soa;

    // Host-side conversion
    std::vector<uint8_t> h_aos(total_bytes);
    std::vector<uint8_t> h_soa(total_bytes);
    HIP_CHECK(hipMemcpy(h_aos.data(), packed_aos, total_bytes, hipMemcpyDeviceToHost));

    for (int r = 0; r < N_out; ++r) {
        const uint8_t* src_row = h_aos.data() + (size_t)r * row_bytes;
        uint8_t* dst_row = h_soa.data() + (size_t)r * row_bytes;

        // d values: first 2 bytes of each block → contiguous
        for (int b = 0; b < num_blocks; ++b) {
            dst_row[b * 2]     = src_row[b * kQ1BlockBytes];
            dst_row[b * 2 + 1] = src_row[b * kQ1BlockBytes + 1];
        }

        // qs values: bytes 2..17 of each block → contiguous
        size_t d_area = (size_t)num_blocks * 2;
        for (int b = 0; b < num_blocks; ++b) {
            for (int j = 0; j < 16; ++j) {
                dst_row[d_area + b * 16 + j] = src_row[b * kQ1BlockBytes + 2 + j];
            }
        }
    }

    HIP_CHECK(hipMemcpy(d_soa, h_soa.data(), total_bytes, hipMemcpyHostToDevice));
}

extern "C" void bonsai_q1_gemv_soa_launch(
    const uint8_t*  packed_weights_soa,
    const uint16_t* act_fp16,
    uint16_t*       out_fp16,
    int             N_out,
    int             K_in,
    void*           stream)
{
    if (N_out <= 0 || K_in <= 0) return;
    if ((K_in & 127) != 0) return;

    dim3 grid((N_out + kRowsPerCTA - 1) / kRowsPerCTA, 1, 1);
    dim3 block(kCTAThreads, 1, 1);
    hipLaunchKernelGGL(bonsai_q1_gemv_soa_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights_soa,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N_out, K_in);
}
