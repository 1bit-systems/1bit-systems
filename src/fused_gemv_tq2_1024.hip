// fused_gemv_tq2_1024.hip — Fused QKV and Gate/Up GEMVs (TQ2_1024 format)
//
// Instead of 7 separate launches per layer (Q, K, V, O, gate, up, down),
// this fuses groups that share the same activation input:
//   fused_qkv:   Q=W_q*x, K=W_k*x, V=W_v*x   (share x, 3 projections)
//   fused_gu:    gate=W_gate*x, up=W_up*x      (share x, 2 projections)
//
// O and down remain single because they use unique activation inputs
// (attention output and silu(gate)*up respectively).
//
// Launch overhead savings:
//   7 launches/layer → 4 launches/layer = 84 launches saved per token
//   At ~7 µs/launch → ~0.6 ms saved per token (22% on TQ2_1024's 2.79 ms)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#ifndef HIP_CHECK
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)
#endif

namespace {

constexpr int kBlockSize       = 128;
constexpr int kWaveSize        = 32;
constexpr int kWaves           = kBlockSize / kWaveSize;
constexpr int kRowsPerWarp     = 2;
constexpr int kRowsPerWG       = kWaves * kRowsPerWarp;  // 8
constexpr int kWeightsPerB     = 1024;
constexpr int kWeightsPerLane  = kWeightsPerB / kWaveSize; // 32
constexpr int kBlockBytes      = 258;  // TQ2_1024: 256B codes + 2B scale
constexpr int kScaleOffset     = 256;
constexpr int kActLDS          = kWeightsPerB; // 1024 fp16 in LDS

} // namespace

// ── Fused QKV: Q = W_q * x, K = W_k * x, V = W_v * x ───────────────────────
// All weights in TQ2_1024 tile8-interleaved format.
// Grid = max(num_tiles_q, num_tiles_kv) workgroups.
// Each WG handles 8 rows. Workgroups with idx < num_tiles_q compute W_q rows;
// those with idx < num_tiles_kv compute W_k and W_v rows.

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void fused_tq2_1024_qkv_kernel(
    const uint8_t* __restrict__ w_q,
    const uint8_t* __restrict__ w_k,
    const uint8_t* __restrict__ w_v,
    const __half*  __restrict__ act,
    __half*        __restrict__ out_q,
    __half*        __restrict__ out_k,
    __half*        __restrict__ out_v,
    int N_q, int N_kv, int K)
{
    __shared__ __half x_lds[kActLDS];

    const int tid    = threadIdx.x;
    const int lane   = tid & (kWaveSize - 1);
    const int wave   = tid >> 5;
    const int tile   = blockIdx.x;
    const int row_b  = tile * kRowsPerWG;

    const int num_blocks = K >> 10;           // K / 1024
    const int tiles_q    = (N_q + kRowsPerWG - 1) / kRowsPerWG;
    const int tiles_kv   = (N_kv + kRowsPerWG - 1) / kRowsPerWG;
    const size_t tile_blk_stride = (size_t)kRowsPerWG * kBlockBytes; // 2064
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    const bool do_q  = (tile < tiles_q);
    const bool do_kv = (tile < tiles_kv);

    // ── Fused Q + K + V GEMVs ───────────────────────────────────────────
    float acc_q[kRowsPerWarp], acc_k[kRowsPerWarp], acc_v[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r)
        acc_q[r] = acc_k[r] = acc_v[r] = 0.0f;

    const int codes_lane_off = lane * 8; // 32 codes × 2-bit = 8 bytes

    for (int b = 0; b < num_blocks; ++b) {
        // Cooperative load into LDS (128 threads × 8 fp16 = 1024 activations)
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            x_lds[tid * 8 + t] = act[b * kWeightsPerB + tid * 8 + t];
        __syncthreads();

        const int act_base = lane * kWeightsPerLane;

        // Pre-load activations for this lane (32 fp16)
        float a[kWeightsPerLane];
        #pragma unroll
        for (int j = 0; j < kWeightsPerLane; ++j)
            a[j] = __half2float(x_lds[act_base + j]);

        // Read weight blocks for Q, K, V (same tile, same block)
        auto read_row = [&](const uint8_t* __restrict__ base, int row_in_wg) -> float {
            const uint8_t* blk = base + (size_t)row_in_wg * kBlockBytes;
            const float d = __half2float(
                *reinterpret_cast<const __half*>(blk + kScaleOffset));
            uint64_t codes;
            __builtin_memcpy(&codes, blk + codes_lane_off, sizeof(codes));
            float p = 0.0f;
            #pragma unroll
            for (int j = 0; j < kWeightsPerLane; ++j) {
                const uint32_t code = (uint32_t)((codes >> (j * 2)) & 3);
                const float val = static_cast<float>(static_cast<int>(code)) - 1.0f;
                p = fmaf(val, a[j], p);
            }
            return d * p;
        };

        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row_in_wg = wave * kRowsPerWarp + r;

            if (do_q) {
                const int row = row_b + wave * kRowsPerWarp + r;
                if (row < N_q) {
                    const uint8_t* base = w_q + (size_t)tile * tile_stride
                                        + (size_t)b * tile_blk_stride;
                    acc_q[r] += read_row(base, row_in_wg);
                }
            }
            if (do_kv) {
                const int row_k = row_b + wave * kRowsPerWarp + r;
                if (row_k < N_kv) {
                    const uint8_t* base_k = w_k + (size_t)tile * tile_stride
                                           + (size_t)b * tile_blk_stride;
                    const uint8_t* base_v = w_v + (size_t)tile * tile_stride
                                           + (size_t)b * tile_blk_stride;
                    acc_k[r] += read_row(base_k, row_in_wg);
                    acc_v[r] += read_row(base_v, row_in_wg);
                }
            }
        }
    }

    // ── Reduction + write ────────────────────────────────────────────────
    auto reduce_write = [&](float* acc, __half* out, int N, int row_base) {
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            float v = acc[r];
            #pragma unroll
            for (int off = kWaveSize / 2; off > 0; off >>= 1)
                v += __shfl_xor_sync(0xffffffffULL, v, off);
            if (lane == 0) {
                const int row = row_base + wave * kRowsPerWarp + r;
                if (row < N) {
                    constexpr float FP16_MAX = 65504.0f;
                    if (v >  FP16_MAX) v =  FP16_MAX;
                    if (v < -FP16_MAX) v = -FP16_MAX;
                    out[row] = __float2half(v);
                }
            }
        }
    };

    if (do_q)   reduce_write(acc_q, out_q, N_q, row_b);
    if (do_kv)  reduce_write(acc_k, out_k, N_kv, row_b);
    if (do_kv)  reduce_write(acc_v, out_v, N_kv, row_b);
}

// ── Fused Gate/Up: gate = W_gate * x, up = W_up * x ────────────────────────
// Same activation reuse pattern as QKV above. Grid = tiles_gs workgroups.

__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
__global__ void fused_tq2_1024_gu_kernel(
    const uint8_t* __restrict__ w_gate,
    const uint8_t* __restrict__ w_up,
    const __half*  __restrict__ act,
    __half*        __restrict__ out_gate,
    __half*        __restrict__ out_up,
    int N_gs, int K)
{
    __shared__ __half x_lds[kActLDS];

    const int tid    = threadIdx.x;
    const int lane   = tid & (kWaveSize - 1);
    const int wave   = tid >> 5;
    const int tile   = blockIdx.x;
    const int row_b  = tile * kRowsPerWG;

    const int num_blocks = K >> 10;
    const int tiles_gs   = (N_gs + kRowsPerWG - 1) / kRowsPerWG;
    const size_t tile_blk_stride = (size_t)kRowsPerWG * kBlockBytes;
    const size_t tile_stride     = (size_t)num_blocks * tile_blk_stride;

    if (tile >= tiles_gs) return;

    float acc_g[kRowsPerWarp], acc_u[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) acc_g[r] = acc_u[r] = 0.0f;

    const int codes_lane_off = lane * 8;

    for (int b = 0; b < num_blocks; ++b) {
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            x_lds[tid * 8 + t] = act[b * kWeightsPerB + tid * 8 + t];
        __syncthreads();

        const int act_base = lane * kWeightsPerLane;
        float a[kWeightsPerLane];
        #pragma unroll
        for (int j = 0; j < kWeightsPerLane; ++j)
            a[j] = __half2float(x_lds[act_base + j]);

        auto read_row = [&](const uint8_t* __restrict__ base, int row_in_wg) -> float {
            const uint8_t* blk = base + (size_t)row_in_wg * kBlockBytes;
            const float d = __half2float(
                *reinterpret_cast<const __half*>(blk + kScaleOffset));
            uint64_t codes;
            __builtin_memcpy(&codes, blk + codes_lane_off, sizeof(codes));
            float p = 0.0f;
            #pragma unroll
            for (int j = 0; j < kWeightsPerLane; ++j) {
                const uint32_t code = (uint32_t)((codes >> (j * 2)) & 3);
                const float val = static_cast<float>(static_cast<int>(code)) - 1.0f;
                p = fmaf(val, a[j], p);
            }
            return d * p;
        };

        for (int r = 0; r < kRowsPerWarp; ++r) {
            const int row_in_wg = wave * kRowsPerWarp + r;
            const int row = row_b + wave * kRowsPerWarp + r;
            if (row < N_gs) {
                const uint8_t* base_g = w_gate + (size_t)tile * tile_stride
                                       + (size_t)b * tile_blk_stride;
                const uint8_t* base_u = w_up   + (size_t)tile * tile_stride
                                       + (size_t)b * tile_blk_stride;
                acc_g[r] += read_row(base_g, row_in_wg);
                acc_u[r] += read_row(base_u, row_in_wg);
            }
        }
    }

    // Reduction + write
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float vg = acc_g[r], vu = acc_u[r];
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) {
            vg += __shfl_xor_sync(0xffffffffULL, vg, off);
            vu += __shfl_xor_sync(0xffffffffULL, vu, off);
        }
        if (lane == 0) {
            const int row = row_b + wave * kRowsPerWarp + r;
            if (row < N_gs) {
                constexpr float F16M = 65504.0f;
                if (vg >  F16M) vg =  F16M; if (vg < -F16M) vg = -F16M;
                if (vu >  F16M) vu =  F16M; if (vu < -F16M) vu = -F16M;
                out_gate[row] = __float2half(vg);
                out_up[row]   = __float2half(vu);
            }
        }
    }
}

// ── C API ───────────────────────────────────────────────────────────────────

extern "C" void fused_tq2_1024_qkv_launch(
    const uint8_t* w_q, const uint8_t* w_k, const uint8_t* w_v,
    const uint16_t* act,
    uint16_t* out_q, uint16_t* out_k, uint16_t* out_v,
    int N_q, int N_kv, int K,
    void* stream)
{
    if (K <= 0 || (K & 1023) != 0) return;
    const int tiles = max((N_q + 7) / 8, (N_kv + 7) / 8);
    dim3 grid(tiles, 1, 1); dim3 block(128, 1, 1);
    hipLaunchKernelGGL(fused_tq2_1024_qkv_kernel, grid, block, 0, (hipStream_t)stream,
        w_q, w_k, w_v,
        reinterpret_cast<const __half*>(act),
        reinterpret_cast<__half*>(out_q),
        reinterpret_cast<__half*>(out_k),
        reinterpret_cast<__half*>(out_v),
        N_q, N_kv, K);
    HIP_CHECK(hipGetLastError());
}

extern "C" void fused_tq2_1024_gu_launch(
    const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* act,
    uint16_t* out_gate, uint16_t* out_up,
    int N_gs, int K,
    void* stream)
{
    if (K <= 0 || (K & 1023) != 0) return;
    const int tiles = (N_gs + 7) / 8;
    dim3 grid(tiles, 1, 1); dim3 block(128, 1, 1);
    hipLaunchKernelGGL(fused_tq2_1024_gu_kernel, grid, block, 0, (hipStream_t)stream,
        w_gate, w_up,
        reinterpret_cast<const __half*>(act),
        reinterpret_cast<__half*>(out_gate),
        reinterpret_cast<__half*>(out_up),
        N_gs, K);
    HIP_CHECK(hipGetLastError());
}
