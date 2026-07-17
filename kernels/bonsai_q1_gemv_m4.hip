// bonsai_q1_gemv_m4.hip — Batched M=4 Q1_0 SoA GEMV for spec-decode verify.
//
// Processes M=4 tokens per weight load → 4× bandwidth efficiency.
// Each block: 1 output row × M=4 tokens, 128 threads.
// Each thread: 4 fp32 accumulators (one per token).
// LDS: 4 × 128 = 512 fp16 activations = 1 KB.
//
// For a full model forward pass with M=4:
//   Weights read once per layer instead of 4× = 4× effective BW.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr int kBlockSize    = 128;
constexpr int kWaveSize     = 32;
constexpr int kWaves        = kBlockSize / kWaveSize;
constexpr int kWeightsPerB  = 128;
constexpr int kQ1BlockBytes = 18;
constexpr int kM = 4;  // batch size
} // namespace

__global__
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
void q1_gemv_soa_m4_kernel(
    const uint8_t* __restrict__ packed_soa,  // SoA weights [N, nb*18]
    const __half*  __restrict__ acts,        // [M, K] = [4, K], row-major
    __half*        __restrict__ out,         // [M, N] = [4, N], row-major
    int N, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int row  = blockIdx.x;
    if (row >= N) return;

    const int nb = K >> 7;
    const size_t row_stride = (size_t)nb * kQ1BlockBytes;

    // SoA pointers
    const __half*  d_row = reinterpret_cast<const __half*>(
        packed_soa + (size_t)row * row_stride);
    const uint8_t* qs_row = packed_soa + (size_t)row * row_stride + (size_t)nb * 2;

    // LDS: M activation vectors × 128 weights each
    __shared__ __half acts_lds[kM][kWeightsPerB];
    // Reduction scratch: M × kWaves
    __shared__ float wave_sums[kM][kWaves];

    // Per-token accumulators
    float acc[kM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nb; ++b) {
        // ── Load M activation vectors into LDS ──
        // Each thread loads M values (one from each token) at position `tid`
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < kM; ++m) {
            acts_lds[m][tid] = acts[(size_t)m * K + (size_t)(b * kWeightsPerB + tid)];
        }
        __syncthreads();

        // ── Read weight: d[b] + qs[b*16 + byte_idx] for this row ──
        const float d = __half2float(d_row[b]);
        const int byte_idx = tid >> 3;
        const int bit_idx  = tid & 7;
        const uint8_t qbyte = qs_row[b * 16 + byte_idx];
        const float sign = ((qbyte >> bit_idx) & 1u) ? 1.0f : -1.0f;

        // ── Accumulate for all M tokens ──
        // Weight value `sign * d` is shared across all M tokens.
        // Only activation differs per token.
        #pragma unroll
        for (int m = 0; m < kM; ++m) {
            const float a = __half2float(acts_lds[m][tid]);
            acc[m] = fmaf(sign * d, a, acc[m]);
        }
    }

    // ── Reduce per-token ──
    #pragma unroll
    for (int m = 0; m < kM; ++m) {
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1)
            acc[m] += __shfl_xor(acc[m], off);
        if (lane == 0) wave_sums[m][wave] = acc[m];
    }
    __syncthreads();

    // Wave-0 folds across tokens
    if (wave == 0) {
        #pragma unroll
        for (int m = 0; m < kM; ++m) {
            float v = (lane < kWaves) ? wave_sums[m][lane] : 0.0f;
            #pragma unroll
            for (int off = kWaveSize / 2; off > 0; off >>= 1)
                v += __shfl_xor(v, off);
            if (lane == 0) {
                constexpr float FP16_MAX = 65504.0f;
                if (v >  FP16_MAX) v =  FP16_MAX;
                if (v < -FP16_MAX) v = -FP16_MAX;
                out[(size_t)m * N + row] = __float2half(v);
            }
        }
    }
}

// ── Host launcher ──
extern "C" void bonsai_q1_gemv_m4_launch(
    const uint8_t*  packed_weights_soa,
    const uint16_t* act_fp16,   // [4, K] row-major
    uint16_t*       out_fp16,   // [4, N] row-major
    int             N,
    int             K,
    void*           stream)
{
    if (N <= 0 || K <= 0) return;
    if (K & 127) return;

    dim3 grid(N, 1, 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(q1_gemv_soa_m4_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_weights_soa,
                       reinterpret_cast<const __half*>(act_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       N, K);
}

// ── Benchmark wrapper ──
#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

int main() {
    printf("═══ Batched M=4 SoA Q1_0 GEMV Benchmark ═══\n\n");

    struct { int rows, cols; const char* name; int count; } shapes[] = {
        {2048, 2048, "Q", 1}, {1024, 2048, "K", 1}, {1024, 2048, "V", 1},
        {2048, 2048, "O", 1}, {6144, 2048, "Gate", 1}, {6144, 2048, "Up", 1},
        {2048, 6144, "Down", 1}
    };
    const int NS = 7;

    double total_ms_m4 = 0;
    size_t total_w = 0;

    for (int s = 0; s < NS; ++s) {
        int N = shapes[s].rows, K = shapes[s].cols;
        size_t bytes = (size_t)N * (size_t)(K/128) * kQ1BlockBytes;

        // Generate SoA weights
        std::vector<uint8_t> h_aos(bytes);
        for (size_t i = 0; i < bytes; ++i) h_aos[i] = (uint8_t)(i*0x9E3779B9u);
        std::vector<uint8_t> h_soa(bytes);
        for (int r = 0; r < N; ++r) {
            const uint8_t* src = h_aos.data() + (size_t)r * (size_t)(K/128) * (size_t)kQ1BlockBytes;
            uint8_t* dst = h_soa.data() + (size_t)r * (size_t)(K/128) * (size_t)kQ1BlockBytes;
            int nb = K/128;
            for (int b = 0; b < nb; ++b) {
                dst[b*2] = src[b*kQ1BlockBytes]; dst[b*2+1] = src[b*kQ1BlockBytes+1];
            }
            size_t d_off = (size_t)nb * 2;
            for (int b = 0; b < nb; ++b)
                for (int j = 0; j < 16; ++j)
                    dst[d_off + b*16 + j] = src[b*kQ1BlockBytes + 2 + j];
        }

        uint8_t *d_w = nullptr;
        uint16_t *d_act = nullptr, *d_out = nullptr;
        HIP_OK(hipMalloc(&d_w, bytes));
        HIP_OK(hipMalloc(&d_act, (size_t)kM * K * 2));
        HIP_OK(hipMalloc(&d_out, (size_t)kM * N * 2));
        HIP_OK(hipMemcpy(d_w, h_soa.data(), bytes, hipMemcpyHostToDevice));
        HIP_OK(hipMemset(d_act, 0, (size_t)kM * K * 2));
        HIP_OK(hipMemset(d_out, 0, (size_t)kM * N * 2));
        HIP_OK(hipDeviceSynchronize());

        hipStream_t st;
        HIP_OK(hipStreamCreate(&st));

        // Warmup
        hipLaunchKernelGGL(q1_gemv_soa_m4_kernel, dim3(N,1), dim3(128,1), 0, st,
            d_w, (__half*)d_act, (__half*)d_out, N, K);
        HIP_OK(hipDeviceSynchronize());

        const int NR = 200;
        hipEvent_t t0, t1;
        HIP_OK(hipEventCreate(&t0));
        HIP_OK(hipEventCreate(&t1));

        HIP_OK(hipEventRecord(t0, st));
        for (int r = 0; r < NR; ++r)
            hipLaunchKernelGGL(q1_gemv_soa_m4_kernel, dim3(N,1), dim3(128,1), 0, st,
                d_w, (__half*)d_act, (__half*)d_out, N, K);
        HIP_OK(hipEventRecord(t1, st));
        HIP_OK(hipEventSynchronize(t1));

        float ms;
        HIP_OK(hipEventElapsedTime(&ms, t0, t1));
        double per_call = ms / (double)NR;
        double bw = (double)bytes / (per_call/1000) / 1e9;
        double tok_s_m4 = (double)kM * 1000.0 / per_call;

        printf("  %-20s %6.3f ms  %5.0f GB/s  %6.0f M4-GEMV/s  (%.0f tok/s)\n",
               shapes[s].name, per_call, bw, 1000.0/per_call, tok_s_m4);

        total_ms_m4 += per_call * shapes[s].count * 28;
        total_w += bytes * shapes[s].count * 28;

        hipFree(d_w); hipFree(d_act); hipFree(d_out);
        HIP_OK(hipEventDestroy(t0)); HIP_OK(hipEventDestroy(t1));
        HIP_OK(hipStreamDestroy(st));
    }

    printf("\n─── Full Model (28 layers, M=4 batch verify) ───\n");
    double gbps = total_w / (total_ms_m4/1000) / 1e9;
    double tok_s = kM * 1000.0 / total_ms_m4;
    printf("  Total GEMV time: %.2f ms\n", total_ms_m4);
    printf("  Effective BW:    %.0f GB/s (%.0f%% of 273)\n", gbps, gbps/273.0*100.0);
    printf("  M=4 verification: %.0f tok/s\n", tok_s);
    printf("  + norms+attn+lm (~3ms): %.0f tok/s\n", kM*1000.0/(total_ms_m4+3.0));

    // Compare to M=1 (from earlier benchmark)
    printf("\n─── Comparison ───\n");
    printf("  M=1 (SoA, per-token): 121 tok/s\n");
    printf("  M=4 (batched verify): %.0f tok/s (%.1fx)\n", kM*1000.0/(total_ms_m4+3.0),
           kM*1000.0/(total_ms_m4+3.0) / 121.0);
    printf("  Target 572 tok/s: %s\n",
           (kM*1000.0/(total_ms_m4+3.0) >= 572.0) ? "✅ EXCEEDED" : "❌ Not yet");

    return 0;
}
