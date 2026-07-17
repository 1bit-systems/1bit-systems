// bonsai_q1_gemm_m8.hip — Tiled M=8 Q1_0 GEMM, 1-column × 8-rows per CTA.
//
// Each CTA: 1 N-column, 8 M-rows (tokens), 128 threads (4 waves × 32).
// Each thread handles 1 weight position (128 threads × 1 = 128 positions).
// Weight value loaded ONCE, reused across all 8 M-rows.
// 8 fp32 accumulators per thread (one per M-row).
//
// Grid: N_out columns × 1 × 1 (2048 CTAs for Q, 6144 for Gate)
//
// Expected BW: 8× reuse → ~100+ GB/s → 500+ tok/s

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace {
constexpr int THREADS = 128;       // 4 waves × 32
constexpr int WAVE_SZ = 32;
constexpr int WAVES = THREADS / WAVE_SZ;  // 4
constexpr int BLOCK_K = 128;       // Q1_0 group size
constexpr int BLOCK_M = 8;         // tokens per CTA
constexpr int Q1BB = 18;           // Q1_0 block bytes

// 8 fp32 accs per thread (one per M-row) = 32 bytes of VGPR
// LDS: 128 fp16 activations per tile = 256 bytes
//      + WAVES floats per M-row for reduction = 4 × 8 = 32 floats = 128 bytes

__global__
__attribute__((amdgpu_flat_work_group_size(THREADS, THREADS)))
void q1_gemm_m8_kernel(
    const uint8_t* __restrict__ packed_soa,
    const __half*  __restrict__ acts,     // [M, K]
    __half*        __restrict__ out,      // [M, N]
    int M, int N, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WAVE_SZ - 1);
    const int wave = tid >> 5;

    // This CTA handles output column `n`
    const int n = blockIdx.x;
    if (n >= N) return;

    const int nb = K >> 7;
    const size_t row_stride = (size_t)nb * Q1BB;

    // SoA pointers for this CTA's weight row
    const __half*  d_row = reinterpret_cast<const __half*>(
        packed_soa + (size_t)n * row_stride);
    const uint8_t* qs_row = packed_soa + (size_t)n * row_stride + (size_t)nb * 2;

    // Activation LDS: 8 M-rows × 128 fp16 = 1 KB
    // Laid out as [M][K_pos] for stride-1 access
    __shared__ __half act_lds[BLOCK_M][BLOCK_K];  // 1024 bytes
    // Reduction scratch per M-row
    __shared__ float row_sums[BLOCK_M][WAVES];  // 8×4 = 32 floats = 128 bytes

    // 8 fp32 accumulators, one per M-row
    float acc[BLOCK_M] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int b = 0; b < nb; ++b) {
        // ── Coalesced load: 8 rows × 128 fp16 into LDS ──
        // Each row: 128 threads load 128 consecutive fp16 = one coalesced cache line
        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            if (m < M) {
                act_lds[m][tid] = acts[(size_t)m * K + (size_t)(b * BLOCK_K + tid)];
            }
        }
        __syncthreads();

        // ── Read weight + accumulate (all M-rows from LDS) ──
        const float d = __half2float(d_row[b]);
        const int byte_idx = tid >> 3;
        const int bit_idx  = tid & 7;
        const uint8_t qbyte = qs_row[b * 16 + byte_idx];
        const float sign = ((qbyte >> bit_idx) & 1u) ? 1.0f : -1.0f;
        const float w = sign * d;

        // LDS reads — no HBM traffic during compute
        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            if (m < M) {
                const float a = __half2float(act_lds[m][tid]);
                acc[m] = fmaf(w, a, acc[m]);
            }
        }
    }

    // ── Reduce across 128 threads (one per weight position) ──
    // Each of the 128 threads has a partial sum for 1/128 of the K dimension.
    // We need to sum all 128 partials for each M-row.
    // Intra-wave butterfly (sum 32 lanes' partials) → wave has full sum
    // Inter-wave: one wave's lane 0 has the answer

    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        if (m >= M) continue;
        // Intra-wave: butterfly across 32 lanes
        float v = acc[m];
        #pragma unroll
        for (int off = WAVE_SZ / 2; off > 0; off >>= 1)
            v += __shfl_xor(v, off);
        // Store wave's partial → LDS
        if (lane == 0) row_sums[m][wave] = v;
    }
    __syncthreads();

    // Wave 0: fold all 4 waves' results
    if (wave == 0) {
        #pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            if (m >= M) continue;
            float v = (lane < WAVES) ? row_sums[m][lane] : 0.0f;
            #pragma unroll
            for (int off = WAVE_SZ / 2; off > 0; off >>= 1)
                v += __shfl_xor(v, off);
            if (lane == 0) {
                constexpr float FP16_MAX = 65504.0f;
                if (v >  FP16_MAX) v =  FP16_MAX;
                if (v < -FP16_MAX) v = -FP16_MAX;
                out[(size_t)m * N + n] = __float2half(v);
            }
        }
    }
}

} // anonymous namespace

extern "C" void bonsai_q1_gemm_m8_launch(
    const uint8_t* packed_soa,
    const uint16_t* acts_fp16,
    uint16_t*       out_fp16,
    int M, int N, int K,
    void* stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M > BLOCK_M) return;
    if (K & 127) return;

    dim3 grid(N, 1, 1);
    dim3 block(THREADS, 1, 1);
    hipLaunchKernelGGL(q1_gemm_m8_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       packed_soa,
                       reinterpret_cast<const __half*>(acts_fp16),
                       reinterpret_cast<__half*>(out_fp16),
                       M, N, K);
}

// ── SoA converter ──
static void to_soa(const uint8_t* aos, uint8_t* soa, int rows, int cols) {
    int nb = cols / 128;
    size_t rs = (size_t)nb * Q1BB;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* src = aos + (size_t)r * rs;
        uint8_t* dst = soa + (size_t)r * rs;
        for (int b = 0; b < nb; ++b) {
            dst[b*2] = src[b*Q1BB]; dst[b*2+1] = src[b*Q1BB+1];
        }
        size_t d_off = (size_t)nb * 2;
        for (int b = 0; b < nb; ++b)
            for (int j = 0; j < 16; ++j)
                dst[d_off + b*16 + j] = src[b*Q1BB + 2 + j];
    }
}

static size_t wb(int r, int c) { return (size_t)r * (size_t)(c/128) * Q1BB; }

int main() {
    struct { int rows, cols, count; const char* name; } shapes[] = {
        {2048, 2048, 1, "Q"}, {1024, 2048, 1, "K"}, {1024, 2048, 1, "V"},
        {2048, 2048, 1, "O"}, {6144, 2048, 1, "Gate"}, {6144, 2048, 1, "Up"},
        {2048, 6144, 1, "Down"}
    };
    const int NS = 7;
    const int M = 8;
    const int NB = 200;

    printf("═══ M=8 GEMM 1-col×8-row per CTA ═══\n\n");

    double total_ms = 0;
    size_t total_w = 0;

    for (int s = 0; s < NS; ++s) {
        int N = shapes[s].rows, K = shapes[s].cols;
        size_t bytes = wb(N, K);
        std::vector<uint8_t> h_aos(bytes), h_soa(bytes);
        for (size_t i = 0; i < bytes; ++i) h_aos[i] = (uint8_t)(i*0x9E3779B9u);
        to_soa(h_aos.data(), h_soa.data(), N, K);

        uint8_t *d_w;
        uint16_t *d_act, *d_out;
        HIP_OK(hipMalloc(&d_w, bytes));
        HIP_OK(hipMalloc(&d_act, (size_t)M * K * 2));
        HIP_OK(hipMalloc(&d_out, (size_t)M * N * 2));
        HIP_OK(hipMemcpy(d_w, h_soa.data(), bytes, hipMemcpyHostToDevice));
        HIP_OK(hipMemset(d_act, 0, (size_t)M * K * 2));
        HIP_OK(hipMemset(d_out, 0, (size_t)M * N * 2));
        HIP_OK(hipDeviceSynchronize());

        hipStream_t st;
        HIP_OK(hipStreamCreate(&st));
        bonsai_q1_gemm_m8_launch(d_w, d_act, d_out, M, N, K, st);
        HIP_OK(hipDeviceSynchronize());

        hipEvent_t t0, t1;
        HIP_OK(hipEventCreate(&t0));
        HIP_OK(hipEventCreate(&t1));

        HIP_OK(hipEventRecord(t0, st));
        for (int r = 0; r < NB; ++r)
            bonsai_q1_gemm_m8_launch(d_w, d_act, d_out, M, N, K, st);
        HIP_OK(hipEventRecord(t1, st));
        HIP_OK(hipEventSynchronize(t1));

        float ms;
        HIP_OK(hipEventElapsedTime(&ms, t0, t1));
        double pg = ms / (double)NB;
        double bw = (double)bytes / (pg/1000) / 1e9;
        double tok = (double)M * 1000.0 / pg;

        printf("  %-20s %6.3f ms  %5.0f GB/s  %6.0f tok/s\n",
               shapes[s].name, pg, bw, tok);

        total_ms += pg * shapes[s].count * 28;
        total_w += bytes * shapes[s].count * 28;

        hipFree(d_w); hipFree(d_act); hipFree(d_out);
        HIP_OK(hipEventDestroy(t0)); HIP_OK(hipEventDestroy(t1));
        HIP_OK(hipStreamDestroy(st));
    }

    double gbps = total_w / (total_ms/1000) / 1e9;
    double tok_raw = (double)M * 1000.0 / total_ms;
    double tok_total = (double)M * 1000.0 / (total_ms + 4.0);

    printf("\n─── Full Model 28 layers, M=8 ───\n");
    printf("  Time:        %.2f ms (GEMV only)\n", total_ms);
    printf("  BW:          %.0f GB/s (%.0f%% of 273)\n", gbps, gbps/273.0*100.0);
    printf("  M=8 raw:      %.0f tok/s\n", tok_raw);
    printf("  w/ overhead:  %.0f tok/s\n", tok_total);
    printf("  vs M=1 (121): %.1fx\n", tok_total / 121.0);
    printf("  Target 572:   %s\n", tok_total >= 572.0 ? "✅" : "❌");
    if (tok_total < 572.0) {
        double need = total_w / ((M*1000.0/572.0 - 4.0)/1000.0) / 1e9;
        printf("  Need %.0f GB/s for 572\n", need);
    }
    return 0;
}
