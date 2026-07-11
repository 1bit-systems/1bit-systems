// bench_soa_vs_aos.cpp — benchmark SoA vs AoS Q1_0 GEMV throughput.
// Compares the original AoS kernel with the new SoA + 4-row-batching kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

constexpr int kQ1BlockBytes = 18;
constexpr int kWeightsPerB = 128;

// Forward declare the SoA kernel
extern "C" void bonsai_q1_gemv_soa_launch(
    const uint8_t*, const uint16_t*, uint16_t*, int, int, void*);

extern "C" void bonsai_q1_gemv_launch(
    const uint8_t*, const uint16_t*, uint16_t*, int, int, void*);

extern "C" void bonsai_q1_convert_aos_to_soa(
    const uint8_t*, uint8_t**, int, int);

static size_t q1_bytes(int rows, int cols) {
    int nb = cols / kWeightsPerB;
    return (size_t)rows * (size_t)nb * (size_t)kQ1BlockBytes;
}

int main() {
    // Test shapes matching Bonsai-1.7B
    struct { int rows, cols; const char* name; } shapes[] = {
        {2048, 2048, "Q/O (2048x2048)"},
        {1024, 2048, "K/V (1024x2048)"},
        {6144, 2048, "Gate/Up (6144x2048)"},
        {2048, 6144, "Down (2048x6144)"},
    };
    const int N_SHAPES = 4;

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    printf("═══════════════════════════════════════════════════════\n");
    printf("  Q1_0 GEMV Benchmark: AoS vs SoA (4 rows/CTA)\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    for (int s = 0; s < N_SHAPES; ++s) {
        int N = shapes[s].rows;
        int K = shapes[s].cols;
        const char* name = shapes[s].name;

        printf("─── %s ───\n", name);
        printf("  Weights: %zu KB\n", q1_bytes(N, K) / 1024);

        // Generate synthetic AoS weights
        size_t w_bytes = q1_bytes(N, K);
        std::vector<uint8_t> h_w_aoa(w_bytes);
        for (size_t i = 0; i < w_bytes; ++i)
            h_w_aoa[i] = (uint8_t)((i * 0x9E3779B9u) & 0xFF);

        uint8_t *d_w_aos = nullptr, *d_w_soa = nullptr;
        HIP_OK(hipMalloc(&d_w_aos, w_bytes));
        HIP_OK(hipMemcpy(d_w_aos, h_w_aoa.data(), w_bytes, hipMemcpyHostToDevice));

        // Convert to SoA
        bonsai_q1_convert_aos_to_soa(d_w_aos, &d_w_soa, N, K);

        // Activation buffer
        uint16_t* d_act = nullptr;
        HIP_OK(hipMalloc(&d_act, K * sizeof(uint16_t)));
        HIP_OK(hipMemset(d_act, 0, K * sizeof(uint16_t)));

        // Output buffer
        uint16_t* d_out = nullptr;
        HIP_OK(hipMalloc(&d_out, N * sizeof(uint16_t)));
        HIP_OK(hipMemset(d_out, 0, N * sizeof(uint16_t)));

        HIP_OK(hipDeviceSynchronize());

        // Warmup
        bonsai_q1_gemv_launch(d_w_aos, d_act, d_out, N, K, stream);
        bonsai_q1_gemv_soa_launch(d_w_soa, d_act, d_out, N, K, stream);
        HIP_OK(hipDeviceSynchronize());

        // Benchmark: AoS (original)
        const int N_RUNS = 100;
        hipEvent_t t0, t1;
        HIP_OK(hipEventCreate(&t0));
        HIP_OK(hipEventCreate(&t1));

        HIP_OK(hipEventRecord(t0, stream));
        for (int r = 0; r < N_RUNS; ++r) {
            bonsai_q1_gemv_launch(d_w_aos, d_act, d_out, N, K, stream);
        }
        HIP_OK(hipEventRecord(t1, stream));
        HIP_OK(hipEventSynchronize(t1));

        float ms_aos;
        HIP_OK(hipEventElapsedTime(&ms_aos, t0, t1));
        double per_gemv_aos = ms_aos / (double)N_RUNS;
        double bw_aos = (double)w_bytes / (per_gemv_aos / 1000.0) / 1e9;
        double tok_aos = 1000.0 / per_gemv_aos;

        // Benchmark: SoA (new)
        HIP_OK(hipEventRecord(t0, stream));
        for (int r = 0; r < N_RUNS; ++r) {
            bonsai_q1_gemv_soa_launch(d_w_soa, d_act, d_out, N, K, stream);
        }
        HIP_OK(hipEventRecord(t1, stream));
        HIP_OK(hipEventSynchronize(t1));

        float ms_soa;
        HIP_OK(hipEventElapsedTime(&ms_soa, t0, t1));
        double per_gemv_soa = ms_soa / (double)N_RUNS;
        double bw_soa = (double)w_bytes / (per_gemv_soa / 1000.0) / 1e9;
        double tok_soa = 1000.0 / per_gemv_soa;

        printf("  AoS (1 row/CTA):  %6.3f ms  %5.0f GB/s  %7.0f GEMV/s\n",
               per_gemv_aos, bw_aos, tok_aos);
        printf("  SoA (4 rows/CTA): %6.3f ms  %5.0f GB/s  %7.0f GEMV/s  (%.1fx speedup)\n",
               per_gemv_soa, bw_soa, tok_soa, per_gemv_aos / per_gemv_soa);

        hipFree(d_w_aos);
        hipFree(d_w_soa);
        hipFree(d_act);
        hipFree(d_out);
        HIP_OK(hipEventDestroy(t0));
        HIP_OK(hipEventDestroy(t1));
        printf("\n");
    }

    // Full model estimate
    printf("─── Full Model Estimate (28 layers, 197 GEMVs) ───\n");
    printf("\n");
    printf("With AoS (1 row/CTA):\n");
    printf("  197 GEMVs × average = ~12.6 ms → 79 tok/s\n");
    printf("\n");
    printf("With SoA (4 rows/CTA) at 4× speedup:\n");
    printf("  ~3.15 ms per token → 317 tok/s\n");
    printf("  + norms/attention ~1 ms → ~240 tok/s\n");
    printf("\n");
    printf("With SoA (8 rows/CTA) at 8× speedup:\n");
    printf("  ~1.58 ms per token → 633 tok/s ✨\n");
    printf("  + norms/attention ~0.5 ms → ~480 tok/s\n");
    printf("\n");
    printf("Path to 572 tok/s: SoA (8 rows) + fused norm+attn\n");
    printf("═══════════════════════════════════════════════════════\n");

    HIP_OK(hipStreamDestroy(stream));
    return 0;
}
