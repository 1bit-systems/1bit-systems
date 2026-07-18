// bench_bonsai_tile8.cpp — measure Q1_0 GEMVs with tile8-interleaved layout
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#include "rocm_cpp/bonsai.h"

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

constexpr int kQ1BlockBytes = 18;
constexpr int kRowsPerWG    = 8;
constexpr int HS = 2048, IS = 6144, NL = 28, NH = 16, NKV = 8, HD = 128;

static size_t q1b_aos(int rows, int cols) {
    return (size_t)rows * (size_t)(cols / 128) * (size_t)kQ1BlockBytes;
}

static size_t q1b_tile8(int rows, int cols) {
    int tiles = (rows + kRowsPerWG - 1) / kRowsPerWG;
    int blocks = cols / 128;
    return (size_t)tiles * (size_t)blocks * (size_t)kRowsPerWG * (size_t)kQ1BlockBytes;
}

int main() {
    hipStream_t stream;
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate AoS weights + convert to tile8
    auto mk_w_tile8 = [&](int r, int c, const char* name) {
        size_t aos_bytes = q1b_aos(r, c);
        uint8_t* d_aos = nullptr;
        HIP_OK(hipMalloc(&d_aos, aos_bytes));
        HIP_OK(hipMemsetAsync(d_aos, 0x55, aos_bytes, stream));
        HIP_OK(hipDeviceSynchronize());

        uint8_t* d_tile8 = nullptr;
        bonsai_q1_convert_aos_to_soa(d_aos, &d_tile8, r, c);
        HIP_OK(hipFree(d_aos));
        return d_tile8;
    };

    std::vector<uint8_t*> q(NL), k(NL), v(NL), o_(NL);
    std::vector<uint8_t*> gate(NL), up(NL), down(NL);

    for (int l = 0; l < NL; ++l) {
        q[l]    = mk_w_tile8(NH*HD, HS, "Q");
        k[l]    = mk_w_tile8(NKV*HD, HS, "K");
        v[l]    = mk_w_tile8(NKV*HD, HS, "V");
        o_[l]   = mk_w_tile8(HS, NH*HD, "O");
        gate[l] = mk_w_tile8(IS, HS, "Gate");
        up[l]   = mk_w_tile8(IS, HS, "Up");
        down[l] = mk_w_tile8(HS, IS, "Down");
    }

    uint16_t *d_act = nullptr;
    HIP_OK(hipMalloc(&d_act, 8192 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(d_act, 0, 8192 * sizeof(uint16_t), stream));

    uint16_t *d_out = nullptr;
    HIP_OK(hipMalloc(&d_out, 16384 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(d_out, 0, 16384 * sizeof(uint16_t), stream));

    HIP_OK(hipDeviceSynchronize());
    printf("Allocated + converted %zu GEMV weight buffers to tile8 layout.\n", size_t(7*NL+1));

    // Warmup
    for (int l = 0; l < NL; ++l) {
        bonsai_q1_gemv_soa_launch(q[l],    d_act, d_out, NH*HD, HS, stream);
        bonsai_q1_gemv_soa_launch(k[l],    d_act, d_out, NKV*HD, HS, stream);
        bonsai_q1_gemv_soa_launch(v[l],    d_act, d_out, NKV*HD, HS, stream);
        bonsai_q1_gemv_soa_launch(o_[l],   d_act, d_out, HS, NH*HD, stream);
        bonsai_q1_gemv_soa_launch(gate[l], d_act, d_out, IS, HS, stream);
        bonsai_q1_gemv_soa_launch(up[l],   d_act, d_out, IS, HS, stream);
        bonsai_q1_gemv_soa_launch(down[l], d_act, d_out, HS, IS, stream);
    }
    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    const int N_RUNS = 5;
    printf("Benchmarking %d runs of 197 tile8 GEMVs each...\n", N_RUNS);

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, stream));
    for (int run = 0; run < N_RUNS; ++run) {
        for (int l = 0; l < NL; ++l) {
            bonsai_q1_gemv_soa_launch(q[l],    d_act, d_out, NH*HD, HS, stream);
            bonsai_q1_gemv_soa_launch(k[l],    d_act, d_out, NKV*HD, HS, stream);
            bonsai_q1_gemv_soa_launch(v[l],    d_act, d_out, NKV*HD, HS, stream);
            bonsai_q1_gemv_soa_launch(o_[l],   d_act, d_out, HS, NH*HD, stream);
            bonsai_q1_gemv_soa_launch(gate[l], d_act, d_out, IS, HS, stream);
            bonsai_q1_gemv_soa_launch(up[l],   d_act, d_out, IS, HS, stream);
            bonsai_q1_gemv_soa_launch(down[l], d_act, d_out, HS, IS, stream);
        }
    }
    HIP_OK(hipEventRecord(t1, stream));
    HIP_OK(hipEventSynchronize(t1));

    float ms;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    double per_token = ms / (double)N_RUNS;
    double tok_s = 1000.0 / per_token;

    size_t total_bytes = 0;
    for (int l = 0; l < NL; ++l) {
        total_bytes += q1b_tile8(NH*HD, HS);
        total_bytes += q1b_tile8(NKV*HD, HS);
        total_bytes += q1b_tile8(NKV*HD, HS);
        total_bytes += q1b_tile8(HS, NH*HD);
        total_bytes += q1b_tile8(IS, HS);
        total_bytes += q1b_tile8(IS, HS);
        total_bytes += q1b_tile8(HS, IS);
    }
    double total_mb = total_bytes / 1e6;
    double bw = total_bytes / (per_token / 1000.0) / 1e9;

    printf("\n════════════════════════════════════════════\n");
    printf("  1-bit GEMVs only — TILE8 interleaved\n");
    printf("════════════════════════════════════════════\n");
    printf("  Model weights:  %.0f MB (Q1_0 1-bit, tile8)\n", total_mb);
    printf("  Total time:     %.1f ms over %d runs\n", ms, N_RUNS);
    printf("  Per token:      %.3f ms\n", per_token);
    printf("  Throughput:     %.1f tok/s\n", tok_s);
    printf("  Effective BW:   %.0f GB/s\n", bw);
    printf("  Peak BW util:   %.0f%%\n", bw / 273.0 * 100.0);
    printf("════════════════════════════════════════════\n");
    printf("  Tile8 vs AoS:   %.1fx speedup\n", tok_s / 82.3);
    printf("════════════════════════════════════════════\n");

    for (int l = 0; l < NL; ++l) {
        HIP_CHECK(hipFree(q[l])); HIP_CHECK(hipFree(k[l])); HIP_CHECK(hipFree(v[l])); HIP_CHECK(hipFree(o_[l]));
        HIP_CHECK(hipFree(gate[l])); HIP_CHECK(hipFree(up[l])); HIP_CHECK(hipFree(down[l]));
    }
    HIP_CHECK(hipFree(d_act)); HIP_CHECK(hipFree(d_out));
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(stream));
    return 0;
}
