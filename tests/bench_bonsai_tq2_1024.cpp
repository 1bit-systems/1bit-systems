// bench_bonsai_tq2_1024.cpp — TQ2_1024 1024-weight block benchmark
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "rocm_cpp/bonsai.h"

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP err %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

constexpr int HS = 2048, IS = 6144, NL = 28, NH = 16, NKV = 8, HD = 128;
constexpr int kBlockBytes_old = 34;

static size_t old_sz(int r, int c) {
    return (size_t)r * (size_t)(c / 128) * kBlockBytes_old;
}

int main() {
    hipStream_t s;
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    auto mk = [&](int r, int c) {
        uint8_t *a = nullptr, *d = nullptr;
        HIP_OK(hipMalloc(&a, old_sz(r, c)));
        HIP_OK(hipMemsetAsync(a, 0x55, old_sz(r, c), s));
        HIP_OK(hipDeviceSynchronize());
        bonsai_tq2_convert_to_1024(a, &d, r, c);
        HIP_OK(hipFree(a));
        return d;
    };

    std::vector<uint8_t*> w;
    for (int l = 0; l < NL; ++l) {
        w.push_back(mk(NH * HD, HS));
        w.push_back(mk(NKV * HD, HS));
        w.push_back(mk(NKV * HD, HS));
        w.push_back(mk(HS, NH * HD));
        w.push_back(mk(IS, HS));
        w.push_back(mk(IS, HS));
        w.push_back(mk(HS, IS));
    }

    uint16_t *a = nullptr, *o = nullptr;
    HIP_OK(hipMalloc(&a, 8192 * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&o, 16384 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(a, 0, 8192 * sizeof(uint16_t), s));
    HIP_OK(hipMemsetAsync(o, 0, 16384 * sizeof(uint16_t), s));
    HIP_OK(hipDeviceSynchronize());

    printf("Allocated %zu TQ2_1024 weight buffers. Warmup...\n", w.size());
    for (int l = 0; l < NL; ++l) {
        int i = l * 7;
        bonsai_tq2_1024_gemv_launch(w[i],   a, o, NH*HD, HS, s);
        bonsai_tq2_1024_gemv_launch(w[i+1], a, o, NKV*HD, HS, s);
        bonsai_tq2_1024_gemv_launch(w[i+2], a, o, NKV*HD, HS, s);
        bonsai_tq2_1024_gemv_launch(w[i+3], a, o, HS, NH*HD, s);
        bonsai_tq2_1024_gemv_launch(w[i+4], a, o, IS, HS, s);
        bonsai_tq2_1024_gemv_launch(w[i+5], a, o, IS, HS, s);
        bonsai_tq2_1024_gemv_launch(w[i+6], a, o, HS, IS, s);
    }
    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, s));
    for (int run = 0; run < 10; ++run) {
        for (int l = 0; l < NL; ++l) {
            int i = l * 7;
            bonsai_tq2_1024_gemv_launch(w[i],   a, o, NH*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+1], a, o, NKV*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+2], a, o, NKV*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+3], a, o, HS, NH*HD, s);
            bonsai_tq2_1024_gemv_launch(w[i+4], a, o, IS, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+5], a, o, IS, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+6], a, o, HS, IS, s);
        }
    }
    HIP_OK(hipEventRecord(t1, s));
    HIP_OK(hipEventSynchronize(t1));

    float ms;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    double pt = ms / 10.0;
    double ts = 1000.0 / pt;

    size_t total = 0;
    for (int l = 0; l < NL; ++l) {
        auto sz = [](int r, int c) -> size_t {
            int tiles = (r + 7) / 8;
            int blocks = c / 1024;
            return (size_t)tiles * blocks * 8 * 258;
        };
        total += sz(NH*HD, HS);
        total += sz(NKV*HD, HS) * 2;
        total += sz(HS, NH*HD);
        total += sz(IS, HS) * 2;
        total += sz(HS, IS);
    }
    double bw = total / (pt / 1000.0) / 1e9;

    printf("\n═══════════════════════════════════════════════\n");
    printf("  TQ2_1024 — 28-layer model (258B blocks)\n");
    printf("═══════════════════════════════════════════════\n");
    printf("  Weights:        ~%.0f MB\n", total / 1e6);
    printf("  Per token:      %.3f ms\n", pt);
    printf("  Throughput:     %.0f tok/s\n", ts);
    printf("  Effective BW:   %.0f GB/s\n", bw);
    printf("  Peak BW util:   %.0f%%\n", bw / 273.0 * 100.0);
    printf("═══════════════════════════════════════════════\n");

    for (auto p : w) hipFree(p);
    hipFree(a); hipFree(o);
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(s));
    return 0;
}
