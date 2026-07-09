// INT8 Quantization Benchmark — measures throughput of quantization/dequantization
// and INT8 GEMM performance across various matrix sizes matching Qwen3-0.6B dimensions
#include "quantization/int8_quant.h"
#include <cstdio>
#include <chrono>
#include <random>
#include <cmath>

using namespace specdecode::quant;

struct BenchResult {
    const char* name;
    double ms;
    double gflops;
    double bandwidth_gbs;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

template <typename F>
BenchResult bench(const char* name, int iterations, F&& fn, double flops_per_call = 0, double bytes_per_call = 0) {
    // Warmup
    for (int i = 0; i < 3; i++) fn();

    double start = now_ms();
    for (int i = 0; i < iterations; i++) fn();
    double elapsed = now_ms() - start;
    double avg_ms = elapsed / iterations;

    return {
        .name = name,
        .ms = avg_ms,
        .gflops = flops_per_call > 0 ? (flops_per_call / 1e9) / (avg_ms / 1000.0) : 0,
        .bandwidth_gbs = bytes_per_call > 0 ? (bytes_per_call / 1e9) / (avg_ms / 1000.0) : 0,
    };
}

int main() {
    printf("═══ INT8 Quantization Benchmark ═══\n\n");

    // Matrix dimensions matching Qwen3-0.6B
    struct Dims { const char* name; int M, N, K; };
    Dims dims[] = {
        {"QKV  (1xH@H→3*HD)",  1, 4096, 1024},  // QKV decode
        {"QKV  (BxH@H→3*HD)",  4, 4096, 1024},  // QKV batch decode
        {"QKV  (BxH@H→3*HD)",  16, 4096, 1024}, // QKV batch prefill
        {"O    (1xHD@HD→H)",   1, 1024, 2048},   // O projection decode
        {"O    (BxHD@HD→H)",   4, 1024, 2048},   // O projection batch
        {"GU   (1xH@H→2*IM)",  1, 6144, 1024},  // Gate+Up decode
        {"GU   (BxH@H→2*IM)",  4, 6144, 1024},  // Gate+Up batch
        {"D    (1xIM@IM→H)",   1, 1024, 3072},   // Down decode
        {"D    (BxIM@IM→H)",   4, 1024, 3072},   // Down batch
        {"LM   (1xH@H→V)",     1, 151936, 1024}, // LM head
        {"LM   (BxH@H→V)",     4, 151936, 1024}, // LM head batch
    };

    std::mt19937 rng(42);
    auto gen_weights = [&](int rows, int cols) {
        std::vector<float> w((size_t)rows * cols);
        for (auto& v : w) v = (float)rng() / (float)rng.max() * 2.0f - 1.0f;
        return w;
    };

    printf("%-30s %10s %10s %12s %12s\n", "Operation", "Time(ms)", "Iterations", "GFLOPS", "BW(GB/s)");
    printf("%s\n", std::string(76, '-').c_str());

    Int8QuantConfig cfg;
    Int8WeightQuant wq(cfg);

    for (auto& d : dims) {
        auto weights = gen_weights(d.N, d.K);

        // Quantize once
        auto packed = wq.quantize(weights, d.N, d.K);

        // Activations
        auto acts = gen_weights(d.M, d.K);
        std::vector<float> A_scales(d.M, 1.0f);
        std::vector<float> result((size_t)d.M * d.N);

        double flops = 2.0 * d.M * d.N * d.K;  // MAC = 2 flops
        double bytes = (double)(d.M * d.K + d.N * d.K + d.M * d.N) * sizeof(float);

        auto r = bench(d.name, 100, [&]() {
            Int8Gemm<>::compute({
                .A = (const int8_t*)packed.data.data(),  // reuse packed data as dummy
                .A_scales = A_scales.data(),
                .B = (const int8_t*)packed.data.data(),
                .B_scales = packed.scales.data(),
                .M = d.M, .N = d.N, .K = d.K,
            }, result);
        }, flops, bytes);

        printf("%-30s %10.3f %10d %12.1f %12.1f\n",
               r.name, r.ms, 100, r.gflops, r.bandwidth_gbs);
    }

    // Quantization throughput benchmark
    printf("\n--- Quantization Throughput ---\n");
    {
        auto weights = gen_weights(4096, 1024);
        auto r = bench("Quantize QKV", 50, [&]() {
            auto p = wq.quantize(weights, 4096, 1024);
            (void)p;
        }, 0, (double)4096 * 1024 * sizeof(float));
        printf("%-30s %10.3f %10d %12s %12.1f\n",
               "Quantize QKV weights", r.ms, 50, "-", r.bandwidth_gbs);
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
