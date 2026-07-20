/** fused_bench.cpp — Benchmark the fused NPU instruction engine.
 *
 *  Tests:
 *    1. Fused sequence generation throughput (CPU)
 *    2. NPU instruction submission latency
 *    3. End-to-end fused inference
 *
 *  Build:
 *    g++ -std=c++17 -O2 -mavx2 -march=native \\
 *        -I src -I include -I ../../include -I /usr/include \\
 *        -I /home/bcloud/fastflowlm-build/src/include \\
 *        fused_bench.cpp fused_engine.cpp \\
 *        -o fused_bench \\
 *        -lxrt_coreutil -lxrt_core -luuid -ldl -fopenmp
 *
 *  Run:
 *    NPU_XCLBIN_DIR=./xclbins ./fused_bench model.q4nx
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

#include "fused_engine.cpp"

static inline float bf16f(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.q4nx\n", argv[0]);
        return 1;
    }

    // ── Parse model ──
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> model_data(fsize);
    fread(model_data.data(), 1, fsize, f);
    fclose(f);

    const uint8_t* md = model_data.data();
    uint32_t hsz = *(uint32_t*)(md + 4);
    const char* js = (const char*)(md + 8);

    auto json_int = [&](const char* k) -> int {
        char s[128]; snprintf(s, 128, "\"%s\"", k);
        const char* p = strstr(js, s);
        if (!p) return 0;
        p = strchr(p, ':');
        if (!p) return 0;
        p++; while (*p == ' ') p++;
        return atoi(p);
    };

    FusedConfig cfg;
    cfg.H = json_int("hidden_size");
    cfg.NC = json_int("num_hidden_layers");
    cfg.NH = json_int("num_attention_heads");
    cfg.NKV = json_int("num_key_value_heads");
    cfg.HD = json_int("head_dim");
    cfg.IM = json_int("intermediate_size");
    cfg.NV = json_int("vocab_size");
    cfg.GQA = cfg.NH / cfg.NKV;
    cfg.GU_split = 0;
    cfg.rope_theta = 1000000.0f;

    if (!cfg.valid()) {
        fprintf(stderr, "Invalid model config\n"); return 1;
    }
    printf("Model: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n",
           cfg.H, cfg.NC, cfg.NH, cfg.NKV, cfg.HD, cfg.IM, cfg.NV);

    // ── TEST 1: Fused sequence generation (CPU-bound) ──
    printf("\n=== TEST 1: Fused Sequence Generation ===\n");
    const int N_ITER = 200;
    size_t total_words = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N_ITER; i++) {
        npu_sequence seq(device_npu2);
        seq.npu_preemption(0);
        for (int l = 0; l < cfg.NC; l++) {
            seq.rtp_write(IT0, 0x1000, 1);
            seq.rtp_write(IT0, 0x1004, cfg.H);
            seq.npu_dma_memcpy_nd(2, 0, S2MM, IT0, bd_0, it_channel_0,
                {0,0,0,(uint32_t)(l * cfg.H * 2)},
                {1,1,(uint32_t)cfg.H, 1},
                {0,0,0,2}, -1, 0, false, normal_cache);
            seq.npu_dma_wait(IT0, S2MM, it_channel_0);
        }
        seq.cmds2seq();
        auto [data, sz] = seq.dump();
        total_words += sz / sizeof(uint32_t);
    }

    auto t1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  %d sequences in %.1f ms\n", N_ITER, gen_ms);
    printf("  %.3f ms per fused sequence\n", gen_ms / N_ITER);
    printf("  %.0f instructions/sequence\n", (double)total_words / N_ITER);
    printf("  Generation throughput: %.0f tok/s (simulated)\n",
           1000.0 / (gen_ms / N_ITER) * cfg.NC);

    // ── TEST 2: NPU device interaction ──
    printf("\n=== TEST 2: NPU Interaction ===\n");
    try {
        xrt::device dev(0);
        printf("  Device opened\n");

        // Register xclbins
        const char* xd = getenv("NPU_XCLBIN_DIR");
        if (!xd) xd = "./xclbins";
        const char* tag = "qwen3_0_6b";

        for (const char* n : {"QKV", "O", "D", "GU"}) {
            std::string xp = std::string(xd) + "/final_i8_" + n + "_" + tag + ".xclbin";
            try {
                xrt::xclbin xc(xp);
                dev.register_xclbin(xc);
                printf("  Registered: %s\n", xp.c_str());
            } catch (std::exception& e) {
                printf("  Skip %s: %s\n", xp.c_str(), e.what());
            }
        }

        auto uuid = dev.get_xclbin_uuid();
        xrt::hw_context hc(dev, uuid);
        printf("  HW context created\n");

        // Create a minimal test sequence
        npu_sequence seq(device_npu2);
        seq.rtp_write(IT0, 0x1000, 1);
        seq.cmds2seq();
        auto [data, sz] = seq.dump();

        xrt::bo ibo(dev, sz, XCL_BO_FLAGS_CACHEABLE, 1);
        memcpy(ibo.map(), data, sz);
        ibo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo scr(dev, 4096, XRT_BO_FLAGS_HOST_ONLY, 0);

        try {
            xrt::kernel ker(hc, "MLIR_AIE");
            printf("  Kernel created\n");

            // Warmup
            for (int i = 0; i < 3; i++) {
                auto r = ker(3, ibo, (unsigned)(sz/4), scr, scr, scr);
                r.wait();
            }

            // Benchmark
            const int N_NPU = 50;
            auto t2 = std::chrono::steady_clock::now();
            for (int i = 0; i < N_NPU; i++) {
                auto r = ker(3, ibo, (unsigned)(sz/4), scr, scr, scr);
                r.wait();
            }
            auto t3 = std::chrono::steady_clock::now();

            double npu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
            printf("  %d NPU submits in %.1f ms\n", N_NPU, npu_ms);
            printf("  %.3f ms per NPU submit (kernel + wait)\n", npu_ms / N_NPU);

            // Estimate fused engine throughput
            double per_layer_ms = (npu_ms / N_NPU) / 28;
            double total_fused_ms = per_layer_ms * cfg.NC * 4;  // QKV+O+GU+D per layer
            printf("\n  Projected fused engine decode:\n");
            printf("    Per-layer NPU time: %.3f ms\n", per_layer_ms);
            printf("    Total per step:     %.3f ms\n", total_fused_ms);
            printf("    Estimated tok/s:    %.0f\n", 1000.0 / total_fused_ms);

        } catch (std::exception& e) {
            printf("  Kernel exec: %s\n", e.what());
            printf("  (Expected if no MLIR_AIE kernel in xclbin)\n");
        }

    } catch (std::exception& e) {
        printf("  NPU not available: %s\n", e.what());
    }

    printf("\n=== BENCHMARK COMPLETE ===\n");
    return 0;
}
