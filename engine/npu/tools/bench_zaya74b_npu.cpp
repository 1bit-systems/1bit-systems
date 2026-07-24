/**
 * bench_zaya74b_npu.cpp — NPU GEMM benchmark at Zaya-74B dimensions (H=4096)
 *
 * Tests whether the universal ("v") xclbins can handle H=4096 dimensions
 * and measures actual NPU throughput for QKV, O, GU, D operations.
 *
 * Build:
 *   g++ -std=c++23 -O3 -march=native -o bench_zaya74b_npu \
 *       engine/npu/tools/bench_zaya74b_npu.cpp \
 *       -I engine/npu/include -I engine/npu/kernel \
 *       -lxrt_coreutil -luuid -lrt -lpthread -fopenmp
 *
 * Run:
 *   sudo ./bench_zaya74b_npu
 *   (requires root for XRT device access)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ─── Zaya1-74B dimensions ───────────────────────────────────────────
static constexpr int H = 4096;           // hidden_size
static constexpr int NH = 16;            // num_attention_heads
static constexpr int NKV = 2;            // num_kv_heads
static constexpr int HD = 128;           // head_dim
static constexpr int IM = 4096;          // intermediate_size

// Derived dimensions
static constexpr int QKV_N = NH * HD + 2 * NKV * HD;  // 2560
static constexpr int QKV_K = H;                        // 4096
static constexpr int O_N = H;                          // 4096
static constexpr int O_K = NH * HD;                    // 2048
static constexpr int GU_N = IM;                        // 4096
static constexpr int GU_K = H;                         // 4096
static constexpr int D_N = H;                          // 4096
static constexpr int D_K = IM;                         // 4096

// NPU GEMM tile sizes (from existing xclbin config)
static constexpr int TILE_M = 1;          // batch=1 (decode)
static constexpr int TILE_K = 256;        // input chunk
static constexpr int TILE_N = 512;        // output tile

struct NPUBench {
    xrt::device* dev = nullptr;

    bool init() {
        try {
            auto devices = xrt::xclbin::enumerate_xclbins();
            if (devices.empty()) {
                fprintf(stderr, "No NPU devices found\n");
                return false;
            }
            dev = new xrt::device(0);
            fprintf(stderr, "NPU device: %s\n", dev->get_info<xrt::info::device::name>().c_str());
            return true;
        } catch (std::exception& e) {
            fprintf(stderr, "NPU init failed: %s\n", e.what());
            return false;
        }
    }

    ~NPUBench() { delete dev; }

    // ─── xclbin path for a given operation ────────────────────
    static const char* xclbin_path(const char* op) {
        static char buf[256];
        snprintf(buf, sizeof(buf), "engine/npu/xclbins/final_i8_%s_v.xclbin", op);
        return buf;
    }
    static const char* insts_path(const char* op) {
        static char buf[256];
        snprintf(buf, sizeof(buf), "engine/npu/xclbins/insts_i8_%s_v.txt", op);
        return buf;
    }

    // ─── Load xclbin + instruction file → run one GEMM ───────
    struct GemmResult {
        bool ok = false;
        double ms = 0;
        const char* label = "";
    };

    GemmResult bench_gemm(const char* label, const char* xclbin_op,
                          int M, int N, int K,
                          int body_records, int chunks_per_record,
                          int weight_chunks)
    {
        GemmResult r;
        r.label = label;

        std::string xp = xclbin_path(xclbin_op);
        std::string ip = insts_path(xclbin_op);

        // Load instruction file
        FILE* f = fopen(ip.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "  Cannot open %s\n", ip.c_str());
            return r;
        }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> ins(sz / 4);
        fread(ins.data(), 4, ins.size(), f);
        fclose(f);

        try {
            // Load xclbin
            auto xc = xrt::xclbin(xp);
            dev->register_xclbin(xc);
            auto hc = xrt::hw_context(*dev, xc.get_uuid());
            auto k = xrt::kernel(hc, "MLIR_AIE");

            // Allocate BOs
            size_t a_size = (size_t)M * K;      // i8 activations
            size_t b_size = (size_t)K * N;      // i8 weights
            size_t c_size = (size_t)M * N * 2;  // i16 output

            auto bI = xrt::bo(*dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.group_id(1));
            auto bA = xrt::bo(*dev, a_size, XRT_BO_FLAGS_HOST_ONLY, k.group_id(3));
            auto bB = xrt::bo(*dev, b_size, XRT_BO_FLAGS_HOST_ONLY, k.group_id(4));
            auto bC = xrt::bo(*dev, c_size, XRT_BO_FLAGS_HOST_ONLY, k.group_id(5));

            // Fill with deterministic data
            int8_t* Am = (int8_t*)bA.map();
            int8_t* Bm = (int8_t*)bB.map();
            for (size_t i = 0; i < a_size; i++) Am[i] = (int8_t)(i & 0x7F);
            for (size_t i = 0; i < b_size; i++) Bm[i] = (int8_t)((i * 7) & 0x7F);

            // Copy instructions
            memcpy(bI.map(), ins.data(), ins.size() * 4);

            // Sync + launch
            bA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bI.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Warmup
            auto r_warm = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
            r_warm.wait();

            // Benchmark: run N iterations
            const int n_iter = 100;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                auto run = k((unsigned)3, bI, (unsigned)ins.size(), bA, bB, bC);
                run.wait();
            }
            auto t1 = std::chrono::steady_clock::now();

            r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_iter;
            r.ok = true;

        } catch (std::exception& e) {
            fprintf(stderr, "  Error: %s\n", e.what());
        }

        return r;
    }

    void run_all() {
        fprintf(stderr, "\n╔════════════════════════════════════════╗\n");
        fprintf(stderr, "║  Zaya-74B NPU GEMM Benchmark          ║\n");
        fprintf(stderr, "╚════════════════════════════════════════╝\n\n");
        fprintf(stderr, "Model: H=%d L=120 NH=%d NKV=%d HD=%d IM=%d\n\n", H, NH, NKV, HD, IM);

        std::vector<GemmResult> results;

        // QKV: M=1, N=QKV_N(2560), K=H(4096)
        // Body records: ceil(N/512) = 2560/512 = 5
        // Chunks/record: ceil(K/256) = 4096/256 = 16
        results.push_back(bench_gemm("QKV", "QKV_v", 1, QKV_N, QKV_K, 5, 16, 0));

        // O: M=1, N=H(4096), K=NH*HD(2048)
        results.push_back(bench_gemm("O", "O_v", 1, O_N, O_K, 8, 8, 0));

        // G/U: M=1, N=IM(4096), K=H(4096) — GU is fused, uses body count for both
        results.push_back(bench_gemm("GU", "GU_v", 1, GU_N, GU_K, 8, 16, 0));

        // D: M=1, N=H(4096), K=IM(4096)
        results.push_back(bench_gemm("D", "D_v", 1, D_N, D_K, 8, 16, 0));

        // ── Results ──
        fprintf(stderr, "\nResults:\n");
        fprintf(stderr, "  %-12s %12s %12s %12s %10s\n",
                "Op", "M×N×K", "MACs", "ms", "GFLOPS");
        fprintf(stderr, "  %s\n", std::string(58, '─').c_str());

        double total_ms = 0;
        int64_t total_macs = 0;

        for (auto& r : results) {
            if (!r.ok) {
                fprintf(stderr, "  ❌ %s: FAILED\n", r.label);
                continue;
            }
            // Estimate MACs
            int64_t macs = 0;
            int M=1, N=0, K=0;
            if (strcmp(r.label, "QKV") == 0) { N=QKV_N; K=QKV_K; }
            else if (strcmp(r.label, "O") == 0) { N=O_N; K=O_K; }
            else if (strcmp(r.label, "GU") == 0) { N=GU_N; K=GU_K; }
            else if (strcmp(r.label, "D") == 0) { N=D_N; K=D_K; }
            macs = (int64_t)M * N * K * 2;  // 2 MACs per multiply-add

            double gflops = macs / (r.ms * 1e6);
            fprintf(stderr, "  %-12s %5d×%-4d×%-4d %12s %8.3f %8.0f\n",
                    r.label, M, N, K,
                    (std::to_string(macs/1000000) + "M").c_str(),
                    r.ms, gflops);

            total_ms += r.ms;
            total_macs += macs;
        }

        // Per-layer and full-model projection
        fprintf(stderr, "\n%s\n", std::string(58, '─').c_str());
        fprintf(stderr, "  Per layer:          %8.3f ms\n", total_ms);

        double total_120 = total_ms * 120;
        double tok_s = 1000.0 / total_120;
        fprintf(stderr, "  120 layers:         %8.3f ms\n", total_120);
        fprintf(stderr, "  NPU decode (est.):  %8.1f tok/s\n", tok_s);

        // Full model with CPU overhead
        double cpu_overhead = 0.2;  // 200µs per layer for CPU ops (norm, attn, residual)
        double full_120 = (total_ms + cpu_overhead) * 120;
        fprintf(stderr, "\nWith CPU overhead (%.2f ms/layer):\n", cpu_overhead);
        fprintf(stderr, "  Full model:         %8.3f ms\n", full_120);
        fprintf(stderr, "  Est. throughput:    %8.1f tok/s\n", 1000.0 / full_120);
    }
};

int main() {
    NPUBench bench;
    if (!bench.init()) return 1;
    bench.run_all();
    return 0;
}
