/**
 * Minimal test for the M=512 K=1024 N=2048 custom matvec xclbin.
 *
 * Tests with simple pattern: A = all 1.0 (512 rows), B = identity-like
 * Expected: C[i,j] = K = 1024 for all i,j (since sum over k of 1.0 * 1.0 = 1024)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "helper.h"
#include "gemm_atb_layout.h"

static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16; float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static inline uint16_t float_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
    uint32_t r = ((bits >> 16) & 1) + 0x7FFF; return (uint16_t)((bits + r) >> 16);
}

static std::vector<uint32_t> load_instructions(const char* path) {
    std::vector<uint32_t> instrs;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); fseek(f, 0, SEEK_SET);
    instrs.resize(sz / 4);
    size_t n = fread(instrs.data(), 4, instrs.size(), f);
    fclose(f);
    printf("  Instructions: %zu words (%ld bytes)\n", n, sz);
    return instrs;
}

int main(int argc, char** argv) {
    const char* xclbin_path = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_512x1024x2048_128x64x128.xclbin";
    const char* instr_path = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_512x1024x2048_128x64x128.txt";
    if (argc > 1) xclbin_path = argv[1];
    if (argc > 2) instr_path = argv[2];

    const int M = 512, K = 1024, N = 2048;
    const int m_tile = 128, k_tile = 64, n_tile = 128;
    const int n_aie_rows = 4;

    printf("=== Test M=%d K=%d N=%d matvec xclbin ===\n\n", M, K, N);

    auto instr_v = load_instructions(instr_path);

    // Open device, load xclbin
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xclbin);
    auto hw_ctx = xrt::hw_context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");
    printf("Kernel loaded\n");

    // Test data: ramp pattern to validate correctness
    std::vector<float> A_float(M * K);
    for (int i = 0; i < M * K; i++) A_float[i] = (float)((i % 7) - 3);  // -3 to +3 ramp
    std::vector<float> B_float(K * N);
    for (int i = 0; i < K * N; i++) B_float[i] = (float)((i % 5) - 2); // -2 to +2 ramp

    // Compute CPU reference
    std::vector<float> C_ref(M * N, 0);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A_float[m * K + k] * B_float[k * N + n];
            C_ref[m * N + n] = sum;
        }
    }

    // Shuffle A: use layout_A_L1_2x1_8x8block (matching MLIR design)
    printf("Shuffling A...\n");
    auto A_shuf = gemm_atb::layout_A_L1_2x1_8x8block(A_float, M, K, m_tile, k_tile);
    std::vector<uint16_t> A_bf16(M * K);
    for (int i = 0; i < M * K; i++) A_bf16[i] = float_to_bf16(A_shuf[i]);

    // Shuffle B: layout_transpose_L1_1x2_8x8block → BFP16 ebs8
    printf("Shuffling B...\n");
    auto B_shuf = gemm_atb::layout_transpose_L1_1x2_8x8block(B_float, K, N, k_tile, n_tile);
    auto B_bfp16 = floatToBfp16(8, K * N, B_shuf.data(), 0);

    // Create BOs
    size_t A_bytes = M * K * sizeof(uint16_t);
    size_t B_bytes = B_bfp16.size();
    size_t C_bytes = M * N * sizeof(uint16_t);
    size_t instr_bytes = instr_v.size() * sizeof(uint32_t);

    printf("A: %zu B: %zu C: %zu Instr: %zu\n", A_bytes, B_bytes, C_bytes, instr_bytes);

    auto bo_instr = xrt::bo(device, instr_bytes, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_A = xrt::bo(device, A_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_B = xrt::bo(device, B_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_C = xrt::bo(device, C_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    // Upload
    memcpy(bo_instr.map(), instr_v.data(), instr_bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_A.map(), A_bf16.data(), A_bytes);
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_B.map(), B_bfp16.data(), B_bytes);
    bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memset(bo_C.map(), 0, C_bytes);
    bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Run
    printf("Running NPU...\n");
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel((unsigned int)3, bo_instr, instr_v.size(), bo_A, bo_B, bo_C);
    ert_cmd_state state = run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    printf("  State: %d %s  Time: %.1f us\n", (int)state,
           state == ERT_CMD_STATE_COMPLETED ? "OK" : "FAIL", us);
    if (state != ERT_CMD_STATE_COMPLETED) return 1;

    // Read back C
    bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t* C_bf16 = (uint16_t*)bo_C.map();

    // BF16 → float → unshuffle
    std::vector<float> C_raw(M * N);
    for (int i = 0; i < M * N; i++) C_raw[i] = bf16_to_float(C_bf16[i]);

    // Try both unshuffle sizes
    // The original 4096×4096 uses (n_aie_rows * m_tile, n_tile) = (4*128, 128) = (512, 128)
    auto C_float = gemm_atb::layout_inverse_C_L1_2x2_8x8block(C_raw, M, N, n_aie_rows * m_tile, n_tile);

    printf("\nOutput top-left 8x8 (vs CPU ref):\n");
    int errors = 0;
    float max_rel = 0, max_abs = 0;
    for (int m = 0; m < 8; m++) {
        printf("  row %2d:", m);
        for (int n = 0; n < 8; n++) {
            float got = C_float[m * N + n];
            float ref = C_ref[m * N + n];
            double rel = fabs((double)got - (double)ref) / (fabs((double)ref) + 1e-30);
            double abs = fabs((double)got - (double)ref);
            if (rel > max_rel) max_rel = rel;
            if (abs > max_abs) max_abs = abs;
            if (rel > 0.5 && abs > 0.5) { if (errors < 5) printf(" [%.0f vs %.0f]", got, ref); errors++; }
            printf(" %8.1f", got);
        }
        printf("\n");
    }
    printf("  CPU ref row 0 first 8: ");
    for (int n = 0; n < 8; n++) printf("%.0f ", C_ref[n]);
    printf("\n");
    printf("  Max rel err: %e  max abs err: %e\n", max_rel, max_abs);
    if (errors > 0) printf("  Total errors: %d\n", errors);

    double gflops = 2.0 * M * K * N / (us * 1000);
    printf("\n%s (%.1f GFLOPS, %.2f TFLOPS)\n",
           errors == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED",
           gflops, gflops / 1000.0);
    return errors ? 1 : 0;
}
