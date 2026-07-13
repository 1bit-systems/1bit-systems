/**
 * NPU GEMM Runner for custom M=128 single-token decode xclbin.
 *
 * Tests a 128x1024 by 1024x2048 matrix multiply on the NPU using
 * a 512x1024x2048 padded xclbin (host pads 128→512, reads first 128 rows).
 *
 * Data format follows torch2aie config1 conventions:
 *   A: BF16 shuffled via layout_A_L1_2x1_8x8block
 *   B: BFP16 ebs8 shuffled via layout_transpose_L1_1x2_8x8block
 *   C: BF16 unshuffled via layout_inverse_C_L1_2x2_8x8block
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

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
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    instrs.resize(sz / 4);
    size_t n = fread(instrs.data(), 4, instrs.size(), f);
    fclose(f);
    printf("  Read %zu instructions from %ld bytes\n", n, sz);
    return instrs;
}

static bool check_result_correct(const float* cpu_C, const float* npu_C, int rows, int cols) {
    double max_rel_err = 0, max_abs_err = 0;
    int bad = 0;
    for (int i = 0; i < rows * cols; i++) {
        float c = cpu_C[i], n = npu_C[i];
        double abs_err = fabs((double)n - (double)c);
        double rel_err = abs_err / (fabs((double)c) + 1e-30);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > 0.5 && abs_err > 0.5) {
            if (bad < 10) printf("  MISMATCH[%d]: cpu=%f npu=%f rel_err=%f\n", i, c, n, rel_err);
            bad++;
        }
    }
    printf("  Max rel error: %e, max abs error: %e\n", max_rel_err, max_abs_err);
    printf("  Mismatches >0.5: %d / %d\n", bad, rows * cols);
    return bad == 0;
}

int main(int argc, char** argv) {
    printf("=== NPU GEMM Runner — M=128 matvec ===\n\n");

    const char* xclbin_path = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/final_512x1024x2048_128x64x128.xclbin";
    const char* instr_path = "/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/build/insts_512x1024x2048_128x64x128.txt";
    if (argc > 1) xclbin_path = argv[1];
    if (argc > 2) instr_path = argv[2];

    // Design dimensions (xclbin expects M_padded=512, but we use M_actual=128)
    const int M_padded = 512;    // xclbin padded M
    const int M_actual = 128;    // actual M we care about
    const int K = 1024;
    const int N = 2048;
    const int m_tile = 128, k_tile = 64, n_tile = 128;
    const int n_aie_rows = 4, n_aie_cols = 8;
    const int mtk = 512;  // A L2 tile K dimension

    printf("Design: M=%d(actual=%d) K=%d N=%d  tile=%dx%dx%d\n",
           M_padded, M_actual, K, N, m_tile, k_tile, n_tile);

    // Load instructions
    auto instr_v = load_instructions(instr_path);
    printf("Loaded %zu instr words\n", instr_v.size());

    // Open device and load xclbin
    auto device = xrt::device(0);
    printf("Opened NPU\n");
    auto uuid = device.load_xclbin(xclbin_path);
    printf("Loaded xclbin\n");
    auto kernel = xrt::kernel(device, uuid, "MLIR_AIE");
    printf("Got kernel\n");

    // Create BOs
    const size_t A_bytes = M_padded * K * sizeof(uint16_t);  // BF16
    const size_t B_bytes = K * N / 8 * 2;  // BFP16 ebs8: each group of 8 BF16 = 2 bytes
    const size_t C_bytes = M_padded * N * sizeof(uint16_t);  // BF16 output
    const size_t instr_bytes = instr_v.size() * sizeof(uint32_t);

    printf("BO sizes: A=%zu B=%zu C=%zu instr=%zu\n", A_bytes, B_bytes, C_bytes, instr_bytes);

    xrt::bo bo_instr = xrt::bo(device, instr_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(1));
    xrt::bo bo_A = xrt::bo(device, A_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    xrt::bo bo_B = xrt::bo(device, B_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    xrt::bo bo_C = xrt::bo(device, C_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    printf("Created BOs\n");

    // Upload instructions
    memcpy(bo_instr.map(), instr_v.data(), instr_bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("Uploaded instructions\n");

    // Generate random test data
    std::vector<float> A_float(M_actual * K, 0);
    srand(42);
    for (auto& v : A_float) v = (float)(rand() % 100) / 50.0f - 1.0f;

    std::vector<float> B_float_colmajor(K * N, 0);  // B is K×N row-major, so we fill it
    for (int i = 0; i < K * N; i++) B_float_colmajor[i] = (float)(rand() % 100) / 50.0f - 1.0f;

    // Compute CPU reference: C = A * B (re-shaped as matvec: M_actual×K * K×N = M_actual×N)
    std::vector<float> C_cpu(M_actual * N, 0);
    for (int m = 0; m < M_actual; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A_float[m * K + k] * B_float_colmajor[k * N + n];
            }
            C_cpu[m * N + n] = sum;
        }
    }
    printf("CPU reference computed\n");

    // ---- Host-side data preparation ----
    // A: pad from [M_actual, K] to [M_padded, K] with zeros, convert to BF16, then shuffle
    std::vector<uint16_t> A_bf16(M_padded * K, 0);
    for (int i = 0; i < M_actual * K; i++) {
        A_bf16[i] = float_to_bf16(A_float[i]);
    }
    // Shuffle A (layout functions return new vectors)
    std::vector<float> A_shuf_float(M_padded * K, 0);
    for (int i = 0; i < M_actual * K; i++) A_shuf_float[i] = A_float[i];
    A_shuf_float = gemm_atb::layout_A_L1_2x1_8x8block(A_shuf_float, M_padded, K, m_tile, mtk);
    std::vector<uint16_t> A_shuf_bf16(M_padded * K);
    for (int i = 0; i < M_padded * K; i++)
        A_shuf_bf16[i] = float_to_bf16(A_shuf_float[i]);

    // B: pack, shuffle, BFP16 encode
    std::vector<float> B_shuf_float(K * N, 0);
    for (int i = 0; i < K * N; i++) B_shuf_float[i] = B_float_colmajor[i];  // already row-major
    B_shuf_float = gemm_atb::layout_transpose_L1_1x2_8x8block(B_shuf_float, K, N, k_tile, n_tile);
    auto B_bfp16 = floatToBfp16(8, K * N, B_shuf_float.data(), 0);

    printf("Data prepared (A BF16, B BFP16)\n");

    // Upload
    memcpy(bo_A.map(), A_shuf_bf16.data(), A_bytes);
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy(bo_B.map(), B_bfp16.data(), B_bytes);
    bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("Data uploaded\n");

    // Run NPU
    size_t ninstr = instr_v.size();
    printf("Running kernel...\n");
    auto run = kernel(3, bo_instr, ninstr, bo_A, bo_B, bo_C);
    run.wait();
    printf("Kernel done\n");

    // Read back
    bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t* C_bf16 = (uint16_t*)bo_C.map();

    // Unshuffle and convert to float (only M_actual rows)
    std::vector<float> C_npu(M_padded * N, 0);
    for (int i = 0; i < M_padded * N; i++)
        C_npu[i] = bf16_to_float(C_bf16[i]);
    C_npu = gemm_atb::layout_inverse_C_L1_2x2_8x8block(C_npu, M_padded, N, 4 * m_tile, n_tile);

    // Check result (first M_actual rows only)
    printf("Checking first %d rows...\n", M_actual);
    bool ok = check_result_correct(C_cpu.data(), C_npu.data(), M_actual, N);

    // Print first few values
    printf("\nFirst 5 NPU results: ");
    for (int i = 0; i < 5 && i < M_actual * N; i++) printf("%.4f ", C_npu[i]);
    printf("\nFirst 5 CPU results: ");
    for (int i = 0; i < 5 && i < M_actual * N; i++) printf("%.4f ", C_cpu[i]);
    printf("\n");

    printf("\n=== %s ===\n", ok ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return ok ? 0 : 1;
}
