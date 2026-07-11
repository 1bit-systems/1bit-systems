// SPDX-License-Identifier: MIT
/*
 * NPU GEMM via pre-compiled xclbin
 *
 * Loads the NPU matrix multiply xclbin and
 * runs a GEMM operation on the NPU via XRT.
 *
 * Build:
 *   g++ -std=gnu++17 -o test_npu_gemm test_npu_gemm.cpp \
 *       -I/usr/include -L/usr/lib/x86_64-linux-gnu \
 *       -lxrt_coreutil
 *
 * Run:
 *   ./test_npu_gemm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

/* ------------------------------------------------------------------ */
/*  Load xclbin from file                                             */
/* ------------------------------------------------------------------ */

static std::vector<char> load_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fprintf(stderr, "Failed to open: %s\n", path.c_str());
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    return buf;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */

int main() {
    printf("=== NPU GEMM via XRT xclbin ===\n\n");

    /* Open NPU device via XRT */
    xrt::device device;
    try {
        device = xrt::device(1);  // NPU is typically device 1
    } catch (...) {
        try {
            device = xrt::device(0);
        } catch (const std::exception& e) {
            printf("FAIL: Cannot open NPU: %s\n", e.what());
            return 1;
        }
    }
    printf("NPU device opened\n");

    /* Find the mm.xclbin for Qwen3-0.6B */
    std::string xclbin_path = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";
    auto xclbin_data = load_file(xclbin_path);
    if (xclbin_data.empty()) {
        printf("FAIL: Cannot load %s\n", xclbin_path.c_str());
        return 1;
    }
    printf("Loaded xclbin: %s (%zu bytes)\n", xclbin_path.c_str(), xclbin_data.size());

    /* Register xclbin with device */
    auto xclbin = xrt::xclbin(xclbin_data);
    try {
        device.register_xclbin(xclbin);
    } catch (const std::exception& e) {
        printf("FAIL: register_xclbin: %s\n", e.what());
        return 1;
    }
    printf("xclbin registered\n");

    /* Get kernel from xclbin */
    auto uuid = xclbin.get_uuid();
    printf("xclbin UUID: %s\n", uuid.to_string().c_str());

    printf("Loading kernel 'MLIR_AIE'...\n");
    xrt::kernel kernel = xrt::kernel(device, uuid, "MLIR_AIE");
    printf("Kernel loaded\n");

    /* Allocate buffers (small GEMM: 64x64) */
    const int M = 64, N = 64, K = 64;
    const size_t size_a = M * K * sizeof(float);
    const size_t size_b = K * N * sizeof(float);
    const size_t size_c = M * N * sizeof(float);

    xrt::bo buf_a = xrt::bo(device, size_a, xrt::bo::flags::host_only, 0);
    xrt::bo buf_b = xrt::bo(device, size_b, xrt::bo::flags::host_only, 0);
    xrt::bo buf_c = xrt::bo(device, size_c, xrt::bo::flags::host_only, 0);

    float *a = buf_a.map<float*>();
    float *b = buf_b.map<float*>();
    float *c = buf_c.map<float*>();

    /* Fill with test data */
    srand(42);
    for (int i = 0; i < M * K; i++) a[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) b[i] = (float)(rand() % 100) / 100.0f;
    memset(c, 0, size_c);

    /* Sync to device */
    buf_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buf_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buf_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    /* Run kernel */
    printf("Running GEMM (%dx%dx%d) on NPU...\n", M, N, K);
    fflush(stdout);

    /* Build instruction buffer for the NPU GEMM */
    /* The DPU kernel expects: opcode, instr_buf, ninstr, bo0-bo4 */
    uint64_t opcode = 0;  /* GEMM opcode */
    std::vector<uint32_t> instr(1024, 0); /* instruction buffer */
    uint32_t ninstr = 0;  /* number of valid instructions */

    printf("Running NPU kernel...\n");
    fflush(stdout);

    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto run = kernel(opcode, instr.data(), ninstr,
                          buf_a, buf_b, buf_c,
                          xrt::bo(), xrt::bo());
        run.wait();
    } catch (const std::exception& e) {
        printf("FAIL: kernel execution: %s\n", e.what());
        printf("(the xclbin requires pre-compiled AIE instructions)\n");
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Kernel completed in %.3f ms\n", ms);

    /* Sync result back */
    buf_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    /* Verify */
    bool pass = true;
    for (int i = 0; i < M && pass; i++) {
        for (int j = 0; j < N && pass; j++) {
            float expected = 0;
            for (int k = 0; k < K; k++)
                expected += a[i * K + k] * b[k * N + j];
            if (fabs(c[i * N + j] - expected) > 0.1f) {
                printf("MISMATCH at [%d,%d]: got %f, expected %f\n",
                       i, j, c[i * N + j], expected);
                pass = false;
            }
        }
    }

    printf("\n%s\n", pass ? "✅ GEMM PASS" : "❌ GEMM FAIL");
    return pass ? 0 : 1;
}
