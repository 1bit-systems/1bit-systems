// test_ternary_npu.cpp — Quick validation of native ternary NPU kernels
// Uses correct XRT NPU API: register_xclbin + hw_context + group_id BOs
//
// Compile: g++ -O2 -std=c++17 -o test_ternary_npu test_ternary_npu.cpp -lxrt_coreutil

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <memory>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// ── BF16 conversion ─────────────────────────────────────────
static float bf16_to_f(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// ── CPU reference ───────────────────────────────────────────
static void cpu_ref(const uint8_t *weights, const uint16_t *scales,
                    const uint16_t *acts, int M, int K_packed, float *out) {
    for (int row = 0; row < M; row++) {
        float acc = 0.0f;
        float scale = bf16_to_f(scales[row]);
        for (int i = 0; i < K_packed; i++) {
            uint8_t byte = weights[row * K_packed + i];
            for (int b = 0; b < 4; b++) {
                uint8_t code = (byte >> (b * 2)) & 3;
                float tern = (code == 2) ? 1.0f : (code == 1) ? 0.0f : -1.0f;
                acc += tern * bf16_to_f(acts[i * 4 + b]);
            }
        }
        out[row] = acc * scale;
    }
}

int main(int argc, char **argv) {
    const char *xclbin_path = "/home/bcloud/1bit-systems/engine/npu/build/build/ternary/design.xclbin";
    const char *insts_path  = "/home/bcloud/1bit-systems/engine/npu/build/build/ternary/design.insts";

    if (argc > 1) xclbin_path = argv[1];
    if (argc > 2) insts_path = argv[2];

    printf("=== Native Ternary NPU Kernel Test ===\n");
    printf("XCLBIN : %s\n", xclbin_path);
    printf("INSTS  : %s\n", insts_path);

    // Dimensions
    const int M = 32, K_PACKED = 64, K_TERNARY = 256;
    int wbytes = M * K_PACKED;           // 2048
    int sbytes = M * 2;                  // 64
    int abytes = K_TERNARY * 2;          // 512
    int in_bytes = wbytes + sbytes + abytes;  // 2624
    int out_bytes = M * 2;               // 64

    // Open device
    auto device = xrt::device(0);
    printf("Device : %s\n", device.get_info<xrt::info::device::name>().c_str());

    // Load xclbin
    auto xclbin = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xclbin);
    auto uuid = xclbin.get_uuid();
    auto hw_ctx = xrt::hw_context(device, uuid);
    auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");

    printf("Kernel : MLIR_AIE (groups: %d)\n", kernel.group_id(5));

    // Load instructions
    FILE *f = fopen(insts_path, "rb");
    if (!f) { printf("ERROR: cannot open %s\n", insts_path); return 1; }
    fseek(f, 0, SEEK_END);
    long isz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> insts(isz / 4);
    fread(insts.data(), 4, insts.size(), f);
    fclose(f);
    printf("Instructions: %zu dwords\n", insts.size());

    // Instruction BO → arg 1 (SRAM), opcode → arg 0, ninstr → arg 2
    auto bo_insts = xrt::bo(device, insts.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bo_insts.map(), insts.data(), insts.size() * 4);
    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Data BOs: bo0 → arg 3, bo1 → arg 4
    auto bo_in = xrt::bo(device, in_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_out = xrt::bo(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    auto in_map = reinterpret_cast<uint8_t*>(bo_in.map());
    auto out_map = reinterpret_cast<uint8_t*>(bo_out.map());

    // Generate test data
    srand(42);
    for (int row = 0; row < M; row++) {
        for (int i = 0; i < K_PACKED; i++) {
            uint8_t byte = 0;
            for (int b = 0; b < 4; b++) {
                int r = rand() % 3;
                uint8_t code = (r == 0) ? 0 : (r == 1) ? 1 : 2;  // 0→-1, 1→0, 2→+1
                byte |= (code << (b * 2));
            }
            in_map[row * K_PACKED + i] = byte;
        }
    }

    uint16_t *scales = (uint16_t*)(in_map + wbytes);
    for (int i = 0; i < M; i++)
        scales[i] = f_to_bf16(0.5f + (rand() % 1000) / 2000.0f);

    uint16_t *acts = (uint16_t*)(in_map + wbytes + sbytes);
    for (int i = 0; i < K_TERNARY; i++)
        acts[i] = f_to_bf16(((rand() % 2000) - 1000) / 1000.0f);

    memset(out_map, 0, out_bytes);

    // CPU reference
    float cpu_out[M];
    cpu_ref(in_map, scales, acts, M, K_PACKED, cpu_out);

    // NPU run
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    // arg 0: opcode (0 = run)
    // arg 1: instructions BO
    // arg 2: ninstr (dword count)
    // arg 3: bo0 (input data)
    // arg 4: bo1 (output data)
    uint64_t opcode = 0;
    uint32_t ninstr = (uint32_t)insts.size();
    run.set_arg(0, opcode);
    run.set_arg(1, bo_insts);
    run.set_arg(2, ninstr);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Compare
    uint16_t *npu_out = (uint16_t*)out_map;
    float max_err = 0;
    printf("\n  Row |        NPU |        CPU |     Error\n");
    printf("  ----+------------+------------+----------\n");
    for (int i = 0; i < M; i++) {
        float npu_val = bf16_to_f(npu_out[i]);
        float err = fabsf(npu_val - cpu_out[i]);
        if (err > max_err) max_err = err;
        if (i < 8 || err > max_err * 0.5f)
            printf("  %3d | %+10.6f | %+10.6f | %8.2e\n", i, npu_val, cpu_out[i], err);
    }

    printf("\n  Max error: %e\n", max_err);
    printf("  %s\n", max_err < 1e-3f ? "✅ PASS" : "❌ FAIL");

    return (max_err < 1e-3f) ? 0 : 1;
}
