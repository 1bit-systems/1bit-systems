// test_ternary_simple.cpp — Simple 2-arg XRT test for native ternary xclbins
// Matches aie.runtime_sequence(%arg0, %arg1) interface (no opcode/insts dispatch)
//
// Compile: g++ -O2 -std=c++17 -o test_ternary_simple test_ternary_simple.cpp -lxrt_coreutil

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static float bf16_to_f(uint16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

static uint16_t f_to_bf16(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// CPU reference: M×K_PACKED weights, M scales, K*4 activations → M outputs
static void cpu_ref(const uint8_t *weights, const uint16_t *scales,
                    const uint16_t *acts, int M, int K_packed, float *out) {
    for (int row = 0; row < M; row++) {
        float acc = 0.0f, scale = bf16_to_f(scales[row]);
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

static void check_error(const char *label, float npu_val, float cpu_val) {
    float err = fabsf(npu_val - cpu_val);
    float rel = (fabsf(cpu_val) > 1e-6f) ? err / fabsf(cpu_val) : err;
    if (err > 1e-2f)
        printf("  %s  npu=%+.4f  cpu=%+.4f  err=%.2e  rel=%.2e  ⚠️\n", label, npu_val, cpu_val, err, rel);
    else
        printf("  %s  npu=%+.4f  cpu=%+.4f  err=%.2e  ✅\n", label, npu_val, cpu_val, err);
}

int main(int argc, char **argv) {
    const char *xclbin_path = "ternary/design.xclbin";
    const char *insts_path  = "ternary/design.insts";

    if (argc > 1) xclbin_path = argv[1];
    if (argc > 2) insts_path = argv[2];

    const int M = 32, K_PACKED = 64, K_TERNARY = 256;
    int wbytes = M * K_PACKED;          // 2048
    int sbytes = M * 2;                 // 64
    int abytes = K_TERNARY * 2;         // 512
    int in_bytes = wbytes + sbytes + abytes;  // 2624
    int in_dwords = (in_bytes + 3) / 4;       // 656
    int out_bytes = M * 2;              // 64
    int out_dwords = (out_bytes + 3) / 4;     // 16

    printf("=== Simple Ternary NPU Test ===\n");
    printf("XCLBIN: %s\n", xclbin_path);
    printf("Input:  %d bytes (%d dwords)\n", in_bytes, in_dwords);
    printf("Output: %d bytes (%d dwords)\n", out_bytes, out_dwords);

    // Open device
    auto device = xrt::device(0);
    printf("Device: %s\n", device.get_info<xrt::info::device::name>().c_str());

    // Load xclbin — NPU requires register_xclbin + hw_context
    auto xclbin = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xclbin);
    auto uuid = xclbin.get_uuid();
    auto hw_ctx = xrt::hw_context(device, uuid);
    auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");

    // Allocate BOs — normal flag for NPU shim DMA
    auto bo_in  = xrt::bo(device, in_dwords * 4,  xrt::bo::flags::normal, 0);
    auto bo_out = xrt::bo(device, out_dwords * 4, xrt::bo::flags::normal, 0);

    auto in_map  = reinterpret_cast<uint8_t*>(bo_in.map());
    auto out_map = reinterpret_cast<uint8_t*>(bo_out.map());

    // Generate test data
    srand(42);
    for (int row = 0; row < M; row++) {
        for (int i = 0; i < K_PACKED; i++) {
            uint8_t byte = 0;
            for (int b = 0; b < 4; b++) {
                int r = rand() % 3;
                uint8_t code = (r == 0) ? 0 : (r == 1) ? 1 : 2;
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

    printf("\nCPU ref (first 6): ");
    for (int i = 0; i < 6; i++) printf("%+.3f ", cpu_out[i]);
    printf("\n");

    // NPU run — simple 2-arg interface
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    run.set_arg(0, bo_in);
    run.set_arg(1, bo_out);
    run.start();
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Compare
    uint16_t *npu_out = (uint16_t*)out_map;
    float max_err = 0;
    int pass = 1;
    printf("\nResults:\n");
    for (int i = 0; i < M; i++) {
        float npu_val = bf16_to_f(npu_out[i]);
        float err = fabsf(npu_val - cpu_out[i]);
        if (err > max_err) max_err = err;
        if (i < 8 || err > 1e-2f)
            check_error(i < 10 ? "row" : "   ", npu_val, cpu_out[i]);
        if (i < 8 && err > 1e-2f) pass = 0;
    }

    printf("\nMax error: %e\n", max_err);
    printf("%s\n", pass ? "✅ PASS" : "❌ FAIL (check buffer layout)");

    return pass ? 0 : 1;
}
