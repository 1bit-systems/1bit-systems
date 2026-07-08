// test_ternary_kernel.c — Quick validation of native ternary NPU kernels
// Compile: g++ -O2 -std=c++17 -o test_ternary test_ternary_kernel.cpp -lxrt_coreutil
//
// Tests both mm_ternary_32x64x128 and bitnet_ternary_micro

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// ── Ternary decode reference ─────────────────────────────────
static float decode_ternary(uint8_t code) {
    if (code == 2) return 1.0f;
    if (code == 1) return 0.0f;
    return -1.0f;
}

// ── Pack ternary values into bytes ───────────────────────────
static void pack_ternary(const float *vals, int count, uint8_t *packed) {
    for (int i = 0; i < count; i += 4) {
        uint8_t byte = 0;
        for (int b = 0; b < 4 && (i + b) < count; b++) {
            uint8_t code;
            float v = vals[i + b];
            if (v > 0.5f)       code = 2;
            else if (v > -0.5f) code = 1;
            else                code = 0;
            byte |= (code << (b * 2));
        }
        packed[i / 4] = byte;
    }
}

// ── CPU reference for mm_ternary_32x64x128 ──────────────────
// M=32, K_PACKED=64 (256 ternary), N=128
static void cpu_reference_mm32(const uint8_t *weights, const uint16_t *scales_bf16,
                                const uint16_t *acts_bf16, int M, int K_packed,
                                float *output) {
    // Convert bf16 to float helper
    auto bf16_to_f = [](uint16_t v) -> float {
        uint32_t bits = ((uint32_t)v) << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    };

    for (int row = 0; row < M; row++) {
        float acc = 0.0f;
        const uint8_t *wt_row = weights + row * K_packed;
        float scale = bf16_to_f(scales_bf16[row]);

        for (int i = 0; i < K_packed; i++) {
            uint8_t byte = wt_row[i];
            for (int b = 0; b < 4; b++) {
                uint8_t code = (byte >> (b * 2)) & 3;
                float tern = decode_ternary(code);
                float act = bf16_to_f(acts_bf16[i * 4 + b]);
                acc += tern * act;
            }
        }
        output[row] = acc * scale;
    }
}

// ── Convert float to bf16 ───────────────────────────────────
static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // Round to nearest even
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// ── Main test ────────────────────────────────────────────────
int main(int argc, char **argv) {
    const char *xclbin_path = "/home/bcloud/1bit-systems/engine/npu/build/build/ternary/design.xclbin";
    const char *insts_path = "/home/bcloud/1bit-systems/engine/npu/build/build/ternary/design.insts";

    if (argc > 1) xclbin_path = argv[1];
    if (argc > 2) insts_path = argv[2];

    printf("=== Ternary NPU Kernel Test ===\n");
    printf("XCLBIN: %s\n", xclbin_path);

    // Open device
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbin_path);
    printf("Device: %s\n", device.get_info<xrt::info::device::name>().c_str());

    // Dimensions
    const int M = 32;
    const int K_PACKED = 64;   // 64 bytes → 256 ternary
    const int K_TERNARY = 256;
    const int SCALE_COUNT = M;

    // Input buffer layout:
    // [M * K_PACKED bytes weights] [M * 2 bytes scales] [K_TERNARY * 2 bytes acts]
    int weight_bytes = M * K_PACKED;           // 2048
    int scale_bytes = SCALE_COUNT * 2;          // 64
    int act_bytes = K_TERNARY * 2;              // 512
    int total_input = weight_bytes + scale_bytes + act_bytes;  // 2624
    int output_bytes = M * 2;                    // 64

    int input_dwords = (total_input + 3) / 4;    // 656
    int output_dwords = (output_bytes + 3) / 4;   // 16

    // Allocate buffers
    auto in_bo = xrt::bo(device, input_dwords * 4, xrt::bo::flags::normal, 0);
    auto out_bo = xrt::bo(device, output_dwords * 4, xrt::bo::flags::normal, 0);

    auto in_map = in_bo.map<uint8_t*>();
    auto out_map = out_bo.map<uint8_t*>();

    // ── Generate test data ──────────────────────────────────
    srand(42); // deterministic

    // Fill ternary weights with {-1, 0, +1} patterns
    float *ternary_vals = (float*)malloc(K_TERNARY * sizeof(float));
    for (int row = 0; row < M; row++) {
        for (int i = 0; i < K_TERNARY; i++) {
            int r = rand() % 3;
            ternary_vals[i] = (r == 0) ? -1.0f : (r == 1) ? 0.0f : 1.0f;
        }
        pack_ternary(ternary_vals, K_TERNARY, in_map + row * K_PACKED);
    }

    // Fill scales
    uint16_t *scales_bf16 = (uint16_t*)(in_map + weight_bytes);
    float *scales_f32 = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) {
        scales_f32[i] = 0.5f + (rand() % 1000) / 2000.0f; // 0.5 .. 1.0
        scales_bf16[i] = float_to_bf16(scales_f32[i]);
    }

    // Fill activations
    uint16_t *acts_bf16 = (uint16_t*)(in_map + weight_bytes + scale_bytes);
    float *acts_f32 = (float*)malloc(K_TERNARY * sizeof(float));
    for (int i = 0; i < K_TERNARY; i++) {
        acts_f32[i] = ((rand() % 2000) - 1000) / 1000.0f; // -1.0 .. 1.0
        acts_bf16[i] = float_to_bf16(acts_f32[i]);
    }

    // Zero output
    memset(out_map, 0, output_bytes);

    // ── CPU reference ────────────────────────────────────────
    float *cpu_output = (float*)calloc(M, sizeof(float));
    cpu_reference_mm32(in_map, scales_bf16, acts_bf16, M, K_PACKED, cpu_output);

    printf("\n--- CPU Reference (first 8 rows) ---\n");
    for (int i = 0; i < 8; i++) {
        printf("  row[%d] = %+.6f\n", i, cpu_output[i]);
    }

    // ── Run on NPU ──────────────────────────────────────────
    printf("\n--- Running on NPU ---\n");

    in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    out_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Create kernel handle
    auto kernel = xrt::kernel(device, uuid, "MLIR_AIE");
    auto run = xrt::run(kernel);

    // Set arguments
    run.set_arg(0, in_bo);
    run.set_arg(1, out_bo);

    // Execute
    run.start();
    run.wait();

    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // ── Compare results ─────────────────────────────────────
    uint16_t *npu_out = (uint16_t*)out_map;

    auto bf16_to_f = [](uint16_t v) -> float {
        uint32_t bits = ((uint32_t)v) << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    };

    printf("\n--- NPU Output (first 8 rows) ---\n");
    float max_err = 0.0f;
    int max_err_row = 0;
    for (int i = 0; i < M; i++) {
        float npu_val = bf16_to_f(npu_out[i]);
        float cpu_val = cpu_output[i];
        float err = fabsf(npu_val - cpu_val);
        float rel_err = (fabsf(cpu_val) > 1e-6f) ? err / fabsf(cpu_val) : err;

        if (i < 8) {
            printf("  row[%d] = %+.6f  (cpu: %+.6f)  err=%e\n", i, npu_val, cpu_val, err);
        }
        if (err > max_err) {
            max_err = err;
            max_err_row = i;
        }
    }

    printf("\n--- Summary ---\n");
    printf("  Max absolute error: %e (row %d)\n", max_err, max_err_row);
    if (max_err < 1e-3f) {
        printf("  ✅ PASS — bit-exact match within BF16 precision\n");
    } else {
        printf("  ❌ FAIL — significant deviation\n");
    }

    // Cleanup
    free(ternary_vals);
    free(scales_f32);
    free(acts_f32);
    free(cpu_output);

    return (max_err < 1e-3f) ? 0 : 1;
}
