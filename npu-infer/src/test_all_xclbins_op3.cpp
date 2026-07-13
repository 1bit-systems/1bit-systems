#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

static inline uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}
static inline float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int main(int argc, char** argv) {
    const char* xclbin_names[] = {"mm", "attn", "layer", "dequant"};
    int xclbin_idx = 0;
    if (argc > 1) {
        for (int i = 0; i < 4; i++) {
            if (strcmp(argv[1], xclbin_names[i]) == 0) { xclbin_idx = i; break; }
        }
    }
    
    printf("=== Testing xclbin: %s ===\n\n", xclbin_names[xclbin_idx]);
    
    xrt::device device(0);
    std::string path = std::string("/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/") + xclbin_names[xclbin_idx] + ".xclbin";
    
    FILE* f = fopen(path.c_str(), "rb");
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsize);
    fread(raw.data(), 1, fsize, f); fclose(f);
    
    xrt::xclbin xclbin(raw);
    device.register_xclbin(xclbin);
    xrt::kernel kernel(device, xclbin.get_uuid(), "MLIR_AIE");
    
    printf("Kernel ready (%s, %ld KB)\n", xclbin_names[xclbin_idx], fsize/1024);
    
    // Create BOs
    size_t BO_SIZE = 4 * 1024 * 1024;
    xrt::bo act_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo ws_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w1_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo w2_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    xrt::bo kv_bo(device, BO_SIZE, xrt::bo::flags::host_only, (xrt::memory_group)0);
    
    printf("BOs ready\n");
    
    // Fill with test patterns
    uint16_t* act = (uint16_t*)act_bo.map();
    for (int i = 0; i < 256; i++) act[i] = float_to_bf16((float)(i+1));
    act_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 4096, 0);
    
    uint16_t* w1 = (uint16_t*)w1_bo.map();
    for (size_t i = 0; i < BO_SIZE/2; i++) w1[i] = float_to_bf16(0.5f);
    w1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, BO_SIZE, 0);
    
    // Run opcode=3 
    printf("Running opcode=3...\n");
    auto t0 = std::chrono::steady_clock::now();
    try {
        auto run = kernel((uint64_t)3, (uint64_t)0, (uint32_t)0, act_bo, ws_bo, w1_bo, w2_bo, kv_bo);
        run.wait();
    } catch (const std::exception& e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    
    act_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 128, 0);
    
    printf("Exec: %.3f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
    printf("Out[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", bf16_to_float(act[i]));
    printf("\n");
    
    // Check if any change
    int diff = 0;
    for (int i = 0; i < 256; i++) if (act[i] != float_to_bf16((float)(i+1))) diff++;
    printf("Non-identity: %d/256\n", diff);
    
    printf("\nDONE\n");
    return 0;
}
