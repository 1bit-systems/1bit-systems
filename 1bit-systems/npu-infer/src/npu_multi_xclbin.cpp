// npu_multi_xclbin.cpp — Load all 4 Qwen3 xclbins and test execution
//
// Tests: mm.xclbin (GEMM), attn.xclbin (Attention), layer.xclbin (Transformer), dequant.xclbin (Dequant)
//
// Build:
//   g++ -std=gnu++17 -O2 -g -o npu_multi_xclbin npu_multi_xclbin.cpp \
//       -lxrt_coreutil -ldl -luuid
//
// Run:
//   sudo ./npu_multi_xclbin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <array>

#include <drm/amdxdna_accel.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

using Clock = std::chrono::high_resolution_clock;

static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    size_t sz = file.tellg();
    file.seekg(0);
    std::vector<char> data(sz);
    file.read(data.data(), sz);
    return data;
}

struct XclbinKernel {
    std::string name;
    xrt::xclbin xclbin;
    xrt::uuid uuid;
    xrt::kernel kernel;
    std::vector<xrt::bo> bos;
};

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NPU Multi-XCLBIN Test — All 4 Qwen3 kernels            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    // Open XRT device once
    printf("Opening XRT device...\n");
    xrt::device device;
    try {
        device = xrt::device(0);
    } catch (...) {
        fprintf(stderr, "  ❌ xrt::device(0) failed\n");
        return 1;
    }
    printf("  ✅ Device opened\n\n");
    
    // List of xclbins to test
    struct XclbinEntry {
        const char* path;
        const char* name;
        const char* kernel_name;
    };
    
    std::vector<XclbinEntry> xclbins = {
        {"/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin",      "mm",      "MLIR_AIE"},
        {"/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/attn.xclbin",    "attn",    "MLIR_AIE"},
        {"/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin",   "layer",   "MLIR_AIE"},
        {"/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/dequant.xclbin", "dequant", "MLIR_AIE"},
    };
    
    // Use separate device instances for each xclbin (they use different hw contexts)
    std::vector<XclbinKernel> kernels;
    
    for (auto& entry : xclbins) {
        printf("Loading %s: %s\n", entry.name, entry.path);
        
        auto data = read_file(entry.path);
        if (data.empty()) {
            fprintf(stderr, "  ❌ Cannot read file\n");
            continue;
        }
        
        try {
            // Each xclbin needs its own device handle to get separate hw context
            xrt::device dev = xrt::device(0);
            
            XclbinKernel k;
            k.name = entry.name;
            k.xclbin = xrt::xclbin(data);
            dev.register_xclbin(k.xclbin);
            k.uuid = k.xclbin.get_uuid();
            k.kernel = xrt::kernel(dev, k.uuid, entry.kernel_name);
            
            // Create 5 BOs
            size_t bo_size = 4 * 1024 * 1024;
            for (int i = 0; i < 5; i++) {
                xrt::bo bo(dev, bo_size, xrt::bo::flags::host_only, 0);
                memset(bo.map(), 0, bo_size);
                bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, bo_size);
                k.bos.push_back(std::move(bo));
            }
            
            printf("  ✅ UUID=%s, 5 BOs created\n", k.uuid.to_string().c_str());
            kernels.push_back(std::move(k));
        } catch (const std::exception& e) {
            fprintf(stderr, "  ❌ Failed: %s\n", e.what());
        }
    }
    
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  Running all kernels with opcode=0..5                ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    
    for (auto& k : kernels) {
        printf("=== %s (opcode test) ===\n", k.name.c_str());
        
        for (uint64_t opcode = 0; opcode <= 5; opcode++) {
            // Reset BOs
            for (int i = 0; i < 5; i++) {
                uint8_t* map = (uint8_t*)k.bos[i].map();
                memset(map, 0x42 + i, 64);
                k.bos[i].sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 64);
            }
            
            auto t0 = Clock::now();
            bool success = true;
            
            try {
                auto run = k.kernel(
                    opcode,           // opcode
                    (uint64_t)0,      // instr
                    (uint32_t)0,      // ninstr
                    k.bos[0], k.bos[1], k.bos[2], k.bos[3], k.bos[4]
                );
                run.wait();
            } catch (const std::exception& e) {
                success = false;
                auto t1 = Clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                printf("  opcode=%lu: ❌ %s (%.3f ms)\n", 
                       (unsigned long)opcode, e.what(), ms);
            }
            
            if (success) {
                auto t1 = Clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                
                // Read back
                for (int i = 0; i < 5; i++) {
                    k.bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 64);
                }
                
                printf("  opcode=%lu: ✅ %.3f ms | ", (unsigned long)opcode, ms);
                for (int i = 0; i < 5; i++) {
                    uint8_t* map = (uint8_t*)k.bos[i].map();
                    printf("BO%d: %02x%02x%02x%02x ", i, map[0], map[1], map[2], map[3]);
                }
                printf("\n");
            }
        }
        printf("\n");
    }
    
    printf("✅ All tests complete!\n");
    return 0;
}
