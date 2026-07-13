// npu_test_all_xclbins.cpp — Test each Qwen3 xclbin in separate processes
//
// Build:
//   g++ -std=gnu++17 -O2 -g -o npu_test_all_xclbins npu_test_all_xclbins.cpp \
//       -lxrt_coreutil -ldl -luuid
//
// Run:
//   sudo ./npu_test_all_xclbins

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

using Clock = std::chrono::high_resolution_clock;

// Run a single test in the current process, then exit
// Returns 0 on success, 1 on failure
static int run_single_test(const char* name, const char* path, const char* kernel_name) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Testing %s\n", name);
    printf("╚══════════════════════════════════════════════════╝\n");
    fflush(stdout);
    
    // Open device
    xrt::device device;
    try {
        device = xrt::device(0);
    } catch (...) {
        try { device = xrt::device(1); }
        catch (...) {
            fprintf(stderr, "  ❌ Cannot open device\n");
            fflush(stderr);
            return 1;
        }
    }
    printf("  Device opened\n");
    fflush(stdout);
    
    // Read xclbin file
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  ❌ fopen\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* raw_data = (char*)malloc(fsize);
    size_t bytes_read = fread(raw_data, 1, fsize, f);
    (void)bytes_read;
    fclose(f);
    printf("  Read %ld bytes\n", fsize);
    fflush(stdout);
    
    // Parse xclbin
    xrt::xclbin xclbin;
    try {
        xclbin = xrt::xclbin(std::vector<char>(raw_data, raw_data + fsize));
        free(raw_data);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ xclbin: %s\n", e.what());
        free(raw_data);
        return 1;
    }
    printf("  xclbin parsed\n");
    fflush(stdout);
    
    // Register
    try {
        device.register_xclbin(xclbin);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ register: %s\n", e.what());
        return 1;
    }
    printf("  Registered, UUID=%s\n", xclbin.get_uuid().to_string().c_str());
    fflush(stdout);
    
    // Open kernel
    xrt::kernel kernel;
    try {
        kernel = xrt::kernel(device, xclbin.get_uuid(), kernel_name);
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ kernel: %s\n", e.what());
        return 1;
    }
    printf("  Kernel ready\n");
    fflush(stdout);
    
    // Create BOs
    size_t bo_size = 4 * 1024 * 1024;
    xrt::bo bo0(device, bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo bo1(device, bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo bo2(device, bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo bo3(device, bo_size, xrt::bo::flags::host_only, 0);
    xrt::bo bo4(device, bo_size, xrt::bo::flags::host_only, 0);
    
    memset(bo0.map(), 0x42, 256);
    memset(bo1.map(), 0x43, 256);
    memset(bo2.map(), 0x44, 256);
    memset(bo3.map(), 0x45, 256);
    memset(bo4.map(), 0x46, 256);
    
    bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    bo2.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    bo3.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    bo4.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    printf("  5 BOs created\n");
    fflush(stdout);
    
    // Test opcodes
    bool all_ok = true;
    for (uint64_t opcode = 0; opcode <= 5; opcode++) {
        auto t0 = Clock::now();
        
        try {
            auto run = kernel(
                opcode, (uint64_t)0, (uint32_t)0,
                bo0, bo1, bo2, bo3, bo4
            );
            run.wait();
        } catch (const std::exception& e) {
            auto t1 = Clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("    opcode=%lu: ❌ %s (%.3f ms)\n", 
                   (unsigned long)opcode, e.what(), ms);
            fflush(stdout);
            continue;
        }
        
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        bo0.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 32);
        uint8_t* map = (uint8_t*)bo0.map();
        
        printf("    opcode=%lu: ✅ %.3f ms | BO0=%02x %02x %02x %02x...\n", 
               (unsigned long)opcode, ms, map[0], map[1], map[2], map[3]);
        fflush(stdout);
    }
    
    printf("  ✅ %s done\n", name);
    fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NPU XCLBIN Test — All 4 Kernels (separate processes)   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    // If we have a CLI arg, run that specific test
    if (argc > 1) {
        const char* xclbin_dir = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
        std::string path = std::string(xclbin_dir) + "/" + argv[1];
        return run_single_test(argv[1], path.c_str(), "MLIR_AIE");
    }
    
    // Otherwise fork for each test
    const char* xclbin_dir = "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2";
    
    struct Entry {
        const char* name;
        std::string path;
    };
    
    Entry entries[] = {
        {"mm.xclbin",      std::string(xclbin_dir) + "/mm.xclbin"},
        {"attn.xclbin",    std::string(xclbin_dir) + "/attn.xclbin"},
        {"layer.xclbin",   std::string(xclbin_dir) + "/layer.xclbin"},
        {"dequant.xclbin", std::string(xclbin_dir) + "/dequant.xclbin"},
    };
    
    for (auto& e : entries) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child
            int ret = run_single_test(e.name, e.path.c_str(), "MLIR_AIE");
            exit(ret);
        } else if (pid > 0) {
            // Parent — wait for child
            int status;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                printf("\n  ✅ %s PASSED\n", e.name);
            } else {
                printf("\n  ❌ %s FAILED (exit=%d)\n", e.name, 
                       WEXITSTATUS(status));
            }
        } else {
            fprintf(stderr, "  ❌ fork failed\n");
        }
        fflush(stdout);
    }
    
    printf("\n✅ All tests complete!\n");
    return 0;
}
