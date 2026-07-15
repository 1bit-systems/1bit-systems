// npu_minimal2.cpp — Add xclbin loading to minimal test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    size_t sz = file.tellg();
    file.seekg(0);
    std::vector<char> data(sz);
    file.read(data.data(), sz);
    return data;
}

int main() {
    printf("Test 1: device\n");
    xrt::device device;
    try { device = xrt::device(0); }
    catch (...) { try { device = xrt::device(1); } catch (...) { return 1; } }
    printf("OK\n");
    
    printf("\nTest 2: load xclbin\n");
    auto data = read_file("/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin");
    printf("  read %zu bytes\n", data.size());
    
    xrt::xclbin xclbin;
    try {
        xclbin = xrt::xclbin(data);
        printf("  xclbin parsed\n");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        return 1;
    }
    
    try {
        device.register_xclbin(xclbin);
        printf("  registered\n");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        return 1;
    }
    
    auto uuid = xclbin.get_uuid();
    printf("  UUID: %s\n", uuid.to_string().c_str());
    
    printf("\nTest 3: open kernel\n");
    xrt::kernel kernel;
    try {
        kernel = xrt::kernel(device, uuid, "MLIR_AIE");
        printf("OK\n");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        return 1;
    }
    
    printf("\nTest 4: create BOs\n");
    size_t sz = 4 * 1024 * 1024;
    xrt::bo bo0(device, sz, xrt::bo::flags::host_only, 0);
    xrt::bo bo1(device, sz, xrt::bo::flags::host_only, 0);
    printf("  BOs created\n");
    
    memset(bo0.map(), 0x42, sz);
    memset(bo1.map(), 0x43, sz);
    printf("  mapped and filled\n");
    
    bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, sz);
    bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, sz);
    printf("  synced to device\n");
    
    printf("\nTest 5: run kernel\n");
    try {
        auto run = kernel(
            (uint64_t)3, (uint64_t)0, (uint32_t)0,
            bo0, bo1, bo0, bo0, bo0
        );
        run.wait();
        printf("  completed\n");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        return 1;
    }
    
    bo0.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 64);
    printf("  BO0[0]=%02x\n", ((uint8_t*)bo0.map())[0]);
    
    printf("\nDONE\n");
    return 0;
}
