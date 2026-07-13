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
    
    printf("Test 2: load xclbin\n");
    auto data = read_file("/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin");
    printf("  read %zu bytes\n", data.size());
    
    {
        xrt::xclbin xclbin(data);
        device.register_xclbin(xclbin);
        printf("  registered\n");
    }
    
    // Test: create BOs and sync BEFORE opening kernel
    printf("Test 3: create BOs without kernel\n");
    size_t sz = 4 * 1024 * 1024;
    xrt::bo bo0(device, sz, xrt::bo::flags::host_only, 0);
    printf("  bo0 created\n");
    memset(bo0.map(), 0x42, 256);
    printf("  mapped\n");
    bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 256);
    printf("  synced OK\n");
    
    printf("Test 4: BO with bigger sync\n");
    memset(bo0.map(), 0x43, sz);
    bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, sz);
    printf("  synced %zu bytes OK\n", sz);
    
    printf("\nDONE\n");
    return 0;
}
