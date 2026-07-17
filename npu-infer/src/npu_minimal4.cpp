#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

int main() {
    printf("Test 1: device\n");
    xrt::device device;
    try { device = xrt::device(0); }
    catch (...) { try { device = xrt::device(1); } catch (...) { return 1; } }
    printf("OK\n");
    
    printf("Test 2: read xclbin\n");
    // Read and parse in one step without std::vector going out of scope
    std::ifstream file("/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin", 
                        std::ios::binary | std::ios::ate);
    size_t sz = file.tellg();
    file.seekg(0);
    char* data = new char[sz];
    file.read(data, sz);
    file.close();
    printf("  read %zu bytes\n", sz);
    
    printf("Test 3: parse xclbin\n");
    xrt::xclbin xclbin;
    try {
        xclbin = xrt::xclbin(std::vector<char>(data, data + sz));
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        delete[] data;
        return 1;
    }
    delete[] data;  // Free xclbin data after parsing
    printf("  parsed\n");
    
    printf("Test 4: register\n");
    try {
        device.register_xclbin(xclbin);
        printf("  registered\n");
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
        return 1;
    }
    
    printf("Test 5: create BO\n");
    xrt::bo bo(device, 4096, xrt::bo::flags::host_only, 0);
    printf("  created\n");
    
    printf("Test 6: sync\n");
    memset(bo.map(), 0x42, 4096);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, 4096);
    printf("  synced OK\n");
    
    printf("\nDONE\n");
    return 0;
}
