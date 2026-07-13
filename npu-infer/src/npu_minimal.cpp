// npu_minimal.cpp — Minimal XRT test to isolate crash
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

int main() {
    printf("Test 1: device\n");
    xrt::device device;
    try { device = xrt::device(0); }
    catch (...) { try { device = xrt::device(1); } catch (...) { return 1; } }
    printf("OK\n");
    
    printf("Test 2: create bo\n");
    size_t sz = 4 * 1024 * 1024;
    try {
        xrt::bo bo(device, sz, xrt::bo::flags::host_only, 0);
        printf("  bo created\n");
        
        printf("Test 3: map\n");
        void* map = bo.map();
        printf("  map=%p\n", map);
        
        printf("Test 4: memset\n");
        memset(map, 0x42, sz);
        printf("OK\n");
        
        printf("Test 5: sync to device\n");
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 0, sz);
        printf("OK\n");
        
        printf("Test 6: sync back\n");
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 64);
        printf("  BO[0]=%02x\n", ((uint8_t*)map)[0]);
        
        printf("Test 7: new scope\n");
        {
            xrt::bo bo2(device, sz, xrt::bo::flags::host_only, 0);
            printf("  bo2 created\n");
        }
        printf("  bo2 destroyed\n");
        
    } catch (const std::exception& e) {
        printf("  EXCEPTION: %s\n", e.what());
    }
    
    printf("DONE\n");
    return 0;
}
