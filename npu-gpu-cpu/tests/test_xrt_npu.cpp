// SPDX-License-Identifier: MIT
/*
 * Test: run NPU kernel via XRT AIE API
 *
 * Uses XRT to open the NPU device, load an AIE control code,
 * and execute it. This demonstrates the NPU kernel dispatch path.
 *
 * Build:
 *   g++ -std=gnu++17 -o test_xrt_npu test_xrt_npu.cpp \
 *       $(pkg-config --cflags --libs xrt) -lxrt_coreutil
 *
 * Run:
 *   ./test_xrt_npu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main() {
    printf("=== NPU Kernel Dispatch via XRT ===\n\n");

    /* Open NPU device (amdxdna) */
    xrt::device device;
    try {
        // XRT device index 1 = NPU (index 0 is usually GPU)
        device = xrt::device(1);
    } catch (const std::exception& e) {
        printf("Failed to open NPU device: %s\n", e.what());
        printf("Trying device 0...\n");
        try {
            device = xrt::device(0);
        } catch (...) {
            printf("No NPU device found via XRT\n");
            return 1;
        }
    }

    std::string dev_name;
    try { dev_name = device.get_info<xrt::info::device::name>(); }
    catch (...) { dev_name = "unknown"; }
    printf("Device: %s\n", dev_name.c_str());

    /* Allocate buffer (64 KiB GTT) */
    const size_t BUF_SIZE = 64 * 1024;
    xrt::bo buf = xrt::bo(device, BUF_SIZE, xrt::bo::flags::normal, 0);
    void *buf_ptr = buf.map();
    printf("Buffer allocated: %zu bytes at %p\n", BUF_SIZE, buf_ptr);

    /* Write test pattern */
    memset(buf_ptr, 0xAB, BUF_SIZE);
    printf("Pattern written\n");

    /* Sync buffer to device */
    buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("Buffer synced to NPU\n");

    // For a real NPU kernel, you would:
    // 1. Load an xclbin (compiled NPU program)
    // 2. Create a kernel object
    // 3. Set kernel arguments (pointers to buffers)
    // 4. Run the kernel
    // 5. Sync buffer back
    //
    // Example (commented out, needs compiled xclbin):
    // auto xclbin = xrt::xclbin("npu_gemm.xclbin");
    // device.register_xclbin(xclbin);
    // auto kernel = xrt::kernel(device, xclbin.get_uuid(), "gemm");
    // auto run = kernel(buf, N, M, K);
    // run.wait();

    printf("\nNPU device detected and buffer allocated successfully.\n");
    printf("Full kernel dispatch requires a compiled NPU xclbin.\n");
    return 0;
}
