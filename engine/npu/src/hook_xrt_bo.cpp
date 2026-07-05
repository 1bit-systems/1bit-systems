/**
 * LD_PRELOAD hook — intercepts xrt::bo construction to capture
 * FLM's weight BOs, then generates + submits fused instructions.
 *
 * Build:
 *   g++ -shared -fPIC -O2 -o hook_xrt_bo.so hook_xrt_bo.cpp -ldl
 *
 * Usage:
 *   LD_PRELOAD=./hook_xrt_bo.so flm serve qwen3:0.6b ...
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <dlfcn.h>
#include <chrono>

// ─── Intercept xrt::bo constructor ───
// xrt::bo has several constructors. The one FLM uses for weights is:
//   xrt::bo(xrt::device&, size_t, xrt::bo::flags, int group_id)
// We intercept the underlying C function.
//
// The xrt::bo object is ~24 bytes internally. The constructor
// is: _ZN3xrt2boC1ERNS_6deviceEmNS0_5flagsEi
//     (xrt::bo::bo(xrt::device&, size_t, flags, int))

// Capture all BOs created by FLM
static std::mutex bo_mutex;
static std::vector<std::pair<void*, size_t>> captured_bos; // (bo_handle, size)
static bool capturing = false;

extern "C" {

// The xrt::bo constructor called by FLM for weight allocation
// We intercept via _ZN3xrt2boC1ERNS_6deviceEmjNS0_5flagsE  (4-arg version)
// or _ZN3xrt2boC1ERNS_6deviceEmNS0_5flagsEi (the common one)
//
// Since xrt::bo uses shared_ptr-like internals, we just intercept
// the construction signature and save the handle.

// Actually xrt::bo::bo is inline in the header, so it doesn't
// produce a symbol. But it calls xrt_core::bo_alloc which IS exported.
// Let me intercept at the xrt_core level.

// The actual allocation function in libxrt_coreutil:
// xrt_core::bo::alloc(xrt::device*, size_t, int, unsigned long long)
void* xrt_core_bo_alloc(void* device, size_t size, int group_id, unsigned long long flags) {
    static auto real_alloc = (void* (*)(void*, size_t, int, unsigned long long))
        dlsym(RTLD_NEXT, "_ZN7xrt_core2bo5allocEPNS_6deviceEmjm");
    
    // Capture large allocations (>1MB = weight buffers)
    if (size > 1024*1024) {
        // Call real allocator first
        printf("[hook] xrt_core::bo::alloc(dev=%p, size=%zu=%.1fMB, gid=%d, flags=0x%llx)\n",
               device, size, size/1048576.0, group_id, (unsigned long long)flags);
        fflush(stdout);
    }
    
    void* result = real_alloc(device, size, group_id, flags);
    
    if (size > 1024*1024) {
        printf("[hook]   → handle=%p\n", result);
        fflush(stdout);
    }
    
    return result;
}

} // extern "C"

// ─── Initialization ───
__attribute__((constructor))
void init() {
    printf("\n=== XRT BO Hook loaded ===\n");
    printf("Intercepting xrt::bo allocations...\n");
    fflush(stdout);
}
