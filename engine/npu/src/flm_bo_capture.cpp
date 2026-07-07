/**
 * Override weak FLM symbols to capture BOs after weight loading.
 *
 * npu_app_manager::npu_app_manager and npu_xclbin_manager::register_xclbin
 * are weak symbols in libqwen3_npu.so. We override them to:
 * 1. Track the xrt::device* used by FLM
 * 2. Capture the xrt::bo objects created during weight loading
 * 3. Generate fused instruction sequences using the captured BOs
 * 4. Submit to NPU via layer.xclbin (1 launch/layer instead of 4)
 *
 * Build:
 *   g++ -shared -fPIC -O2 -o flm_bo_capture.so flm_bo_capture.cpp \
 *       -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -ldl
 *
 * Run:
 *   LD_PRELOAD=./flm_bo_capture.so flm serve qwen3:0.6b
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <dlfcn.h>
#include <chrono>

// ─── Forward declarations ───
namespace xrt {
class device;
class bo;
}

// We store BO info as opaque handles to avoid needing xrt::bo definition
struct BOHandle {
    void* impl;  // xrt::bo internal impl pointer
    size_t size;
    int group_id;
    void* bo_self; // the this pointer of xrt::bo
};

// ─── Global state ───
static FILE* logf = nullptr;
static std::mutex mtx;
static xrt::device* g_xrt_device = nullptr;
static std::vector<BOHandle> g_weight_bos;  // captured weight BOs
static bool g_weights_loaded = false;

// ─── Override weak npu_app_manager constructor ───
// Original: npu_app_manager::npu_app_manager(npu_device, xrt::device*, string, bool)
// This is a WEAK symbol — our strong definition overrides it.
// We save the device pointer and call the original via RTLD_NEXT.

struct npu_app_manager {
    // We don't need to know the layout — we just call the original
    // constructor from the library via dlsym.
};

// Constructor type: npu_app_manager(void* this, int npu_dev, xrt::device* dev, 
//                                     const std::string& xclbin, bool flag)
typedef void (*app_mgr_ctor_t)(void*, int, void*, const std::string&, bool);
static app_mgr_ctor_t real_app_mgr_ctor = nullptr;

// Our override — strong symbol replaces weak
extern "C" void _ZN15npu_app_managerC1E10npu_devicePN3xrt6deviceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb(
    void* self, int npu_dev, xrt::device* dev, 
    const std::string& xclbin, bool flag)
{
    // Lazily resolve real constructor on first call
    if (!real_app_mgr_ctor) {
        // Open the library directly to get the original constructor
        void* lib = dlopen("/opt/fastflowlm/lib/flm/libqwen3_npu.so", RTLD_LAZY | RTLD_NOLOAD);
        if (lib) {
            real_app_mgr_ctor = (app_mgr_ctor_t)dlsym(lib, 
                "_ZN15npu_app_managerC1E10npu_devicePN3xrt6deviceENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb");
            dlclose(lib);
        }
    }
    
    // Save device for later use
    g_xrt_device = (xrt::device*)dev;
    
    printf("[capture] npu_app_manager ctor: dev=%p xclbin=%s\n", 
           (void*)dev, xclbin.c_str());
    fflush(stdout);
    
    // Call original constructor (captured from the library, not via RTLD_NEXT)
    if (real_app_mgr_ctor)
        real_app_mgr_ctor(self, npu_dev, dev, xclbin, flag);
    else
        printf("[capture] WARNING: no original ctor found, NPU may not work\n");
}

// ─── Override npu_xclbin_manager::register_xclbin ───
// Also a weak symbol. We intercept to capture BOs after registration.

typedef void (*mgr_reg_t)(void*, const std::string&);
static mgr_reg_t real_mgr_reg = nullptr;

extern "C" void _ZN18npu_xclbin_manager15register_xclbinENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    void* self, const std::string& path)
{
    if (!real_mgr_reg)
        real_mgr_reg = (mgr_reg_t)dlsym(RTLD_NEXT,
            "_ZN18npu_xclbin_manager15register_xclbinENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    
    printf("[capture] register_xclbin: path=%s\n", path.c_str());
    fflush(stdout);
    
    if (real_mgr_reg)
        real_mgr_reg(self, path);
}

// ─── Override xrt::bo::sync to track when weights are uploaded ───
// xrt::bo::sync(xclBOSyncDirection, size_t, size_t)
// mangled: _ZN3xrt2bo4syncE18xclBOSyncDirectionmm

typedef void (*bo_sync_t)(void*, int, size_t, size_t);
static bo_sync_t real_bo_sync = nullptr;

extern "C" void _ZN3xrt2bo4syncE18xclBOSyncDirectionmm(
    void* self, int dir, size_t offset, size_t size)
{
    if (!real_bo_sync)
        real_bo_sync = (bo_sync_t)dlsym(RTLD_NEXT, 
            "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm");
    
    // Log large uploads to device (BO_TO_DEVICE = 0)
    if (dir == 0 && size > 1024*1024) {
        printf("[capture] bo->sync(TO_DEVICE, off=%zu, sz=%zu=%.1fMB) self=%p impl=%p\n",
               offset, size, size/1048576.0, self, *(void**)self);
        fflush(stdout);
    }
    
    if (real_bo_sync)
        real_bo_sync(self, dir, offset, size);
}

// ─── Poll for completion and dump state ───
__attribute__((destructor))
void fini() {
    printf("\n=== FLM BO Capture Summary ===\n");
    printf("  XRT device: %p\n", (void*)g_xrt_device);
    printf("  Weights loaded: %s\n", g_weights_loaded ? "yes" : "no");
    printf("  Captured BOs: %zu\n", g_weight_bos.size());
    printf("==============================\n");
    fflush(stdout);
}

__attribute__((constructor))
void init() {
    printf("\n=== FLM BO Capture Loaded ===\n");
    printf("Overriding weak FLM symbols to capture weight BOs...\n");
    fflush(stdout);
}
