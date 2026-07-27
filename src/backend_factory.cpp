// backend_factory.cpp — Auto-detect and create best available backend
// Uses dlsym to load GPU/NPU backends at runtime (like ORT EP discovery).
// CPU backend is statically linked (pure C++, no GPU deps).
// This means the backend_manager library never links against HIP, Vulkan, or XRT.

#include "backend.h"
#include "backend_detect.h"
#include "backend_cpu_stub.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <dlfcn.h>
#include <sys/stat.h>
#include <dirent.h>

// Any /dev/accel/accelN node present — the index isn't stable across driver
// resets (the XDNA driver has been observed renumbering accel0 -> accel1
// after an IOMMU page fault / device reset), so don't hardcode accel0.
static bool any_accel_device_present() {
    DIR* d = opendir("/dev/accel");
    if (!d) return false;
    bool found = false;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (strncmp(e->d_name, "accel", 5) == 0) { found = true; break; }
    }
    closedir(d);
    return found;
}

// ── dlsym-based backend loading ──
// Loads create_*_backend() from the rocm_cpp shared library or standalone .so.
// This is exactly how ONNX Runtime loads Execution Providers.

static void* open_backend_lib(const char* lib_name) {
    void* lib = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        // Try alternate names
        std::string alt = std::string("lib") + lib_name;
        lib = dlopen(alt.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            alt = std::string("./") + lib_name;
            lib = dlopen(alt.c_str(), RTLD_NOW | RTLD_LOCAL);
        }
    }
    return lib;
}

static Backend* try_load_backend(const char* lib_name, const char* symbol) {
    void* lib = open_backend_lib(lib_name);
    if (!lib) return nullptr;

    auto* creator = (Backend* (*)())dlsym(lib, symbol);
    if (!creator) {
        dlclose(lib);
        return nullptr;
    }

    Backend* b = creator();
    if (!b) {
        dlclose(lib);
        return nullptr;
    }

    return b;
}

// ── Forward declarations (CPU and Mamba1 are always linked) ──
extern Backend* create_cpu_backend();

// Mamba1 GPU backend — uses HIP kernels from mamba1_engine.hip
// Linked directly when ROCM_CPP_STATIC_HIP is defined.
extern "C" Backend* create_mamba1_backend();

// Zamba2 GPU backend — uses Mamba2 SSD kernels from mamba2_kernels.hip
extern "C" Backend* create_zamba2_backend();

// ── Runtime backend creation via dlsym ──

// Check for a backend symbol statically linked into the main executable
// itself (e.g. ROCM_CPP_STATIC_HIP builds link src/backend_hip.cpp directly
// rather than loading librocm_cpp.so). dlopen(NULL) doesn't create a new
// handle — it just bumps the refcount on the already-loaded executable, so
// this is cheap and safe to call from a detection probe (issue #330: the
// actual creation path in BackendManager::create_instance_rt() already does
// this exact check, but the detection probes here didn't, so a statically
// linked build reported the backend as unavailable even though it could be
// created).
static bool has_static_symbol(const char* symbol) {
    void* self = dlopen(NULL, RTLD_NOW | RTLD_LOCAL);
    if (!self) return false;
    bool found = dlsym(self, symbol) != nullptr;
    dlclose(self);
    return found;
}
// These try to load from either the rocm_cpp shared library or standalone modules.
// If the library isn't available (e.g., no ROCm installed), they return nullptr.

static Backend* try_create_hip() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_hip_backend");
    if (!b) b = try_load_backend("libhip_backend.so", "create_hip_backend");
    return b;
}

static Backend* try_create_vulkan() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_vulkan_backend");
    if (!b) b = try_load_backend("libvulkan_backend.so", "create_vulkan_backend");
    return b;
}

static Backend* try_create_cuda() {
    Backend* b = try_load_backend("libcuda_backend.so", "create_cuda_backend");
    if (!b) b = try_load_backend("librocm_cpp.so", "create_cuda_backend");
    if (!b && has_static_symbol("create_cuda_backend")) {
        return ((Backend*(*)())dlsym(dlopen(NULL, RTLD_NOW | RTLD_LOCAL), "create_cuda_backend"))();
    }
    return b;
}

static Backend* try_create_metal() {
    Backend* b = try_load_backend("libmetal_backend.so", "create_metal_backend");
    if (!b) b = try_load_backend("librocm_cpp.so", "create_metal_backend");
    if (!b && has_static_symbol("create_metal_backend")) {
        return ((Backend*(*)())dlsym(dlopen(NULL, RTLD_NOW | RTLD_LOCAL), "create_metal_backend"))();
    }
    return b;
}

static Backend* try_create_npu() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_npu_backend");
    if (!b) b = try_load_backend("libnpu_backend.so", "create_npu_backend");
    return b;
}

// ── Mamba1 detection ──
bool is_mamba1_architecture(const ModelConfig& cfg) {
    return cfg.arch == RCPP_ARCH_MAMBA || cfg.arch == RCPP_ARCH_ZAMBA;
}

bool is_zamba2_architecture(const ModelConfig& cfg) {
    return cfg.arch == RCPP_ARCH_ZAMBA2;
}

// ── Auto-detect available backends ──
bool has_hip_gpu() {
    // Check lightweight indicators first — avoid dlopen("librocm_cpp.so")
    // which triggers HSA runtime init that opens /dev/accel/accel0 on
    // Strix Halo, preventing standalone NPU tools from accessing the device
    // even when the GPU backend isn't actively inferring. See issue #1029.
    struct stat st;
    if (stat("/dev/dri/renderD128", &st) == 0) return true;
    if (stat("/dev/dri/renderD129", &st) == 0) return true;
    if (stat("/dev/dri/renderD130", &st) == 0) return true;
    if (has_static_symbol("create_hip_backend")) return true;
    // Last resort: probe the shared library (will trigger HSA init)
    void* lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    return false;
}

bool has_vulkan() {
    // Check lightweight indicators first — avoid dlopen("librocm_cpp.so")
    // which triggers HSA runtime init on Strix Halo. See issue #1029.
    // 1. Static symbol check (fast, no library load)
    if (has_static_symbol("create_vulkan_backend")) return true;

    // 2. Check render nodes first — fast stat() call, no library load.
    //    On systems with a GPU (Strix Halo, dGPU), this succeeds instantly.
    struct stat st;
    if (stat("/dev/dri/renderD128", &st) == 0) return true;
    if (stat("/dev/dri/renderD129", &st) == 0) return true;

    // 3. Probe via libvulkan — lightweight symbol check only.
    //    dlopen("libvulkan.so") can hang on some systems if the Vulkan
    //    loader tries to initialize GPU drivers, so only do this if
    //    render nodes weren't found (headless/VM fallback).
    void* lib = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_LAZY);
    if (lib) {
        bool has_syms =
            dlsym(lib, "vkCreateInstance") &&
            dlsym(lib, "vkEnumeratePhysicalDevices") &&
            dlsym(lib, "vkDestroyInstance");
        dlclose(lib);
        if (has_syms) return true;
    }

    // 4. Last resort: probe the full shared library (will trigger HSA init)
    lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    return false;
}

bool has_npu() {
    // Check lightweight indicators first — avoid dlopen("librocm_cpp.so")
    // which triggers HSA runtime init that opens /dev/accel/accel0 on
    // Strix Halo. The sysfs and accel device checks are cheap file ops
    // that don't load any NPU/GPU runtime. See issue #1029.
    if (any_accel_device_present()) return true;
    std::ifstream drv("/sys/bus/pci/drivers/amdxdna/uevent");
    if (drv) return true;
    if (has_static_symbol("create_npu_backend")) return true;
    // Check via XRT (lighter than loading full librocm_cpp.so)
    void* lib = dlopen("libxrt_coreutil.so", RTLD_LAZY);
    if (!lib) lib = dlopen("libxrt_coreutil.so.2", RTLD_LAZY);
    if (lib) { dlclose(lib); return true; }
    // Last resort: probe the full shared library (will trigger HSA init)
    lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    return false;
}

bool has_avx512() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo) return false;
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.find("avx512") != std::string::npos) return true;
    return false;
}

bool has_cuda() {
    // Probe for CUDA-capable GPU via nvidia-ml or cuda runtime
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libcuda.so", RTLD_LAZY);
    if (!lib) {
        // Check for /dev/nvidia* devices as fallback
        struct stat st;
        if (stat("/dev/nvidia0", &st) == 0) return true;
        return false;
    }
    // Check that core CUDA symbols exist
    bool has_cuInit = dlsym(lib, "cuInit") != nullptr;
    bool has_cuDeviceGetCount = dlsym(lib, "cuDeviceGetCount") != nullptr;
    dlclose(lib);
    if (has_cuInit && has_cuDeviceGetCount) {
        // Quick probe: try to initialize CUDA and get device count
        typedef int (*cuInit_t)(unsigned int);
        typedef int (*cuDeviceGetCount_t)(int*);
        lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (lib) {
            auto cuInit = (cuInit_t)dlsym(lib, "cuInit");
            auto cuGetCount = (cuDeviceGetCount_t)dlsym(lib, "cuDeviceGetCount");
            if (cuInit && cuGetCount) {
                if (cuInit(0) == 0) {
                    int ndev = 0;
                    if (cuGetCount(&ndev) == 0 && ndev > 0) {
                        dlclose(lib);
                        return true;
                    }
                }
            }
            dlclose(lib);
        }
        return true; // symbols exist, assume GPU available
    }
    return false;
}

bool has_metal() {
#ifdef __APPLE__
    // On macOS, check for Metal framework availability
    void* lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (!lib) lib = dlopen("Metal", RTLD_LAZY);
    if (!lib) return false;
    
    // Check that MTLDevice creation works by probing the symbol
    bool has_dev = dlsym(lib, "MTLCreateSystemDefaultDevice") != nullptr;
    dlclose(lib);
    
    if (has_dev) {
        // Quick probe using dlopen + dlsym to call MTLCreateSystemDefaultDevice
        // On Apple Silicon, this always returns a valid device
        return true;
    }
    return has_dev;
#else
    return false;
#endif
}

/// Detect all available backends, sorted by preference.
BackendType detect_backends() {
    printf("\n🔍 Detecting available compute backends...\n");

    struct Probe { BackendType type; const char* name; bool (*check)(); };
    Probe probes[] = {
        {BackendType::HIP_GPU,    "HIP GPU (ROCm)",     has_hip_gpu},
        {BackendType::CUDA_GPU,   "CUDA GPU (NVIDIA)",  has_cuda},
        {BackendType::VULKAN,     "Vulkan GPU",          has_vulkan},
        {BackendType::METAL_GPU,  "Metal GPU (Apple)",  has_metal},
        {BackendType::NPU_XRT,    "NPU XDNA (XRT)",     has_npu},
        {BackendType::CPU_AVX512, "CPU AVX-512",        has_avx512},
        {BackendType::CPU_SCALAR, "CPU (scalar)",       []()->bool{return true;}},
        {BackendType::GENERIC,   "Generic CPU",          []()->bool{return true;}},
    };

    BackendType best = BackendType::NONE;
    for (auto& p : probes) {
        bool avail = p.check();
        printf("  %s: %s\n", p.name, avail ? "✅ detected" : "❌ not available");
        if (avail && best == BackendType::NONE) best = p.type;
    }

    if (best == BackendType::NONE) best = BackendType::CPU_SCALAR;
    printf("  → Selected: %s\n\n", backend_name(best));
    return best;
}

/// Create the best available backend.
Backend* create_best_backend() {
    BackendType best = detect_backends();
    return create_backend(best);
}

// ── Architecture-aware HIP creation ──
// Creates a specialized backend depending on the model architecture:
//   - Mamba1 (Zamba, BlackMamba) → Mamba1 GPU backend (mamba1_engine.hip)
//   - Zamba2 (Zamba2-1.2B/2.7B/7B) → Zamba2 GPU backend (mamba2_kernels.hip)
//   - Everything else → generic HIP backend.
Backend* create_backend_for_arch(BackendType type, const ModelConfig* cfg) {
    if (type == BackendType::HIP_GPU && cfg && is_mamba1_architecture(*cfg)) {
        // Mamba1 models use the specialized Mamba1 GPU backend
        auto* b = create_mamba1_backend();
        if (b) { printf("  Created Mamba1 GPU backend\n"); return b; }
        printf("  Mamba1 GPU backend unavailable\n");
        return nullptr;
    }
    if (type == BackendType::HIP_GPU && cfg && is_zamba2_architecture(*cfg)) {
        // Zamba2 models use the specialized Zamba2 GPU backend
        auto* b = create_zamba2_backend();
        if (b) { printf("  Created Zamba2 GPU backend\n"); return b; }
        printf("  Zamba2 GPU backend unavailable\n");
        return nullptr;
    }
    return create_backend(type);
}

/// Create a specific backend by type.
/// Uses dlsym for GPU/NPU backends (loaded from librocm_cpp.so or standalone .so).
/// CPU backend is always linked in (pure C++).
Backend* create_backend(BackendType type) {
    switch (type) {
        case BackendType::HIP_GPU: {
            auto* b = try_create_hip();
            if (b) { printf("  Created HIP GPU backend\n"); return b; }
            printf("  HIP backend unavailable (install ROCm)\n");
            return nullptr;
        }
        case BackendType::CUDA_GPU: {
            auto* b = try_create_cuda();
            if (b) { printf("  Created CUDA GPU backend\n"); return b; }
            printf("  CUDA backend unavailable (install CUDA Toolkit)\n");
            return nullptr;
        }
        case BackendType::VULKAN: {
            auto* b = try_create_vulkan();
            if (b) { printf("  Created Vulkan GPU backend\n"); return b; }
            printf("  Vulkan backend unavailable\n");
            return nullptr;
        }
        case BackendType::METAL_GPU: {
            auto* b = try_create_metal();
            if (b) { printf("  Created Metal GPU backend\n"); return b; }
            printf("  Metal backend unavailable\n");
            return nullptr;
        }
        case BackendType::NPU_XRT: {
            auto* b = try_create_npu();
            if (b) { printf("  Created NPU XRT backend\n"); return b; }
            printf("  NPU backend unavailable (need XRT)\n");
            return nullptr;
        }
        case BackendType::CPU_AVX512:
        case BackendType::CPU_SCALAR:
            return create_cpu_backend();
        case BackendType::GENERIC:
            return create_generic_backend();
        default:
            return nullptr;
    }
}
