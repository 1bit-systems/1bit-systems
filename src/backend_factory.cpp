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

// ── Forward declarations (CPU is always linked) ──
extern Backend* create_cpu_backend();

// ── Runtime backend creation via dlsym ──
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

static Backend* try_create_npu() {
    Backend* b = try_load_backend("librocm_cpp.so", "create_npu_backend");
    if (!b) b = try_load_backend("libnpu_backend.so", "create_npu_backend");
    return b;
}

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

// ── Auto-detect available backends ──
bool has_hip_gpu() {
    void* lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    if (has_static_symbol("create_hip_backend")) return true;
    // Check for render nodes via file I/O instead of popen (fixes #67)
    struct stat st;
    if (stat("/dev/dri/renderD128", &st) == 0) return true;
    if (stat("/dev/dri/renderD129", &st) == 0) return true;
    if (stat("/dev/dri/renderD130", &st) == 0) return true;
    return false;
}

bool has_vulkan() {
    // Check if librocm_cpp is available (it contains the Vulkan backend)
    void* lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    if (has_static_symbol("create_vulkan_backend")) return true;

    // Probe via libvulkan — just check that the loader exists and we can
    // enumerate physical devices. Use a minimal vkCreateInstance call
    // with real Vulkan structs loaded via dlsym to avoid header deps.
    lib = dlopen("libvulkan.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_LAZY);
    if (!lib) return false;

    // Check that core symbols exist (sign of a working Vulkan loader)
    bool has_syms =
        dlsym(lib, "vkCreateInstance") &&
        dlsym(lib, "vkEnumeratePhysicalDevices") &&
        dlsym(lib, "vkDestroyInstance");

    dlclose(lib);
    return has_syms;
}

bool has_npu() {
    void* lib = dlopen("librocm_cpp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib) { dlclose(lib); return true; }
    if (has_static_symbol("create_npu_backend")) return true;
    // Check via XRT
    lib = dlopen("libxrt_coreutil.so", RTLD_LAZY);
    if (!lib) lib = dlopen("libxrt_coreutil.so.2", RTLD_LAZY);
    if (!lib) {
        struct stat st;
        if (stat("/dev/accel/accel0", &st) == 0 || stat("/sys/class/accel/accel0", &st) == 0)
            return true;
        // Check via sysfs file I/O instead of popen (fixes #67)
        std::ifstream drv("/sys/bus/pci/drivers/amdxdna/uevent");
        if (drv) return true;
        return false;
    }
    dlclose(lib);
    return true;
}

bool has_avx512() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo) return false;
    std::string line;
    while (std::getline(cpuinfo, line))
        if (line.find("avx512") != std::string::npos) return true;
    return false;
}

/// Detect all available backends, sorted by preference.
BackendType detect_backends() {
    printf("\n🔍 Detecting available compute backends...\n");

    struct Probe { BackendType type; const char* name; bool (*check)(); };
    Probe probes[] = {
        {BackendType::HIP_GPU,    "HIP GPU (ROCm)",   has_hip_gpu},
        {BackendType::VULKAN,     "Vulkan GPU",        has_vulkan},
        {BackendType::NPU_XRT,    "NPU XDNA (XRT)",    has_npu},
        {BackendType::CPU_AVX512, "CPU AVX-512",       has_avx512},
        {BackendType::CPU_SCALAR, "CPU (scalar)",      []()->bool{return true;}},
        {BackendType::GENERIC,   "Generic CPU",       []()->bool{return true;}},
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
        case BackendType::VULKAN: {
            auto* b = try_create_vulkan();
            if (b) { printf("  Created Vulkan GPU backend\n"); return b; }
            printf("  Vulkan backend unavailable\n");
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
