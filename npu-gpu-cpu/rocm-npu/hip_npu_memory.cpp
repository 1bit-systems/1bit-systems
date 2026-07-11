// SPDX-License-Identifier: MIT
/*
 * hip_npu_memory.cpp — NPU memory management for HIP shim
 *
 * Allocates GTT buffers on amdgpu, exports as dma-buf, imports into NPU.
 * This is the zero-copy path verified in Phase 0.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
#include <amdgpu.h>

#include <hip/hip_runtime_api.h>

#include "npu_aql.h"

/* ------------------------------------------------------------------ */
/*  Buffer tracking                                                    */
/* ------------------------------------------------------------------ */

#include <map>
#include <mutex>

struct NPUBuffer {
    void        *cpu_ptr;    /* CPU mmap of dma-buf */
    int         dmabuf_fd;  /* dma-buf file descriptor */
    size_t      size;
    amdgpu_bo_handle bo;    /* amdgpu BO handle */
};

static std::mutex g_buf_mutex;
static std::map<void*, NPUBuffer> g_buffers;

/* ------------------------------------------------------------------ */
/*  GTT allocation helpers                                            */
/* ------------------------------------------------------------------ */

static amdgpu_device_handle g_amdgpu_dev = nullptr;

static int ensure_amdgpu() {
    if (g_amdgpu_dev) return 0;

    int fd = open("/dev/dri/renderD128", O_RDWR);
    if (fd < 0) return -1;

    uint32_t maj, min;
    int r = amdgpu_device_initialize(fd, &maj, &min, &g_amdgpu_dev);
    if (r) { close(fd); return -1; }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  hipMalloc — allocate GTT buffer, export to NPU                    */
/* ------------------------------------------------------------------ */

static hipError_t npu_malloc(void **ptr, size_t size) {
    if (ensure_amdgpu() < 0) return hipErrorMemoryAllocation;

    /* Allocate GTT BO */
    struct amdgpu_bo_alloc_request req = {};
    req.alloc_size      = (size + 4095) & ~4095;
    req.preferred_heap  = AMDGPU_GEM_DOMAIN_GTT;
    req.flags           = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;

    amdgpu_bo_handle bo;
    int r = amdgpu_bo_alloc(g_amdgpu_dev, &req, &bo);
    if (r) return hipErrorMemoryAllocation;

    /* CPU map */
    void *cpu_ptr;
    r = amdgpu_bo_cpu_map(bo, &cpu_ptr);
    if (r) { amdgpu_bo_free(bo); return hipErrorMemoryAllocation; }

    /* Export as dma-buf */
    uint32_t export_handle;
    r = amdgpu_bo_export(bo, amdgpu_bo_handle_type_dma_buf_fd, &export_handle);
    if (r) { amdgpu_bo_cpu_unmap(bo); amdgpu_bo_free(bo); return hipErrorMemoryAllocation; }

    int dmabuf_fd = (int)export_handle;

    /* If NPU is active, import into NPU */
    if (g_npu_available) {
        int npu_fd = open("/dev/accel/accel0", O_RDWR);
        if (npu_fd >= 0) {
            struct drm_prime_handle args = {};
            args.fd = dmabuf_fd;
            ioctl(npu_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &args);
            close(npu_fd);
        }
    }

    /* Track buffer */
    NPUBuffer buf = { cpu_ptr, dmabuf_fd, req.alloc_size, bo };
    std::lock_guard<std::mutex> lock(g_buf_mutex);
    g_buffers[cpu_ptr] = buf;

    *ptr = cpu_ptr;
    return hipSuccess;
}

/* ------------------------------------------------------------------ */
/*  hipFree — release NPU buffer                                      */
/* ------------------------------------------------------------------ */

static hipError_t npu_free(void *ptr) {
    if (!ptr) return hipSuccess;

    std::lock_guard<std::mutex> lock(g_buf_mutex);
    auto it = g_buffers.find(ptr);
    if (it == g_buffers.end()) return hipErrorInvalidValue;

    close(it->second.dmabuf_fd);
    amdgpu_bo_cpu_unmap(it->second.bo);
    amdgpu_bo_free(it->second.bo);
    g_buffers.erase(it);
    return hipSuccess;
}

/* ------------------------------------------------------------------ */
/*  hipMemcpy — copy between host and NPU buffer (CPU-side for now)   */
/* ------------------------------------------------------------------ */

static hipError_t npu_memcpy(void *dst, const void *src, size_t size,
                             hipMemcpyKind kind) {
    switch (kind) {
    case hipMemcpyHostToDevice:
    case hipMemcpyDeviceToHost:
    case hipMemcpyDeviceToDevice:
        memcpy(dst, src, size);
        return hipSuccess;
    default:
        return hipErrorInvalidValue;
    }
}

/* ------------------------------------------------------------------ */
/*  Intercepted HIP memory APIs                                       */
/* ------------------------------------------------------------------ */

extern "C" {

bool g_npu_available = false;

hipError_t hipMalloc(void **ptr, size_t size) {
    if (!ptr) return hipErrorInvalidValue;
    if (size == 0) { *ptr = nullptr; return hipSuccess; }

    int deviceId = 0;
    LOAD_REAL(hipGetDevice);
    LOAD_REAL(hipGetDeviceCount);

    int gpu_count = 0;
    real_hipGetDeviceCount(&gpu_count);
    real_hipGetDevice(&deviceId);

    if (deviceId >= gpu_count) {
        return npu_malloc(ptr, size);
    }

    /* GPU — delegate to real HIP */
    LOAD_REAL(hipMalloc);
    return real_hipMalloc(ptr, size);
}

hipError_t hipFree(void *ptr) {
    if (!ptr) return hipSuccess;

    std::lock_guard<std::mutex> lock(g_buf_mutex);
    if (g_buffers.find(ptr) != g_buffers.end()) {
        lock.unlock();
        return npu_free(ptr);
    }

    LOAD_REAL(hipFree);
    return real_hipFree(ptr);
}

hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) {
    /* Check if src or dst is an NPU buffer */
    {
        std::lock_guard<std::mutex> lock(g_buf_mutex);
        bool src_is_npu = g_buffers.find(const_cast<void*>(src)) != g_buffers.end();
        bool dst_is_npu = g_buffers.find(dst) != g_buffers.end();

        if (src_is_npu || dst_is_npu) {
            return npu_memcpy(dst, src, size, kind);
        }
    }

    LOAD_REAL(hipMemcpy);
    return real_hipMemcpy(dst, src, size, kind);
}

hipError_t hipMemcpyDtoD(void *dst, const void *src, size_t size) {
    LOAD_REAL(hipMemcpyDtoD);
    return real_hipMemcpyDtoD(dst, src, size);
}

hipError_t hipMemcpyHtoD(void *dst, const void *src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoH(void *dst, const void *src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

} /* extern "C" */
