// SPDX-License-Identifier: MIT
// Minimal NPU HIP shim — just intercepts hipGetDeviceCount to add NPU

#define _GNU_SOURCE
#define __HIP_PLATFORM_AMD__ 1

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <hip/hip_runtime_api.h>

/* Resolve next function in chain */
template<typename T>
static T resolve(const char *name) {
    return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}

extern "C" {

hipError_t hipGetDeviceCount(int *count) {
    static auto real = resolve<decltype(&hipGetDeviceCount)>("hipGetDeviceCount");
    int gpu_count = 0;
    hipError_t ret = real(&gpu_count);
    if (ret != hipSuccess) return ret;

    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd >= 0) {
        close(fd);
        gpu_count += 1;  /* NPU adds one more device */
        fprintf(stderr, "hip_npu: added NPU device, total=%d\n", gpu_count);
    }
    *count = gpu_count;
    return hipSuccess;
}

static int g_real_gpu_count = -1;

hipError_t hipGetDeviceProperties(hipDeviceProp_t *props, int deviceId) {
    static auto real = resolve<decltype(&hipGetDeviceProperties)>("hipGetDeviceProperties");

    /* Cache the real GPU count on first call */
    if (g_real_gpu_count < 0) {
        static auto real_count = resolve<decltype(&hipGetDeviceCount)>("hipGetDeviceCount");
        real_count(&g_real_gpu_count);
    }

    /* NPU is always the last device */
    if (deviceId >= g_real_gpu_count) {
        memset(props, 0, sizeof(*props));
        strncpy(props->name, "AMD XDNA 2 NPU (aie2)", sizeof(props->name) - 1);
        strncpy(props->gcnArchName, "aie2", sizeof(props->gcnArchName) - 1);
        props->major                     = 2;
        props->minor                     = 0;
        props->totalGlobalMem            = 128ULL * 1024 * 1024 * 1024;
        props->multiProcessorCount       = 8;
        props->clockRate                = 1000;
        props->integrated                = 1;
        props->canMapHostMemory          = 1;
        props->warpSize                  = 1;
        props->maxThreadsPerBlock        = 1;
        props->maxThreadsDim[0]          = 1;
        props->maxThreadsDim[1]          = 1;
        props->maxThreadsDim[2]          = 1;
        props->maxGridSize[0]            = 1;
        props->maxGridSize[1]            = 1;
        props->maxGridSize[2]            = 1;
        return hipSuccess;
    }
    return real(props, deviceId);
}

} /* extern "C" */
