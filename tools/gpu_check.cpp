#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    int nd; hipGetDeviceCount(&nd);
    hipSetDevice(0);
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    printf("Device: %s\n", p.name);
    printf("integrated: %d\n", p.integrated);
    printf("concurrentManagedAccess: %d\n", p.concurrentManagedAccess);
    printf("pageableMemoryAccess: %d\n", p.pageableMemoryAccess);
    printf("isLargeBar: %d\n", p.isLargeBar);
    printf("totalGlobalMem: %.0f MB\n", p.totalGlobalMem/(1024.*1024.));
    printf("l2CacheSize: %d\n", p.l2CacheSize);
    printf("multiProcessorCount: %d\n", p.multiProcessorCount);
    printf("memoryClockRate: %d kHz\n", p.memoryClockRate);
    printf("memoryBusWidth: %d bits\n", p.memoryBusWidth);
    size_t free_, tot;
    hipMemGetInfo(&free_, &tot);
    printf("Free: %.0f / Total: %.0f MB\n", free_/(1024.*1024.), tot/(1024.*1024.));
    return 0;
}
