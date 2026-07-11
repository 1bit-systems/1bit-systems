// List HIP devices and their properties
#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
    int count = 0;
    hipGetDeviceCount(&count);
    printf("HIP devices: %d\n", count);

    for (int i = 0; i < count; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);
        printf("  Device %d: %s\n", i, props.name);
        printf("    arch:       %s\n", props.gcnArchName);
        printf("    compute:    %d.%d\n", props.major, props.minor);
        printf("    memory:     %.0f MB\n", props.totalGlobalMem / 1e6);
        printf("    PCI domain: 0x%x\n", props.pciDomainID);
        printf("    PCI bus:    0x%x\n", props.pciBusID);
        printf("    multiProc:  %d\n", props.multiProcessorCount);
        printf("    isIntegrat: %d\n", props.integrated);
        printf("    maxThreads: %d\n", props.maxThreadsPerBlock);
    }
    return 0;
}
