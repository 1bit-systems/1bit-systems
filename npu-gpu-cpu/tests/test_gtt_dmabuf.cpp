// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * Test: dma-buf zero-copy between amdgpu (TTM_PL_TT) and amdxdna (NPU)
 *
 * 1. Allocate GTT BO on amdgpu     → AMDGPU_GEM_DOMAIN_GTT
 * 2. Export to dma-buf fd          → drmPrimeHandleToFD
 * 3. Import into amdxdna            → DRM_IOCTL_PRIME_FD_TO_HANDLE
 * 4. mmap both sides
 * 5. GPU writes pattern, NPU reads, verify zero-copy
 *
 * Build:
 *   g++ -std=gnu++17 -o test_gtt_dmabuf test_gtt_dmabuf.cpp \
 *       -I/usr/include/libdrm -ldrm -ldrm_amdgpu
 *
 * Run:
 *   sudo ./test_gtt_dmabuf
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <vector>

#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
#include <drm/amdxdna_accel.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>

/* ------------------------------------------------------------------ */
/*  amdgpu helpers (libdrm_amdgpu)                                    */
/* ------------------------------------------------------------------ */

struct amdgpu_bo {
    amdgpu_device_handle dev;
    amdgpu_bo_handle     bo;
    amdgpu_va_handle     va;
    uint64_t             alloc_size;
    uint32_t             handle;
};

static int amdgpu_open(struct amdgpu_bo *abo, int fd) {
    uint32_t maj, min;
    int ret = amdgpu_device_initialize(fd, &maj, &min, &abo->dev);
    if (ret) {
        fprintf(stderr, "amdgpu_device_initialize failed: %d\n", ret);
        return -1;
    }
    return 0;
}

static int amdgpu_alloc_gtt(struct amdgpu_bo *abo, size_t size) {
    struct amdgpu_bo_alloc_request req = {};
    req.alloc_size      = size;
    req.preferred_heap  = AMDGPU_GEM_DOMAIN_GTT;
    req.flags           = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;

    int ret = amdgpu_bo_alloc(abo->dev, &req, &abo->bo);
    if (ret) {
        fprintf(stderr, "amdgpu_bo_alloc(GTT) failed: %d\n", ret);
        return -1;
    }

    uint32_t export_handle;
    ret = amdgpu_bo_export(abo->bo, amdgpu_bo_handle_type_dma_buf_fd, &export_handle);
    if (ret) {
        fprintf(stderr, "amdgpu_bo_export dma-buf failed: %d\n", ret);
        amdgpu_bo_free(abo->bo);
        return -1;
    }
    abo->handle = (int)export_handle; /* dma-buf fd */

    abo->alloc_size = size;

    /* mmap the GPU BO for CPU/GPU access */
    ret = amdgpu_bo_cpu_map(abo->bo, (void**)&abo->va);
    if (ret) {
        fprintf(stderr, "amdgpu_bo_cpu_map failed: %d\n", ret);
        close(abo->handle);
        amdgpu_bo_free(abo->bo);
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  amdxdna helpers (raw DRM ioctls)                                  */
/* ------------------------------------------------------------------ */

struct amdxdna_bo {
    int     fd;       /* /dev/accel/accel0 fd */
    uint32_t handle;  /* GEM handle after import */
    void    *map;     /* CPU mmap */
    size_t   size;
};

static int amdxdna_open(struct amdxdna_bo *xbo) {
    xbo->fd = open("/dev/accel/accel0", O_RDWR);
    if (xbo->fd < 0) {
        perror("open /dev/accel/accel0");
        return -1;
    }
    return 0;
}

static int amdxdna_import_dmabuf(struct amdxdna_bo *xbo, int dma_buf_fd, size_t size) {
    struct drm_prime_handle args = {};
    args.fd = dma_buf_fd;
    args.flags = 0;

    int ret = ioctl(xbo->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &args);
    if (ret) {
        perror("amdxdna DRM_IOCTL_PRIME_FD_TO_HANDLE");
        return -1;
    }

    xbo->handle = args.handle;
    xbo->size   = size;

    /* mmap the dma-buf fd directly (exporter's mmap) */
    xbo->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    dma_buf_fd, 0);
    if (xbo->map == MAP_FAILED) {
        perror("mmap dma-buf fd");
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Test: write from GPU side, read from NPU side                     */
/* ------------------------------------------------------------------ */

static int test_zero_copy(volatile uint32_t *gpu_ptr,
                          volatile uint32_t *npu_ptr,
                          size_t num_words) {
    /* GPU writes known pattern */
    for (size_t i = 0; i < num_words; i++)
        gpu_ptr[i] = 0xDEADBEEF;

    /* NPU reads and verifies */
    for (size_t i = 0; i < num_words; i++) {
        if (npu_ptr[i] != 0xDEADBEEF) {
            fprintf(stderr, "MISMATCH at word %zu: GPU wrote 0x%08X, NPU sees 0x%08X\n",
                    i, (unsigned)0xDEADBEEF, (unsigned)npu_ptr[i]);
            return -1;
        }
    }

    /* NPU writes different pattern */
    for (size_t i = 0; i < num_words; i++)
        npu_ptr[i] = 0xCAFEBABE;

    /* GPU reads and verifies */
    for (size_t i = 0; i < num_words; i++) {
        if (gpu_ptr[i] != 0xCAFEBABE) {
            fprintf(stderr, "MISMATCH at word %zu: NPU wrote 0x%08X, GPU sees 0x%08X\n",
                    i, (unsigned)0xCAFEBABE, (unsigned)gpu_ptr[i]);
            return -1;
        }
    }

    return 0; /* PASS */
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    const size_t SIZE = 64 * 1024; /* 64 KiB */
    const size_t NUM_WORDS = SIZE / sizeof(uint32_t);

    printf("=== dma-buf zero-copy test: amdgpu GTT ↔ amdxdna NPU ===\n\n");

    /* Open amdgpu device */
    int amdgpu_fd = open("/dev/dri/renderD128", O_RDWR);
    if (amdgpu_fd < 0) {
        perror("open /dev/dri/renderD128");
        return 1;
    }

    struct amdgpu_bo gpu_bo = {};
    if (amdgpu_open(&gpu_bo, amdgpu_fd) < 0)
        return 1;

    /* Allocate GTT BO and export to dma-buf fd */
    printf("1. Allocating %zu bytes on amdgpu in GTT domain... ", SIZE);
    if (amdgpu_alloc_gtt(&gpu_bo, SIZE) < 0)
        return 1;
    printf("OK (dma-buf fd=%d, CPU ptr=%p)\n", gpu_bo.handle, gpu_bo.va);

    /* Open amdxdna device */
    printf("2. Opening amdxdna NPU device... ");
    struct amdxdna_bo npu_bo = {};
    if (amdxdna_open(&npu_bo) < 0)
        return 1;
    printf("OK (fd=%d)\n", npu_bo.fd);

    /* Import the dma-buf into amdxdna */
    printf("3. Importing dma-buf fd %d into amdxdna... ", gpu_bo.handle);
    if (amdxdna_import_dmabuf(&npu_bo, gpu_bo.handle, SIZE) < 0)
        return 1;
    printf("OK (GEM handle=%u, mmap=%p)\n", npu_bo.handle, npu_bo.map);

    /* Verify both sides map the same physical memory */
    printf("4. Checking if GPU and NPU point to same buffer...\n");
    printf("   GPU va=%p, NPU mmap=%p, diff=%ld bytes\n",
           gpu_bo.va, npu_bo.map,
           (long)((uint8_t*)npu_bo.map - (uint8_t*)gpu_bo.va));

    /* Zero-copy test: GPU writes, NPU reads */
    printf("\n5. Zero-copy test: GPU→NPU and NPU→GPU...\n");
    int ret = test_zero_copy((volatile uint32_t*)gpu_bo.va,
                             (volatile uint32_t*)npu_bo.map,
                             NUM_WORDS);

    if (ret == 0) {
        printf("\n✅ PASS: Zero-copy verified — GPU and NPU access same physical pages\n");
        printf("   GPU wrote 0xDEADBEEF → NPU read it back\n");
        printf("   NPU wrote 0xCAFEBABE → GPU read it back\n");
    } else {
        printf("\n❌ FAIL: Data mismatch — pages may not be shared\n");
    }

    /* Cleanup */
    munmap(npu_bo.map, npu_bo.size);
    close(npu_bo.fd);
    amdgpu_bo_cpu_unmap(gpu_bo.bo);
    close(gpu_bo.handle);
    amdgpu_bo_free(gpu_bo.bo);
    amdgpu_device_deinitialize(gpu_bo.dev);
    close(amdgpu_fd);

    return ret ? 1 : 0;
}
