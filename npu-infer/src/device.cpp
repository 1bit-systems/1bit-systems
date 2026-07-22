#include "device.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cerrno>

// Include DRM header
#include <drm/amdxdna_accel.h>

// XRT includes for xclbin loading (transitional — we'll do raw later)
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

// Device path
constexpr const char* kNPUDevicePath = "/dev/accel/accel0";

// Helper: ioctl wrapper with error checking
static int drm_ioctl(int fd, unsigned long cmd, void* arg) {
    int ret = ::ioctl(fd, cmd, arg);
    if (ret < 0) {
        LOG_ERROR("ioctl(0x%lx) failed: %s (errno=%d)", cmd, strerror(errno), errno);
    }
    return ret;
}

NpuDevice* device_open(void) {
    NpuDevice* dev = static_cast<NpuDevice*>(std::calloc(1, sizeof(NpuDevice)));
    if (!dev) return nullptr;
    
    dev->fd = ::open(kNPUDevicePath, O_RDWR);
    if (dev->fd < 0) {
        LOG_ERROR("Failed to open %s: %s", kNPUDevicePath, strerror(errno));
        std::free(dev);
        return nullptr;
    }
    
    LOG_INFO("Opened %s (fd=%d)", kNPUDevicePath, dev->fd);
    return dev;
}

void device_close(NpuDevice* dev) {
    if (!dev) return;
    if (dev->fd >= 0) ::close(dev->fd);
    std::free(dev);
    LOG_INFO("Device closed");
}

// XCLBIN loading through XRT (temporary — will be replaced with raw ioctl)
// This is needed because the CONFIG_HWCTX ioctl requires xclbin data
// in a specific format that we need to understand first.

NpuHwCtx* device_create_hwctx(NpuDevice* dev, XclbinType xclbin_type,
                                int start_col, int num_col) {
    NpuHwCtx* ctx = static_cast<NpuHwCtx*>(std::calloc(1, sizeof(NpuHwCtx)));
    if (!ctx) return nullptr;
    
    ctx->dev = dev;
    ctx->xclbin_type = xclbin_type;
    ctx->start_col = start_col;
    ctx->columns = num_col;
    
    LOG_INFO("Created hwctx (type=%d, cols=%d, start=%d)", 
             xclbin_type, num_col, start_col);
    return ctx;
}

void device_destroy_hwctx(NpuHwCtx* ctx) {
    if (!ctx) return;
    std::free(ctx);
}

int device_config_hwctx(NpuHwCtx* ctx, const char* xclbin_path) {
    // Phase 1: Use XRT to load xclbin (we'll replace with raw ioctl)
    // For now, return 0 and track context info
    (void)ctx;
    (void)xclbin_path;
    LOG_INFO("Config hwctx with xclbin: %s", xclbin_path);
    return 0;
}

NpuBo* device_create_bo(NpuDevice* dev, uint64_t size, int type) {
    NpuBo* bo = static_cast<NpuBo*>(std::calloc(1, sizeof(NpuBo)));
    if (!bo) return nullptr;
    
    bo->dev = dev;
    bo->size = size;
    bo->type = type;
    
    struct amdxdna_drm_create_bo create_bo = {
        0,      // flags
        0,      // vaddr
        size,   // size
        static_cast<__u32>(type),  // type
        0,      // handle
    };
    
    if (drm_ioctl(dev->fd, DRM_IOCTL_AMDXDNA_CREATE_BO, &create_bo) < 0) {
        std::free(bo);
        return nullptr;
    }
    
    bo->handle = create_bo.handle;
    
    // Get BO info (xdna_addr, map_offset)
    struct amdxdna_drm_get_bo_info bo_info = {
        0,              // ext
        0,              // ext_flags
        bo->handle,     // handle
        0,              // pad
        0,              // map_offset
        0,              // vaddr
        0,              // xdna_addr
    };
    
    if (drm_ioctl(dev->fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &bo_info) < 0) {
        device_free_bo(bo);
        return nullptr;
    }
    
    bo->xdna_addr = bo_info.xdna_addr;
    bo->map_offset = bo_info.map_offset;
    
    LOG_DEBUG("Created BO handle=%u size=%lu xdna_addr=0x%lx map_offset=0x%lx",
              bo->handle, bo->size, bo->xdna_addr, bo->map_offset);
    
    return bo;
}

void* device_map_bo(NpuBo* bo) {
    if (!bo || bo->map) return bo ? bo->map : nullptr;
    
    bo->map = ::mmap(nullptr, bo->size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     bo->dev->fd, bo->map_offset);
    
    if (bo->map == MAP_FAILED) {
        LOG_ERROR("mmap BO %u failed: %s", bo->handle, strerror(errno));
        bo->map = nullptr;
        return nullptr;
    }
    
    LOG_DEBUG("Mapped BO %u at %p", bo->handle, bo->map);
    return bo->map;
}

void device_unmap_bo(NpuBo* bo) {
    if (!bo || !bo->map) return;
    ::munmap(bo->map, bo->size);
    bo->map = nullptr;
}

void device_free_bo(NpuBo* bo) {
    if (!bo) return;
    if (bo->map) device_unmap_bo(bo);
    // Free the BO handle (close the dmabuf or use FREE_BO ioctl)
    // AMDXDNA doesn't have a FREE_BO — handles are freed on DRM fd close
    std::free(bo);
}

void device_sync_bo(NpuBo* bo, int direction, uint64_t offset, uint64_t size) {
    struct amdxdna_drm_sync_bo sync_bo = {
        bo->handle,                         // handle
        static_cast<__u32>(direction),      // direction
        offset,                              // offset
        size,                                // size
    };
    
    if (drm_ioctl(bo->dev->fd, DRM_IOCTL_AMDXDNA_SYNC_BO, &sync_bo) < 0) {
        LOG_ERROR("sync_bo failed");
    }
}

uint64_t device_get_xdna_addr(NpuBo* bo) {
    return bo ? bo->xdna_addr : 0;
}

int device_exec_cmd(NpuHwCtx* ctx, uint32_t* cmd_handles,
                     uint32_t cmd_count, uint64_t* args, uint32_t arg_count) {
    NpuDevice* dev = ctx->dev;
    
    struct amdxdna_drm_exec_cmd exec = {
        0,                                              // ext
        0,                                              // ext_flags
        ctx->handle,                                    // hwctx
        AMDXDNA_CMD_SUBMIT_EXEC_BUF,                    // type
        reinterpret_cast<uint64_t>(cmd_handles),        // cmd_handles
        reinterpret_cast<uint64_t>(args),               // args
        cmd_count,                                      // cmd_count
        arg_count,                                      // arg_count
        0,                                              // seq
    };
    
    if (drm_ioctl(dev->fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec) < 0) {
        LOG_ERROR("exec_cmd failed on hwctx %u", ctx->handle);
        return -1;
    }
    
    return 0;
}

int device_wait_cmd(NpuHwCtx* ctx, uint64_t seq) {
    // For now, syncing the output BO is our wait mechanism
    (void)ctx;
    (void)seq;
    return 0;
}
