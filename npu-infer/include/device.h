#ifndef NPU_INFER_DEVICE_H
#define NPU_INFER_DEVICE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "common.h"

// Forward declarations
typedef struct NpuDevice NpuDevice;
typedef struct NpuBo NpuBo;
typedef struct NpuHwCtx NpuHwCtx;

// Buffer object
struct NpuBo {
    NpuDevice* dev;
    uint32_t handle;       // DRM BO handle
    uint64_t size;         // Size in bytes
    uint64_t xdna_addr;    // NPU device virtual address
    uint64_t map_offset;   // For mmap
    void*     map;         // Mapped address (NULL if not mapped)
    int       type;        // BO type
};

// Hardware context
struct NpuHwCtx {
    NpuDevice* dev;
    uint32_t   handle;      // DRM hwctx handle
    int        columns;     // Number of AIE columns
    int        start_col;   // Starting column
    XclbinType xclbin_type; // What this context is for
};

// NPU device
struct NpuDevice {
    int fd;                // /dev/accel/accel0 file descriptor
    int num_ctx;           // Number of active contexts
    int num_bo;            // Number of active BOs
    uint32_t next_ctx_id;  // For tracking
};

// Device lifecycle
NpuDevice* device_open(void);
void       device_close(NpuDevice* dev);

// Hardware context
NpuHwCtx*  device_create_hwctx(NpuDevice* dev, XclbinType xclbin_type,
                                int start_col, int num_col);
void       device_destroy_hwctx(NpuHwCtx* ctx);
int        device_config_hwctx(NpuHwCtx* ctx, const char* xclbin_path);

// Buffer objects
NpuBo*     device_create_bo(NpuDevice* dev, uint64_t size, int type);
void*      device_map_bo(NpuBo* bo);
void       device_unmap_bo(NpuBo* bo);
void       device_free_bo(NpuBo* bo);
void       device_sync_bo(NpuBo* bo, int direction, uint64_t offset, uint64_t size);
uint64_t   device_get_xdna_addr(NpuBo* bo);

// Command execution
int        device_exec_cmd(NpuHwCtx* ctx, uint32_t* cmd_handles,
                            uint32_t cmd_count, uint64_t* args, uint32_t arg_count);

// Wait for completion
int        device_wait_cmd(NpuHwCtx* ctx, uint64_t seq);

#endif // NPU_INFER_DEVICE_H
