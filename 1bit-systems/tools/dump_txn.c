// LD_PRELOAD library to intercept AMD NPU EXEC_CMD ioctl and dump txn data
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <string.h>
#include <stdlib.h>

// DRM_IOCTL_BASE for compute accelerators
#define DRM_COMMAND_BASE 0x40
#define DRM_AMDXDNA_EXEC_CMD 7
#define DRM_IOCTL_AMDXDNA_EXEC_CMD \
    (((DRM_AMDXDNA_EXEC_CMD) << 0) | \
     ((4) << 16) | \
     ((sizeof(struct amdxdna_drm_exec_cmd)) << 0) | \
     (DRM_COMMAND_BASE + DRM_AMDXDNA_EXEC_CMD))

// We need the actual struct definition
struct amdxdna_drm_exec_cmd {
    uint64_t ext;
    uint64_t ext_flags;
    uint32_t hwctx;
    uint32_t type;
    uint64_t cmd_handles;
    uint64_t args;
    uint32_t cmd_count;
    uint32_t arg_count;
    uint64_t seq;
};

static int (*real_ioctl)(int fd, unsigned long request, void *arg) = NULL;

int ioctl(int fd, unsigned long request, void *arg) {
    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    
    int ret = real_ioctl(fd, request, arg);
    
    // Check if this is an AMDXDNA EXEC_CMD
    // DRM_IOCTL_NR(request) gives the ioctl number
    unsigned int nr = request & 0xff;
    unsigned int dir = (request >> 30) & 3;
    
    if (nr == 7 && dir == 2) { // EXEC_CMD is write (dir=2)
        struct amdxdna_drm_exec_cmd *cmd = (struct amdxdna_drm_exec_cmd *)arg;
        fprintf(stderr, "\n[txn-dump] EXEC_CMD: hwctx=%u type=%u cmd_count=%u seq=%lu\n", 
                cmd->hwctx, cmd->type, cmd->cmd_count, (unsigned long)cmd->seq);
        
        // If type 0 (SUBMIT_EXEC_BUF), dump the command buffer
        if (cmd->type == 0 && cmd->cmd_handles != 0) {
            fprintf(stderr, "[txn-dump]  cmd_handles ptr=%p\n", (void*)cmd->cmd_handles);
        }
        
        // Dump more info about what's being submitted
        fprintf(stderr, "[txn-dump]  args ptr=%p arg_count=%u\n", 
                (void*)cmd->args, cmd->arg_count);
        
        // Try to read the command buffer data from the args pointer
        // (this is a userspace pointer - we can read it directly)
        if (cmd->args && cmd->arg_count > 0) {
            uint32_t *data = (uint32_t*)(uintptr_t)cmd->args;
            fprintf(stderr, "[txn-dump]  First 16 words:");
            for (int i = 0; i < 16 && i < (int)cmd->arg_count; i++) {
                fprintf(stderr, " %08x", data[i]);
            }
            fprintf(stderr, "\n");
            
            // Check if this looks like attention-related (check for specific patterns)
            if (cmd->arg_count >= 10) {
                uint32_t w8 = data[8];
                if (w8 == 0x00010000 || w8 == 0x00020000 || w8 == 0x00030000) {
                    fprintf(stderr, "[txn-dump]  ⚡ QKV/O/GU/D instruction\n");
                } else if (w8 >= 0x00040000) {
                    fprintf(stderr, "[txn-dump]  *** POTENTIAL ATTENTION INSTRUCTION ***\n");
                    fprintf(stderr, "[txn-dump]  Dumping ALL %u words:\n", cmd->arg_count);
                    for (uint32_t i = 0; i < cmd->arg_count && i < 100; i++) {
                        fprintf(stderr, "  [%4u] %08x\n", i, data[i]);
                    }
                    if (cmd->arg_count > 100)
                        fprintf(stderr, "  ... (%u more words)\n", cmd->arg_count - 100);
                }
            }
        }
    }
    
    return ret;
}
