// dump_txn.cpp — LD_PRELOAD library to intercept AMD NPU EXEC_CMD ioctl
// g++ -std=c++17 -fPIC -shared -o dump_txn.so dump_txn.cpp -ldl
#define _GNU_SOURCE
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/ioctl.h>

namespace {

constexpr int DRM_COMMAND_BASE       = 0x40;
constexpr int DRM_AMDXDNA_EXEC_CMD   = 7;

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

static int (*real_ioctl)(int fd, unsigned long request, void *arg) = nullptr;

}  // anonymous namespace

int ioctl(int fd, unsigned long request, void *arg) {
    if (!real_ioctl) real_ioctl = reinterpret_cast<decltype(real_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));

    int ret = real_ioctl(fd, request, arg);

    unsigned int nr  = request & 0xff;
    unsigned int dir = (request >> 30) & 3;

    if (nr == 7 && dir == 2) {  // EXEC_CMD is write
        auto *cmd = static_cast<amdxdna_drm_exec_cmd *>(arg);
        std::fprintf(stderr,
            "\n[txn-dump] EXEC_CMD: hwctx=%u type=%u cmd_count=%u seq=%lu\n",
            cmd->hwctx, cmd->type, cmd->cmd_count,
            static_cast<unsigned long>(cmd->seq));

        if (cmd->type == 0 && cmd->cmd_handles != 0) {
            std::fprintf(stderr, "[txn-dump]  cmd_handles ptr=%p\n",
                reinterpret_cast<void *>(cmd->cmd_handles));
        }
        std::fprintf(stderr, "[txn-dump]  args ptr=%p arg_count=%u\n",
            reinterpret_cast<void *>(cmd->args), cmd->arg_count);

        if (cmd->args && cmd->arg_count > 0) {
            auto *data = reinterpret_cast<uint32_t *>(static_cast<uintptr_t>(cmd->args));
            std::fprintf(stderr, "[txn-dump]  First 16 words:");
            for (uint32_t i = 0; i < 16 && i < cmd->arg_count; i++)
                std::fprintf(stderr, " %08x", data[i]);
            std::fprintf(stderr, "\n");

            if (cmd->arg_count >= 10) {
                uint32_t w8 = data[8];
                if (w8 == 0x00010000 || w8 == 0x00020000 || w8 == 0x00030000) {
                    std::fprintf(stderr, "[txn-dump]  ⚡ QKV/O/GU/D instruction\n");
                } else if (w8 >= 0x00040000) {
                    std::fprintf(stderr,
                        "[txn-dump]  *** POTENTIAL ATTENTION INSTRUCTION ***\n");
                    std::fprintf(stderr, "[txn-dump]  Dumping ALL %u words:\n",
                        cmd->arg_count);
                    for (uint32_t i = 0; i < cmd->arg_count && i < 100; i++)
                        std::fprintf(stderr, "  [%4u] %08x\n", i, data[i]);
                    if (cmd->arg_count > 100)
                        std::fprintf(stderr, "  ... (%u more words)\n",
                            cmd->arg_count - 100);
                }
            }
        }
    }

    return ret;
}
