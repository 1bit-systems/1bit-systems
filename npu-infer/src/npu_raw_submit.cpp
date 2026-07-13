// npu_raw_submit.cpp -- NPU GEMM submission tool
// Supports two modes:
//   opcode=3 (default): run pre-compiled AIE program from xclbin
//   opcode=0: run with instruction buffer (insts.bin) for dynamic GEMM shapes
//
// Build:
//   g++ -std=gnu++17 -O2 -g -o npu_raw_submit npu_raw_submit.cpp \
//       -lxrt_coreutil -ldl -luuid
//
// Usage (opcode=3, backward compatible):
//   sudo ./npu_raw_submit [xclbin_path]
//
// Usage (opcode=0, instruction buffer):
//   sudo ./npu_raw_submit <xclbin_path> <insts.bin> <A.bin> <B.bin> <M> <K> <N> [C.bin]
//
// Examples:
//   sudo ./npu_raw_submit                                          (opcode=3, default xclbin)
//   sudo ./npu_raw_submit /path/to/mm.xclbin                       (opcode=3, custom xclbin)
//   sudo ./npu_raw_submit mm.xclbin insts.bin A.bin B.bin 128 1024 256 C.bin   (opcode=0)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>

#include <drm/amdxdna_accel.h>

// XRT headers (for xclbin + kernel execution)
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

#define NPU_DEVICE "/dev/accel/accel0"

using Clock = std::chrono::high_resolution_clock;

// ── helpers ────────────────────────────────────────────────────────

static int drm_ioctl(int fd, unsigned long cmd, void *arg) {
    int ret = ioctl(fd, cmd, arg);
    if (ret < 0) {
        fprintf(stderr, "  ❌ ioctl(0x%lx): %s\n", cmd, strerror(errno));
    }
    return ret;
}

/// Read entire file into a byte vector.
static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    size_t sz = file.tellg();
    file.seekg(0);
    std::vector<char> data(sz);
    file.read(data.data(), sz);
    return data;
}

/// Read a float32 binary file into a float vector.
static std::vector<float> read_float_bin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    size_t bytes = file.tellg();
    if (bytes % sizeof(float) != 0) {
        fprintf(stderr, "  ❌ File size %zu not multiple of float size\n", bytes);
        return {};
    }
    file.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

static double ms_since(Clock::time_point t0) {
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── main ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Default xclbin path
    const char* default_xclbin =
        "/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/mm.xclbin";

    // ── Parse CLI arguments ──────────────────────────────────────
    //
    // Mode detection: if argv[2] is provided, we're in opcode=0 mode.
    // Otherwise opcode=3 (backward compatible with original tool).
    //
    // Args:
    //   argv[1] (optional) : xclbin path
    //   argv[2] (optional) : insts.bin path  → triggers opcode=0
    //   argv[3]             : A matrix binary (float32)
    //   argv[4]             : B matrix binary (float32)
    //   argv[5]             : M dimension
    //   argv[6]             : K dimension
    //   argv[7]             : N dimension
    //   argv[8] (optional) : Output C matrix path (default: C.bin)

    const char* xclbin_path   = argc > 1 ? argv[1] : default_xclbin;
    const char* insts_path    = argc > 2 ? argv[2] : nullptr;

    bool use_inst_buf = (insts_path != nullptr);

    // These are only used in opcode=0 mode
    const char* a_path      = nullptr;
    const char* b_path      = nullptr;
    int M = 0, K = 0, N = 0;
    const char* c_out_path  = "C.bin";

    if (use_inst_buf) {
        if (argc < 8) {
            fprintf(stderr, "Usage (opcode=0 mode):\n");
            fprintf(stderr, "  %s <xclbin> <insts.bin> <A.bin> <B.bin> <M> <K> <N> [C.bin]\n",
                    argv[0]);
            return 1;
        }
        a_path = argv[3];
        b_path = argv[4];
        M = atoi(argv[5]);
        K = atoi(argv[6]);
        N = atoi(argv[7]);
        if (argc > 8) c_out_path = argv[8];

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  NPU Raw Submission — opcode=0 (instruction buffer)     ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
        printf("  xclbin: %s\n", xclbin_path);
        printf("  insts:  %s\n", insts_path);
        printf("  A:      %s\n", a_path);
        printf("  B:      %s\n", b_path);
        printf("  M=%d K=%d N=%d  output: %s\n", M, K, N, c_out_path);
    } else {
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  NPU Raw Submission — opcode=3 (pre-compiled)           ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
        printf("  xclbin: %s\n", xclbin_path);
    }
    printf("\n");

    // ── Step 1: Open XRT device ──────────────────────────────────
    printf("Step 1: Opening XRT device...\n");
    xrt::device device;
    try {
        device = xrt::device(0);
    } catch (...) {
        fprintf(stderr, "  ❌ xrt::device(0) failed\n");
        return 1;
    }
    printf("  ✅ Device opened\n");

    // ── Step 2: Load xclbin ──────────────────────────────────────
    printf("\nStep 2: Loading xclbin: %s\n", xclbin_path);
    auto xclbin_data = read_file(xclbin_path);
    if (xclbin_data.empty()) {
        fprintf(stderr, "  ❌ Cannot read xclbin\n");
        return 1;
    }
    printf("  ✅ Read %zu bytes\n", xclbin_data.size());

    xrt::xclbin xclbin;
    xrt::uuid uuid;
    try {
        xclbin = xrt::xclbin(xclbin_data);
        device.register_xclbin(xclbin);
        uuid = xclbin.get_uuid();
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ xclbin load failed: %s\n", e.what());
        return 1;
    }
    printf("  ✅ xclbin loaded, UUID=%s\n", uuid.to_string().c_str());

    // ── Step 3: Open kernel ──────────────────────────────────────
    printf("\nStep 3: Opening kernel MLIR_AIE...\n");
    xrt::kernel kernel;
    try {
        kernel = xrt::kernel(device, uuid, "MLIR_AIE");
    } catch (const std::exception& e) {
        fprintf(stderr, "  ❌ kernel open failed: %s\n", e.what());
        return 1;
    }
    printf("  ✅ Kernel ready\n");

    // ── Step 4: Create BOs ───────────────────────────────────────
    printf("\nStep 4: Creating BOs...\n");

    if (use_inst_buf) {
        // ───────── opcode=0 mode: instruction-buffer submission ─────────

        // 4a. Read instructions from insts.bin
        auto insts_data = read_file(insts_path);
        if (insts_data.empty()) {
            fprintf(stderr, "  ❌ Cannot read insts.bin: %s\n", insts_path);
            return 1;
        }

        // Validate size is 4-byte aligned
        if (insts_data.size() % 4 != 0) {
            fprintf(stderr, "  ❌ insts.bin size %zu not multiple of 4\n",
                    insts_data.size());
            return 1;
        }

        uint32_t ninstr = insts_data.size() / 4;
        printf("  Instructions: %u words (%zu bytes)\n", ninstr, insts_data.size());

        // Validate header: expected 0x06040100 at words 0..3
        const uint32_t* insts_u32 = reinterpret_cast<const uint32_t*>(insts_data.data());
        if (insts_data.size() >= 16) {
            uint32_t h0 = insts_u32[0];
            uint32_t h1 = insts_u32[1];
            uint32_t h2 = insts_u32[2];
            uint32_t h3 = insts_u32[3];
            printf("  Header: 0x%08x 0x%08x 0x%08x 0x%08x\n", h0, h1, h2, h3);
            // Typical MLIR-AIE instruction header starts with 0x06040100 pattern
            if ((h0 & 0xFF) != 0 || (h0 & 0xFF00) != 0x0100) {
                printf("  ⚠ Header does not match expected pattern (0x06040100...)\n");
                printf("    This may still be valid for this xclbin.\n");
            }
        }

        // 4b. Create instruction BO (on SRAM for performance)
        xrt::bo instr_bo;
        try {
            instr_bo = xrt::bo(device, insts_data.size(),
                               XCL_BO_FLAGS_CACHEABLE,
                               kernel.group_id(1));
        } catch (...) {
            // Fallback: host-only BO with default group
            try {
                instr_bo = xrt::bo(device, insts_data.size(),
                                   xrt::bo::flags::host_only, 0);
            } catch (const std::exception& e) {
                fprintf(stderr, "  ❌ Failed to create instr BO: %s\n", e.what());
                return 1;
            }
        }
        memcpy(instr_bo.map(), insts_data.data(), insts_data.size());
        instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, insts_data.size(), 0);
        printf("  ✅ Instruction BO created: %zu bytes, addr=0x%llx\n",
               insts_data.size(),
               (unsigned long long)instr_bo.address());

        // 4c. Create BOs for A, B matrices and C output
        //     A: M×K float32,  B: K×N float32,  C: M×N float32
        size_t a_size = (size_t)M * K * sizeof(float);
        size_t b_size = (size_t)K * N * sizeof(float);
        size_t c_size = (size_t)M * N * sizeof(float);

        // Round up to 4KB alignment for NPU BOs
        size_t bo_size_a = ((a_size + 4095) / 4096) * 4096;
        size_t bo_size_b = ((b_size + 4095) / 4096) * 4096;
        size_t bo_size_c = ((c_size + 4095) / 4096) * 4096;

        // Read matrix data from files
        auto a_data = read_float_bin(a_path);
        auto b_data = read_float_bin(b_path);

        if (a_data.size() != (size_t)M * K) {
            fprintf(stderr, "  ❌ A matrix: expected %d x %d = %d floats, got %zu\n",
                    M, K, M * K, a_data.size());
            return 1;
        }
        if (b_data.size() != (size_t)K * N) {
            fprintf(stderr, "  ❌ B matrix: expected %d x %d = %d floats, got %zu\n",
                    K, N, K * N, b_data.size());
            return 1;
        }

        // Create A BO
        xrt::bo bo_A(device, bo_size_a, xrt::bo::flags::host_only, 0);
        memcpy(bo_A.map(), a_data.data(), a_size);
        bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_size_a, 0);
        printf("  ✅ A BO: %zu bytes (alloc %zu)\n", a_size, bo_size_a);

        // Create B BO
        xrt::bo bo_B(device, bo_size_b, xrt::bo::flags::host_only, 0);
        memcpy(bo_B.map(), b_data.data(), b_size);
        bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_size_b, 0);
        printf("  ✅ B BO: %zu bytes (alloc %zu)\n", b_size, bo_size_b);

        // Create C BO (zero-initialized)
        xrt::bo bo_C(device, bo_size_c, xrt::bo::flags::host_only, 0);
        memset(bo_C.map(), 0, bo_size_c);
        bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_size_c, 0);
        printf("  ✅ C BO: %zu bytes (alloc %zu)\n", c_size, bo_size_c);

        // Create extra BOs (scratch, etc.) — kernel expects 5 buffer handles
        size_t extra_size = 4096;  // small scratch buffers
        xrt::bo bo_extra1(device, extra_size, xrt::bo::flags::host_only, 0);
        xrt::bo bo_extra2(device, extra_size, xrt::bo::flags::host_only, 0);
        printf("  ✅ Extra BOs created\n");

        // ── Step 5: Run kernel with instruction buffer ─────────────
        printf("\nStep 5: Running kernel (opcode=0, %u instructions)...\n", ninstr);

        auto t0 = Clock::now();
        auto run = kernel(
            (uint64_t)0,       // opcode = 0 (dynamic instruction submission)
            instr_bo,          // instr = BO object (not physical address!)
            (uint32_t)ninstr,  // ninstr = instruction count
            bo_A, bo_B, bo_C, bo_extra1, bo_extra2
        );
        run.wait();
        double elapsed_ms = ms_since(t0);
        printf("  ✅ Completed in %.3f ms\n", elapsed_ms);

        // ── Step 6: Read results ──────────────────────────────────
        printf("\nStep 6: Reading results -> %s\n", c_out_path);

        bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_size, 0);
        float* c_map = static_cast<float*>(bo_C.map());

        // Write output to file
        std::ofstream c_out(c_out_path, std::ios::binary);
        if (!c_out) {
            fprintf(stderr, "  ❌ Cannot write output: %s\n", c_out_path);
            return 1;
        }
        c_out.write(reinterpret_cast<const char*>(c_map), c_size);
        c_out.close();

        // Print first few values
        printf("  C[0..7]: ");
        for (int i = 0; i < 8 && i < M * N; i++) {
            printf("%.6f ", c_map[i]);
        }
        printf("\n");

        // Compute simple checksum
        double sum = 0.0;
        for (size_t i = 0; i < (size_t)M * N; i++) sum += c_map[i];
        printf("  Sum of all elements: %.6f\n", sum);

        printf("\n✅ Done! Output written to %s\n", c_out_path);

    } else {
        // ───────── opcode=3 mode: pre-compiled AIE program ─────────
        //
        // IMPORTANT: XRT sync(dir, sz, offset) — sz is the size to sync,
        // NOT the offset. The original code had (dir, 0, bo_size) which
        // was a no-op (sync 0 bytes at offset=4MB). Fixed to (dir, sz, 0).

        size_t bo_size = 4 * 1024 * 1024;

        std::vector<xrt::bo> bos;
        for (int i = 0; i < 5; i++) {
            xrt::bo bo = xrt::bo(device, bo_size, xrt::bo::flags::host_only, 0);
            void* map = bo.map();
            memset(map, 0x42 + i, bo_size);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_size, 0);
            bos.push_back(std::move(bo));
            printf("  ✅ BO[%d] created (%zu bytes)\n", i, bo_size);
        }

        // Step 5: Run kernel with pre-compiled AIE
        printf("\nStep 5: Running kernel (opcode=3)...\n");

        auto t0 = Clock::now();
        auto run = kernel(
            (uint64_t)3,    // opcode
            (uint64_t)0,    // instr_ptr = NULL
            (uint32_t)0,    // ninstr = 0
            bos[0], bos[1], bos[2], bos[3], bos[4]
        );
        run.wait();
        double elapsed_ms = ms_since(t0);
        printf("  ✅ Completed in %.3f ms\n", elapsed_ms);

        // Step 6: Read results
        printf("\nStep 6: Reading results...\n");
        for (int i = 0; i < 5; i++) {
            bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE, 0, 64);
            uint8_t* map = (uint8_t*)bos[i].map();
            printf("  BO[%d]: ", i);
            for (int j = 0; j < 16; j++) printf("%02x ", map[j]);
            printf("\n");
        }

        printf("\n✅ Done!\n");
    }

    return 0;
}
