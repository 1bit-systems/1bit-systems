// SPDX-License-Identifier: MIT
/*
 * Direct NPU GEMM via XRT + Captured IRON xclbin
 *
 * Loads the xclbin captured from experiment_10_phase5 and submits
 * GEMM directly via XRT — bypassing IRON ObjectFifo DMA entirely.
 *
 * KEY INSIGHT: The xclbin is compiled by IRON but we load and run it
 * with raw XRT, giving us full control over buffer allocation (GTT
 * dma-buf) and kernel submission timing.
 *
 * Build:
 *   g++ -std=gnu++17 -O2 -o tests/test_direct_npu_gemm \
 *       tests/test_direct_npu_gemm.cpp \
 *       -I/usr/include/libdrm -I/usr/include \
 *       -L/usr/lib/x86_64-linux-gnu \
 *       -lxrt_coreutil -ldrm -ldrm_amdgpu
 *
 * Run:
 *   ./tests/test_direct_npu_gemm [xclbin_path] [insts_path]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <drm/drm.h>
#include <drm/amdgpu_drm.h>
#include <libdrm/amdgpu.h>

#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

using Clock = std::chrono::high_resolution_clock;

/* ================================================================== */
/*  BFP16 Packing Library                                              */
/* ================================================================== */
constexpr int R = 8, S = 8, T = 8;
constexpr int TILE_ELEMS = R * S;

void pack_A_bf16(const float *A, float *A_packed, int M, int K) {
    int M_tiles = M / R, K_tiles = K / S;
    for (int z = 0; z < M_tiles; z++)
        for (int ki = 0; ki < K_tiles; ki++) {
            int off = (z * K_tiles + ki) * TILE_ELEMS;
            for (int r = 0; r < R; r++)
                for (int c = 0; c < S; c++)
                    A_packed[off + r * S + c] = A[(z * R + r) * K + (ki * S + c)];
        }
}

// Pack B[K×N] into column-major TILE order, row-major within 8×8 tiles
// This is the pre-packed format - kernel processes without transpose
void pack_B_bf16_colmaj(const float *B, float *B_packed, int K, int N) {
    int K_tiles = K / S, N_tiles = N / T;
    for (int nj = 0; nj < N_tiles; nj++)
        for (int ki = 0; ki < K_tiles; ki++) {
            int off = (nj * K_tiles + ki) * TILE_ELEMS;
            for (int r = 0; r < S; r++)
                for (int c = 0; c < T; c++)
                    B_packed[off + r * T + c] = B[(ki * S + r) * N + (nj * T + c)];
        }
}

// cpu reference
void cpu_gemm(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

static std::vector<char> load_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    return buf;
}

/* ================================================================== */
/*  GTT dma-buf zero-copy allocation                                   */
/* ================================================================== */
struct GTTBuffer {
    amdgpu_device_handle dev;
    amdgpu_bo_handle     bo;
    void                *cpu_ptr;
    int                  dmabuf_fd;
    int                  npu_handle;
    size_t               size;
};

int gtt_alloc(GTTBuffer *b, size_t size, int fd_amdgpu, int fd_npu) {
    memset(b, 0, sizeof(*b)); b->dmabuf_fd = -1; b->size = size;
    uint32_t maj, min;
    if (amdgpu_device_initialize(fd_amdgpu, &maj, &min, &b->dev) < 0) return -1;

    struct amdgpu_bo_alloc_request req = {};
    req.alloc_size = size; req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
    req.flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;
    if (amdgpu_bo_alloc(b->dev, &req, &b->bo)) return -1;
    if (amdgpu_bo_cpu_map(b->bo, &b->cpu_ptr)) { amdgpu_bo_free(b->bo); return -1; }

    uint32_t h;
    amdgpu_bo_export(b->bo, amdgpu_bo_handle_type_dma_buf_fd, &h);
    b->dmabuf_fd = (int)h;

    struct drm_prime_handle prime = {}; prime.fd = b->dmabuf_fd;
    if (ioctl(fd_npu, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime) < 0) return -1;
    b->npu_handle = prime.handle;
    return 0;
}

void gtt_free(GTTBuffer *b) {
    if (b->dmabuf_fd >= 0) close(b->dmabuf_fd);
    if (b->cpu_ptr) amdgpu_bo_cpu_unmap(b->bo);
    if (b->bo) amdgpu_bo_free(b->bo);
    if (b->dev) amdgpu_device_deinitialize(b->dev);
}

/* ================================================================== */
/*  main                                                              */
/* ================================================================== */
int main(int argc, char **argv) {
    const char *xclbin_path = (argc > 1) ? argv[1]
        : "/home/bcloud/strixhalo-npu-setup/saved_xclbins/phase5_bf16_512x8192x512.xclbin";
    const char *insts_path = (argc > 2) ? argv[2]
        : "/home/bcloud/strixhalo-npu-setup/saved_xclbins/phase5_bf16_512x8192x512_insts.bin";

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Direct NPU GEMM via Captured IRON xclbin + XRT      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* ---- Open devices ---- */
    int fd_npu = open("/dev/accel/accel0", O_RDWR);
    int fd_amdgpu = open("/dev/dri/renderD128", O_RDWR);
    if (fd_npu < 0 || fd_amdgpu < 0) {
        printf("FAIL: device open\n"); return 1;
    }
    printf("✅ Devices opened (NPU + GPU)\n");

    /* ---- Open NPU via XRT ---- */
    xrt::device device;
    try { device = xrt::device(1); }
    catch (...) { try { device = xrt::device(0); }
        catch (...) { printf("FAIL: XRT device\n"); return 1; } }
    printf("✅ XRT device opened\n");

    /* ---- Load xclbin ---- */
    auto xclbin_data = load_file(xclbin_path);
    if (xclbin_data.empty()) { printf("FAIL: load %s\n", xclbin_path); return 1; }
    auto xclbin = xrt::xclbin(xclbin_data);
    try { device.register_xclbin(xclbin); }
    catch (const std::exception &e) {
        printf("FAIL: register_xclbin: %s\n", e.what()); return 1;
    }
    printf("✅ xclbin registered (%s, %.0f KB)\n", xclbin_path,
           xclbin_data.size() / 1024.0);

    /* ---- Get kernel ---- */
    auto uuid = xclbin.get_uuid();
    printf("   UUID: %s\n", uuid.to_string().c_str());

    // IRON kernels are named "gemm_bf16_f32_bfp16_packed", "zero_f32_packed"
    // The MLIR module has the compute tiles; find any compute kernel
    printf("Available kernels:\n");
    // XRT auto-discovers kernels from xclbin

    /* ---- GEMM dimensions ---- */
    int M = 64, N = 64, K = 64;  // small test
    printf("\nGEMM: %d×%d×%d (bf16→f32)\n", M, N, K);

    /* ---- Allocate XRT BOs (host_only = system memory, zero-copy) ---- */
    size_t sz_A = M * K * sizeof(float);
    size_t sz_B = K * N * sizeof(float);
    size_t sz_C = M * N * sizeof(float);

    xrt::bo bo_A = xrt::bo(device, sz_A, xrt::bo::flags::host_only, 0);
    xrt::bo bo_B = xrt::bo(device, sz_B, xrt::bo::flags::host_only, 0);
    xrt::bo bo_C = xrt::bo(device, sz_C, xrt::bo::flags::host_only, 0);

    float *A = bo_A.map<float*>();
    float *B_raw = bo_B.map<float*>();
    float *C = bo_C.map<float*>();

    /* ---- Fill with test data ---- */
    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K*N; i++) B_raw[i] = (float)(rand() % 100) / 100.0f;
    memset(C, 0, sz_C);

    /* ---- CPU reference ---- */
    std::vector<float> C_ref(M * N);
    cpu_gemm(A, B_raw, C_ref.data(), M, N, K);

    /* ---- Pre-pack B on CPU ---- */
    std::vector<float> B_packed(K * N);
    pack_B_bf16_colmaj(B_raw, B_packed.data(), K, N);
    memcpy(B_raw, B_packed.data(), sz_B);

    /* ---- Sync to device ---- */
    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    /* ---- Load insts from file ---- */
    auto insts_data = load_file(insts_path);
    if (insts_data.empty()) {
        printf("⚠️  No insts.bin — kernel may need AIE instructions\n");
        printf("   The xclbin is valid but requires instruction buffer.\n");
        printf("   IRON generates insts at runtime via ObjectFifo DMA.\n\n");
        printf("   To use without IRON, we need either:\n");
        printf("   1. A pre-compiled xclbin with embedded instructions, or\n");
        printf("   2. XAIE transaction generation (like the MLIR-aie toolchain)\n");
        close(fd_npu); close(fd_amdgpu);
        return 0;
    }

    printf("✅ Loaded insts.bin (%zu bytes = %zu words)\n",
           insts_data.size(), insts_data.size() / 4);

    /* ---- Try to find kernel in xclbin ---- */
    // The IRON xclbin may have a different kernel structure than the standard toolchain
    // Let's try the common kernel names
    printf("\nTrying kernel execution...\n");

    try {
        // Try MLIR_AIE (common name)
        auto kernel = xrt::kernel(device, uuid, "MLIR_AIE");
        printf("Found kernel: MLIR_AIE\n");

        uint64_t opcode = 0;
        auto *instr_ptr = reinterpret_cast<uint32_t*>(insts_data.data());
        uint32_t ninstr = insts_data.size() / 4;

        auto run = kernel(opcode, instr_ptr, ninstr,
                          bo_A, bo_B, bo_C,
                          xrt::bo(), xrt::bo());
        run.wait();
        printf("✅ Kernel executed!\n");
    } catch (const std::exception &e) {
        printf("Kernel 'MLIR_AIE' not found: %s\n", e.what());
    }

    /* ---- Sync back ---- */
    bo_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    /* ---- Verify ---- */
    bool pass = true; double max_err = 0;
    for (int i = 0; i < M && pass; i++)
        for (int j = 0; j < N && pass; j++) {
            double err = fabs(C[i*N+j] - C_ref[i*N+j]);
            if (err > max_err) max_err = err;
            if (err > 0.1f) pass = false;
        }

    printf("\n%s (max err=%.6f)\n", pass ? "✅ PASS" : "❌ FAIL", max_err);
    printf("xclbin: %s\n", xclbin_path);
    printf("insts:  %s\n", insts_path);

    close(fd_npu); close(fd_amdgpu);
    return pass ? 0 : 1;
}
