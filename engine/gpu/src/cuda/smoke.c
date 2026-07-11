// Standalone C smoke test for the ZINC CUDA backend primitive layer.
// Exercises the full cuda_shim.h ABI on real hardware without Zig or the rest
// of ZINC: device selection, staged buffers + H2D/D2H, NVRTC runtime compile,
// the buffers+push dispatch ABI, and both sync and async commit paths.
//
// Build (on the box):
//   gcc -O2 -I/usr/local/cuda/include cuda_shim.c smoke.c -o smoke \
//       -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lnvrtc \
//       -Wl,-rpath,/usr/local/cuda/lib64

#include "cuda_shim.h"
#include <stdio.h>

// Kernels authored to the ZINC dispatch ABI: bound buffers first (as device
// pointers), then a single by-value push-constant struct.
static const char* KSRC =
    "struct Push { int n; };\n"
    "extern \"C\" __global__ void vadd(const float* a, const float* b, float* c, struct Push pc){\n"
    "  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < pc.n) c[i] = a[i] + b[i];\n"
    "}\n"
    "extern \"C\" __global__ void dp4a_k(const int* a, const int* b, int* out, struct Push pc){\n"
    "  int acc = 0; acc = __dp4a(a[0], b[0], acc); out[0] = acc;\n"
    "}\n";

struct Push { int n; };

int main(void) {
    // Pick the highest-compute-capability device (the RTX 5090, sm_120).
    int best = -1; unsigned best_cc = 0;
    for (int idx = 0; idx < 8; idx++) {
        CudaCtx* c = cuda_init(idx);
        if (!c) break;
        unsigned cc = cuda_compute_capability(c);
        char nm[128]; cuda_device_name(c, nm, sizeof nm);
        printf("dev[%d]: %-28s cc=%u SMs=%u vram=%.1fGB\n",
               idx, nm, cc, cuda_sm_count(c), cuda_total_memory(c) / 1e9);
        if (cc > best_cc) { best_cc = cc; best = idx; }
        cuda_destroy(c);
    }
    if (best < 0) { printf("FAIL: no CUDA device\n"); return 1; }
    printf("=> using dev[%d] (cc=%u)\n", best, best_cc);

    CudaCtx* c = cuda_init(best);
    if (!c) { printf("FAIL: cuda_init: %s\n", cuda_last_error()); return 1; }

    // --- staged buffers + upload + NVRTC compile + dispatch + commit_and_wait ---
    const int N = 1024;
    void *ha, *hb;
    CudaBuf* a = cuda_create_buffer_staged(c, N * sizeof(float), &ha);
    CudaBuf* b = cuda_create_buffer_staged(c, N * sizeof(float), &hb);
    CudaBuf* out = cuda_create_buffer(c, N * sizeof(float));
    for (int i = 0; i < N; i++) { ((float*)ha)[i] = (float)i; ((float*)hb)[i] = 2.0f * (float)i; }
    cuda_upload(c, a, ha, N * sizeof(float));
    cuda_upload(c, b, hb, N * sizeof(float));

    CudaPipe* vadd = cuda_create_pipeline(c, KSRC, "vadd", NULL, 0);
    if (!vadd) { printf("FAIL: nvrtc vadd: %s\n", cuda_last_error()); return 2; }
    CudaPipe* dp = cuda_create_pipeline(c, KSRC, "dp4a_k", NULL, 0);
    if (!dp) { printf("FAIL: nvrtc dp4a: %s\n", cuda_last_error()); return 2; }
    printf("nvrtc: compiled vadd (max_threads=%u) + dp4a_k for sm_%u\n",
           cuda_pipeline_max_threads(vadd), best_cc);

    struct Push push = { N };
    uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
    CudaBuf* bufs[3] = { a, b, out };
    CudaCmd* cmd = cuda_begin_command(c);
    cuda_dispatch(cmd, vadd, grid, block, bufs, 3, &push, sizeof push, 0);
    cuda_commit_and_wait(cmd);

    float hc[1024];
    cuda_download(c, out, hc, N * sizeof(float));
    printf("vadd: c[1]=%.1f (expect 3.0)  c[100]=%.1f (expect 300.0)\n", hc[1], hc[100]);

    // --- dp4a via the abstraction + the ASYNC commit path ---
    int pa = (1 & 0xff) | (2 << 8) | (3 << 16) | (4 << 24);
    int pb = (5 & 0xff) | (6 << 8) | (7 << 16) | (8 << 24);
    CudaBuf* da = cuda_create_buffer(c, 4);
    CudaBuf* db = cuda_create_buffer(c, 4);
    CudaBuf* dout = cuda_create_buffer(c, 4);
    cuda_upload(c, da, &pa, 4);
    cuda_upload(c, db, &pb, 4);
    CudaBuf* bufs2[3] = { da, db, dout };
    uint32_t one[3] = { 1, 1, 1 };
    CudaCmd* cmd2 = cuda_begin_command(c);
    cuda_dispatch(cmd2, dp, one, one, bufs2, 3, &push, sizeof push, 0);
    cuda_commit_async(cmd2);   // return immediately…
    cuda_wait(cmd2);           // …then block on the event
    int hr = 0;
    cuda_download(c, dout, &hr, 4);
    printf("dp4a (via abstraction, async path): %d (expect 70)\n", hr);

    int ok = (hc[1] == 3.0f && hc[100] == 300.0f && hr == 70);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");

    cuda_free_buffer(a); cuda_free_buffer(b); cuda_free_buffer(out);
    cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_buffer(dout);
    cuda_free_pipeline(vadd); cuda_free_pipeline(dp);
    cuda_destroy(c);
    return ok ? 0 : 1;
}
