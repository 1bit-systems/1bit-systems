// Numeric validation for the ZINC CUDA kernels (src/shaders/cuda/kernels.cu).
// Reads the .cu at runtime, NVRTC-compiles each kernel via the shim, runs it on
// the GPU, and compares against an independent CPU reference. Standalone — no
// repo / Zig needed.
//
// Build (on the box, from ~/cuda_proto with kernels.cu present):
//   gcc -O2 -I. -I/usr/local/cuda/include kernels_test.c cuda_shim.c -o kernels_test \
//       -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lcuda -lnvrtc -lm \
//       -Wl,-rpath,/usr/local/cuda/lib64

#include "cuda_shim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ---- host mirrors of the kernel math (the ground-truth Q4_K spec) -----------
static float half_to_float_h(uint16_t h) {
    unsigned sign = (unsigned)(h >> 15) & 1u;
    unsigned exp = (unsigned)(h >> 10) & 0x1Fu;
    unsigned mant = (unsigned)h & 0x3FFu;
    unsigned f;
    if (exp == 0u) {
        if (mant == 0u) { f = sign << 31; }
        else { int e = 1; while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
               mant &= 0x3FFu; f = (sign << 31) | ((unsigned)(127 - 15 + e) << 23) | (mant << 13); }
    } else if (exp == 0x1Fu) { f = (sign << 31) | (0xFFu << 23) | (mant << 13); }
    else { f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13); }
    float out; memcpy(&out, &f, 4); return out;
}

static void get_scale_min_k4_h(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63u; *m = q[j + 4] & 63u; }
    else { *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Canonical reference implementation dequant_row_q4_K for one 256-elem block (36 u32).
static void deq_q4k_block_h(const uint32_t* blk, float* out) {
    uint32_t dd = blk[0];
    float d = half_to_float_h((uint16_t)(dd & 0xFFFF));
    float dmin = half_to_float_h((uint16_t)(dd >> 16));
    const uint8_t* scales = (const uint8_t*)(blk + 1);
    const uint8_t* qs = (const uint8_t*)(blk + 4);
    int is = 0; const uint8_t* q = qs; float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_h(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_h(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

// Canonical reference implementation dequant_row_q5_K for one 256-elem block (176 bytes).
static void deq_q5k_block_h(const unsigned char* blk, float* out) {
    float d = half_to_float_h((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = half_to_float_h((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* qlp = blk + 48;
    int is = 0; uint8_t u1 = 1, u2 = 2; float* y = out;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4_h(is + 0, scales, &sc, &m); float d1 = d * sc, m1 = dmin * m;
        get_scale_min_k4_h(is + 1, scales, &sc, &m); float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((qlp[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((qlp[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        qlp += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}

// Canonical reference implementation dequant_row_q6_K for one 256-elem block (210 bytes).
static void deq_q6k_block_h(const unsigned char* blk, float* out) {
    float d = half_to_float_h((uint16_t)(blk[208] | (blk[209] << 8)));
    const uint8_t* ql = blk + 0;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* qlh = ql + (n / 128) * 64;
        const uint8_t* qhh = qh + (n / 128) * 32;
        const int8_t* sch = sc + (n / 128) * 8;
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
            y[n + l +  0] = d * sch[is + 0] * q1;
            y[n + l + 32] = d * sch[is + 2] * q2;
            y[n + l + 64] = d * sch[is + 4] * q3;
            y[n + l + 96] = d * sch[is + 6] * q4;
        }
    }
}

// ---- deterministic PRNG + helpers -------------------------------------------
static uint32_t rng = 0x9e3779b9u;
static uint32_t xrand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(void) { return (float)(xrand() & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f; }
// "nice" positive normal half: exp field 7..13 -> magnitudes ~0.004..0.25.
static uint16_t nice_half(void) { uint16_t e = 7 + (uint16_t)(xrand() % 7); return (uint16_t)((e << 10) | (xrand() & 0x3FF)); }

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1); if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, n, f); buf[rd] = 0; fclose(f); return buf;
}

static int pick_best_device(void) {
    int best = -1; unsigned best_cc = 0;
    for (int i = 0; i < 8; i++) {
        CudaCtx* c = cuda_init(i); if (!c) break;
        unsigned cc = cuda_compute_capability(c);
        if (cc > best_cc) { best_cc = cc; best = i; }
        cuda_destroy(c);
    }
    return best;
}

int main(void) {
    int dev = pick_best_device();
    if (dev < 0) { printf("FAIL: no CUDA device\n"); return 1; }
    CudaCtx* c = cuda_init(dev);
    char nm[128]; cuda_device_name(c, nm, sizeof nm);
    printf("device: %s (cc=%u)\n", nm, cuda_compute_capability(c));

    char* src = read_file("kernels.cu");
    if (!src) { printf("FAIL: cannot read kernels.cu (run from ~/cuda_proto)\n"); return 1; }
    int all_ok = 1;

    // ===== Test 1: rms_norm =====
    {
        const unsigned tokens = 3, N = 2048; const float eps = 1e-5f;
        float* x = malloc((size_t)tokens * N * 4);
        float* w = malloc((size_t)N * 4);
        float* yref = malloc((size_t)tokens * N * 4);
        float* ygpu = malloc((size_t)tokens * N * 4);
        for (unsigned i = 0; i < tokens * N; i++) x[i] = frand();
        for (unsigned i = 0; i < N; i++) w[i] = frand() * 0.5f + 1.0f;
        for (unsigned t = 0; t < tokens; t++) {
            double ss = 0; for (unsigned i = 0; i < N; i++) { float v = x[t * N + i]; ss += (double)v * v; }
            float rinv = 1.0f / sqrtf((float)(ss / N) + eps);
            for (unsigned i = 0; i < N; i++) yref[t * N + i] = w[i] * (x[t * N + i] * rinv);
        }
        CudaBuf* dx = cuda_create_buffer(c, (size_t)tokens * N * 4);
        CudaBuf* dw = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)tokens * N * 4);
        cuda_upload(c, dx, x, (size_t)tokens * N * 4);
        cuda_upload(c, dw, w, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "rms_norm", NULL, 0);
        if (!p) { printf("FAIL rms_norm compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; float eps; } push = { N, eps };
        uint32_t grid[3] = { tokens, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { dx, dw, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)tokens * N * 4);
        float maxrel = 0;
        for (unsigned i = 0; i < tokens * N; i++) {
            float a = yref[i], b = ygpu[i], r = fabsf(a - b) / (fabsf(a) + 1e-4f);
            if (r > maxrel) maxrel = r;
        }
        int ok = maxrel < 1e-3f; all_ok &= ok;
        printf("rms_norm [%ux%u]: max_rel_err=%.2e -> %s\n", tokens, N, maxrel, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dx); cuda_free_buffer(dw); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(x); free(w); free(yref); free(ygpu);
    }

    // ===== Test 2: dmmv_q4k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; unsigned nblk = M * bpr;
        uint32_t* a = malloc((size_t)nblk * 36 * 4);
        for (unsigned bi = 0; bi < nblk; bi++) {
            uint32_t* blk = a + (size_t)bi * 36;
            blk[0] = nice_half() | ((uint32_t)nice_half() << 16);
            uint8_t* scales = (uint8_t*)(blk + 1); for (int k = 0; k < 12; k++) scales[k] = xrand() & 0xFF;
            uint8_t* qs = (uint8_t*)(blk + 4); for (int k = 0; k < 128; k++) qs[k] = xrand() & 0xFF;
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4);
        float deq[256];
        for (unsigned row = 0; row < M; row++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) {
                deq_q4k_block_h(a + (size_t)(row * bpr + b) * 36, deq);
                for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e];
            }
            yref[row] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, (size_t)nblk * 36 * 4);
        CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, (size_t)nblk * 36 * 4);
        cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q4k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q4k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, a_off, x_off, y_off, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        float* ygpu = malloc((size_t)M * 4); cuda_download(c, dy, ygpu, (size_t)M * 4);
        float maxrel = 0;
        for (unsigned r = 0; r < M; r++) {
            float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f);
            printf("  row %u: ref=%.3f gpu=%.3f\n", r, yref[r], ygpu[r]);
            if (rr > maxrel) maxrel = rr;
        }
        int ok = maxrel < 2e-3f; all_ok &= ok;
        printf("dmmv_q4k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, maxrel, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 3: swiglu =====
    {
        const unsigned N = 4096;
        float* gate = malloc((size_t)N * 4); float* up = malloc((size_t)N * 4);
        float* yref = malloc((size_t)N * 4); float* ygpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { gate[i] = frand() * 4.0f; up[i] = frand(); }
        for (unsigned i = 0; i < N; i++) { float g = gate[i]; yref[i] = (g / (1.0f + expf(-g))) * up[i]; }
        CudaBuf* dg = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* du = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dy = cuda_create_buffer(c, (size_t)N * 4);
        cuda_upload(c, dg, gate, (size_t)N * 4); cuda_upload(c, du, up, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "swiglu", NULL, 0);
        if (!p) { printf("FAIL swiglu compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { dg, du, dy };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(yref[i] - ygpu[i]) / (fabsf(yref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("swiglu [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dg); cuda_free_buffer(du); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(gate); free(up); free(yref); free(ygpu);
    }

    // ===== Test 4: scale_accumulate (a += scale*b) =====
    {
        const unsigned N = 4096; const float scale = 0.37f;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { a0[i] = frand(); b[i] = frand(); }
        for (unsigned i = 0; i < N; i++) aref[i] = a0[i] + scale * b[i];
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)N * 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "scale_accumulate", NULL, 0);
        if (!p) { printf("FAIL scale_accumulate compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; float scale; } push = { N, scale };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[2] = { da, db };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("scale_accumulate [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    // ===== Test 5: sigmoid_scale_acc (a += sigmoid(c0)*b) =====
    {
        const unsigned N = 4096; const float cgate = 0.6f;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) { a0[i] = frand(); b[i] = frand(); }
        float g = 1.0f / (1.0f + expf(-cgate));
        for (unsigned i = 0; i < N; i++) aref[i] = a0[i] + g * b[i];
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)N * 4);
        CudaBuf* dc = cuda_create_buffer(c, 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)N * 4); cuda_upload(c, dc, &cgate, 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "sigmoid_scale_acc", NULL, 0);
        if (!p) { printf("FAIL sigmoid_scale_acc compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 };
        CudaBuf* bufs[3] = { da, db, dc };
        CudaCmd* cmd = cuda_begin_command(c);
        cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0);
        cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("sigmoid_scale_acc [%u]: max_rel_err=%.2e -> %s\n", N, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_buffer(dc); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    // ===== Test 6: dmmv_f32 =====
    {
        const unsigned M = 5, K = 512;
        float* w = malloc((size_t)M * K * 4); float* x = malloc((size_t)K * 4);
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4);
        for (unsigned i = 0; i < M * K; i++) w[i] = frand();
        for (unsigned i = 0; i < K; i++) x[i] = frand();
        for (unsigned r = 0; r < M; r++) { double acc = 0; for (unsigned k = 0; k < K; k++) acc += (double)w[r * K + k] * x[k]; yref[r] = (float)acc; }
        CudaBuf* dw = cuda_create_buffer(c, (size_t)M * K * 4); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, dw, w, (size_t)M * K * 4); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_f32", NULL, 0);
        if (!p) { printf("FAIL dmmv_f32 compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { dw, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_f32 [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dw); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(w); free(x); free(yref); free(ygpu);
    }

    // ===== Test 7: dmmv_q8_0 =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 32; size_t bytes = (size_t)M * bpr * 34;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 34;
            uint16_t d = nice_half(); blk[0] = d & 0xFF; blk[1] = d >> 8;
            for (int i = 0; i < 32; i++) blk[2 + i] = (unsigned char)(xrand() & 0xFF);
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4);
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) {
                unsigned char* blk = a + (size_t)(r * bpr + b) * 34;
                uint16_t db = (uint16_t)(blk[0] | (blk[1] << 8)); float d = half_to_float_h(db);
                for (int i = 0; i < 32; i++) { signed char q = (signed char)blk[2 + i]; acc += (double)(d * (float)q) * x[b * 32 + i]; }
            }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q8_0", NULL, 0);
        if (!p) { printf("FAIL dmmv_q8_0 compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q8_0 [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 8: dmmv_q5k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; size_t bytes = (size_t)M * bpr * 176;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 176;
            uint16_t d = nice_half(), dm = nice_half();
            blk[0] = d & 0xFF; blk[1] = d >> 8; blk[2] = dm & 0xFF; blk[3] = dm >> 8;
            for (int k = 4; k < 176; k++) blk[k] = (unsigned char)(xrand() & 0xFF);
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4); float deq[256];
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) { deq_q5k_block_h(a + (size_t)(r * bpr + b) * 176, deq); for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e]; }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q5k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q5k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q5k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 9: dmmv_q6k =====
    {
        const unsigned M = 5, K = 512; unsigned bpr = K / 256; size_t bytes = (size_t)M * bpr * 210;
        unsigned char* a = malloc(bytes);
        for (unsigned bi = 0; bi < M * bpr; bi++) {
            unsigned char* blk = a + (size_t)bi * 210;
            for (int k = 0; k < 208; k++) blk[k] = (unsigned char)(xrand() & 0xFF);
            uint16_t d = nice_half(); blk[208] = d & 0xFF; blk[209] = d >> 8;
        }
        float* x = malloc((size_t)K * 4); for (unsigned i = 0; i < K; i++) x[i] = frand();
        float* yref = malloc((size_t)M * 4); float* ygpu = malloc((size_t)M * 4); float deq[256];
        for (unsigned r = 0; r < M; r++) {
            double acc = 0;
            for (unsigned b = 0; b < bpr; b++) { deq_q6k_block_h(a + (size_t)(r * bpr + b) * 210, deq); for (int e = 0; e < 256; e++) acc += (double)deq[e] * x[b * 256 + e]; }
            yref[r] = (float)acc;
        }
        CudaBuf* da = cuda_create_buffer(c, bytes); CudaBuf* dx = cuda_create_buffer(c, (size_t)K * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)M * 4);
        cuda_upload(c, da, a, bytes); cuda_upload(c, dx, x, (size_t)K * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "dmmv_q6k", NULL, 0);
        if (!p) { printf("FAIL dmmv_q6k compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned M, K, aoff, xoff, yoff, acc; } push = { M, K, 0, 0, 0, 0 };
        uint32_t grid[3] = { M, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, dx, dy };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)M * 4);
        float mr = 0; for (unsigned r = 0; r < M; r++) { float rr = fabsf(yref[r] - ygpu[r]) / (fabsf(yref[r]) + 1e-2f); if (rr > mr) mr = rr; }
        int ok = mr < 2e-3f; all_ok &= ok;
        printf("dmmv_q6k [M=%u K=%u]: max_rel_err=%.2e -> %s\n", M, K, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_pipeline(p);
        free(a); free(x); free(yref); free(ygpu);
    }

    // ===== Test 10: softmax_topk =====
    {
        const unsigned NE = 128, K = 8;
        float* logits = malloc((size_t)NE * 4);
        for (unsigned i = 0; i < NE; i++) logits[i] = frand() * 4.0f;
        // CPU ref: top-k by logit, then renormalized softmax over the winners.
        float lc[128]; memcpy(lc, logits, (size_t)NE * 4);
        unsigned ref_id[8]; float ref_logit[8];
        for (unsigned ki = 0; ki < K; ki++) {
            float b = -1e30f; unsigned bi = 0;
            for (unsigned i = 0; i < NE; i++) if (lc[i] > b) { b = lc[i]; bi = i; }
            ref_id[ki] = bi; ref_logit[ki] = b; lc[bi] = -1e30f;
        }
        float maxl = -1e30f; for (unsigned i = 0; i < K; i++) maxl = fmaxf(maxl, ref_logit[i]);
        float ws = 0, ref_w[8];
        for (unsigned i = 0; i < K; i++) { ref_w[i] = expf(ref_logit[i] - maxl); ws += ref_w[i]; }
        for (unsigned i = 0; i < K; i++) ref_w[i] /= ws;
        CudaBuf* dl = cuda_create_buffer(c, (size_t)NE * 4); CudaBuf* dout = cuda_create_buffer(c, (size_t)2 * K * 4);
        cuda_upload(c, dl, logits, (size_t)NE * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "softmax_topk", NULL, 0);
        if (!p) { printf("FAIL softmax_topk compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned ne, k; } push = { NE, K };
        uint32_t grid[3] = { 1, 1, 1 }, block[3] = { 64, 1, 1 }; CudaBuf* bufs[2] = { dl, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        unsigned go[16]; cuda_download(c, dout, go, (size_t)2 * K * 4);
        int ids_ok = 1; float wmax = 0;
        for (unsigned i = 0; i < K; i++) {
            if (go[i] != ref_id[i]) ids_ok = 0;
            float gw; memcpy(&gw, &go[K + i], 4);
            float e = fabsf(gw - ref_w[i]); if (e > wmax) wmax = e;
        }
        int ok = ids_ok && wmax < 1e-5f; all_ok &= ok;
        printf("softmax_topk [NE=%u K=%u]: ids_match=%d w_max_err=%.2e -> %s\n", NE, K, ids_ok, wmax, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dl); cuda_free_buffer(dout); cuda_free_pipeline(p); free(logits);
    }

    // ===== Test 11: rope (partial rotation) =====
    {
        const unsigned n_heads = 3, stride = 128, rope_dim = 64, position = 5;
        float freq_base = 1000000.0f; unsigned fbb; memcpy(&fbb, &freq_base, 4);
        unsigned total = n_heads * stride, half = rope_dim / 2;
        float* x = malloc((size_t)total * 4); for (unsigned i = 0; i < total; i++) x[i] = frand();
        float* yref = malloc((size_t)total * 4); float* ygpu = malloc((size_t)total * 4);
        for (unsigned h = 0; h < n_heads; h++) {
            unsigned base = h * stride;
            for (unsigned i = 0; i < half; i++) {
                float xi = x[base + i], xih = x[base + i + half];
                float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)rope_dim);
                float th = (float)position * freq, ct = cosf(th), st = sinf(th);
                yref[base + i] = xi * ct - xih * st;
                yref[base + i + half] = xi * st + xih * ct;
            }
            for (unsigned i = rope_dim; i < stride; i++) yref[base + i] = x[base + i];
        }
        CudaBuf* dx = cuda_create_buffer(c, (size_t)total * 4); CudaBuf* dy = cuda_create_buffer(c, (size_t)total * 4); CudaBuf* df = cuda_create_buffer(c, 4);
        cuda_upload(c, dx, x, (size_t)total * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "rope", NULL, 0);
        if (!p) { printf("FAIL rope compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned stride, rope_dim, n_heads, position, fbb, asb; } push = { stride, rope_dim, n_heads, position, fbb, 0 };
        uint32_t grid[3] = { n_heads, 1, 1 }, block[3] = { 64, 1, 1 }; CudaBuf* bufs[3] = { dx, dy, df };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dy, ygpu, (size_t)total * 4);
        float mr = 0; for (unsigned i = 0; i < total; i++) { float r = fabsf(yref[i] - ygpu[i]) / (fabsf(yref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("rope [heads=%u stride=%u rope_dim=%u]: max_rel_err=%.2e -> %s\n", n_heads, stride, rope_dim, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dx); cuda_free_buffer(dy); cuda_free_buffer(df); cuda_free_pipeline(p);
        free(x); free(yref); free(ygpu);
    }

    // ===== Test 12: argmax =====
    {
        const unsigned N = 4096;
        float* logits = malloc((size_t)N * 4);
        for (unsigned i = 0; i < N; i++) logits[i] = frand() * 10.0f;
        unsigned ref = 0; float best = -1e30f;
        for (unsigned i = 0; i < N; i++) if (logits[i] > best) { best = logits[i]; ref = i; }
        CudaBuf* dl = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* dt = cuda_create_buffer(c, 4);
        cuda_upload(c, dl, logits, (size_t)N * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "argmax", NULL, 0);
        if (!p) { printf("FAIL argmax compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N; } push = { N };
        uint32_t grid[3] = { 1, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[2] = { dl, dt };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 2, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        unsigned gt; cuda_download(c, dt, &gt, 4);
        int ok = (gt == ref); all_ok &= ok;
        printf("argmax [N=%u]: ref=%u gpu=%u -> %s\n", N, ref, gt, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dl); cuda_free_buffer(dt); cuda_free_pipeline(p); free(logits);
    }

    // ===== Test 13: moe_weighted_acc =====
    {
        const unsigned N = 2048, n_used = 8;
        float* a0 = malloc((size_t)N * 4); float* b = malloc((size_t)n_used * N * 4);
        float* aref = malloc((size_t)N * 4); float* agpu = malloc((size_t)N * 4);
        unsigned routing[16]; float wts[8]; float wsum = 0;
        for (unsigned i = 0; i < N; i++) a0[i] = frand();
        for (unsigned j = 0; j < n_used * N; j++) b[j] = frand();
        for (unsigned j = 0; j < n_used; j++) { wts[j] = fabsf(frand()) + 0.1f; wsum += wts[j]; }
        for (unsigned j = 0; j < n_used; j++) { wts[j] /= wsum; routing[j] = j; memcpy(&routing[n_used + j], &wts[j], 4); }
        for (unsigned i = 0; i < N; i++) { float s = 0; for (unsigned j = 0; j < n_used; j++) s += wts[j] * b[(size_t)j * N + i]; aref[i] = a0[i] + s; }
        CudaBuf* da = cuda_create_buffer(c, (size_t)N * 4); CudaBuf* db = cuda_create_buffer(c, (size_t)n_used * N * 4); CudaBuf* dr = cuda_create_buffer(c, 16 * 4);
        cuda_upload(c, da, a0, (size_t)N * 4); cuda_upload(c, db, b, (size_t)n_used * N * 4); cuda_upload(c, dr, routing, 16 * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "moe_weighted_acc", NULL, 0);
        if (!p) { printf("FAIL moe_weighted_acc compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned N, nu, ss; } push = { N, n_used, N };
        uint32_t grid[3] = { (N + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[3] = { da, db, dr };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 3, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, da, agpu, (size_t)N * 4);
        float mr = 0; for (unsigned i = 0; i < N; i++) { float r = fabsf(aref[i] - agpu[i]) / (fabsf(aref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("moe_weighted_acc [N=%u n_used=%u]: max_rel_err=%.2e -> %s\n", N, n_used, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(da); cuda_free_buffer(db); cuda_free_buffer(dr); cuda_free_pipeline(p);
        free(a0); free(b); free(aref); free(agpu);
    }

    // ===== Test 14: ssm_conv1d (validates out + in-place state update) =====
    {
        const unsigned cc = 512, d_conv = 4, state_offset = 1; unsigned d_conv_1 = d_conv - 1;
        float* ci = malloc((size_t)cc * 4); float* ker = malloc((size_t)cc * d_conv * 4); float* st0 = malloc((size_t)d_conv_1 * cc * 4);
        float* out_ref = malloc((size_t)cc * 4); float* st_ref = malloc((size_t)d_conv_1 * cc * 4);
        float* out_gpu = malloc((size_t)cc * 4); float* st_gpu = malloc((size_t)d_conv_1 * cc * 4);
        for (unsigned i = 0; i < cc; i++) ci[i] = frand();
        for (unsigned i = 0; i < cc * d_conv; i++) ker[i] = frand();
        for (unsigned i = 0; i < d_conv_1 * cc; i++) st0[i] = frand();
        memcpy(st_ref, st0, (size_t)d_conv_1 * cc * 4);
        for (unsigned ch = 0; ch < cc; ch++) {
            float sum = 0;
            for (unsigned ki = 0; ki < d_conv; ki++) {
                float kw = ker[ch * d_conv + ki]; float sv;
                if (ki < d_conv_1) { unsigned slot = state_offset + ki; if (slot >= d_conv_1) slot -= d_conv_1; sv = st0[slot * cc + ch]; }
                else sv = ci[ch];
                sum += kw * sv;
            }
            out_ref[ch] = sum / (1.0f + expf(-sum));
            st_ref[state_offset * cc + ch] = ci[ch];
        }
        CudaBuf* dci = cuda_create_buffer(c, (size_t)cc * 4); CudaBuf* dk = cuda_create_buffer(c, (size_t)cc * d_conv * 4);
        CudaBuf* dst = cuda_create_buffer(c, (size_t)d_conv_1 * cc * 4); CudaBuf* dout = cuda_create_buffer(c, (size_t)cc * 4);
        cuda_upload(c, dci, ci, (size_t)cc * 4); cuda_upload(c, dk, ker, (size_t)cc * d_conv * 4); cuda_upload(c, dst, st0, (size_t)d_conv_1 * cc * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "ssm_conv1d", NULL, 0);
        if (!p) { printf("FAIL ssm_conv1d compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned cc, dc, f16, so; } push = { cc, d_conv, 0, state_offset };
        uint32_t grid[3] = { (cc + 255) / 256, 1, 1 }, block[3] = { 256, 1, 1 }; CudaBuf* bufs[4] = { dci, dk, dst, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 4, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dout, out_gpu, (size_t)cc * 4); cuda_download(c, dst, st_gpu, (size_t)d_conv_1 * cc * 4);
        float mo = 0; for (unsigned i = 0; i < cc; i++) { float r = fabsf(out_ref[i] - out_gpu[i]) / (fabsf(out_ref[i]) + 1e-4f); if (r > mo) mo = r; }
        float msv = 0; for (unsigned i = 0; i < d_conv_1 * cc; i++) { float r = fabsf(st_ref[i] - st_gpu[i]) / (fabsf(st_ref[i]) + 1e-4f); if (r > msv) msv = r; }
        int ok = (mo < 1e-3f && msv < 1e-3f); all_ok &= ok;
        printf("ssm_conv1d [cc=%u d_conv=%u]: out_err=%.2e state_err=%.2e -> %s\n", cc, d_conv, mo, msv, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dci); cuda_free_buffer(dk); cuda_free_buffer(dst); cuda_free_buffer(dout); cuda_free_pipeline(p);
        free(ci); free(ker); free(st0); free(out_ref); free(st_ref); free(out_gpu); free(st_gpu);
    }

    // ===== Test 15: ssm_gated_norm =====
    {
        const unsigned dt_rank = 4, head_v_dim = 128, d_state = 128, norm_per_head = 1; unsigned d_inner = dt_rank * head_v_dim;
        float* o = malloc((size_t)d_inner * 4); float* z = malloc((size_t)d_inner * 4); float* nw = malloc((size_t)d_inner * 4);
        float* ref = malloc((size_t)d_inner * 4); float* gpu = malloc((size_t)d_inner * 4);
        for (unsigned i = 0; i < d_inner; i++) { o[i] = frand(); z[i] = frand() * 2.0f; nw[i] = frand() * 0.5f + 1.0f; }
        for (unsigned h = 0; h < dt_rank; h++) {
            unsigned base = h * head_v_dim;
            double ss = 0; for (unsigned i = 0; i < head_v_dim; i++) { float v = o[base + i]; ss += (double)v * v; }
            float rinv = 1.0f / sqrtf((float)(ss / head_v_dim) + 1e-6f);
            for (unsigned i = 0; i < head_v_dim; i++) {
                float nv = o[base + i] * rinv; unsigned ni = norm_per_head ? base + i : i % d_state; nv *= nw[ni];
                float zv = z[base + i]; ref[base + i] = nv * (zv / (1.0f + expf(-zv)));
            }
        }
        CudaBuf* doo = cuda_create_buffer(c, (size_t)d_inner * 4); CudaBuf* dz = cuda_create_buffer(c, (size_t)d_inner * 4);
        CudaBuf* dnw = cuda_create_buffer(c, (size_t)d_inner * 4); CudaBuf* dout = cuda_create_buffer(c, (size_t)d_inner * 4);
        cuda_upload(c, doo, o, (size_t)d_inner * 4); cuda_upload(c, dz, z, (size_t)d_inner * 4); cuda_upload(c, dnw, nw, (size_t)d_inner * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "ssm_gated_norm", NULL, 0);
        if (!p) { printf("FAIL ssm_gated_norm compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned di, dt, hv, ds, nph; } push = { d_inner, dt_rank, head_v_dim, d_state, norm_per_head };
        uint32_t grid[3] = { dt_rank, 1, 1 }, block[3] = { 128, 1, 1 }; CudaBuf* bufs[4] = { doo, dz, dnw, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 4, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dout, gpu, (size_t)d_inner * 4);
        float mr = 0; for (unsigned i = 0; i < d_inner; i++) { float r = fabsf(ref[i] - gpu[i]) / (fabsf(ref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("ssm_gated_norm [heads=%u hv=%u]: max_rel_err=%.2e -> %s\n", dt_rank, head_v_dim, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(doo); cuda_free_buffer(dz); cuda_free_buffer(dnw); cuda_free_buffer(dout); cuda_free_pipeline(p);
        free(o); free(z); free(nw); free(ref); free(gpu);
    }

    // ===== Test 16: kv_cache_write =====
    {
        const unsigned kv_dim = 1024, dst_offset = 2048, cache_sz = 8192;
        float* ks = malloc((size_t)kv_dim * 4); float* vs = malloc((size_t)kv_dim * 4);
        float* kd = malloc((size_t)cache_sz * 4); float* vd = malloc((size_t)cache_sz * 4);
        float* kref = malloc((size_t)cache_sz * 4); float* vref = malloc((size_t)cache_sz * 4);
        float* kg = malloc((size_t)cache_sz * 4); float* vg = malloc((size_t)cache_sz * 4);
        for (unsigned i = 0; i < kv_dim; i++) { ks[i] = frand(); vs[i] = frand(); }
        for (unsigned i = 0; i < cache_sz; i++) { kd[i] = frand(); vd[i] = frand(); }
        memcpy(kref, kd, (size_t)cache_sz * 4); memcpy(vref, vd, (size_t)cache_sz * 4);
        for (unsigned i = 0; i < kv_dim; i++) { kref[dst_offset + i] = ks[i]; vref[dst_offset + i] = vs[i]; }
        CudaBuf* dks = cuda_create_buffer(c, (size_t)kv_dim * 4); CudaBuf* dvs = cuda_create_buffer(c, (size_t)kv_dim * 4);
        CudaBuf* dkd = cuda_create_buffer(c, (size_t)cache_sz * 4); CudaBuf* dvd = cuda_create_buffer(c, (size_t)cache_sz * 4);
        cuda_upload(c, dks, ks, (size_t)kv_dim * 4); cuda_upload(c, dvs, vs, (size_t)kv_dim * 4);
        cuda_upload(c, dkd, kd, (size_t)cache_sz * 4); cuda_upload(c, dvd, vd, (size_t)cache_sz * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "kv_cache_write", NULL, 0);
        if (!p) { printf("FAIL kv_cache_write compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned kv, off; } push = { kv_dim, dst_offset };
        uint32_t grid[3] = { (kv_dim + 63) / 64, 1, 1 }, block[3] = { 64, 1, 1 };
        CudaBuf* bufs[4] = { dks, dkd, dvs, dvd };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 4, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        cuda_download(c, dkd, kg, (size_t)cache_sz * 4); cuda_download(c, dvd, vg, (size_t)cache_sz * 4);
        float me = 0; for (unsigned i = 0; i < cache_sz; i++) { me = fmaxf(me, fabsf(kref[i] - kg[i])); me = fmaxf(me, fabsf(vref[i] - vg[i])); }
        int ok = (me == 0.0f); all_ok &= ok;
        printf("kv_cache_write [kv_dim=%u off=%u]: max_abs_err=%.2e -> %s\n", kv_dim, dst_offset, me, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dks); cuda_free_buffer(dvs); cuda_free_buffer(dkd); cuda_free_buffer(dvd); cuda_free_pipeline(p);
        free(ks); free(vs); free(kd); free(vd); free(kref); free(vref); free(kg); free(vg);
    }

    // ===== Test 17: naive_attention (GQA + mixed sinks) =====
    {
        const unsigned head_dim = 64, n_heads = 4, n_kv_heads = 2, seq_len = 48;
        float scale = 0.125f; unsigned sbits; memcpy(&sbits, &scale, 4);
        float* q = malloc((size_t)n_heads * head_dim * 4);
        float* k = malloc((size_t)seq_len * n_kv_heads * head_dim * 4);
        float* v = malloc((size_t)seq_len * n_kv_heads * head_dim * 4);
        float* sinks = malloc((size_t)n_heads * 4);
        float* ref = malloc((size_t)n_heads * head_dim * 4); float* gpu = malloc((size_t)n_heads * head_dim * 4);
        for (unsigned i = 0; i < n_heads * head_dim; i++) q[i] = frand();
        for (unsigned i = 0; i < seq_len * n_kv_heads * head_dim; i++) { k[i] = frand(); v[i] = frand(); }
        for (unsigned h = 0; h < n_heads; h++) sinks[h] = (h % 2 == 0) ? NAN : (frand() * 2.0f);
        for (unsigned h = 0; h < n_heads; h++) {
            unsigned kvh = h / (n_heads / n_kv_heads); float* qh = q + h * head_dim;
            float* sc = malloc((size_t)seq_len * 4); float mx = -1e30f;
            for (unsigned i = 0; i < seq_len; i++) {
                float* ki = k + ((size_t)i * n_kv_heads + kvh) * head_dim; float dot = 0;
                for (unsigned d = 0; d < head_dim; d++) dot += qh[d] * ki[d];
                sc[i] = dot * scale; if (sc[i] > mx) mx = sc[i];
            }
            float sum = 0; for (unsigned i = 0; i < seq_len; i++) { sc[i] = expf(sc[i] - mx); sum += sc[i]; }
            float rescale = 1.0f, fsum = sum, sv = sinks[h];
            if (!isnan(sv)) { float smax = fmaxf(mx, sv); rescale = (sum > 0) ? expf(mx - smax) : 0; fsum = sum * rescale + expf(sv - smax); }
            float inv = (fsum > 0) ? 1.0f / fsum : 0;
            for (unsigned d = 0; d < head_dim; d++) {
                float acc = 0; for (unsigned i = 0; i < seq_len; i++) { float* vi = v + ((size_t)i * n_kv_heads + kvh) * head_dim; acc += sc[i] * vi[d]; }
                ref[h * head_dim + d] = acc * rescale * inv;
            }
            free(sc);
        }
        CudaBuf* dq = cuda_create_buffer(c, (size_t)n_heads * head_dim * 4);
        CudaBuf* dk = cuda_create_buffer(c, (size_t)seq_len * n_kv_heads * head_dim * 4);
        CudaBuf* dv = cuda_create_buffer(c, (size_t)seq_len * n_kv_heads * head_dim * 4);
        CudaBuf* dsink = cuda_create_buffer(c, (size_t)n_heads * 4);
        CudaBuf* dout = cuda_create_buffer(c, (size_t)n_heads * head_dim * 4);
        cuda_upload(c, dq, q, (size_t)n_heads * head_dim * 4);
        cuda_upload(c, dk, k, (size_t)seq_len * n_kv_heads * head_dim * 4);
        cuda_upload(c, dv, v, (size_t)seq_len * n_kv_heads * head_dim * 4);
        cuda_upload(c, dsink, sinks, (size_t)n_heads * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "naive_attention", NULL, 0);
        if (!p) { printf("FAIL naive_attention compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned hd, nh, nkv, sl, sbits, soff; } push = { head_dim, n_heads, n_kv_heads, seq_len, sbits, 0 };
        uint32_t grid[3] = { n_heads, 1, 1 }, block[3] = { 128, 1, 1 };
        CudaBuf* bufs[5] = { dq, dk, dv, dsink, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 5, &push, sizeof push, (unsigned)(seq_len * 4)); cuda_commit_and_wait(cmd);
        cuda_download(c, dout, gpu, (size_t)n_heads * head_dim * 4);
        float mr = 0; for (unsigned i = 0; i < n_heads * head_dim; i++) { float r = fabsf(ref[i] - gpu[i]) / (fabsf(ref[i]) + 1e-4f); if (r > mr) mr = r; }
        int ok = mr < 1e-3f; all_ok &= ok;
        printf("naive_attention [H=%u KVH=%u hd=%u seq=%u, mixed sinks]: max_rel_err=%.2e -> %s\n", n_heads, n_kv_heads, head_dim, seq_len, mr, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dq); cuda_free_buffer(dk); cuda_free_buffer(dv); cuda_free_buffer(dsink); cuda_free_buffer(dout); cuda_free_pipeline(p);
        free(q); free(k); free(v); free(sinks); free(ref); free(gpu);
    }

    // ===== Test 18: ssm_delta_net (autoregressive selective scan, multi-token) =====
    {
        const unsigned dt_rank = 4, hv = 64, d_state = 64, n_group = 2, n_tok = 3;
        unsigned qk_dim = d_state * n_group, d_inner = dt_rank * hv, k_len = (hv < d_state) ? hv : d_state;
        unsigned conv_stride = 2 * qk_dim + d_inner, ab_stride = dt_rank, y_stride = d_inner;
        size_t conv_n = (size_t)n_tok * conv_stride, ab_n = (size_t)n_tok * dt_rank;
        size_t state_n = (size_t)dt_rank * hv * hv, out_n = (size_t)n_tok * d_inner;
        float* conv = malloc(conv_n * 4); float* dtb = malloc((size_t)dt_rank * 4);
        float* al = malloc(ab_n * 4); float* be = malloc(ab_n * 4); float* sa = malloc((size_t)dt_rank * 4);
        float* st0 = malloc(state_n * 4);
        for (size_t i = 0; i < conv_n; i++) conv[i] = frand();
        for (unsigned i = 0; i < dt_rank; i++) { dtb[i] = frand() * 0.1f; sa[i] = -(0.5f + 0.5f * fabsf(frand())); } // ssm_a < 0 => decay
        for (size_t i = 0; i < ab_n; i++) { al[i] = frand(); be[i] = frand(); }
        for (size_t i = 0; i < state_n; i++) st0[i] = frand() * 0.1f;
        // CPU reference scan
        float* st = malloc(state_n * 4); memcpy(st, st0, state_n * 4);
        float* out_ref = malloc(out_n * 4);
        float* sqv = malloc((size_t)k_len * 4); float* skv = malloc((size_t)k_len * 4);
        for (unsigned t = 0; t < n_tok; t++) {
            for (unsigned h = 0; h < dt_rank; h++) {
                unsigned k_hi = (n_group == dt_rank) ? h : (h % n_group);
                unsigned q_off = t * conv_stride + k_hi * d_state;
                unsigned k_off = t * conv_stride + qk_dim + k_hi * d_state;
                unsigned v_off = t * conv_stride + 2 * qk_dim + h * hv;
                double sumq = 0, sumk = 0;
                for (unsigned i = 0; i < k_len; i++) { float q = conv[q_off + i], k = conv[k_off + i]; sumq += (double)q * q; sumk += (double)k * k; }
                float invq = 1.0f / sqrtf(fmaxf((float)sumq, 1e-12f)), invk = 1.0f / sqrtf(fmaxf((float)sumk, 1e-12f));
                float q_scale = invq / sqrtf((float)d_state);
                for (unsigned i = 0; i < k_len; i++) { sqv[i] = conv[q_off + i] * q_scale; skv[i] = conv[k_off + i] * invk; }
                float a = al[t * dt_rank + h] + dtb[h]; float sp = logf(1.0f + expf(a));
                float g = expf(sp * sa[h]); float bb = 1.0f / (1.0f + expf(-be[t * dt_rank + h]));
                for (unsigned row = 0; row < hv; row++) {
                    float* strow = st + ((size_t)h * hv + row) * hv;
                    float v = conv[v_off + row];
                    for (unsigned col = 0; col < hv; col++) strow[col] *= g;
                    float sk = 0; for (unsigned col = 0; col < k_len; col++) sk += strow[col] * skv[col];
                    float dd = bb * (v - sk);
                    float o = 0; for (unsigned col = 0; col < k_len; col++) { strow[col] += skv[col] * dd; o += strow[col] * sqv[col]; }
                    out_ref[t * y_stride + h * hv + row] = o;
                }
            }
        }
        // GPU
        CudaBuf* dconv = cuda_create_buffer(c, conv_n * 4); CudaBuf* ddtb = cuda_create_buffer(c, (size_t)dt_rank * 4);
        CudaBuf* dal = cuda_create_buffer(c, ab_n * 4); CudaBuf* dbe = cuda_create_buffer(c, ab_n * 4); CudaBuf* dsa = cuda_create_buffer(c, (size_t)dt_rank * 4);
        CudaBuf* dst = cuda_create_buffer(c, state_n * 4); CudaBuf* dout = cuda_create_buffer(c, out_n * 4);
        cuda_upload(c, dconv, conv, conv_n * 4); cuda_upload(c, ddtb, dtb, (size_t)dt_rank * 4); cuda_upload(c, dal, al, ab_n * 4);
        cuda_upload(c, dbe, be, ab_n * 4); cuda_upload(c, dsa, sa, (size_t)dt_rank * 4); cuda_upload(c, dst, st0, state_n * 4);
        CudaPipe* p = cuda_create_pipeline(c, src, "ssm_delta_net", NULL, 0);
        if (!p) { printf("FAIL ssm_delta_net compile: %s\n", cuda_last_error()); return 2; }
        struct { unsigned d_inner, dt_rank, hv, d_state, n_group, saf16, dbf16, has_dtb, has_sa, n_tok, conv_st, ab_st, y_st; } push =
            { d_inner, dt_rank, hv, d_state, n_group, 0, 0, 1, 1, n_tok, conv_stride, ab_stride, y_stride };
        uint32_t grid[3] = { dt_rank, hv, 1 }, block[3] = { hv, 1, 1 };
        CudaBuf* bufs[7] = { dconv, ddtb, dal, dbe, dsa, dst, dout };
        CudaCmd* cmd = cuda_begin_command(c); cuda_dispatch(cmd, p, grid, block, bufs, 7, &push, sizeof push, 0); cuda_commit_and_wait(cmd);
        float* out_gpu = malloc(out_n * 4); float* st_gpu = malloc(state_n * 4);
        cuda_download(c, dout, out_gpu, out_n * 4); cuda_download(c, dst, st_gpu, state_n * 4);
        float mo = 0; for (size_t i = 0; i < out_n; i++) { float r = fabsf(out_ref[i] - out_gpu[i]) / (fabsf(out_ref[i]) + 1e-4f); if (r > mo) mo = r; }
        float ms = 0; for (size_t i = 0; i < state_n; i++) { float r = fabsf(st[i] - st_gpu[i]) / (fabsf(st[i]) + 1e-4f); if (r > ms) ms = r; }
        int ok = (mo < 2e-3f && ms < 2e-3f); all_ok &= ok;
        printf("ssm_delta_net [heads=%u hv=%u dstate=%u ngrp=%u ntok=%u]: out_err=%.2e state_err=%.2e -> %s\n", dt_rank, hv, d_state, n_group, n_tok, mo, ms, ok ? "PASS" : "FAIL");
        cuda_free_buffer(dconv); cuda_free_buffer(ddtb); cuda_free_buffer(dal); cuda_free_buffer(dbe); cuda_free_buffer(dsa); cuda_free_buffer(dst); cuda_free_buffer(dout); cuda_free_pipeline(p);
        free(conv); free(dtb); free(al); free(be); free(sa); free(st0); free(st); free(out_ref); free(sqv); free(skv); free(out_gpu); free(st_gpu);
    }

    printf("RESULT: %s\n", all_ok ? "ALL PASS" : "FAIL");
    cuda_destroy(c);
    free(src);
    return all_ok ? 0 : 1;
}
