/* GPU matmul test using hipBLAS (Radeon 8060S, 55 TFLOPS)
 * Build: hipcc -O3 -o gpu_test gpu_test.cu -lhipblas
 * The APU has unified memory — no PCIe transfer overhead.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#define H 1024
#define NH 16
#define NKV 8
#define HD 128
#define IM 3072
#define NV 151936
#define QT (NH*HD + 2*NKV*HD)

#define I8_ROW_B 5120
#define TILE_R 32
#define TILE_C 256

static float bf16_f32(uint16_t v) { uint32_t b = (uint32_t)v << 16; float f; memcpy(&f, &b, 4); return f; }

/* Dequantize a weight matrix from I8 to FP16 on host */
static float* dequant_to_f32(const uint8_t* i8_data, int i8_rows, int in_dim) {
    int ntc = in_dim / TILE_C;
    int ntr = i8_rows / ntc;
    int out_rows = ntr * TILE_R;
    int out_cols = ntc * TILE_C;
    float* out = (float*)calloc((size_t)out_rows * out_cols, 4);
    
    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* rd = i8_data + ir * I8_ROW_B;
        int tr = ir / ntc, tc = ir % ntc;
        const uint16_t* sc = (const uint16_t*)rd;
        const uint16_t* zp = (const uint16_t*)(rd + 512);
        const uint8_t* pk = rd + 1024;
        for (int lr = 0; lr < TILE_R; lr++) {
            int lane = lr / 16, lr2 = lr % 16, bi = lr2 / 2, ns = lr % 2;
            const uint8_t* ld = pk + lane * (TILE_C * 8);
            for (int c = 0; c < TILE_C; c++) {
                int g = c / 32;
                float s = bf16_f32(sc[g * 32 + lr]), z = bf16_f32(zp[g * 32 + lr]);
                uint8_t bv = ld[c * 8 + bi];
                int cd = (ns == 0) ? (bv & 0x0F) : ((bv >> 4) & 0x0F);
                out[(tr * TILE_R + lr) * out_cols + (tc * TILE_C + c)] = (float)cd * s + z;
            }
        }
    }
    return out;
}

int main() {
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    // Allocate test data (unified memory — zero copy)
    float *d_in, *d_out;
    hipMalloc(&d_in, H * 4);
    hipMalloc(&d_out, QT * 4);
    
    // Random input
    float h_in[H];
    for (int i = 0; i < H; i++) h_in[i] = (float)(rand() % 1000) / 1000.0f;
    hipMemcpy(d_in, h_in, H * 4, hipMemcpyHostToDevice);
    
    // Load actual weights
    int fd = open("/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx", O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* map = (uint8_t*)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    uint64_t hdr; memcpy(&hdr, map, 8); size_t ds = 8 + (size_t)hdr;
    
    // Find q_proj for layer 0
    const char* js = (const char*)(map + 8);
    auto find = [&](const char* k) -> uint64_t {
        const char* p = strstr(js, k); if (!p) return 0;
        const char* d = strstr(p, "\"data_offsets\""); if (!d) return 0;
        const char* b = strchr(d, '['); return b ? strtoull(b + 1, NULL, 10) : 0;
    };
    
    uint64_t qoff = find("model.layers.0.self_attn.q_proj.weight");
    uint8_t* wt = map + ds + qoff;
    
    // Dequantize to FP32
    float* q_f32 = dequant_to_f32(wt, 256, H);  // Q: out=2048, in=1024
    
    // Copy to GPU (unified mem — can use device pointer directly)
    float *d_w, *d_out_q;
    hipMalloc(&d_w, (size_t)(NH*HD) * H * 4);
    hipMalloc(&d_out_q, (size_t)(NH*HD) * 4);
    hipMemcpy(d_w, q_f32, (size_t)(NH*HD) * H * 4, hipMemcpyHostToDevice);
    
    // Warmup
    float alpha = 1.0f, beta = 0.0f;
    hipblasSgemv(handle, HIPBLAS_OP_T, H, NH*HD, &alpha, d_w, H, d_in, 1, &beta, d_out_q, 1);
    hipDeviceSynchronize();
    
    // Benchmark
    int iters = 1000;
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    for (int i = 0; i < iters; i++) {
        hipblasSgemv(handle, HIPBLAS_OP_T, H, NH*HD, &alpha, d_w, H, d_in, 1, &beta, d_out_q, 1);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms; hipEventElapsedTime(&ms, start, stop);
    
    float* q_out = (float*)malloc((size_t)(NH*HD) * 4);
    hipMemcpy(q_out, d_out_q, (size_t)(NH*HD) * 4, hipMemcpyDeviceToHost);
    
    double sum = 0;
    for (int i = 0; i < NH*HD; i++) sum += q_out[i];
    printf("Q matmul: %.3f ms avg (%.0f tok/s theor)\n", ms / iters, 1000.0f / (ms / iters * 28 * 7));
    printf("Q[0]=%f sum=%.2f\n", q_out[0], sum);
    
    // All 7 matmuls per layer: QKV(3) + O(1) + GU(2) + D(1) = 7
    // 28 layers × 7 = 196 matmuls per token
    printf("One decode step (196 matmuls): %.3f ms (%.0f tok/s)\n",
           ms / iters * 196, 1000.0f / (ms / iters * 196));
    
    free(q_f32); free(q_out);
    hipFree(d_w); hipFree(d_out_q); hipFree(d_in); hipFree(d_out);
    hipblasDestroy(handle);
    munmap(map, st.st_size);
    return 0;
}
