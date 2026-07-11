// Benchmark: Fused Ternary GEMV vs Dequantize+GEMM
// Tests the Wave32 kernel on gfx1151
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>

// Kernel launches from ternary_gemv.hip
extern "C" {
void ternary_gemv(const uint32_t* packed, const float* x, const float* scales,
                  float* y, int M, int K, hipStream_t stream);
void ternary_gemv_batched(const uint32_t* packed, const float* x, const float* scales,
                          float* y, int M, int K, int B, hipStream_t stream);
}

#define HC(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

void pack_ternary(uint32_t* out, const int8_t* vals, int count) {
    // Pack ternary values: -1=0b10, 0=0b00, +1=0b01
    int u32s = count / 16;
    for (int i = 0; i < u32s; i++) {
        uint32_t word = 0;
        for (int v = 0; v < 16; v++) {
            int8_t val = vals[i * 16 + v];
            uint32_t bits;
            if (val == 1)       bits = 0x1;  // +1
            else if (val == -1) bits = 0x2;  // -1
            else                bits = 0x0;  // 0
            word |= (bits << (v * 2));
        }
        out[i] = word;
    }
}

void bench(int M, int K, int warmup, int runs) {
    int packed_k = K / 16;

    // Generate random ternary weights and activation
    std::vector<int8_t> h_weights(M * K);
    std::vector<float> h_x(K);
    std::vector<float> h_scales(M);

    srand(42);
    for (int i = 0; i < M * K; i++) {
        int r = rand() % 3;
        h_weights[i] = (r == 0) ? -1 : (r == 1) ? 0 : 1;
    }
    for (int i = 0; i < K; i++) h_x[i] = ((float)(rand() % 1000) - 500) / 500.0f;
    for (int i = 0; i < M; i++) h_scales[i] = 1.0f;

    // Pack weights
    std::vector<uint32_t> h_packed(M * packed_k);
    for (int r = 0; r < M; r++) {
        pack_ternary(h_packed.data() + r * packed_k, h_weights.data() + r * K, K);
    }

    // Allocate GPU
    uint32_t* d_packed;
    float *d_x, *d_scales, *d_y;
    HC(hipMalloc(&d_packed, M * packed_k * sizeof(uint32_t)));
    HC(hipMalloc(&d_x, K * sizeof(float)));
    HC(hipMalloc(&d_scales, M * sizeof(float)));
    HC(hipMalloc(&d_y, M * sizeof(float)));

    HC(hipMemcpy(d_packed, h_packed.data(), M * packed_k * sizeof(uint32_t), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_x, h_x.data(), K * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_scales, h_scales.data(), M * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    HC(hipStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ternary_gemv(d_packed, d_x, d_scales, d_y, M, K, stream);
    }
    HC(hipStreamSynchronize(stream));

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        ternary_gemv(d_packed, d_x, d_scales, d_y, M, K, stream);
    }
    HC(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_us = (ms / runs) * 1000.0;

    // Correctness check — compute reference on CPU
    std::vector<float> h_y(M);
    HC(hipMemcpy(h_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> ref(M, 0.0f);
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            ref[r] += (float)h_weights[r * K + c] * h_x[c];
        }
        ref[r] *= h_scales[r];
    }

    float max_err = 0.0f;
    for (int r = 0; r < M; r++) {
        float err = fabsf(h_y[r] - ref[r]);
        if (err > max_err) max_err = err;
    }

    // Estimate tok/s for full model
    // BitNet-2B: 30 layers, each has ~4 linear ops (q,k,v,o) + 2 FFN = 6 GEMV per layer
    // = 180 GEMV per token
    double gemvs_per_token = 180.0;
    double us_per_token = avg_us * gemvs_per_token;
    double tok_per_sec = 1e6 / us_per_token;

    printf("  %4d x %4d : %7.1f us  err=%.6f  est ~%.0f tok/s (180 GEMV/tok)\n",
           M, K, avg_us, max_err, tok_per_sec);

    HC(hipFree(d_packed));
    HC(hipFree(d_x));
    HC(hipFree(d_scales));
    HC(hipFree(d_y));
    HC(hipStreamDestroy(stream));
}

int main() {
    printf("=== Fused Ternary GEMV — Wave32 gfx1151 ===\n");

    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("GPU: %s (%s), CUs: %d\n\n", p.name, p.gcnArchName, p.multiProcessorCount);

    int w = 10, r = 100;
    printf("Fused ternary GEMV — %d warmup, %d runs\n\n", w, r);

    // BitNet-2B-4T sizes
    printf("BitNet-2B-4T shapes:\n");
    bench(2560, 2560, w, r);   // Q,K,V,O projections
    bench(6912, 2560, w, r);   // FFN up
    bench(2560, 6912, w, r);   // FFN down
    bench(128256, 2560, w, r); // LM head (vocab projection)

    printf("\nGeneric sizes:\n");
    bench(4096, 4096, w, r);   // 7B model
    bench(11008, 4096, w, r);  // 7B FFN

    printf("\n=== Done ===\n");
    return 0;
}
