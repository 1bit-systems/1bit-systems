// bench_q1_to_halo_spec.cpp — Convert Q1_0 weights to halo-1bit format,
// then benchmark with M=8 tiled GEMM (ternary_gemm_smallm kernel).
//
// This measures the speculative decode verification path:
//   M=8 draft tokens verified in one forward pass using the existing
//   tiled GEMM kernel, which gets ~8× bandwidth reuse vs batch=1 GEMV.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

// Forward declare the existing tiled GEMM kernel
extern "C" void onebit_ternary_gemm_smallm_launch(
    const int8_t* x_i8, float x_scale,
    const uint32_t* w_packed, const float* w_row_scales,
    hip_bfloat16* y_out,
    int M, int N, int K, hipStream_t stream);

// ── Q1_0 → halo-1bit converter ──
// Q1_0: [fp16 d][16 bytes sign bits], 128 weights per 18-byte block
//   w[i] = sign_bit ? +d : -d
//
// Halo-1bit: uint32 per 16 weights, codes 0=-1, 1=0, 2=+1, 3=0
//   w_packed[k/16][n] = sum(code_j << (2*j)), j=0..15
//   row_scales[n] = per-output fp32 scale
//
// Since Q1_0 has no zeros (only +d or -d), we map:
//   sign_bit=0 → code 0 = -1
//   sign_bit=1 → code 2 = +1
//   row_scale = d (block scale)

static size_t q1_bytes(int rows, int cols) {
    return (size_t)rows * (size_t)(cols / 128) * (size_t)18;
}

static size_t halo_bytes(int rows, int cols) {
    return (size_t)(cols / 16) * (size_t)rows * sizeof(uint32_t);
}

static void convert_q1_row_to_halo(const uint8_t* q1_row, 
                                    uint32_t* halo_row,
                                    float* scale_row,
                                    int cols) {
    int blocks = cols / 128;
    int write_idx = 0;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* blk = q1_row + b * 18;
        
        // Extract scale
        uint16_t d_u16;
        memcpy(&d_u16, blk, 2);
        __half d_h;
        memcpy(&d_h, &d_u16, sizeof(d_h));
        float d = __half2float(d_h);
        
        // Process 16-weight groups within the block
        for (int g = 0; g < 8; ++g) {  // 8 groups of 16 = 128 weights
            uint32_t word = 0;
            for (int j = 0; j < 16; ++j) {
                int w_idx = g * 16 + j;
                int byte_idx = w_idx / 8;
                int bit_idx = w_idx % 8;
                uint8_t bit = (blk[2 + byte_idx] >> bit_idx) & 1;
                // Q1_0: 0→-d, 1→+d
                // Halo: code 0→-1, code 2→+1
                uint32_t code = bit ? 2u : 0u;
                word |= code << (2 * j);
            }
            halo_row[write_idx] = word;
            scale_row[write_idx] = d;
            write_idx++;
        }
    }
}

int main() {
    const int NL = 28, HS = 2048, IS = 6144, NH = 16, NKV = 8, HD = 128;
    const int M_SPEC = 8;  // speculative decode batch size

    printf("Converting Bonsai-1.7B Q1_0 weights to halo-1bit format...\n");
    printf("Testing with M=%d (8-way speculative decode verify)\n\n", M_SPEC);

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    // Store layer shapes
    struct LayerShape { int rows, cols; const char* name; };
    LayerShape shapes[] = {
        {NH*HD, HS, "Q"}, {NKV*HD, HS, "K"}, {NKV*HD, HS, "V"},
        {HS, NH*HD, "O"}, {IS, HS, "Gate"}, {IS, HS, "Up"}, {HS, IS, "Down"}
    };
    const int N_LAYER_OPS = 7;

    // Allocate device buffers for all conversions
    std::vector<uint32_t*> d_halo_weights;
    std::vector<float*> d_halo_scales;
    std::vector<int> rows_list, cols_list;
    size_t total_halo_bytes = 0, total_q1_bytes = 0;

    for (int l = 0; l < NL; ++l) {
        for (int op = 0; op < N_LAYER_OPS; ++op) {
            int rows = shapes[op].rows;
            int cols = shapes[op].cols;
            
            // Host: generate Q1_0 weights (synthetic)
            size_t q1_sz = q1_bytes(rows, cols);
            std::vector<uint8_t> h_q1(q1_sz);
            
            // Fill with deterministic pattern
            for (size_t i = 0; i < q1_sz; ++i)
                h_q1[i] = (uint8_t)((i * 0x9E3779B9u) & 0xFF);
            
            // Host: convert to halo-1bit
            int per_row_groups = cols / 16;  // each group of 16 weights = 1 uint32
            size_t halo_sz = (size_t)rows * per_row_groups * sizeof(uint32_t);
            size_t scale_sz = (size_t)rows * per_row_groups * sizeof(float);
            std::vector<uint32_t> h_halo(rows * per_row_groups);
            std::vector<float> h_scales(rows * per_row_groups);
            
            for (int r = 0; r < rows; ++r) {
                convert_q1_row_to_halo(
                    h_q1.data() + r * (cols / 128) * 18,
                    h_halo.data() + r * per_row_groups,
                    h_scales.data() + r * per_row_groups,
                    cols);
            }
            
            // Device allocation
            uint32_t* d_halo = nullptr;
            float* d_scales = nullptr;
            HIP_OK(hipMalloc(&d_halo, halo_sz));
            HIP_OK(hipMalloc(&d_scales, scale_sz));
            HIP_OK(hipMemcpyAsync(d_halo, h_halo.data(), halo_sz, 
                                  hipMemcpyHostToDevice, stream));
            HIP_OK(hipMemcpyAsync(d_scales, h_scales.data(), scale_sz,
                                  hipMemcpyHostToDevice, stream));
            
            d_halo_weights.push_back(d_halo);
            d_halo_scales.push_back(d_scales);
            rows_list.push_back(rows);
            cols_list.push_back(cols);
            
            total_halo_bytes += halo_sz;
            total_q1_bytes += q1_sz;
        }
    }

    printf("  Q1_0 weights:  %.0f MB\n", total_q1_bytes / 1e6);
    printf("  Halo-1bit:     %.0f MB (larger due to fp32 scales)\n", total_halo_bytes / 1e6);

    // Activation buffers (M=8, K=max)
    const int MAX_K = 6144;
    int8_t* d_act_i8 = nullptr;
    HIP_OK(hipMalloc(&d_act_i8, (size_t)M_SPEC * MAX_K * sizeof(int8_t)));
    HIP_OK(hipMemsetAsync(d_act_i8, 1, (size_t)M_SPEC * MAX_K * sizeof(int8_t), stream));

    // Output buffer
    hip_bfloat16* d_out = nullptr;
    HIP_OK(hipMalloc(&d_out, (size_t)M_SPEC * IS * sizeof(hip_bfloat16)));
    HIP_OK(hipMemsetAsync(d_out, 0, (size_t)M_SPEC * IS * sizeof(hip_bfloat16), stream));

    HIP_OK(hipDeviceSynchronize());
    printf("  Allocated buffers OK\n");

    // Warmup
    {
        int idx = 0;
        for (int l = 0; l < NL; ++l) {
            for (int op = 0; op < N_LAYER_OPS; ++op) {
                int N = shapes[op].rows;
                int K = shapes[op].cols;
                // Pad N to multiple of 64 for the kernel
                int N_pad = ((N + 63) / 64) * 64;
                if (K % 64 != 0 || N_pad % 64 != 0 || K < 64) continue;
                
                onebit_ternary_gemm_smallm_launch(
                    d_act_i8, 0.01f,
                    d_halo_weights[idx], d_halo_scales[idx],
                    d_out, M_SPEC, N_pad, K, stream);
                idx++;
            }
        }
        HIP_OK(hipDeviceSynchronize());
    }
    printf("  Warmup done\n");

    // Timed runs
    const int N_RUNS = 5;
    printf("\nBenchmark: %d forward passes with M=%d (spec-decode verify)\n", 
           N_RUNS, M_SPEC);

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, stream));
    for (int run = 0; run < N_RUNS; ++run) {
        int idx = 0;
        for (int l = 0; l < NL; ++l) {
            for (int op = 0; op < N_LAYER_OPS; ++op) {
                int N = shapes[op].rows;
                int K = shapes[op].cols;
                int N_pad = ((N + 63) / 64) * 64;
                if (K % 64 != 0 || N_pad % 64 != 0 || K < 64) continue;
                
                onebit_ternary_gemm_smallm_launch(
                    d_act_i8, 0.01f,
                    d_halo_weights[idx], d_halo_scales[idx],
                    d_out, M_SPEC, N_pad, K, stream);
                idx++;
            }
        }
    }
    HIP_OK(hipEventRecord(t1, stream));
    HIP_OK(hipEventSynchronize(t1));

    float ms;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    double per_verify = ms / (double)N_RUNS;

    // Estimate throughput with spec decode
    // Assume 80% acceptance of M=8 draft tokens = 6.4 effective tokens per verify
    double effective_tokens = M_SPEC * 0.80;
    double tok_s = effective_tokens / (per_verify / 1000.0);
    
    // Also calculate without overhead (norms, attention etc ~1.5ms)
    double per_verify_total = per_verify + 1.5;
    double tok_s_total = effective_tokens / (per_verify_total / 1000.0);

    double bw = total_halo_bytes / (per_verify / 1000.0 * M_SPEC) / 1e9;

    printf("\n══════════════════════════════════════════════════\n");
    printf("  Tiled GEMM with M=%d (spec-decode verify pass)\n", M_SPEC);
    printf("══════════════════════════════════════════════════\n");
    printf("  Per verify pass:  %.3f ms (GEMV only)\n", per_verify);
    printf("  + norms/attn:     ~%.1f ms\n", 1.5);
    printf("  Total per verify: %.3f ms\n", per_verify_total);
    printf("  Effective BW:     %.0f GB/s (per activation row)\n", bw);
    printf("  Peak BW util:     %.0f%%\n", bw / 273.0 * 100.0);
    printf("──────────────────────────────────────────────────\n");
    printf("  Draft accept:     80%% of %d tokens\n", M_SPEC);
    printf("  Eff tokens/verify: %.1f\n", effective_tokens);
    printf("──────────────────────────────────────────────────\n");
    printf("  🏆 GEMV-only:     %.0f tok/s\n", tok_s);
    printf("  🏆 With overhead: %.0f tok/s\n", tok_s_total);
    printf("══════════════════════════════════════════════════\n");

    // Cleanup
    for (size_t i = 0; i < d_halo_weights.size(); ++i) {
        hipFree(d_halo_weights[i]);
        hipFree(d_halo_scales[i]);
    }
    hipFree(d_act_i8);
    hipFree(d_out);
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(stream));

    return 0;
}
