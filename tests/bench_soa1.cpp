#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

constexpr int kBlockSize    = 128;
constexpr int kWaveSize     = 32;
constexpr int kWaves        = kBlockSize / kWaveSize;
constexpr int kWeightsPerB  = 128;
constexpr int kQ1BlockBytes = 18;

__global__
__attribute__((amdgpu_flat_work_group_size(kBlockSize, kBlockSize)))
void q1_gemv_soa1_kernel(
    const uint8_t* __restrict__ packed_soa,
    const __half*  __restrict__ act,
    __half*        __restrict__ out,
    int N_out, int K_in)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid >> 5;
    const int row  = blockIdx.x;
    if (row >= N_out) return;

    const int nb  = K_in >> 7;
    const size_t row_stride = (size_t)nb * (size_t)kQ1BlockBytes;

    const __half*  d_row = reinterpret_cast<const __half*>(packed_soa + (size_t)row * row_stride);
    const uint8_t* qs_row = packed_soa + (size_t)row * row_stride + (size_t)nb * 2;

    __shared__ __half x_lds[kWeightsPerB];
    __shared__ float  wave_sums[kWaves];
    float acc = 0.0f;

    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        x_lds[tid] = act[b * kWeightsPerB + tid];
        __syncthreads();
        const float d = __half2float(d_row[b]);
        const int byte_idx = tid >> 3, bit_idx = tid & 7;
        const uint8_t qbyte = qs_row[b * 16 + byte_idx];
        const float sign = ((qbyte >> bit_idx) & 1u) ? 1.0f : -1.0f;
        const float a = __half2float(x_lds[tid]);
        acc = fmaf(sign * d, a, acc);
    }

    #pragma unroll
    for (int off = kWaveSize / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
    if (lane == 0) wave_sums[wave] = acc;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < kWaves) ? wave_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = kWaveSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
        if (lane == 0) out[row] = __float2half(v);
    }
}

static void convert_aos_to_soa(const uint8_t* aos, uint8_t* soa, int rows, int cols) {
    int nb = cols / 128;
    size_t rs = (size_t)nb * kQ1BlockBytes;
    for (int r = 0; r < rows; ++r) {
        const uint8_t* src = aos + (size_t)r * rs;
        uint8_t* dst = soa + (size_t)r * rs;
        for (int b = 0; b < nb; ++b) {
            dst[b*2] = src[b*kQ1BlockBytes];
            dst[b*2+1] = src[b*kQ1BlockBytes+1];
        }
        size_t d_off = (size_t)nb * 2;
        for (int b = 0; b < nb; ++b)
            for (int j = 0; j < 16; ++j)
                dst[d_off + b*16 + j] = src[b*kQ1BlockBytes + 2 + j];
    }
}

static size_t wb(int r, int c) {
    return (size_t)r * (size_t)(c/128) * kQ1BlockBytes;
}

struct Shape { int r, c; const char* n; };

int main() {
    Shape shapes[] = {{2048,2048,"Q/O(2048x2048)"},{1024,2048,"K/V(1024x2048)"},
                      {6144,2048,"Gate/Up(6144x2048)"},{2048,6144,"Down(2048x6144)"}};

    printf("═══ 1-row/CTA SoA Q1_0 GEMV Benchmark ═══\n\n");
    
    // Full model estimate
    int shapes_model[] = {2048,2048, 1024,2048, 1024,2048, 2048,2048, 6144,2048, 6144,2048, 2048,6144};
    int num_ops = 7;
    double total_model_time_aos = 0, total_model_time_soa = 0;
    size_t total_w = 0;

    for (int s = 0; s < 4; ++s) {
        int N = shapes[s].r, K = shapes[s].c;
        size_t bytes = wb(N, K);
        std::vector<uint8_t> h_aos(bytes), h_soa(bytes);
        for (size_t i = 0; i < bytes; ++i) h_aos[i] = (uint8_t)(i*0x9E3779B9u);
        convert_aos_to_soa(h_aos.data(), h_soa.data(), N, K);

        uint8_t *d_aos, *d_soa;
        uint16_t *d_act, *d_out;
        HIP_OK(hipMalloc(&d_aos, bytes));
        HIP_OK(hipMalloc(&d_soa, bytes));
        HIP_OK(hipMalloc(&d_act, K*2));
        HIP_OK(hipMalloc(&d_out, N*2));
        HIP_OK(hipMemcpy(d_aos, h_aos.data(), bytes, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_soa, h_soa.data(), bytes, hipMemcpyHostToDevice));
        HIP_OK(hipMemset(d_act, 0, K*2));
        HIP_OK(hipDeviceSynchronize());

        hipStream_t st;
        HIP_OK(hipStreamCreate(&st));

        // Warmup
        hipLaunchKernelGGL(q1_gemv_soa1_kernel, dim3(N,1), dim3(128,1), 0, st, d_soa, (__half*)d_act, (__half*)d_out, N, K);
        HIP_OK(hipDeviceSynchronize());

        const int NR = 200;
        hipEvent_t t0, t1;
        HIP_OK(hipEventCreate(&t0));
        HIP_OK(hipEventCreate(&t1));

        // SoA
        HIP_OK(hipEventRecord(t0, st));
        for (int r = 0; r < NR; ++r)
            hipLaunchKernelGGL(q1_gemv_soa1_kernel, dim3(N,1), dim3(128,1), 0, st, d_soa, (__half*)d_act, (__half*)d_out, N, K);
        HIP_OK(hipEventRecord(t1, st));
        HIP_OK(hipEventSynchronize(t1));
        float ms_soa;
        HIP_OK(hipEventElapsedTime(&ms_soa, t0, t1));
        double pg_soa = ms_soa / (double)NR;
        double bw_soa = (double)bytes / (pg_soa/1000) / 1e9;

        printf("  %-20s %6.3f ms  %5.0f GB/s  %7.0f GEMV/s",
               shapes[s].n, pg_soa, bw_soa, 1000.0/pg_soa);
        printf("\n");

        // Count how many times this shape appears in full model
        int count = 0;
        for (int i = 0; i < num_ops*2; i+=2) {
            if (shapes_model[i] == N && shapes_model[i+1] == K) count++;
        }
        total_model_time_soa += pg_soa * count * 28;  // 28 layers
        total_w += bytes * count * 28;

        hipFree(d_aos); hipFree(d_soa); hipFree(d_act); hipFree(d_out);
        HIP_OK(hipEventDestroy(t0)); HIP_OK(hipEventDestroy(t1));
        HIP_OK(hipStreamDestroy(st));
    }

    printf("\n─── Full Model (28 layers, %d GEMVs) ───\n", num_ops*28);
    printf("  Total GEMV-only time: %.2f ms\n", total_model_time_soa);
    printf("  GEMV-only tok/s:      %.0f\n", 1000.0/total_model_time_soa);
    printf("  + norms+attn (~1.5ms): %.0f tok/s\n", 1000.0/(total_model_time_soa+1.5));
    printf("  Weight BW:             %.0f MB\n", total_w/1e6);
    printf("  Effective BW:          %.0f GB/s\n", total_w/(total_model_time_soa/1000)/1e9);
    printf("\n─── Path to 572 ───\n");
    printf("  Need: %.1f ms/token → %.0f GB/s (%.0f%% of 273 GB/s)\n",
           1000.0/572.0, total_w/((1000.0/572.0)/1000)/1e9,
           total_w/((1000.0/572.0)/1000)/1e9 / 273.0 * 100.0);
    return 0;
}
