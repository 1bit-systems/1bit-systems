// TWLA TQ2 INT4 bridge — dequantize INT4 activations and dispatch to TQ2
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

extern "C" void bonsai_tq2_1024_gemv_launch(
    const uint8_t*, const __half*, __half*, int, int, void*);

__global__ void int4_to_fp16_kernel(
    const uint8_t* act_i4, __half* out, float scale, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    uint8_t byte = act_i4[i / 2];
    int8_t nibble = (i & 1) ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
    if (nibble >= 8) nibble -= 16;
    out[i] = __float2half((float)nibble * scale);
}

extern "C" void twla_tq2_int4_launch(
    const void* packed, const uint8_t* act_i4, float act_scale,
    void* out, int N_out, int K_in, void* stream)
{
    __half* temp;
    hipMallocAsync(&temp, K_in * 2, (hipStream_t)stream);
    constexpr int B = 256;
    hipLaunchKernelGGL(int4_to_fp16_kernel,
        dim3((K_in+B-1)/B), dim3(B), 0, (hipStream_t)stream,
        act_i4, temp, act_scale, K_in);
    bonsai_tq2_1024_gemv_launch(
        (const uint8_t*)packed, temp, (__half*)out, N_out, K_in, (hipStream_t)stream);
    hipFreeAsync(temp, (hipStream_t)stream);
}
