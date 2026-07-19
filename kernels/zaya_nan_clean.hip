// NaN-clearing kernel: replace NaN values with 0
extern "C" __global__ void nan_clean_k(__half* __restrict__ buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(buf[i]);
    if (isnan(v) || isinf(v)) buf[i] = __float2half(0.0f);
}
