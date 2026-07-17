// Simple K projection kernel — no warp shuffle, just pure dot product
// K[row] = sum(hidden[c] * wt[row * H + c]) for all c in 0..H-1
__global__ void k_proj_k(__half* out, const __half* in, const __half* wt, int K, int HDIM) {
    int row = blockIdx.x;  // One block per output row
    if (row >= K) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < HDIM; c += blockDim.x)
        sum += __half2float(in[c]) * __half2float(wt[(size_t)row * H + c]);
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    for (int o = blockDim.x/2; o > 0; o >>= 1) {
        __syncthreads();
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
    }
    if (threadIdx.x == 0) out[row] = __float2half(sh[0]);
}
