// GPU argmax kernel — finds max value and its index in a fp16 array
// Launch with 1 block of 256 threads
// Does hierarchical reduction: each warp finds local max, then warp 0 reduces
__global__ void argmax_kernel(const __half* __restrict__ data, int n, int* __restrict__ out_idx, float* __restrict__ out_val) {
    int tx = threadIdx.x;
    int lane = tx % 32;
    int warp = tx / 32;
    
    // Each thread scans a portion
    float my_val = -1e30f;
    int my_idx = 0;
    for (int i = tx; i < n; i += blockDim.x) {
        float v = __half2float(data[i]);
        if (v > my_val) { my_val = v; my_idx = i; }
    }
    
    // Warp reduce
    for (int o = 16; o > 0; o >>= 1) {
        float other_val = __shfl_xor(my_val, o);
        int other_idx = __shfl_xor(my_idx, o);
        if (other_val > my_val) { my_val = other_val; my_idx = other_idx; }
    }
    
    __shared__ float s_val[8];
    __shared__ int s_idx[8];
    if (lane == 0) { s_val[warp] = my_val; s_idx[warp] = my_idx; }
    __syncthreads();
    
    if (warp == 0) {
        float best_val = (lane < 8) ? s_val[lane] : -1e30f;
        int best_idx = (lane < 8) ? s_idx[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) {
            float other_val = __shfl_xor(best_val, o);
            int other_idx = __shfl_xor(best_idx, o);
            if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
        }
        if (lane == 0) { *out_idx = best_idx; *out_val = best_val; }
    }
}
