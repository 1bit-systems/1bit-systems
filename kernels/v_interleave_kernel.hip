// Zaya V interleave kernel — combines current + delayed V projections into contiguous buffer
// v1[KD/2] (current token), v2[KD/2] (delayed token) → v[KD]
__global__ void v_interleave_kernel(__half* v_out, const __half* v1, const __half* v2, int half_kd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_kd) return;
    v_out[i] = v1[i];
    v_out[half_kd + i] = v2[i];
}
