// mamba2_kernels.hip — Mamba2 GPU kernels for AMD ROCm/HIP
// Implements the Mamba2 selective scan + conv1d on AMD Strix Halo (gfx1151).
//
// Kernel strategy:
//   - mamba2_selective_scan: 1 block per head-row, parallelizes over d_state
//   - mamba2_conv1d: 1 block per channel, each warp handles the conv kernel
//   - Layer fusions decomposed into readable steps
//
// Reference:
//   state-spaces/mamba: mamba_ssm/selective_scan_interface.py
//   Dao & Gu, "Transformers are SSMs: Generalized Models..." (2024)

#include "mamba2_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

// ── Mamba2 Selective Scan Kernel ──
// Each block processes one head of one sequence position.
// Grid: (L, B, heads_per_group * n_group)  (3D grid)
// Block: (d_state/THREADS_PER_STATE, 1, 1)
// Shared memory: d_state * sizeof(float) for state cache

#define SSM_THREADS 64  // threads per block = d_state (typically 64)

__global__ void mamba2_selective_scan_kernel(
    const float* __restrict__ x,        // [B, L, d_inner]
    const float* __restrict__ dt,       // [B, L, n_head]
    const float* __restrict__ A,        // [n_head]
    const float* __restrict__ B,        // [B, L, n_group, d_state]
    const float* __restrict__ C,        // [B, L, n_group, d_state]
    const float* __restrict__ D,        // [n_head]
    float* __restrict__ y,              // [B, L, d_inner]
    float* __restrict__ final_state,    // [B, d_state, d_inner]
    int B_dim, int L, int d_inner, int d_state, int n_head, int n_group, int head_dim
) {
    // 3D grid: (n_head, L, B)
    int head_id = blockIdx.x;
    int t       = blockIdx.y;
    int batch   = blockIdx.z;

    if (head_id >= n_head || t >= L || batch >= B_dim) return;

    // Which group this head belongs to
    int heads_per_group = n_head / n_group;
    int group_id = head_id / heads_per_group;
    int head_in_group = head_id % heads_per_group;

    // Offsets
    int x_offset     = (batch * L + t) * d_inner + head_id * head_dim;
    int dt_offset    = (batch * L + t) * n_head + head_id;
    int b_offset     = ((batch * L + t) * n_group + group_id) * d_state;
    int c_offset     = ((batch * L + t) * n_group + group_id) * d_state;
    int y_offset     = (batch * L + t) * d_inner + head_id * head_dim;
    int state_offset = (batch * d_inner + head_id * head_dim) * d_state;

    // Shared memory for state (per-block)
    __shared__ float s_state[SSM_THREADS];
    __shared__ float s_dt;
    __shared__ float s_A;
    __shared__ float s_D;
    __shared__ float s_x_head;

    int tid = threadIdx.x;

    // ── Load parameters (broadcast to all threads) ──
    if (tid == 0) {
        s_dt = dt[dt_offset];
        s_A  = A[head_id];
        s_D  = D[head_id];
        s_x_head = x[x_offset];  // head_dim elements, but scan is per-element
    }
    __syncthreads();

    // Load B and C for this head's group
    float B_val = (tid < d_state) ? B[b_offset + tid] : 0.0f;
    float C_val = (tid < d_state) ? C[c_offset + tid] : 0.0f;

    // ── Discretize ──
    // softplus(dt + dt_bias) — dt_bias is baked into dt already in the fused kernel
    float dt_val = s_dt;
    // Numerically stable softplus
    float dt_softplus = dt_val > 20.0f ? dt_val : log1pf(expf(dt_val));

    // A = -exp(A_log)
    float A_val = -expf(s_A);
    float A_bar = expf(dt_softplus * A_val);  // in (0,1)

    // ── Load previous state from global memory ──
    // final_state layout: [B, d_state, d_inner] = [B][d_inner][d_state]
    // For single-token decode, we load the persistent state
    // For prefill (L > 1), we accumulate in shared memory first

    float prev_state = 0.0f;
    if (t == 0) {
        // First token: load from persistent state (or zero-init)
        // In decode mode, we just use whatever is in ssm_state
        // For now, we use the incoming final_state as initial
        prev_state = (tid < d_state) ? final_state[state_offset + tid] : 0.0f;
    } else {
        // Subsequent tokens in prefill: accumulate in registers
        // (In practice, prefill uses a separate kernel that scans within registers)
        prev_state = (tid < d_state) ? final_state[state_offset + tid] : 0.0f;
    }

    // ── Selective scan step ──
    // state_new = A_bar * state_old + dt_softplus * B * x
    float state_new;
    if (tid < d_state) {
        state_new = A_bar * prev_state + dt_softplus * B_val * s_x_head;
    } else {
        state_new = 0.0f;
    }

    // Store updated state
    if (tid < d_state) {
        final_state[state_offset + tid] = state_new;
    }

    // ── Output: y = C @ state_new + D * x ──
    // Parallel reduction of C * state_new across d_state
    float c_dot = (tid < d_state) ? C_val * state_new : 0.0f;

    // Warp-level reduction for d_state <= 64
    // Using shuffle for faster reduction within warp
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        c_dot += __shfl_down(c_dot, offset);
    }

    // If d_state > 32, reduction across warps via shared memory
    if (d_state > 32) {
        if (tid % warpSize == 0) {
            s_state[tid / warpSize] = c_dot;
        }
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            int n_warps = (d_state + warpSize - 1) / warpSize;
            for (int w = 0; w < n_warps; ++w) {
                sum += s_state[w];
            }
            s_state[0] = sum;
        }
        __syncthreads();
        c_dot = s_state[0];
    } else {
        // Broadcast the result from lane 0
        c_dot = __shfl(c_dot, 0);
    }

    // ── Write output ──
    // y = c_dot * x + D * x  (for each element in head_dim)
    // Each thread handles some head_dim elements
    int elems_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    int start_elem = tid * elems_per_thread;
    int end_elem = min(start_elem + elems_per_thread, head_dim);

    for (int e = start_elem; e < end_elem; ++e) {
        float x_val = x[x_offset + e];
        y[y_offset + e] = c_dot * x_val + s_D * x_val;
    }
}

// ── Mamba2 Conv1D Kernel ──
// Depthwise 1D convolution along sequence.
// Grid: (conv_dim, B) — each block handles one channel
// Block: (L, 1, 1) — each thread handles one position
__global__ void mamba2_conv1d_kernel(
    const float* __restrict__ x,        // [B, L, conv_dim]
    const float* __restrict__ w,        // [d_conv, conv_dim]
    const float* __restrict__ b,        // [conv_dim]
    float* __restrict__ y,              // [B, L, conv_dim]
    float* __restrict__ state,          // [B, d_conv-1, conv_dim]
    int B_dim, int L, int d_conv, int conv_dim
) {
    int channel = blockIdx.x;
    int batch   = blockIdx.y;
    int t       = threadIdx.x;

    if (channel >= conv_dim || batch >= B_dim || t >= L) return;

    float acc = b[channel];
    float w0 = w[0 * conv_dim + channel];

    // Current input
    float x_t = x[(batch * L + t) * conv_dim + channel];
    acc += w0 * x_t;

    // Past state
    for (int k = 1; k < d_conv; ++k) {
        float state_val;
        if (t >= k) {
            // Use input from previous positions
            state_val = x[(batch * L + (t - k)) * conv_dim + channel];
        } else {
            // Use persistent state for positions before the sequence
            int state_idx = (d_conv - 1) - k + t;
            state_val = state[(batch * (d_conv - 1) + state_idx) * conv_dim + channel];
        }
        acc += w[k * conv_dim + channel] * state_val;
    }

    y[(batch * L + t) * conv_dim + channel] = acc;

    // Update persistent state with last d_conv-1 positions
    if (t == L - 1) {
        for (int k = 0; k < d_conv - 1; ++k) {
            int src_pos = max(0, L - 1 - k);
            state[(batch * (d_conv - 1) + k) * conv_dim + channel] =
                x[(batch * L + src_pos) * conv_dim + channel];
        }
    }
}

// ── Simple GEMV (matrix-vector multiply) for single-token decode ──
// y[M] = A[M,N] @ x[N]
__global__ void gemv_kernel(
    const float* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float sum = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < N; j++) sum += A[(long)i * N + j] * x[j];
    y[i] = sum;
}

// ── SiLU activation (in-place) ──
__global__ void silu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

// ── Element-wise multiply: y = y * x ──
__global__ void mul_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= x[i];
}

// ── Apply dt_bias: dt += dt_bias (in-place) ──
__global__ void add_dt_bias_kernel(float* dt, const float* dt_bias, int n_head) {
    int i = threadIdx.x;
    if (i < n_head) dt[i] += dt_bias[i];
}

// ── GPU Mamba2 decode block: orchestrates all sub-kernels for one token ──
void mamba2_gpu_decode_block(
    const float* x_in, const float* in_proj_w,
    const float* conv1d_w, const float* conv1d_b,
    const float* dt_bias, const float* A_log, const float* D,
    const float* out_proj_w,
    float* conv_state, float* ssm_state, float* y_out,
    float* tmp,
    int d_model, int d_inner, int d_state, int d_conv,
    int n_head, int n_group, int head_dim, int conv_dim,
    hipStream_t stream)
{
    int d_in_proj = d_inner + conv_dim + n_head;
    int threads = 256;

    // 1. in_proj: tmp[0..d_in_proj] = in_proj_w @ x_in
    int blocks_in = (d_in_proj + threads - 1) / threads;
    hipLaunchKernelGGL(gemv_kernel, dim3(blocks_in), dim3(threads), 0, stream,
        in_proj_w, x_in, tmp, d_in_proj, d_model);

    // Split pointers into tmp:
    float* z   = tmp;                    // [d_inner]
    float* xBC = tmp + d_inner;          // [conv_dim]
    float* dt  = tmp + d_inner + conv_dim; // [n_head]

    // 2. conv1d: xBC' = silu(conv1d(xBC) + conv1d_b)
    //    Uses L=1 (single token decode)
    //    conv1d_kernel expects [B, L, conv_dim], B=1, L=1
    dim3 conv_grid(conv_dim, 1);
    dim3 conv_block(1, 1);
    float* conv_out = tmp + d_in_proj;  // reuse tmp tail, conv_dim elements
    hipLaunchKernelGGL(mamba2_conv1d_kernel, conv_grid, conv_block, 0, stream,
        xBC, conv1d_w, conv1d_b, conv_out, conv_state,
        1, 1, d_conv, conv_dim);

    // silu activation in-place on conv_out
    int blocks_silu = (conv_dim + threads - 1) / threads;
    hipLaunchKernelGGL(silu_kernel, dim3(blocks_silu), dim3(threads), 0, stream,
        conv_out, conv_dim);

    // Copy conv_out back to xBC position for split (or use separate pointers)
    float* xBC_act = conv_out;  // [conv_dim]
    // Split xBC_act into x, B, C within shared tmp
    float* x_seg  = xBC_act;                         // [d_inner]
    float* B_seg  = xBC_act + d_inner;               // [n_group * d_state]
    float* C_seg  = xBC_act + d_inner + n_group * d_state; // [n_group * d_state]

    // 3. Apply dt_bias: dt += dt_bias
    hipLaunchKernelGGL(add_dt_bias_kernel, dim3(1), dim3(n_head), 0, stream,
        dt, dt_bias, n_head);

    // 4. selective_scan for each head (L=1, B=1)
    dim3 scan_grid(n_head, 1, 1);
    dim3 scan_block(d_state > 32 ? d_state : 64);  // at least d_state threads
    hipLaunchKernelGGL(mamba2_selective_scan_kernel, scan_grid, scan_block, 0, stream,
        x_seg, dt, A_log, B_seg, C_seg, D,
        tmp + d_in_proj + conv_dim,  // y_inner output after scan
        ssm_state,
        1, 1, d_inner, d_state, n_head, n_group, head_dim);

    // 5. Gate: y_inner *= silu(z)
    float* y_inner = tmp + d_inner;  // reuse xBC area after conv is done
    int blocks_gate = (d_inner + threads - 1) / threads;
    hipLaunchKernelGGL(silu_kernel, dim3(blocks_gate), dim3(threads), 0, stream,
        z, d_inner);  // silu in-place on z
    // Multiply: y_inner = y_inner * z
    hipLaunchKernelGGL(mul_kernel, dim3(blocks_gate), dim3(threads), 0, stream,
        y_inner, z, d_inner);

    // 6. out_proj: y_out[d_model] = out_proj_w @ y_inner
    int blocks_out = (d_model + threads - 1) / threads;
    hipLaunchKernelGGL(gemv_kernel, dim3(blocks_out), dim3(threads), 0, stream,
        out_proj_w, y_inner, y_out, d_model, d_inner);
}
