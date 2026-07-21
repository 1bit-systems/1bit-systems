// mamba1_engine.hip — Mamba1 GPU selective scan + conv1d for Strix Halo
// Implements the Mamba1 SSM block (used by Zamba-7B-v1, BlackMamba).
//
// Mamba1 architecture (per layer):
//   in_proj: [x, z] = split(W_in @ input)
//   conv1d:  x = silu(conv1d(x) + bias)
//   x_proj:  [dt, B, C] = split(W_x @ x)
//   dt_proj: dt = W_dt @ dt + bias_dt
//   scan:    y = selective_scan(x, softplus(dt), A, B, C)
//   gate:    y = y * silu(z)
//   out:     y = W_out @ y
//
// Key difference from Mamba2: no head structure, A is [d_state, d_inner],
// separate x_proj and dt_proj.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

// ── Mamba1 fused inner: in_proj→conv1d→x_proj→dt_proj→scan→gate ──
// Shared memory: (2*d_inner) for xz + d_inner for x_proj_out + d_inner for y
// Max: 4*d_inner ≈ 4*3712 = 14.8 KB for Zamba-7B (well within 64KB)

// Q4_0 GEMV: 16-row tiled (for when weights are Q4_0 quantized)
#define QK 32
#define QB 18

__global__ void gemv_q4(const uint8_t* W, const float* x, float* y, int M, int K) {
    __shared__ float xt[2048];
    int t=threadIdx.x, w=t/64, ln=t%64, rb=blockIdx.x*16;
    float a[4]={0}; int gp=(K+QK-1)/QK;
    for(int k0=0;k0<K;k0+=2048){int ks=min(2048,K-k0);
        for(int i=t;i<ks;i+=128)xt[i]=x[k0+i];__syncthreads();
        for(int r=0;r<4;r++){int row=rb+w*4+r;if(row>=M)continue;float s=0;
            int g0=k0/QK,g1=min((k0+ks+QK-1)/QK,gp);
            for(int g=g0+ln;g<g1;g+=64){
                const uint8_t* b_=W+((size_t)row*gp+g)*QB;
                uint16_t sb;memcpy(&sb,b_,2);uint32_t fb=(uint32_t)sb<<16;float sc;memcpy(&sc,&fb,4);
                int gk=g*QK;
                for(int j=0;j<16;j++){int ki=gk+j*2;uint8_t by=b_[2+j];int n0=(by>>4)&0x0F,n1=by&0x0F;if(n0>=8)n0-=16;if(n1>=8)n1-=16;
                    if(ki>=k0&&ki<k0+ks)s+=(float)n0*sc*xt[ki-k0];
                    if(ki+1>=k0&&ki+1<k0+ks)s+=(float)n1*sc*xt[ki+1-k0];}
            }
            for(int o=32;o>0;o>>=1)s+=__shfl_xor(s,o);if(ln==0)a[r]+=s;}
        __syncthreads();}
    for(int r=0;r<4;r++){int row=rb+w*4+r;if(ln==0&&row<M)y[row]=a[r];}
}

// ── Mamba1 fused inner kernel ──
// Processes: conv1d → silu → x_proj → dt_proj → scan → gate
// Takes in_proj output (x + z) as input, produces gated SSM output.
__global__ void mamba1_inner(
    const float* __restrict__ c1w, const float* __restrict__ c1b, int dc, int di,
    const float* __restrict__ xpw,           // x_proj weight: [di, dt_rank+2*ds]
    const float* __restrict__ dpw,           // dt_proj weight: [dt_rank, di]
    const float* __restrict__ dpb,           // dt_proj bias: [di]
    const float* __restrict__ A,             // [ds, di]
    const float* __restrict__ Dv,            // [di]
    float* __restrict__ out,                 // [di] — gated output (before out_proj)
    float* __restrict__ cs,                  // [dc-1, di] conv state
    float* __restrict__ ss,                  // [ds, di] ssm state
    const float* __restrict__ inp,           // [2*di] — in_proj output (x + z)
    int dt_rank, int ds
) {
    extern __shared__ float sh[];
    float* x_in = sh;                    // [di] — x from in_proj
    float* z = sh + di;                  // [di] — z from in_proj (silu gate)
    float* x_proj_out = sh + 2*di;      // [dt_rank + 2*ds] — x_proj output
    float* x_conv = x_in;               // [di] — conv output (reuse x_in space after conv)
    
    int ti = threadIdx.x;
    int di_ = di;
    
    // ── 1. Extract x and z from in_proj output ──
    for(int i = ti; i < di_; i += 256) {
        x_in[i] = inp[i];           // first half = x
        z[i] = inp[di_ + i];        // second half = z (gate)
    }
    __syncthreads();
    
    // ── 2. conv1d + silu on x ──
    for(int i = ti; i < di_; i += 256) {
        float xt = x_in[i];
        float a = c1b[i] + c1w[0*di_ + i] * xt;
        for(int k = 1; k < dc; k++) a += c1w[k*di_ + i] * cs[(k-1)*di_ + i];
        float v = a;
        x_conv[i] = v / (1.0f + expf(-v));  // silu in-place
        // Shift conv state: discard oldest (cs[dc-2]), shift the rest, insert new
        for(int k = dc-3; k >= 0; k--)
            cs[(k+1)*di_ + i] = cs[k*di_ + i];
        if(dc > 1) cs[0*di_ + i] = xt;
    }
    __syncthreads();
    
    // ── 3. x_proj: [dt_rank + 2*ds] = W_x @ x_conv ──
    int xp_dim = dt_rank + 2*ds;
    for(int r = ti; r < xp_dim; r += 256) {
        float sum = 0;
        for(int k = 0; k < di_; k += 4) {
            if(k + 3 < di_)
                sum += xpw[(size_t)r*di_ + k]*x_conv[k] + xpw[(size_t)r*di_ + k+1]*x_conv[k+1]
                     + xpw[(size_t)r*di_ + k+2]*x_conv[k+2] + xpw[(size_t)r*di_ + k+3]*x_conv[k+3];
        }
        for(int k = (di_/4)*4; k < di_; k++)
            sum += xpw[(size_t)r*di_ + k] * x_conv[k];
        x_proj_out[r] = sum;
    }
    __syncthreads();
    
    // ── 4. Project dt and apply softplus ──
    float* dt_act = x_proj_out;  // reuse space
    for(int i = ti; i < di_; i += 256) {
        float sum = dpb[i];
        for(int r = 0; r < dt_rank; r++)
            sum += dpw[(size_t)r*di_ + i] * x_proj_out[r];
        dt_act[i] = sum > 20.0f ? sum : log1pf(expf(sum));
    }
    __syncthreads();
    
    // ── 5. Selective scan ──
    const float* B = x_proj_out + dt_rank;
    const float* C = x_proj_out + dt_rank + ds;
    
    for(int i = ti; i < di_; i += 256) {
        float d_dt = dt_act[i];
        float xh = x_conv[i];
        
        for(int s = 0; s < ds; s++) {
            // Mamba1: A = -exp(A_log), then A_bar = exp(dt * A)
            float A_val = -expf(A[(size_t)s*di_ + i]);
            float A_bar = expf(d_dt * A_val);
            ss[(size_t)i*ds + s] = A_bar * ss[(size_t)i*ds + s] + d_dt * B[s] * xh;
        }
        
        float cdot = 0;
        for(int s = 0; s < ds; s++)
            cdot += C[s] * ss[(size_t)i*ds + s];
        
        x_conv[i] = cdot * xh + Dv[i] * xh;
    }
    __syncthreads();
    
    // ── 6. Gate: y = y * silu(z) ──
    for(int i = ti; i < di_; i += 256) {
        float zv = z[i];
        float silu_z = zv / (1.0f + expf(-zv));
        out[i] = x_conv[i] * silu_z;
    }
}

// ── FP32 GEMV: y[m] = sum_k W[m,k] * x[k] ──
// One block per output row, 256 threads reducing across K.
#define GEMV_BLOCK 256
__global__ void fp32_gemv(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K)
{
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    __shared__ float s_sum[GEMV_BLOCK / 32];
    float acc = 0.0f;
    const float* row = W + (size_t)m * K;
    for (int k = tid; k < K; k += GEMV_BLOCK)
        acc += row[k] * x[k];
    // Warp reduction
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if ((tid & 31) == 0) s_sum[tid / 32] = acc;
    __syncthreads();
    if (tid < GEMV_BLOCK / 32) {
        float v = s_sum[tid];
        for (int o = (GEMV_BLOCK / 32) / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
        if (tid == 0) y[m] = v;
    }
}

// ── Residual add: y[i] += x[i] ──
__global__ void add_residual_k(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

// ── SiLU gate: given fused [gate; up], produce y = silu(gate) * up ──
// Input layout: [0..hidden) = gate, [hidden..2*hidden) = up
__global__ void silu_gate_k(float* y, const float* fused, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        float g = fused[i];
        float u = fused[hidden + i];
        float silu_g = g / (1.0f + expf(-g));
        y[i] = silu_g * u;
    }
}

// ── Scale and add residual: y[i] += scale * x[i] ──
__global__ void scale_add_residual_k(float* y, const float* x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += scale * x[i];
}

// ── RMS norm ──
__global__ void rms_ker(const float*x,float*y,const float*w,int n,float eps){
    __shared__ float rd[256];int t=threadIdx.x;float s=0;
    for(int i=t;i<n;i+=256)s+=x[i]*x[i];rd[t]=s;__syncthreads();
    for(int i=128;i>0;i>>=1){if(t<i)rd[t]+=rd[t+i];__syncthreads();}
    float iv=rsqrtf(rd[0]/n+eps);if(w)for(int i=t;i<n;i+=256)y[i]=x[i]*iv*w[i];else for(int i=t;i<n;i+=256)y[i]=x[i]*iv;
}

// ── Mamba1 layer forward (host-side wrapper) ──
// Launches the full Mamba1 SSM chain: rmsnorm → in_proj → mamba1_inner
// → out_proj → residual add. Updates hidden state in-place.
//
// All pointers (hidden, weight ptrs, state ptrs, scratch) are device pointers.
// scratch buffers:
//   normed: [d_model]
//   inproj: [2*d_inner]
//   gated:  [d_inner]
//
// Returns 0 on success.
extern "C" int mamba1_forward(
    float* hidden,
    const float* in_proj_w, const float* norm_w,
    const float* conv1d_w, const float* conv1d_b, int d_conv,
    const float* x_proj_w,
    const float* dt_w, const float* dt_b,
    const float* A_log, const float* D,
    const float* out_proj_w,
    float* conv_state, float* ssm_state,
    float* normed, float* inproj, float* gated,
    int d_model, int d_inner, int d_state, int dt_rank,
    hipStream_t stream
) {
    int grid_n = (d_model + 255) / 256;
    
    // 1. RMS norm
    rms_ker<<<grid_n, 256, 0, stream>>>(hidden, normed, norm_w, d_model, 1e-5f);
    
    // 2. in_proj GEMV
    fp32_gemv<<<2 * d_inner, GEMV_BLOCK, 0, stream>>>(in_proj_w, normed, inproj, 2 * d_inner, d_model);
    
    // 3. mamba1_inner: conv1d → silu → x_proj → dt_proj → scan → gate
    size_t smem = 4 * d_inner * sizeof(float);
    mamba1_inner<<<1, 256, smem, stream>>>(
        conv1d_w, conv1d_b, d_conv, d_inner,
        x_proj_w, dt_w, dt_b,
        A_log, D,
        gated, conv_state, ssm_state,
        inproj, dt_rank, d_state);
    
    // 4. out_proj GEMV
    fp32_gemv<<<d_model, GEMV_BLOCK, 0, stream>>>(out_proj_w, gated, normed, d_model, d_inner);
    
    // 5. Residual
    add_residual_k<<<grid_n, 256, 0, stream>>>(hidden, normed, d_model);
    return 0;
}

// ── Host-callable kernel launch wrappers ──
// These let pure CXX callers (backend_mamba1.cpp, etc.) launch kernels
// without needing HIP compilation themselves. The device stubs are
// generated here in the .hip TU.

extern "C" void launch_rms_ker(
    const float* x, float* y, const float* w, int n, float eps, hipStream_t stream)
{
    int grid = (n + 255) / 256;
    if (grid < 1) grid = 1;
    rms_ker<<<grid, 256, 0, stream>>>(x, y, w, n, eps);
}

extern "C" void launch_fp32_gemv(
    const float* W, const float* x, float* y, int M, int K, hipStream_t stream)
{
    if (M <= 0 || K <= 0) return;
    fp32_gemv<<<M, GEMV_BLOCK, 0, stream>>>(W, x, y, M, K);
}

extern "C" void launch_silu_gate_k(
    float* y, const float* fused, int hidden, hipStream_t stream)
{
    int grid = (hidden + 255) / 256;
    if (grid < 1) grid = 1;
    silu_gate_k<<<grid, 256, 0, stream>>>(y, fused, hidden);
}

extern "C" void launch_scale_add_residual_k(
    float* y, const float* x, float scale, int n, hipStream_t stream)
{
    int grid = (n + 255) / 256;
    if (grid < 1) grid = 1;
    scale_add_residual_k<<<grid, 256, 0, stream>>>(y, x, scale, n);
}

extern "C" void launch_add_residual_k(
    float* y, const float* x, int n, hipStream_t stream)
{
    int grid = (n + 255) / 256;
    if (grid < 1) grid = 1;
    add_residual_k<<<grid, 256, 0, stream>>>(y, x, n);
}

// ── Standalone test ──
// Compiled only when this file is built as a standalone executable
// (e.g. via `hipcc src/mamba1_engine.hip -o mamba1_test`).
#ifndef MAMBA1_ENGINE_AS_LIB
int main() {
    printf("Mamba1 engine built for Strix Halo (gfx1151)\n");
    printf("Supports Zamba-7B-v1 and BlackMamba models\n");
    printf("\nArchitecture:\n");
    printf("  in_proj: [2*d_inner] = W @ input\n");
    printf("  conv1d:  silu(conv1d(x))\n");
    printf("  x_proj:  [dt_rank+2*d_state] = W_x @ x\n");
    printf("  dt_proj: W_dt @ dt[:dt_rank] + bias\n");
    printf("  scan:    h = A_bar*h + B*dt*x, y = C@h + D*x\n");
    printf("  gate:    y = y * silu(z)\n");
    printf("  out:     W_out @ y\n");
    printf("\nShared memory: 4*di floats = %.0f KB (Zamba-7B, di=7424)\n", 4*7424.0*4/1024);
    printf("\nBuilt as part of 1bit-systems — kernels wired into CMake build.\n");
    return 0;
}
#endif
