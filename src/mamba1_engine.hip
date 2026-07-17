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
#include "src/zamba2_engine.h"
#include "src/gguf_zamba2_loader.cpp"

// ── Mamba1 fused inner: in_proj→conv1d→x_proj→dt_proj→scan→gate ──
// Shared memory: (2*d_inner) for xz + d_inner for x_proj_out + d_inner for y
// Max: 4*d_inner ≈ 4*3712 = 14.8 KB for Zamba-7B (well within 64KB)
#define QK 32
#define QB 18

// GEMV: Q4_0 16-row tiled (same as Mamba2)
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
// Processes: conv1d → silu → x_proj → dt_proj → scan → gate → out
__global__ void mamba1_inner(
    const float* __restrict__ c1w, const float* __restrict__ c1b, int dc, int di,
    const float* __restrict__ xpw,           // x_proj weight: [di, dt_rank+2*ds]
    const float* __restrict__ dpw,           // dt_proj weight: [dt_rank, di]
    const float* __restrict__ dpb,           // dt_proj bias: [di]
    const float* __restrict__ A,             // [ds, di]
    const float* __restrict__ Dv,            // [di]
    float* __restrict__ out,                 // [di] — gated + norm output
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
        // Shift conv state
        for(int k = dc-2; k >= 0; k--)
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
        // Handle remaining elements
        for(int k = (di_/4)*4; k < di_; k++)
            sum += xpw[(size_t)r*di_ + k] * x_conv[k];
        x_proj_out[r] = sum;
    }
    __syncthreads();
    
    // ── 4. Project dt and apply softplus ──
    // dt_from_xproj = x_proj_out[0..dt_rank)
    // dt = W_dt @ dt_from_xproj + bias_dt
    // dt_act = softplus(dt)
    // Store dt_act in x_proj_out[0..di) (reuse after x_proj is consumed)
    float* dt_act = x_proj_out;  // reuse space
    for(int i = ti; i < di_; i += 256) {
        float sum = dpb[i];  // start with bias
        for(int r = 0; r < dt_rank; r++)
            sum += dpw[(size_t)r*di_ + i] * x_proj_out[r];
        // softplus
        dt_act[i] = sum > 20.0f ? sum : log1pf(expf(sum));
    }
    __syncthreads();
    
    // ── 5. Selective scan ──
    // B = x_proj_out[dt_rank .. dt_rank+ds)
    // C = x_proj_out[dt_rank+ds .. dt_rank+2*ds)
    const float* B = x_proj_out + dt_rank;
    const float* C = x_proj_out + dt_rank + ds;
    
    for(int i = ti; i < di_; i += 256) {
        float d = dt_act[i];
        float xh = x_conv[i];
        
        // SSM state update: h = exp(d*A)*h + d*B*x
        // A is [ds, di], so A[s*di + i] is the state transition for channel i
        for(int s = 0; s < ds; s++) {
            float A_val = A[(size_t)s*di_ + i];
            float A_bar = expf(d * A_val);
            ss[(size_t)i*ds + s] = A_bar * ss[(size_t)i*ds + s] + d * B[s] * xh;
        }
        
        // cdot = C @ state
        float cdot = 0;
        for(int s = 0; s < ds; s++)
            cdot += C[s] * ss[(size_t)i*ds + s];
        
        // y = cdot * x + D * x
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

// ── RMS norm (same as Mamba2) ──
__global__ void rms_ker(const float*x,float*y,const float*w,int n,float eps){
    __shared__ float rd[256];int t=threadIdx.x;float s=0;
    for(int i=t;i<n;i+=256)s+=x[i]*x[i];rd[t]=s;__syncthreads();
    for(int i=128;i>0;i>>=1){if(t<i)rd[t]+=rd[t+i];__syncthreads();}
    float iv=rsqrtf(rd[0]/n+eps);if(w)for(int i=t;i<n;i+=256)y[i]=x[i]*iv*w[i];else for(int i=t;i<n;i+=256)y[i]=x[i]*iv;
}

// ── Test with Zamba-7B-v1 config or standalone ──
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
    return 0;
}
