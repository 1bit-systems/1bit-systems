// CCA Attention GPU Test - Zaya CCA Attention GPU Kernel + CPU Reference
// Build: /opt/rocm-therock/bin/hipcc -O3 --offload-arch=gfx1151 test_cca_attn.cpp -o test_cca_attn

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <numeric>
#include <cstdlib>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__); abort();}} while(0)

// Model config
constexpr int H = 2048, NQ = 8, NKV = 2, HD = 128;
constexpr int GQA = NQ / NKV, QD = NQ * HD, KD = NKV * HD, QKV = QD + KD;
constexpr int DC = 2, NGRP = 10, GC = QKV / NGRP, NROT = 64;
constexpr int WARP_SIZE = 32;
constexpr float RMD_EPS = 1e-5f, FP16_MAX = 65504.0f;

struct CCAWeights {
    std::vector<float> wq, wk, wv1, wv2, wo;
    std::vector<float> cdw, cdb, cgw, cgb;
    std::vector<float> k_scale, norm_w;
};

// ── CPU Reference ──
static void matmul(float* out, const float* a, const float* b, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += a[m*K+k] * b[k*N+n];
            out[m*N+n] = s;
        }
}

static void cca_attn_ref(float* out, const float* x, const CCAWeights& w, int pos) {
    float* buf = new float[H*10 + QD*10 + KD*10 + QKV*10];
    float *normed = buf, *q = buf+H, *k = buf+H+QD, *qp = buf+H+QD+KD;
    float *kp = qp+QD, *qk = kp+KD, *pad = qk+QKV, *c1 = pad+QKV*3, *c2 = c1+QKV*2;
    float *qc = c2+QKV, *kc = qc+QD, *v1 = kc+KD, *v = v1+KD/2;
    float *attn = v+KD, *rout = attn+QD;
    
    float ss = 0; for (int i = 0; i < H; i++) ss += x[i]*x[i];
    float rn = 1.0f / sqrtf(ss/H + RMD_EPS);
    for (int i = 0; i < H; i++) normed[i] = x[i] * rn * w.norm_w[i];
    
    matmul(q, normed, w.wq.data(), 1, QD, H);
    matmul(k, normed, w.wk.data(), 1, KD, H);
    memcpy(qp, q, QD*4); memcpy(kp, k, KD*4);
    memcpy(qk, q, QD*4); memcpy(qk+QD, k, KD*4);
    
    memset(pad, 0, QKV*3*4);
    for (int i = 0; i < QKV; i++) pad[i*3+2] = qk[i];
    for (int c = 0; c < QKV; c++)
        for (int l = 0; l < 2; l++) {
            float s = 0;
            for (int d = 0; d < DC; d++) s += pad[c*3+l+d] * w.cdw[c*DC+d];
            c1[c*2+l] = s + w.cdb[c];
        }
    for (int g = 0; g < NGRP; g++)
        for (int co = 0; co < GC; co++) {
            int co2 = g*GC+co; float s = 0;
            for (int ci = 0; ci < GC; ci++)
                for (int d = 0; d < DC; d++)
                    s += c1[(g*GC+ci)*2+d] * w.cgw[co2*GC*DC + ci*DC + d];
            c2[co2] = s + w.cgb[co2];
        }
    
    memcpy(qc, c2, QD*4); memcpy(kc, c2+QD, KD*4);
    for (int h = 0; h < NQ; h++)
        for (int d = 0; d < HD; d++)
            qc[h*HD+d] += 0.5f*qp[h*HD+d] + 0.5f*kp[(h/GQA)*HD+d];
    for (int kh = 0; kh < NKV; kh++) {
        float qm[HD] = {0};
        for (int g = 0; g < GQA; g++)
            for (int d = 0; d < HD; d++) qm[d] += qp[(kh*GQA+g)*HD+d];
        for (int d = 0; d < HD; d++) qm[d] /= GQA;
        for (int d = 0; d < HD; d++)
            kc[kh*HD+d] += 0.5f*qm[d] + 0.5f*kp[kh*HD+d];
    }
    
    for (int h = 0; h < NQ; h++) {
        float s = 0; for (int d = 0; d < HD; d++) s += qc[h*HD+d]*qc[h*HD+d];
        float r = sqrtf(HD) * powf(s+1e-12f, -0.5f);
        for (int d = 0; d < HD; d++) qc[h*HD+d] *= r;
    }
    for (int h = 0; h < NKV; h++) {
        float s = 0; for (int d = 0; d < HD; d++) s += kc[h*HD+d]*kc[h*HD+d];
        float r = sqrtf(HD) * powf(s+1e-12f, -0.5f) * w.k_scale[h];
        for (int d = 0; d < HD; d++) kc[h*HD+d] *= r;
    }
    
    for (int h = 0; h < NQ; h++)
        for (int d = 0; d < NROT; d+=2) {
            float th = powf(500000.0f, -2.0f*d/HD);
            float cv = cosf(pos*th), sv = sinf(pos*th);
            float q0=qc[h*HD+d], q1=qc[h*HD+d+1];
            qc[h*HD+d]=q0*cv-q1*sv; qc[h*HD+d+1]=q0*sv+q1*cv;
        }
    for (int h = 0; h < NKV; h++)
        for (int d = 0; d < NROT; d+=2) {
            float th = powf(500000.0f, -2.0f*d/HD);
            float cv = cosf(pos*th), sv = sinf(pos*th);
            float k0=kc[h*HD+d], k1=kc[h*HD+d+1];
            kc[h*HD+d]=k0*cv-k1*sv; kc[h*HD+d+1]=k0*sv+k1*cv;
        }
    
    matmul(v1, normed, w.wv1.data(), 1, KD/2, H);
    memcpy(v, v1, (KD/2)*4); memset(v+KD/2, 0, (KD/2)*4);
    
    memset(attn, 0, QD*4);
    for (int h = 0; h < NQ; h++) {
        int kh = h/GQA; float sc = 0;
        for (int d = 0; d < HD; d++) sc += qc[h*HD+d] * kc[kh*HD+d];
        sc *= 1.0f/sqrtf(HD);
        for (int d = 0; d < HD; d++) attn[h*HD+d] = expf(sc) * v[kh*HD+d];
    }
    matmul(rout, attn, w.wo.data(), 1, H, QD);
    memcpy(out, rout, H*4);
    delete[] buf;
}

// ── GPU kernel: Q projection (matmul) ──
// out[M] = sum_j(in[j] * weight[j*M + i])
// Each block processes one row of output
__global__ void proj_kernel(const __half* in, const __half* weight, __half* out,
                             int M, int N, int K) {
    // out is [M] = in [K] @ weight [K, M]
    // One block per output element, each thread computes partial product
    // Actually simpler: split output across threads, each accumulates over K
    extern __shared__ float s_part[];
    int tx = threadIdx.x;
    int elem = blockIdx.x * blockDim.x + tx;
    if (elem >= M) return;
    
    float sum = 0;
    for (int k = 0; k < K; k++)
        sum += (float)in[k] * (float)weight[k * M + elem];
    out[elem] = __float2half(sum);
}

// ── GPU kernel: depthwise conv 1D ──
// in: [C, L_in], w: [C, K], b: [C], out: [C, L_out]
// L_in = L_out + K - 1 (with padding=2 left, L_in=3, L_out=1 for DC=2 single-token)
__global__ void conv_dw_kernel(const float* in, const float* w, const float* b,
                                float* out, int C, int L_out, int K) {
    int tx = threadIdx.x;
    int c = blockIdx.x;
    if (c >= C) return;
    for (int l = tx; l < L_out; l += blockDim.x) {
        float s = 0;
        for (int d = 0; d < K; d++)
            s += in[c * (L_out + K - 1) + l + d] * w[c * K + d];
        out[c * L_out + l] = s + (b ? b[c] : 0);
    }
}

// ── GPU kernel: grouped conv 1D ──
// in: [C, L_in], w: [C, C/G, K], b: [C], out: [C, L_out], groups=G
__global__ void conv_grp_kernel(const float* in, const float* w, const float* b,
                                 float* out, int C, int L_out, int G, int K) {
    int tx = threadIdx.x;
    int c = blockIdx.x;
    if (c >= C) return;
    int g = c / (C / G);
    int co = c % (C / G);
    int cg = g * (C / G);
    for (int l = tx; l < L_out; l += blockDim.x) {
        float s = 0;
        for (int ci = 0; ci < C/G; ci++)
            for (int d = 0; d < K; d++)
                s += in[(cg + ci) * (L_out + K - 1) + l + d] * w[c * (C/G) * K + ci * K + d];
        out[c * L_out + l] = s + (b ? b[c] : 0);
    }
}

// ── GPU: L2 normalize Q (per head) ──
// q: [N_HEADS * HEAD_DIM], out: same, each head normalized to ||out|| = sqrt(HD)
__global__ void l2norm_q_kernel(__half* q, int n_heads, int hd) {
    int h = blockIdx.x;
    if (h >= n_heads) return;
    __shared__ float red[32];
    int tx = threadIdx.x;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;
    float ss = 0;
    for (int d = tx; d < hd; d += blockDim.x)
        ss += (float)q[h*hd+d] * (float)q[h*hd+d];
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (lane < (blockDim.x/WARP_SIZE)) ? red[lane] : 0;
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
        if (lane == 0) {
            float r = sqrtf((float)hd) * rsqrtf(ss + 1e-12f);
            for (int d = 0; d < hd; d++)
                q[h*hd+d] = __float2half((float)q[h*hd+d] * r);
        }
    }
}

// ── GPU: L2 normalize K (per head, with temperature) ──
__global__ void l2norm_k_kernel(__half* k, const float* k_scale, int n_kv, int hd) {
    int h = blockIdx.x;
    if (h >= n_kv) return;
    __shared__ float red[32];
    int tx = threadIdx.x;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;
    float ss = 0;
    for (int d = tx; d < hd; d += blockDim.x)
        ss += (float)k[h*hd+d] * (float)k[h*hd+d];
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (lane < (blockDim.x/WARP_SIZE)) ? red[lane] : 0;
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
        if (lane == 0) {
            float r = sqrtf((float)hd) * rsqrtf(ss + 1e-12f) * k_scale[h];
            for (int d = 0; d < hd; d++)
                k[h*hd+d] = __float2half((float)k[h*hd+d] * r);
        }
    }
}

// ── GPU: partial RoPE on Q and K ──
__global__ void rope_kernel(__half* q, __half* k, int n_q, int n_kv, int hd, int n_rot, int pos) {
    int tx = threadIdx.x;
    for (int h = 0; h < n_q; h++) {
        for (int d = tx*2; d < n_rot; d += blockDim.x*2) {
            float theta = powf(500000.0f, -2.0f * d / hd);
            float cv = cosf(pos * theta), sv = sinf(pos * theta);
            float q0 = (float)q[h*hd+d], q1 = (float)q[h*hd+d+1];
            q[h*hd+d] = __float2half(q0*cv - q1*sv);
            q[h*hd+d+1] = __float2half(q0*sv + q1*cv);
        }
    }
    for (int h = 0; h < n_kv; h++) {
        for (int d = tx*2; d < n_rot; d += blockDim.x*2) {
            float theta = powf(500000.0f, -2.0f * d / hd);
            float cv = cosf(pos * theta), sv = sinf(pos * theta);
            float k0 = (float)k[h*hd+d], k1 = (float)k[h*hd+d+1];
            k[h*hd+d] = __float2half(k0*cv - k1*sv);
            k[h*hd+d+1] = __float2half(k0*sv + k1*cv);
        }
    }
}

// ── GPU: GQA attention + output projection for single token ──
// Q: [N_Q * HD], K: [N_KV * HD], V: [N_KV * HD], wo: [H * N_Q*HD]
// out: [H]
__global__ void gqa_kernel(const __half* q, const __half* k, const __half* v,
                            const __half* wo, __half* out,
                            int n_q, int n_kv, int hd) {
    int tx = threadIdx.x;
    int n_head_out = n_q; // each Q head produces weighted V
    
    // Each thread block computes one output element
    // Shared memory for partial attention results
    extern __shared__ float s_attn[];
    
    // Compute attention score for each Q head (single token)
    float max_s = -1e10f;
    float sum_exp = 0;
    int elem = blockIdx.x;
    if (elem >= H) return;
    
    float val = 0;
    for (int h = 0; h < n_q; h++) {
        int kh = h / (n_q / n_kv);
        float s = 0;
        for (int d = 0; d < hd; d++)
            s += (float)q[h*hd+d] * (float)k[kh*hd+d];
        s *= 1.0f / sqrtf((float)hd);
        if (s > max_s) max_s = s;
    }
    for (int h = 0; h < n_q; h++) {
        int kh = h / (n_q / n_kv);
        float s = 0;
        for (int d = 0; d < hd; d++)
            s += (float)q[h*hd+d] * (float)k[kh*hd+d];
        sum_exp += expf(s * (1.0f/sqrtf((float)hd)) - max_s);
    }
    for (int h = 0; h < n_q; h++) {
        int kh = h / (n_q / n_kv);
        float s = 0;
        for (int d = 0; d < hd; d++)
            s += (float)q[h*hd+d] * (float)k[kh*hd+d];
        float prob = expf(s * (1.0f/sqrtf((float)hd)) - max_s) / sum_exp;
        for (int d = 0; d < hd; d++) {
            float vv = (float)v[kh*hd+d];
            val += prob * vv * (float)wo[elem * n_q * hd + h * hd + d];
        }
    }
    if (tx == 0) out[elem] = __float2half(val);
}

// ── GPU Norm-only kernel ──
__global__ void norm_kernel(const __half* x, const __half* w, __half* out, int n) {
    const int warps = blockDim.x / WARP_SIZE;
    __shared__ float red[32]; // max warps
    int tx = threadIdx.x;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;
    
    float ss = 0;
    for (int i = tx; i < n; i += blockDim.x) ss += (float)x[i] * (float)x[i];
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    
    // Second-level reduction: warp 0 sums all partials from shared memory
    if (warp == 0) {
        float total = 0;
        for (int w = 0; w < warps; w++) total += red[w];
        if (lane == 0) red[0] = total;
    }
    __syncthreads();
    
    float r = 1.0f / sqrtf(red[0] / n + RMD_EPS);
    for (int i = tx; i < n; i += blockDim.x)
        out[i] = __float2half((float)x[i] * r * (float)w[i]);
}

// ── Test ──
int main() {
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        fprintf(stderr, "no HIP device available, skipping\n");
        return 77;
    }
    printf("=== CCA Attention GPU Test ===\n");
    std::mt19937 rng(42);
    auto rand = [&](int n, float s=1.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = (float)(int)rng() * (s / (float)0x7FFFFFFF);
        return v;
    };
    
    auto x_in = rand(H, 2.0f);
    CCAWeights w;
    w.wq = rand(QD*H, 0.1f); w.wk = rand(KD*H, 0.1f);
    w.wv1 = rand((KD/2)*H, 0.1f); w.wv2 = rand((KD/2)*H, 0.1f);
    w.wo = rand(H*QD, 0.1f); w.cdw = rand(QKV*DC, 1.0f);
    w.cdb = rand(QKV, 0.5f); w.cgw = rand(QKV*GC*DC, 1.0f);
    w.cgb = rand(QKV, 0.5f); w.k_scale = {11.625f, 2.890625f};
    w.norm_w = rand(H, 0.2f);
    for (auto& v : w.norm_w) v = 0.9f + v;
    
    std::vector<float> ref(H);
    cca_attn_ref(ref.data(), x_in.data(), w, 0);
    float ref_norm = 0;
    for (auto& v : ref) ref_norm += v*v;
    printf("Reference: norm=%.4f [0:8]:", sqrtf(ref_norm));
    for (int i = 0; i < 8; i++) printf(" %.4f", ref[i]);
    printf("\n");
    
    // GPU Norm test
    __half *d_x, *d_w, *d_out;
    HIP_OK(hipMalloc(&d_x, H*sizeof(__half)));
    HIP_OK(hipMalloc(&d_w, H*sizeof(__half)));
    HIP_OK(hipMalloc(&d_out, H*sizeof(__half)));
    
    std::vector<__half> hx(H), hw(H);
    for (int i = 0; i < H; i++) hx[i] = __float2half(x_in[i]);
    for (int i = 0; i < H; i++) hw[i] = __float2half(w.norm_w[i]);
    HIP_OK(hipMemcpy(d_x, hx.data(), H*sizeof(__half), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_w, hw.data(), H*sizeof(__half), hipMemcpyHostToDevice));
    
    int block = 256;
    norm_kernel<<<1, block>>>(d_x, d_w, d_out, H);
    HIP_OK(hipDeviceSynchronize());
    
    std::vector<__half> gpu_out(H);
    HIP_OK(hipMemcpy(gpu_out.data(), d_out, H*sizeof(__half), hipMemcpyDeviceToHost));
    
    // Compare RMSNorm output
    float max_diff = 0;
    float* normed_ref = new float[H];
    float ss = 0; for (int i = 0; i < H; i++) ss += x_in[i]*x_in[i];
    float rn = 1.0f / sqrtf(ss/H + RMD_EPS);
    for (int i = 0; i < H; i++) normed_ref[i] = x_in[i] * rn * w.norm_w[i];
    
    for (int i = 0; i < H; i++) {
        float diff = fabsf(__half2float(gpu_out[i]) - normed_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("RMSNorm GPU vs CPU: max_diff=%.6f %s\n", max_diff, max_diff < 0.1f ? "PASS" : "FAIL");
    
    // ── GPU Q projection test ──
    {
        __half *d_neg, *d_wq, *d_q;
        HIP_OK(hipMalloc(&d_neg, H*sizeof(__half)));  // normed input
        HIP_OK(hipMalloc(&d_wq, QD*H*sizeof(__half)));
        HIP_OK(hipMalloc(&d_q, QD*sizeof(__half)));
        HIP_OK(hipMemcpy(d_neg, gpu_out.data(), H*sizeof(__half), hipMemcpyHostToDevice));
        
        std::vector<__half> h_wq(QD*H);
        for (int i = 0; i < QD*H; i++) h_wq[i] = __float2half(w.wq[i]);
        HIP_OK(hipMemcpy(d_wq, h_wq.data(), QD*H*sizeof(__half), hipMemcpyHostToDevice));
        
        int block = 256;
        int grid = (QD + block - 1) / block;
        proj_kernel<<<grid, block, 0>>>(d_neg, d_wq, d_q, QD, 0, H);
        HIP_OK(hipDeviceSynchronize());
        
        std::vector<__half> gpu_q(QD);
        HIP_OK(hipMemcpy(gpu_q.data(), d_q, QD*sizeof(__half), hipMemcpyDeviceToHost));
        
        // Reference Q from CPU normed
        float* ref_q = new float[QD];
        matmul(ref_q, normed_ref, w.wq.data(), 1, QD, H);
        
        float q_max_diff = 0;
        for (int i = 0; i < QD; i++) {
            float diff = fabsf(__half2float(gpu_q[i]) - ref_q[i]);
            if (diff > q_max_diff) q_max_diff = diff;
        }
        printf("Q proj GPU vs CPU: max_diff=%.6f %s\n", q_max_diff, q_max_diff < 0.1f ? "PASS" : "FAIL");
        if (q_max_diff > max_diff) max_diff = q_max_diff;
        delete[] ref_q;
        HIP_OK(hipFree(d_neg)); HIP_OK(hipFree(d_wq)); HIP_OK(hipFree(d_q));
    }
    
    // ── GPU conv depthwise test ──
    {
        float* d_in; float* d_cdw; float* d_cdb; float* d_c1;
        HIP_OK(hipMalloc(&d_in, QKV*3*sizeof(float)));
        HIP_OK(hipMalloc(&d_cdw, QKV*DC*sizeof(float)));
        HIP_OK(hipMalloc(&d_cdb, QKV*sizeof(float)));
        HIP_OK(hipMalloc(&d_c1, QKV*2*sizeof(float)));
        
        // Pack Q+K into padded input
        std::vector<float> qk(QKV);
        float* ref_q = new float[QD];
        float* ref_k = new float[KD];
        matmul(ref_q, normed_ref, w.wq.data(), 1, QD, H);
        matmul(ref_k, normed_ref, w.wk.data(), 1, KD, H);
        memcpy(qk.data(), ref_q, QD*4);
        memcpy(qk.data()+QD, ref_k, KD*4);
        
        std::vector<float> pad(QKV*3, 0);
        for (int i = 0; i < QKV; i++) pad[i*3+2] = qk[i];
        HIP_OK(hipMemcpy(d_in, pad.data(), QKV*3*sizeof(float), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_cdw, w.cdw.data(), QKV*DC*sizeof(float), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_cdb, w.cdb.data(), QKV*sizeof(float), hipMemcpyHostToDevice));
        
        conv_dw_kernel<<<QKV, 256>>>(d_in, d_cdw, d_cdb, d_c1, QKV, 2, DC);
        HIP_OK(hipDeviceSynchronize());
        
        std::vector<float> gpu_c1(QKV*2);
        HIP_OK(hipMemcpy(gpu_c1.data(), d_c1, QKV*2*sizeof(float), hipMemcpyDeviceToHost));
        
        // CPU reference
        std::vector<float> ref_c1(QKV*2);
        for (int c = 0; c < QKV; c++)
            for (int l = 0; l < 2; l++) {
                float s = 0;
                for (int d = 0; d < DC; d++) s += pad[c*3+l+d] * w.cdw[c*DC+d];
                ref_c1[c*2+l] = s + w.cdb[c];
            }
        
        float cd_max_diff = 0;
        for (int i = 0; i < QKV*2; i++) {
            float diff = fabsf(gpu_c1[i] - ref_c1[i]);
            if (diff > cd_max_diff) cd_max_diff = diff;
        }
        printf("Conv depthwise GPU vs CPU: max_diff=%.6f %s\n", cd_max_diff, cd_max_diff < 0.1f ? "PASS" : "FAIL");
        if (cd_max_diff > max_diff) max_diff = cd_max_diff;
        
        delete[] ref_q; delete[] ref_k;
        HIP_OK(hipFree(d_in)); HIP_OK(hipFree(d_cdw)); HIP_OK(hipFree(d_cdb)); HIP_OK(hipFree(d_c1));
    }
    
    // ── GPU conv grouped test ──
    {
        float *d_c1, *d_cgw, *d_cgb, *d_c2;
        HIP_OK(hipMalloc(&d_c1, QKV*2*sizeof(float)));
        HIP_OK(hipMalloc(&d_cgw, QKV*GC*DC*sizeof(float)));
        HIP_OK(hipMalloc(&d_cgb, QKV*sizeof(float)));
        HIP_OK(hipMalloc(&d_c2, QKV*sizeof(float)));
        
        // Compute c1 from CPU reference
        float* ref_q = new float[QD]; float* ref_k = new float[KD];
        matmul(ref_q, normed_ref, w.wq.data(), 1, QD, H);
        matmul(ref_k, normed_ref, w.wk.data(), 1, KD, H);
        std::vector<float> qk(QKV);
        memcpy(qk.data(), ref_q, QD*4); memcpy(qk.data()+QD, ref_k, KD*4);
        std::vector<float> pad(QKV*3,0);
        for (int i = 0; i < QKV; i++) pad[i*3+2] = qk[i];
        std::vector<float> h_c1(QKV*2);
        for (int c = 0; c < QKV; c++)
            for (int l = 0; l < 2; l++) {
                float s = 0;
                for (int d = 0; d < DC; d++) s += pad[c*3+l+d] * w.cdw[c*DC+d];
                h_c1[c*2+l] = s + w.cdb[c];
            }
        
        HIP_OK(hipMemcpy(d_c1, h_c1.data(), QKV*2*sizeof(float), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_cgw, w.cgw.data(), QKV*GC*DC*sizeof(float), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_cgb, w.cgb.data(), QKV*sizeof(float), hipMemcpyHostToDevice));
        
        conv_grp_kernel<<<QKV, 256>>>(d_c1, d_cgw, d_cgb, d_c2, QKV, 1, NGRP, DC);
        HIP_OK(hipDeviceSynchronize());
        
        std::vector<float> gpu_c2(QKV);
        HIP_OK(hipMemcpy(gpu_c2.data(), d_c2, QKV*sizeof(float), hipMemcpyDeviceToHost));
        
        // CPU reference for grouped conv
        std::vector<float> ref_c2(QKV, 0);
        for (int g = 0; g < NGRP; g++)
            for (int co = 0; co < GC; co++) {
                int co2 = g*GC+co; float s = 0;
                for (int ci = 0; ci < GC; ci++)
                    for (int d = 0; d < DC; d++)
                        s += h_c1[(g*GC+ci)*2+d] * w.cgw[co2*GC*DC + ci*DC + d];
                ref_c2[co2] = s + w.cgb[co2];
            }
        
        float cg_max_diff = 0;
        for (int i = 0; i < QKV; i++) {
            float diff = fabsf(gpu_c2[i] - ref_c2[i]);
            if (diff > cg_max_diff) cg_max_diff = diff;
        }
        printf("Conv grouped GPU vs CPU: max_diff=%.6f %s\n", cg_max_diff, cg_max_diff < 0.1f ? "PASS" : "FAIL");
        if (cg_max_diff > max_diff) max_diff = cg_max_diff;
        
        delete[] ref_q; delete[] ref_k;
        HIP_OK(hipFree(d_c1)); HIP_OK(hipFree(d_cgw)); HIP_OK(hipFree(d_cgb)); HIP_OK(hipFree(d_c2));
    }
    
    // ── GPU L2 norm + RoPE test ──
    {
        // Prepare Q and K data from CPU reference (after conv + means)
        float* ref_q = new float[QD]; float* ref_k = new float[KD];
        matmul(ref_q, normed_ref, w.wq.data(), 1, QD, H);
        matmul(ref_k, normed_ref, w.wk.data(), 1, KD, H);
        
        // Conv c2
        std::vector<float> qk(QKV);
        memcpy(qk.data(), ref_q, QD*4); memcpy(qk.data()+QD, ref_k, KD*4);
        std::vector<float> pad(QKV*3,0);
        for (int i = 0; i < QKV; i++) pad[i*3+2] = qk[i];
        std::vector<float> c1(QKV*2), c2(QKV);
        for (int c = 0; c < QKV; c++)
            for (int l = 0; l < 2; l++) {
                float s = 0;
                for (int d = 0; d < DC; d++) s += pad[c*3+l+d] * w.cdw[c*DC+d];
                c1[c*2+l] = s + w.cdb[c];
            }
        for (int g = 0; g < NGRP; g++)
            for (int co = 0; co < GC; co++) {
                int co2 = g*GC+co; float s = 0;
                for (int ci = 0; ci < GC; ci++)
                    for (int d = 0; d < DC; d++)
                        s += c1[(g*GC+ci)*2+d] * w.cgw[co2*GC*DC + ci*DC + d];
                c2[co2] = s + w.cgb[co2];
            }
        
        // Add means
        std::vector<float> qc(QD), kc(KD);
        memcpy(qc.data(), c2.data(), QD*4); memcpy(kc.data(), c2.data()+QD, KD*4);
        for (int h = 0; h < NQ; h++)
            for (int d = 0; d < HD; d++)
                qc[h*HD+d] += 0.5f*ref_q[h*HD+d] + 0.5f*ref_k[(h/GQA)*HD+d];
        for (int kh = 0; kh < NKV; kh++) {
            float qm[HD] = {0};
            for (int g = 0; g < GQA; g++)
                for (int d = 0; d < HD; d++) qm[d] += ref_q[(kh*GQA+g)*HD+d];
            for (int d = 0; d < HD; d++) qm[d] /= GQA;
            for (int d = 0; d < HD; d++)
                kc[kh*HD+d] += 0.5f*qm[d] + 0.5f*ref_k[kh*HD+d];
        }
        
        // Copy to GPU (fp16)
        __half *d_q, *d_k;
        float *d_k_scale;
        HIP_OK(hipMalloc(&d_q, QD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_k, KD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_k_scale, NKV*sizeof(float)));
        
        std::vector<__half> h_q(QD), h_k(KD);
        for (int i = 0; i < QD; i++) h_q[i] = __float2half(qc[i]);
        for (int i = 0; i < KD; i++) h_k[i] = __float2half(kc[i]);
        HIP_OK(hipMemcpy(d_q, h_q.data(), QD*sizeof(__half), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_k, h_k.data(), KD*sizeof(__half), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_k_scale, w.k_scale.data(), NKV*sizeof(float), hipMemcpyHostToDevice));
        
        // L2 norm Q
        l2norm_q_kernel<<<NQ, 64>>>(d_q, NQ, HD);
        // L2 norm K
        l2norm_k_kernel<<<NKV, 64>>>(d_k, d_k_scale, NKV, HD);
        HIP_OK(hipDeviceSynchronize());
        
        // Read back and compare
        std::vector<__half> gpu_q(QD), gpu_k(KD);
        HIP_OK(hipMemcpy(gpu_q.data(), d_q, QD*sizeof(__half), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(gpu_k.data(), d_k, KD*sizeof(__half), hipMemcpyDeviceToHost));
        
        // CPU reference for L2 norm
        std::vector<float> ref_qc(QD), ref_kc(KD);
        memcpy(ref_qc.data(), qc.data(), QD*4); memcpy(ref_kc.data(), kc.data(), KD*4);
        for (int h = 0; h < NQ; h++) {
            float s = 0; for (int d = 0; d < HD; d++) s += ref_qc[h*HD+d]*ref_qc[h*HD+d];
            float r = sqrtf(HD) * powf(s+1e-12f, -0.5f);
            for (int d = 0; d < HD; d++) ref_qc[h*HD+d] *= r;
        }
        for (int h = 0; h < NKV; h++) {
            float s = 0; for (int d = 0; d < HD; d++) s += ref_kc[h*HD+d]*ref_kc[h*HD+d];
            float r = sqrtf(HD) * powf(s+1e-12f, -0.5f) * w.k_scale[h];
            for (int d = 0; d < HD; d++) ref_kc[h*HD+d] *= r;
        }
        
        float ln_max_diff = 0;
        for (int i = 0; i < QD; i++) {
            float diff = fabsf(__half2float(gpu_q[i]) - ref_qc[i]);
            if (diff > ln_max_diff) ln_max_diff = diff;
        }
        for (int i = 0; i < KD; i++) {
            float diff = fabsf(__half2float(gpu_k[i]) - ref_kc[i]);
            if (diff > ln_max_diff) ln_max_diff = diff;
        }
        printf("L2 norm Q+K GPU vs CPU: max_diff=%.6f %s\n", ln_max_diff, ln_max_diff < 0.1f ? "PASS" : "FAIL");
        if (ln_max_diff > max_diff) max_diff = ln_max_diff;
        
        // RoPE
        int pos = 0;
        rope_kernel<<<1, 64>>>(d_q, d_k, NQ, NKV, HD, NROT, pos);
        HIP_OK(hipDeviceSynchronize());
        
        HIP_OK(hipMemcpy(gpu_q.data(), d_q, QD*sizeof(__half), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(gpu_k.data(), d_k, KD*sizeof(__half), hipMemcpyDeviceToHost));
        
        // CPU RoPE
        for (int h = 0; h < NQ; h++)
            for (int d = 0; d < NROT; d+=2) {
                float th = powf(500000.0f, -2.0f*d/HD);
                float cv = cosf(pos*th), sv = sinf(pos*th);
                float q0=ref_qc[h*HD+d], q1=ref_qc[h*HD+d+1];
                ref_qc[h*HD+d]=q0*cv-q1*sv; ref_qc[h*HD+d+1]=q0*sv+q1*cv;
            }
        for (int h = 0; h < NKV; h++)
            for (int d = 0; d < NROT; d+=2) {
                float th = powf(500000.0f, -2.0f*d/HD);
                float cv = cosf(pos*th), sv = sinf(pos*th);
                float k0=ref_kc[h*HD+d], k1=ref_kc[h*HD+d+1];
                ref_kc[h*HD+d]=k0*cv-k1*sv; ref_kc[h*HD+d+1]=k0*sv+k1*cv;
            }
        
        float rp_max_diff = 0;
        for (int i = 0; i < QD; i++) {
            float diff = fabsf(__half2float(gpu_q[i]) - ref_qc[i]);
            if (diff > rp_max_diff) rp_max_diff = diff;
        }
        for (int i = 0; i < KD; i++) {
            float diff = fabsf(__half2float(gpu_k[i]) - ref_kc[i]);
            if (diff > rp_max_diff) rp_max_diff = diff;
        }
        printf("RoPE GPU vs CPU: max_diff=%.6f %s\n", rp_max_diff, rp_max_diff < 0.1f ? "PASS" : "FAIL");
        if (rp_max_diff > max_diff) max_diff = rp_max_diff;
        
        delete[] ref_q; delete[] ref_k;
        HIP_OK(hipFree(d_q)); HIP_OK(hipFree(d_k)); HIP_OK(hipFree(d_k_scale));
    }
    
    // ── GPU GQA attention + output projection test ──
    {
        // Prepare Q, K, V from CPU reference (after all CCA processing)
        float* ref_q = new float[QD]; float* ref_k = new float[KD];
        matmul(ref_q, normed_ref, w.wq.data(), 1, QD, H);
        matmul(ref_k, normed_ref, w.wk.data(), 1, KD, H);
        
        std::vector<float> qk(QKV);
        memcpy(qk.data(), ref_q, QD*4); memcpy(qk.data()+QD, ref_k, KD*4);
        std::vector<float> pad(QKV*3,0);
        for (int i = 0; i < QKV; i++) pad[i*3+2] = qk[i];
        std::vector<float> c1(QKV*2), c2(QKV);
        for (int c = 0; c < QKV; c++)
            for (int l = 0; l < 2; l++) {
                float s = 0;
                for (int d = 0; d < DC; d++) s += pad[c*3+l+d] * w.cdw[c*DC+d];
                c1[c*2+l] = s + w.cdb[c];
            }
        for (int g = 0; g < NGRP; g++)
            for (int co = 0; co < GC; co++) {
                int co2 = g*GC+co; float s = 0;
                for (int ci = 0; ci < GC; ci++)
                    for (int d = 0; d < DC; d++)
                        s += c1[(g*GC+ci)*2+d] * w.cgw[co2*GC*DC + ci*DC + d];
                c2[co2] = s + w.cgb[co2];
            }
        
        // Means + L2 norm + RoPE
        std::vector<float> qc(QD), kc(KD);
        memcpy(qc.data(), c2.data(), QD*4); memcpy(kc.data(), c2.data()+QD, KD*4);
        for (int h = 0; h < NQ; h++)
            for (int d = 0; d < HD; d++)
                qc[h*HD+d] += 0.5f*ref_q[h*HD+d] + 0.5f*ref_k[(h/GQA)*HD+d];
        for (int kh = 0; kh < NKV; kh++) {
            float qm[HD] = {0};
            for (int g = 0; g < GQA; g++)
                for (int d = 0; d < HD; d++) qm[d] += ref_q[(kh*GQA+g)*HD+d];
            for (int d = 0; d < HD; d++) qm[d] /= GQA;
            for (int d = 0; d < HD; d++)
                kc[kh*HD+d] += 0.5f*qm[d] + 0.5f*ref_k[kh*HD+d];
        }
        
        // V projection
        float* v = new float[KD];
        float* v1 = new float[KD/2];
        matmul(v1, normed_ref, w.wv1.data(), 1, KD/2, H);
        memcpy(v, v1, (KD/2)*4);
        memset(v+KD/2, 0, (KD/2)*4);
        
        // L2 norm on CPU for reference
        for (int h = 0; h < NQ; h++) {
            float s = 0; for (int d = 0; d < HD; d++) s += qc[h*HD+d]*qc[h*HD+d];
            float r = sqrtf(HD) * powf(s+1e-12f, -0.5f);
            for (int d = 0; d < HD; d++) qc[h*HD+d] *= r;
        }
        for (int h = 0; h < NKV; h++) {
            float s = 0; for (int d = 0; d < HD; d++) s += kc[h*HD+d]*kc[h*HD+d];
            float r = sqrtf(HD) * powf(s+1e-12f, -0.5f) * w.k_scale[h];
            for (int d = 0; d < HD; d++) kc[h*HD+d] *= r;
        }
        // RoPE (pos=0)
        int pos = 0;
        for (int h = 0; h < NQ; h++)
            for (int d = 0; d < NROT; d+=2) {
                float th = powf(500000.0f,-2.0f*d/HD);
                float cv=cosf(pos*th), sv=sinf(pos*th);
                float q0=qc[h*HD+d], q1=qc[h*HD+d+1];
                qc[h*HD+d]=q0*cv-q1*sv; qc[h*HD+d+1]=q0*sv+q1*cv;
            }
        for (int h = 0; h < NKV; h++)
            for (int d = 0; d < NROT; d+=2) {
                float th = powf(500000.0f,-2.0f*d/HD);
                float cv=cosf(pos*th), sv=sinf(pos*th);
                float k0=kc[h*HD+d], k1=kc[h*HD+d+1];
                kc[h*HD+d]=k0*cv-k1*sv; kc[h*HD+d+1]=k0*sv+k1*cv;
            }
        
        // CPU reference: GQA + output proj
        float* cpu_out = new float[H];
        float max_s = -1e10f;
        for (int h = 0; h < NQ; h++) {
            int kh = h/GQA;
            float s = 0; for (int d=0; d<HD; d++) s+=qc[h*HD+d]*kc[kh*HD+d];
            s *= 1.0f/sqrtf(HD);
            if (s > max_s) max_s = s;
        }
        float sum_exp = 0;
        float scores[NQ];
        for (int h = 0; h < NQ; h++) {
            int kh = h/GQA;
            scores[h] = 0;
            for (int d=0; d<HD; d++) scores[h] += qc[h*HD+d]*kc[kh*HD+d];
            scores[h] *= 1.0f/sqrtf(HD);
            sum_exp += expf(scores[h] - max_s);
        }
        for (int out_idx = 0; out_idx < H; out_idx++) {
            float val = 0;
            for (int h = 0; h < NQ; h++) {
                int kh = h/GQA;
                float prob = expf(scores[h] - max_s) / sum_exp;
                for (int d = 0; d < HD; d++)
                    val += prob * v[kh*HD+d] * w.wo[out_idx*QD + h*HD + d];
            }
            cpu_out[out_idx] = val;
        }
        
        // GPU
        __half *d_q, *d_k, *d_v, *d_wo, *d_out;
        HIP_OK(hipMalloc(&d_q, QD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_k, KD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_v, KD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_wo, H*QD*sizeof(__half)));
        HIP_OK(hipMalloc(&d_out, H*sizeof(__half)));
        
        std::vector<__half> h_q(QD), h_k(KD), h_v(KD), h_wo(H*QD);
        for (int i = 0; i < QD; i++) h_q[i] = __float2half(qc[i]);
        for (int i = 0; i < KD; i++) h_k[i] = __float2half(kc[i]);
        for (int i = 0; i < KD; i++) h_v[i] = __float2half(v[i]);
        for (int i = 0; i < H*QD; i++) h_wo[i] = __float2half(w.wo[i]);
        HIP_OK(hipMemcpy(d_q, h_q.data(), QD*sizeof(__half), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_k, h_k.data(), KD*sizeof(__half), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_v, h_v.data(), KD*sizeof(__half), hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(d_wo, h_wo.data(), H*QD*sizeof(__half), hipMemcpyHostToDevice));
        
        size_t shared_mem = 0; // no shared needed for this simple kernel
        gqa_kernel<<<H, 1, shared_mem>>>(d_q, d_k, d_v, d_wo, d_out, NQ, NKV, HD);
        HIP_OK(hipDeviceSynchronize());
        
        std::vector<__half> gpu_out(H);
        HIP_OK(hipMemcpy(gpu_out.data(), d_out, H*sizeof(__half), hipMemcpyDeviceToHost));
        
        float gqa_max_diff = 0;
        for (int i = 0; i < H; i++) {
            float diff = fabsf(__half2float(gpu_out[i]) - cpu_out[i]);
            if (diff > gqa_max_diff) gqa_max_diff = diff;
        }
        printf("GQA+output GPU vs CPU: max_diff=%.6f %s\n", gqa_max_diff, gqa_max_diff < 1.0f ? "PASS" : "FAIL");
        if (gqa_max_diff > max_diff) max_diff = gqa_max_diff;
        
        delete[] ref_q; delete[] ref_k; delete[] v; delete[] v1; delete[] cpu_out;
        HIP_OK(hipFree(d_q)); HIP_OK(hipFree(d_k)); HIP_OK(hipFree(d_v)); HIP_OK(hipFree(d_wo)); HIP_OK(hipFree(d_out));
    }
    
    printf("\nDone. Max error: %.6f\n", max_diff);
    delete[] normed_ref;
    HIP_OK(hipFree(d_x)); HIP_OK(hipFree(d_w)); HIP_OK(hipFree(d_out));
    return max_diff < 0.5f ? 0 : 1;
}
