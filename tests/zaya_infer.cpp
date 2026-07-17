// zaya_infer.cpp — Standalone Zaya inference on GPU
// Integrates CCA attention + TQ1 MoE GPU kernels
//
// Build:
//   /opt/rocm-7.2.4/bin/hipcc -O3 --offload-arch=gfx1151 \
//     zaya_infer.cpp ../kernels/zaya_moe_ternary_gemv.hip \
//     -o zaya_infer -L/opt/rocm-7.2.4/lib -lamdhip64
//
// Weights: run 'python3 export_weights.py' first
//
// export_weights.py:
//   Reads safetensors, writes .bin float32 files to /tmp/zaya_weights/

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <fstream>
#include <cassert>

#define HIP_OK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__); abort();}} while(0)

// ── Config ──
constexpr int H = 2048, NQ = 8, NKV = 2, HD = 128;
constexpr int GQA = NQ / NKV, QD = NQ * HD, KD = NKV * HD, QKV = QD + KD;
constexpr int DC = 2, NGRP = 10, GC = QKV / NGRP, NROT = 64;
constexpr int N_EXP = 16, TOP_K = 1, N_FF = 2048, N_FF_EXP = 256;
constexpr int N_LAYERS = 40;
constexpr int VOCAB = 262272;
constexpr float RMD_EPS = 1e-5f;
constexpr int WARP_SIZE = 32;
constexpr float FP16_MAX = 65504.0f;

// ── Weight store ──
struct Tensor { std::vector<float> data; std::vector<int> shape; };
struct Weights {
    std::unordered_map<std::string, Tensor> tensors;
    
    void load_dir(const std::string& dir) {
        char path[512]; snprintf(path, 512, "%s/manifest.json", dir.c_str());
        FILE* f = fopen(path, "r");
        if (!f) { printf("No manifest at %s\n", path); return; }
        fclose(f);
        
        // Load .bin files matching known tensor names
        // This is a simple approach - load files we know we need
        printf("Loading weights from %s...\n", dir.c_str());
    }
    
    void add(const std::string& name, const float* data, int n) {
        tensors[name].data.assign(data, data + n);
    }
    
    const float* get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) { fprintf(stderr,"Missing: %s\n",name.c_str()); exit(1); }
        return it->second.data.data();
    }
    
    bool has(const std::string& name) const { return tensors.find(name) != tensors.end(); }
};

static std::string L(int il) { return std::to_string(il); }

// ── Forward declarations of GPU kernels ──
// (Included from the test file - these need to be in a shared header)

// ── CCA Attention kernels (from test_cca_attn.cpp) ──
__global__ void norm_kernel(const __half* x, const __half* w, __half* out, int n) {
    const int warps = blockDim.x / WARP_SIZE;
    __shared__ float red[32];
    int tx = threadIdx.x;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;
    float ss = 0;
    for (int i = tx; i < n; i += blockDim.x) ss += (float)x[i] * (float)x[i];
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        ss = (lane < (blockDim.x/WARP_SIZE)) ? red[lane] : 0;
        for (int o = WARP_SIZE/2; o > 0; o >>= 1) ss += __shfl_xor(ss, o);
        if (lane == 0) red[0] = ss;
    }
    __syncthreads();
    float r = 1.0f / sqrtf(red[0] / n + RMD_EPS);
    for (int i = tx; i < n; i += blockDim.x)
        out[i] = __float2half((float)x[i] * r * (float)w[i]);
}

__global__ void proj_kernel(const __half* in, const __half* w, __half* out, int M, int K) {
    int tx = threadIdx.x;
    int elem = blockIdx.x * blockDim.x + tx;
    if (elem >= M) return;
    float sum = 0;
    for (int k = 0; k < K; k++) sum += (float)in[k] * (float)w[k * M + elem];
    out[elem] = __float2half(sum);
}

__global__ void cca_attn_full_kernel(
    const __half* x,          // [H] normed input
    const __half* wq,         // [QD * H]
    const __half* wk,         // [KD * H]
    const __half* wv1,        // [(KD/2) * H]
    const __half* wo,         // [H * QD]
    const float* cdw,         // [QKV * DC] conv depthwise weight
    const float* cdb,         // [QKV]
    const float* cgw,         // [QKV * GC * DC] conv grouped weight
    const float* cgb,         // [QKV]
    const float* k_scale,     // [NKV]
    __half* attn_out,         // [H] output
    int pos)                  // RoPE position
{
    // Shared memory for intermediate values
    extern __shared__ float shm[];
    float* qk = shm;          // [QKV]
    float* qp = qk + QKV;     // [QKV]
    float* pad = qp + QKV;    // [QKV * 3]
    // Use 32KB shared - should be enough for ~40KB total
    
    int tx = threadIdx.x;
    int warp = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;
    
    // 1. Q, K projections
    // Each thread computes one output element
    // Q proj
    for (int i = tx; i < QD; i += blockDim.x) {
        float s = 0;
        for (int k = 0; k < H; k++) s += (float)x[k] * (float)wq[k * QD + i];
        qk[i] = s;
    }
    // K proj
    for (int i = tx; i < KD; i += blockDim.x) {
        float s = 0;
        for (int k = 0; k < H; k++) s += (float)x[k] * (float)wk[k * KD + i];
        qk[QD + i] = s;
    }
    __syncthreads();
    
    // Save pre-conv values for means
    if (tx < QKV) qp[tx] = qk[tx];
    __syncthreads();
    
    // 2. Conv prep: pad left by 2
    if (tx < QKV) {
        pad[tx * 3 + 0] = 0; pad[tx * 3 + 1] = 0; pad[tx * 3 + 2] = qk[tx];
    } else if (tx < QKV * 3) {
        pad[tx] = 0;
    }
    __syncthreads();
    
    // 3. Depthwise conv: [QKV, 3] -> [QKV, 2]
    // Each thread processes one channel
    if (tx < QKV) {
        for (int l = 0; l < 2; l++) {
            float s = 0;
            for (int d = 0; d < DC; d++) s += pad[tx*3 + l+d] * cdw[tx*DC + d];
            qk[tx] = s + cdb[tx];  // reuse qk for c1 (but we need c1 and c2)
            __syncthreads();
        }
        // Actually we need separate buffers - let's use qp for c1
    }
    // ... This gets complex with shared memory management
}
