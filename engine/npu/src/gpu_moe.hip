// GPU MoE FFN wrapper — compiled separately with hipcc.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// Kernel files. They #define H=2048, N_EXP=16, etc. — undef after.
#define N_EXP 8
#include "kernels/zaya_gpu_router.hip"
#include "kernels/zaya_moe_expert_ffn.hip"
#undef H
#undef NQ
#undef NKV
#undef HD
#undef QD
#undef KD
#undef QKV
#undef N_LAYERS
#undef VOCAB
#undef N_EXP
#undef N_EXP_T
#undef N_FF
#undef RTR_H

__global__ void silu_mul_k(__half* out, const __half* g, const __half* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float)g[i]; out[i] = __float2half((v / (1.0f + expf(-v))) * (float)u[i]);
}

struct GpuMoeCtx {
    hipStream_t st;
    __half *d_hs, *d_tmp, *d_ao, *d_moe_out;
    float *d_prev_rs, *d_expert_wt;
    int *d_expert_idx;
    __half *d_embed;
    int n_layers;
    struct LayerW { __half *gu, *dn; float *gdw, *gdb, *rfn, *rf1, *rf1b, *rf2, *rf2b, *rout, *bb; int has_eda; float eda_scale[4]; };
    std::vector<LayerW> layers;

    bool init(int H, int num_layers) {
        n_layers = num_layers;
        HIP_CHECK(hipStreamCreate(&st));
        HIP_CHECK(hipMalloc(&d_hs, (size_t)H * 2));
        int tmp_sz = (H > 4096 ? H : 4096) * 2;
        HIP_CHECK(hipMalloc(&d_tmp, (size_t)tmp_sz));
        HIP_CHECK(hipMalloc(&d_ao, (size_t)H * 2));
        HIP_CHECK(hipMalloc(&d_moe_out, (size_t)H * 2));
        HIP_CHECK(hipMalloc(&d_prev_rs, (size_t)n_layers * 256 * 4));
        HIP_CHECK(hipMemset(d_prev_rs, 0, (size_t)n_layers * 256 * 4));
        HIP_CHECK(hipMalloc(&d_expert_idx, 4));
        HIP_CHECK(hipMalloc(&d_expert_wt, 4));
        layers.resize(n_layers);
        return true;
    }

    void load_layer(int il, int H, const float* gu_cpu, const float* dn_cpu) {
        auto& l = layers[il];
        int gu_sz = 2 * 2048 * H, dn_sz = 2048 * H;
        std::vector<__half> gu_half(gu_sz), dn_half(dn_sz);
        for (int i = 0; i < gu_sz; i++) gu_half[i] = __float2half(gu_cpu[i]);
        for (int i = 0; i < dn_sz; i++) dn_half[i] = __float2half(dn_cpu[i]);
        HIP_CHECK(hipMalloc(&l.gu, (size_t)gu_sz * 2));
        HIP_CHECK(hipMalloc(&l.dn, (size_t)dn_sz * 2));
        HIP_CHECK(hipMemcpy(l.gu, gu_half.data(), (size_t)gu_sz * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(l.dn, dn_half.data(), (size_t)dn_sz * 2, hipMemcpyHostToDevice));
        // Router weights (stub — zero-initialized for testing)
        auto mf = [&](float*& dp, int n) { HIP_CHECK(hipMalloc(&dp, (size_t)n * 4)); HIP_CHECK(hipMemset(dp, 0, (size_t)n * 4)); };
        mf(l.gdw, H * 256); mf(l.gdb, 256); mf(l.rfn, 256);
        mf(l.rf1, 256 * 256); mf(l.rf1b, 256); mf(l.rf2, 256 * 256); mf(l.rf2b, 256);
        mf(l.rout, 256 * 8); mf(l.bb, 8);
        l.has_eda = 0; memset(l.eda_scale, 0, 4 * 4);
    }

    void run_moe(int il, int H, const float* hidden_f32, float* output_f32) {
        auto& l = layers[il];
        std::vector<__half> hidden_half(H);
        for (int i = 0; i < H; i++) hidden_half[i] = __float2half(hidden_f32[i]);
        HIP_CHECK(hipMemcpy(d_hs, hidden_half.data(), (size_t)H * 2, hipMemcpyHostToDevice));
        int gb = (4096 + 15) / 16, sb = (2048 + 255) / 256, db = (H + 15) / 16;
        eda_router_gpu_kernel<<<1, 256, 0, st>>>(d_hs, d_prev_rs + (size_t)il * 256,
            0, 0.0f, l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
            l.rf2, l.rf2b, l.rout, l.bb, d_prev_rs + (size_t)il * 256,
            d_expert_idx, d_expert_wt);
        wmma_gateup_kernel<<<gb, 128, 0, st>>>(d_tmp, d_hs, l.gu, d_expert_idx);
        silu_mul_k<<<sb, 256, 0, st>>>(d_ao, d_tmp, d_tmp + 2048, 2048);
        wmma_down_kernel<<<db, 128, 0, st>>>(d_moe_out, d_ao, l.dn, d_expert_idx);
        std::vector<__half> out_half(H);
        HIP_CHECK(hipMemcpy(out_half.data(), d_moe_out, (size_t)H * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipStreamSynchronize(st));
        for (int i = 0; i < H; i++) output_f32[i] = (float)out_half[i];
    }
};

extern "C" {
    GpuMoeCtx* gpu_moe_create(int H, int num_layers) { auto* ctx = new GpuMoeCtx(); ctx->init(H, num_layers); return ctx; }
    void gpu_moe_destroy(GpuMoeCtx* ctx) { delete ctx; }
    void gpu_moe_load(GpuMoeCtx* ctx, int il, int H, const float* gu, const float* dn) { ctx->load_layer(il, H, gu, dn); }
    void gpu_moe_run(GpuMoeCtx* ctx, int il, int H, const float* in, float* out) { ctx->run_moe(il, H, in, out); }
}
