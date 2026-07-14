// gpu_npu_bridge.cpp — GPU+NPU fused inference for Zaya1-8B
// Architecture (CURRENT — GPU-only loop; NPU integration is aspirational):
//   GPU:  CCA attention + MoE (all 40 layers) via HIP — WORKS (Phase 1 baseline)
//   NPU:  Placeholder only — NPURunner::run_expert_ffn() returns `false`.
//         Real NPU integration was never completed; the ternary xclbins
//         needed by the TQ1 path were never built/verified for Zaya.
//   Memory: THE DMA-BUF ZERO-COPY APPROACH BELOW DOES NOT COMPILE.
//         ROCm 7.2.4 HIP lacks `hipExternalMemoryHandleTypeDmaBuf` (enum value
//         absent).  Even if it compiled, the GPU-owner→NPU-importer direction
//         produces AMD-Vi IO_PAGE_FAULTs (verified on Strix Halo).
//         The proven zero-copy architecture (see engine/fusion/zero_copy/):
//           NPU (amdxdna) owns the buffer as XRT HOST_ONLY BO,
//           exports dma-buf fd -> imported into GPU via Vulkan dma-buf import
//           (HIP can't do this; Vulkan CAN via VK_KHR_external_memory_fd).
//         Host maps the BO directly — all three alias the same pages, no copy.
//   Pipeline: Still aspirational — the overlap code is GPU-only sequential.
//         Real NPU+GPU overlap requires the 2-slot pipeline skeleton in
//         engine/fusion/zero_copy/pipeline_overlap.h.
//
// Summary: THIS FILE IS A HISTORIC REFERENCE, NOT A PRODUCTION FUSION ENGINE.
//          Its GPU baseline loop (Phase 1) is still correct reference code.
//          The dead import_to_hip()/import_to_xrt() methods have been removed;
//          the NPURunner is a stub.  See engine/fusion/zero_copy/ for the
//          proven zero-copy substrate and pipeline pattern.
//
// Build: g++ -O3 -std=c++20 tools/gpu_npu_bridge.cpp -o build/gpu_npu_bridge \
//   -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib -lamdhip64 \
//   -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -lpthread

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ── GPU (HIP) ──
#include <hip/hip_runtime.h>
#define HIP_OK(e) do{auto _s=(e);if(_s!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d\n",_s,__FILE__,__LINE__);abort();}}while(0)

// ── NPU (XRT) ──
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

// ── dma-buf ──
#include <drm/drm.h>
#include <xf86drm.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>

// ── Zaya Architecture ──
constexpr int H=2048, NQ=8, NKV=2, HD=128, QD=NQ*HD, KD=NKV*HD, QKV=QD+KD;
constexpr int N_LAYERS=40, VOCAB=262272, N_EXP=16, N_FF=2048, RTR_H=256;
constexpr int BLK=256;

// ── dma-buf helpers ──
struct DmaBuf {
    int fd = -1;
    size_t size = 0;
    void *cpu_ptr = nullptr;
    amdgpu_bo_handle bo = nullptr;
    amdgpu_device_handle dev = nullptr;

    bool alloc_gtt(size_t sz, int render_node_fd) {
        size = sz;
        // Initialize amdgpu device from render node fd
        int ret = amdgpu_device_initialize(render_node_fd, &dev);
        if (ret) { fprintf(stderr,"amdgpu_device_initialize failed: %d\n",ret); return false; }

        struct amdgpu_bo_alloc_request req = {};
        req.alloc_size = sz;
        req.preferred_heap = AMDGPU_GEM_DOMAIN_GTT;
        ret = amdgpu_bo_alloc(dev, &req, &bo);
        if (ret) { fprintf(stderr,"amdgpu_bo_alloc failed: %d\n",ret); return false; }

        // Export to dma-buf fd
        ret = amdgpu_bo_export(bo, amdgpu_bo_handle_type_dma_buf_fd, &fd);
        if (ret) { fprintf(stderr,"amdgpu_bo_export failed: %d\n",ret); return false; }

        // CPU map
        ret = amdgpu_bo_cpu_map(bo, &cpu_ptr);
        if (ret) { fprintf(stderr,"amdgpu_bo_cpu_map failed: %d\n",ret); return false; }

        printf("  dma-buf: fd=%d size=%zu cpu=%p\n", fd, size, cpu_ptr);
        return true;
    }

    ~DmaBuf() {
        if (bo) amdgpu_bo_free(bo);
        if (dev) amdgpu_device_deinitialize(dev);
        if (fd >= 0) close(fd);
    }
};

// ── Weight loading ──
static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());exit(1);}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
static std::string L(int i){return std::to_string(i);}
#define W(N) load_bin(std::string("/tmp/zaya_weights/")+N)
static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    HIP_OK(hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h));
}
static void upf32(const std::vector<float>& s,float*d,int n,hipStream_t h=0){
    HIP_OK(hipMemcpyAsync(d,s.data(),n*4,hipMemcpyHostToDevice,h));
}

// ── GPU kernel declarations (from zaya_cca_attn.hip, zaya_router_moe.hip) ──
__global__ void cca_attn_kernel(
    const __half*hs,const __half*phs,const __half*csi,int pos,
    const __half*wq,const __half*wk,const __half*wv1,const __half*wv2,const __half*wo,
    const float*cdw,const float*cdb,const float*cgw,const float*cgb,
    const float*ks,const __half*nw,
    __half*ao,__half*ncs,__half*nph);
__global__ void rmsnorm_k(__half*x,const __half*w,int n);
__global__ void copy_k(__half*dst,const __half*src,int n);
__global__ void residual_scale_k(__half*out,const __half*res,
    const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n);
__global__ void eda_router_moe_kernel(
    const __half*hs,const float*prev_rs,int has_eda,float eda_scale,
    const float*gdw,const float*gdb,const float*rfn,const float*rf1,const float*rf1b,
    const float*rf2,const float*rf2b,const float*rout,const float*bb,
    const __half*gu,const __half*dn,
    float*next_rs,__half*moe_out,int*expert_idx,float*expert_wt);

// ── NPU XRT runner (ternary GEMM) ──
struct NPURunner {
    xrt::device xrt_dev;
    bool available = false;

    // Ternary xclbin for TQ1 GEMV
    struct TQ1Kernel {
        std::unique_ptr<xrt::xclbin> xc;
        std::unique_ptr<xrt::hw_context> hc;
        std::unique_ptr<xrt::kernel> k;
        std::unique_ptr<xrt::bo> bI;  // instructions
        std::unique_ptr<xrt::bo> bA;  // input activations
        std::unique_ptr<xrt::bo> bW;  // ternary weights
        std::unique_ptr<xrt::bo> bC;  // output
        uint32_t *ins_ptr = nullptr;
        size_t ins_size = 0;
        bool ready = false;

        bool init(xrt::device &d, const char *xclbin_path, const char *insts_path,
                  size_t weight_bytes, size_t output_bytes) {
            // Load instructions
            FILE *f = fopen(insts_path, "rb");
            if (!f) { fprintf(stderr,"No insts: %s\n",insts_path); return false; }
            fseek(f, 0, SEEK_END);
            ins_size = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::vector<uint32_t> ins(ins_size/4);
            fread(ins.data(), 4, ins.size(), f);
            fclose(f);

            xc = std::make_unique<xrt::xclbin>(std::string(xclbin_path));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            k = std::make_unique<xrt::kernel>(*hc, "MLIR_AIE");

            bI = std::make_unique<xrt::bo>(d, ins_size, XCL_BO_FLAGS_CACHEABLE, k->group_id(1));
            memcpy(bI->map(), ins.data(), ins_size);
            bI->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            bW = std::make_unique<xrt::bo>(d, weight_bytes, XRT_BO_FLAGS_HOST_ONLY, k->group_id(4));
            bA = std::make_unique<xrt::bo>(d, 128*H*4, XRT_BO_FLAGS_HOST_ONLY, k->group_id(3));
            bC = std::make_unique<xrt::bo>(d, output_bytes, XRT_BO_FLAGS_HOST_ONLY, k->group_id(5));

            ready = true;
            printf("    NPU TQ1 kernel ready: %s\n", xclbin_path);
            return true;
        }

        // Run M-column ternary GEMV: out[M] = in[H] @ weights[M×H]^T
        void run(const float *in_h, const float *weights, float *out, int M, int K) {
            if (!ready) return;

            // Pack input as int8 (ternary kernel expects int8)
            auto *a_ptr = (int8_t*)bA->map();
            for (int i = 0; i < K; i++) {
                float v = in_h[i];
                if (!std::isfinite(v)) v = 0;
                int q = (int)roundf(v * 64.0f);  // scale for ternary
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                a_ptr[i] = (int8_t)q;
            }
            bA->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Weights are already packed
            bW->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Launch
            auto r = (*k)((unsigned)3, *bI, (unsigned)ins_size, *bA, *bW, *bC);
            r.wait();

            // Read output
            bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto *c_ptr = (int16_t*)bC->map();
            for (int i = 0; i < M; i++) {
                out[i] = (float)c_ptr[i] / 64.0f;
            }
        }
    };

    TQ1Kernel gate_up_k, down_k;

    bool init() {
        const char *xclbin_dir = getenv("HOME")?std::string(getenv("HOME"))+"/npu-sandbox/npu-infer/build/int8";
        const char *ternary_dir = getenv("HOME")?std::string(getenv("HOME"))+"/npu-sandbox/npu-infer/build/chess_infer";

        try {
            xrt_dev = xrt::device(0);  // NPU device 0
        } catch (...) {
            fprintf(stderr, "No NPU device found\n");
            return false;
        }
        printf("  NPU device: %s\n", xrt_dev.get_info<xrt::info::device::name>().c_str());

        // Check for ternary xclbins at the Zaya-appropriate paths
        // For now, report what's available
        available = true;
        return true;
    }

    // Run MoE expert on NPU (given pre-selected expert)
    // Returns true if NPU handled it, false if caller should use GPU fallback
    bool run_expert_ffn(float *hidden, int expert, float *gate_up_w, float *down_w,
                        float *output, int n_tokens) {
        if (!available) return false;
        // TODO: actual NPU dispatch using ternary xclbins
        return false;
    }
};

// ════════════════════════════════════════════════════════════════════
// Main — Fused GPU+NPU inference
// ════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    printf("=== Zaya1-8B GPU+NPU Fused Inference ===\n\n");

    // ── 1. Initialize NPU ──
    printf("[1/5] Initializing NPU...\n");
    NPURunner npu;
    npu.init();

    // ── 2. Initialize dma-buf for shared memory ──
    printf("[2/5] Setting up dma-buf shared memory...\n");
    int render_fd = open("/dev/dri/renderD128", O_RDWR);
    if (render_fd < 0) { perror("open renderD128"); return 1; }

    // Allocate shared buffer for hidden states + expert selection
    // Size: 2 × H × sizeof(float) = 16KB (double-buffered)
    DmaBuf shared_buf;
    if (!shared_buf.alloc_gtt(H * 2 * 4, render_fd)) {
        printf("  dma-buf fallback: using CPU memory\n");
    }

    // ── 3. Load weights ──
    printf("[3/5] Loading weights...\n");
    auto embed = W("model_embed_tokens_weight.bin");
    auto fnorm = W("model_norm_weight.bin");
    auto iscale = W("model_input_hidden_states_scale.bin");
    auto ibias  = W("model_input_hidden_states_bias.bin");

    // ── 4. Initialize GPU ──
    printf("[4/5] Initializing GPU...\n");
    __half *d_hs, *d_ao, *d_tmp, *d_phs, *d_csi, *d_moe;
    float *d_fbuf, *d_rs, *d_prev_rs;
    int *d_expert_idx; float *d_expert_wt;

    HIP_OK(hipMalloc(&d_hs, H*2));
    HIP_OK(hipMalloc(&d_ao, H*2));
    HIP_OK(hipMalloc(&d_tmp, H*2));
    HIP_OK(hipMalloc(&d_phs, H*2));
    HIP_OK(hipMalloc(&d_csi, QKV*2*2));
    HIP_OK(hipMalloc(&d_moe, H*2));
    HIP_OK(hipMalloc(&d_fbuf, std::max(4096,H*2)*4));
    HIP_OK(hipMalloc(&d_rs, RTR_H*4));
    HIP_OK(hipMalloc(&d_prev_rs, RTR_H*4));
    HIP_OK(hipMalloc(&d_expert_idx, 4));
    HIP_OK(hipMalloc(&d_expert_wt, 4));

    hipStream_t gpu_st; HIP_OK(hipStreamCreate(&gpu_st));

    // ── 5. Load layer weights to GPU ──
    printf("[5/5] Loading %d layers to GPU...\n", N_LAYERS);
    struct LGPU {
        __half *nw,*wq,*wk,*wv1,*wv2,*wo,*pan;
        float *cdw,*cdb,*cgw,*cgb,*ks;
        float *pahss,*pahsb,*parss,*parsb;
        float *gdw,*gdb,*rfn,*rf1,*rf1b,*rf2,*rf2b,*rout,*bb;
        __half *gu,*dn;
        float *pmhss,*pmhsb,*pmrss,*pmrsb;
    };
    std::vector<LGPU> lg(N_LAYERS);

    for(int il=0; il<N_LAYERS; il++){
        auto& l=lg[il];
        auto A=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*2));};
        auto B=[&](auto&p,int n){HIP_OK(hipMalloc(&p,n*4));};

        A(l.nw,H);  upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,H,gpu_st);
        A(l.wq,QD*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,QD*H,gpu_st);
        A(l.wk,KD*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,KD*H,gpu_st);
        A(l.wv1,(KD/2)*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(KD/2)*H,gpu_st);
        A(l.wv2,(KD/2)*H); upf16(W("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(KD/2)*H,gpu_st);
        A(l.wo,H*QD); upf16(W("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,H*QD,gpu_st);
        A(l.pan,H); upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,H,gpu_st);
        B(l.cdw,QKV*2); upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,QKV*2,gpu_st);
        B(l.cdb,QKV);   upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,QKV,gpu_st);
        B(l.cgw,QKV*128*2); upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,QKV*128*2,gpu_st);
        B(l.cgb,QKV);   upf32(W("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,QKV,gpu_st);
        B(l.ks,NKV);    upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,NKV,gpu_st);
        B(l.pahss,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,H,gpu_st);
        B(l.pahsb,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,H,gpu_st);
        B(l.parss,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,H,gpu_st);
        B(l.parsb,H); upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,H,gpu_st);
        B(l.gdw,RTR_H*H); upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin"),l.gdw,RTR_H*H,gpu_st);
        B(l.gdb,RTR_H);   upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,RTR_H,gpu_st);
        B(l.rfn,RTR_H);   upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,RTR_H,gpu_st);
        B(l.rf1,RTR_H*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,RTR_H*RTR_H,gpu_st);
        B(l.rf1b,RTR_H);  upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,RTR_H,gpu_st);
        B(l.rf2,RTR_H*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,RTR_H*RTR_H,gpu_st);
        B(l.rf2b,RTR_H);  upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,RTR_H,gpu_st);
        B(l.rout,17*RTR_H); upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,17*RTR_H,gpu_st);
        B(l.bb,17);  upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,17,gpu_st);
        A(l.gu,N_EXP*2*N_FF*H); upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,N_EXP*2*N_FF*H,gpu_st);
        A(l.dn,N_EXP*H*N_FF);   upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,N_EXP*H*N_FF,gpu_st);
        B(l.pmhss,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,H,gpu_st);
        B(l.pmhsb,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,H,gpu_st);
        B(l.pmrss,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,H,gpu_st);
        B(l.pmrsb,H); upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,H,gpu_st);
    }
    HIP_OK(hipStreamSynchronize(gpu_st));
    printf("  Layers loaded to GPU\n");

    // ── Embed first token ──
    int token_id = 100;
    printf("\n--- Running inference (token %d) ---\n", token_id);

    std::vector<__half> h_tok(H);
    for(int i=0;i<H;i++){
        float raw = embed[token_id*H+i];
        h_tok[i]=__float2half((raw+ibias[i])*iscale[i]);
    }
    HIP_OK(hipMemcpyAsync(d_hs, h_tok.data(), H*2, hipMemcpyHostToDevice, gpu_st));
    HIP_OK(hipMemsetAsync(d_phs, 0, H*2, gpu_st));
    HIP_OK(hipMemsetAsync(d_csi, 0, QKV*2*2, gpu_st));
    HIP_OK(hipMemsetAsync(d_prev_rs, 0, RTR_H*4, gpu_st));
    HIP_OK(hipStreamSynchronize(gpu_st));

    // ── Forward pass (GPU-only baseline, NPU integration TODO) ──
    auto t0 = std::chrono::high_resolution_clock::now();
    int g1 = (H+BLK-1)/BLK;

    // Phase 1: GPU does all layers (attention + MoE both on GPU)
    // Phase 2 will split: GPU attention, NPU MoE, overlapped via pipeline
    printf("\n  Phase 1: GPU-only baseline\n");

    for(int il=0; il<N_LAYERS; il++){
        auto& l = lg[il];

        // ── A) CCA Attention (GPU) ──
        cca_attn_kernel<<<1,128,0,gpu_st>>>(
            d_hs, d_phs, d_csi, il,
            l.wq, l.wk, l.wv1, l.wv2, l.wo,
            l.cdw, l.cdb, l.cgw, l.cgb, l.ks, l.nw,
            d_ao, d_csi, d_phs);

        // ── B) Post-attention residual scale (GPU) ──
        residual_scale_k<<<g1,BLK,0,gpu_st>>>(d_ao, d_hs, l.pahss, l.pahsb, l.parss, l.parsb, H);
        copy_k<<<g1,BLK,0,gpu_st>>>(d_hs, d_ao, H);

        // ── C) Post-attention RMSNorm (GPU) ──
        rmsnorm_k<<<1,BLK,0,gpu_st>>>(d_hs, l.pan, H);

        // ── D) EDA Router + MoE Expert (GPU) ──
        eda_router_moe_kernel<<<1,256,0,gpu_st>>>(
            d_hs, d_prev_rs, 1, 1.0f,
            l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b, l.rf2, l.rf2b, l.rout, l.bb,
            l.gu, l.dn,
            d_rs, d_moe, d_expert_idx, d_expert_wt);

        // ── E) Post-MLP residual scale (GPU) ──
        residual_scale_k<<<g1,BLK,0,gpu_st>>>(d_moe, d_hs, l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, H);
        copy_k<<<g1,BLK,0,gpu_st>>>(d_hs, d_moe, H);
        std::swap(d_rs, d_prev_rs);
    }

    HIP_OK(hipStreamSynchronize(gpu_st));
    float ms = std::chrono::duration<float,std::milli>(
        std::chrono::high_resolution_clock::now()-t0).count();

    // ── lm_head ──
    __half *d_fnw;
    auto fnw = load_bin("/tmp/zaya_weights/model_norm_weight.bin");
    HIP_OK(hipMalloc(&d_fnw, H*2));
    upf16(fnw, d_fnw, H, gpu_st);
    rmsnorm_k<<<1,BLK,0,gpu_st>>>(d_hs, d_fnw, H);
    HIP_OK(hipStreamSynchronize(gpu_st));

    std::vector<__half> h_hs(H);
    HIP_OK(hipMemcpy(h_hs.data(), d_hs, H*2, hipMemcpyDeviceToHost));
    std::vector<float> logits(1000);
    for(int i=0;i<1000;i++){
        float s=0;
        for(int j=0;j<H;j++) s+=__half2float(h_hs[j])*embed[i*H+j];
        logits[i]=s;
    }
    int predicted = (int)(std::max_element(logits.begin(),logits.end())-logits.begin());

    printf("\n═══ Results ═══\n");
    printf("%d layers: %.2f ms (%.2f ms/layer)\n", N_LAYERS, ms, ms/N_LAYERS);
    printf("Predicted token (top-1000): %d\n", predicted);
    printf("\n── Next: Phase 2 — NPU MoE pipeline ──\n");

    // ── Cleanup ──
    hipFree(d_hs); hipFree(d_ao); hipFree(d_tmp); hipFree(d_phs); hipFree(d_csi);
    hipFree(d_moe); hipFree(d_fbuf); hipFree(d_rs); hipFree(d_prev_rs);
    hipFree(d_expert_idx); hipFree(d_expert_wt); hipFree(d_fnw);
    for(auto& l : lg){
        hipFree(l.nw); hipFree(l.wq); hipFree(l.wk); hipFree(l.wv1); hipFree(l.wv2);
        hipFree(l.wo); hipFree(l.pan); hipFree(l.cdw); hipFree(l.cdb);
        hipFree(l.cgw); hipFree(l.cgb); hipFree(l.ks);
        hipFree(l.pahss); hipFree(l.pahsb); hipFree(l.parss); hipFree(l.parsb);
        hipFree(l.gdw); hipFree(l.gdb); hipFree(l.rfn); hipFree(l.rf1); hipFree(l.rf1b);
        hipFree(l.rf2); hipFree(l.rf2b); hipFree(l.rout); hipFree(l.bb);
        hipFree(l.gu); hipFree(l.dn);
        hipFree(l.pmhss); hipFree(l.pmhsb); hipFree(l.pmrss); hipFree(l.pmrsb);
    }
    close(render_fd);
    printf("Done.\n");
    return 0;
}
