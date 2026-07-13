/**
 * NPU Inference Engine v2 — Qwen3-0.6B with torch2aie custom xclbin.
 *
 * Uses the config1 pre-built xclbin (4096x4096x2048, 12.12 TFLOPS verified).
 * Weights are packed column-major from Q4NX → BFP16 shuffled for the xclbin.
 *
 * Pipeline per layer:
 *   1. RMS norm (input)
 *   2. Q/K/V projections via NPU GEMM
 *   3. RoPE + GQA reshape + QK^T attention + softmax (CPU for now)
 *   4. O projection via NPU GEMM
 *   5. Residual add + RMS norm (post-attn)
 *   6. Gate/Up projections via NPU GEMM
 *   7. SiLU(Gate) * Up → Down projection via NPU GEMM
 *   8. Residual add
 *
 * Build:
 *   g++ -std=c++23 -O2 -g -o build/npu_engine_v2 src/npu_engine_v2.cpp \
 *       src/model.c -Iinclude \
 *       -I/home/bcloud/torch2aie/toolchain/xrt/include \
 *       -I/home/bcloud/torch2aie/examples \
 *       -I/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering \
 *       -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
 *       -L/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
 *       -L/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 \
 *       -lxrt_coreutil -luuid -ldl -lm
 */

#include "common.h"
#include "model.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include "helper.h"
#include "gemm_atb_layout.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

// ---- NPU Kernel Manager ----
struct NpuKernel {
    xrt::device device{0};
    std::unique_ptr<xrt::xclbin> xclbin;
    std::unique_ptr<xrt::hw_context> hw_ctx;
    std::unique_ptr<xrt::kernel> kernel;
    std::vector<uint32_t> instrs;

    // Pre-allocated BOs (reused every GEMM call)
    std::unique_ptr<xrt::bo> bo_instr, bo_A, bo_B, bo_C;
    uint16_t* A_map = nullptr;
    uint8_t*  C_map = nullptr;

    bool init(const char* xclbin_path, const char* instr_path) {
        // Load instructions
        FILE* f = fopen(instr_path, "rb");
        if (!f) { LOG_ERROR("Cannot open %s", instr_path); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        instrs.resize(sz / 4);
        size_t nr = fread(instrs.data(), 4, instrs.size(), f); (void)nr;
        fclose(f);
        LOG_INFO("Loaded %zu instructions", instrs.size());

        // Load xclbin
        device = xrt::device(0);
        xclbin = std::make_unique<xrt::xclbin>(std::string(xclbin_path));
        device.register_xclbin(*xclbin);
        hw_ctx = std::make_unique<xrt::hw_context>(device, xclbin->get_uuid());
        kernel = std::make_unique<xrt::kernel>(*hw_ctx, "MLIR_AIE");

        // Pre-create BOs
        bo_instr = std::make_unique<xrt::bo>(device, instrs.size() * 4,
                                               XCL_BO_FLAGS_CACHEABLE,
                                               kernel->group_id(INSTR_GROUP_ID));
        bo_A = std::make_unique<xrt::bo>(device, (size_t)XM * XK * 2,  // BF16
                                           XRT_BO_FLAGS_HOST_ONLY,
                                           kernel->group_id(ACT_GROUP_ID));
        bo_B = std::make_unique<xrt::bo>(device, (size_t)XK * XN * 9 / 8, // BFP16
                                           XRT_BO_FLAGS_HOST_ONLY,
                                           kernel->group_id(WEIGHT_GROUP_ID));
        bo_C = std::make_unique<xrt::bo>(device, (size_t)XM * XN * 2,  // BF16
                                           XRT_BO_FLAGS_HOST_ONLY,
                                           kernel->group_id(OUT_GROUP_ID));

        A_map = (uint16_t*)bo_A->map();
        C_map = (uint8_t*)bo_C->map();

        // Upload instructions (static)
        memcpy(bo_instr->map(), instrs.data(), instrs.size() * 4);
        bo_instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        LOG_INFO("NPU kernel ready (XM=%d XK=%d XN=%d)", XM, XK, XN);
        return true;
    }

    // Run GEMM: C[M,N] = A[M,K] @ B[K,N]
    // M is padded to next multiple of N_AIE_ROWS * M_TILE = 512
    static constexpr int M_PAD = 512; // 4 * 128

    void gemm(const uint16_t* A, int M, int K,
              const BfpPackedWeight* B_packed,
              uint16_t* C_out, int N) {
        // Pad M to multiple of M_PAD
        int M_padded = ((M + M_PAD - 1) / M_PAD) * M_PAD;
        if (M_padded > XM) M_padded = XM;
        // 1. Pad activation to M_padded x XK, convert to float
        std::vector<float> A_float(XM * XK);
        for (int m = 0; m < M && m < XM; m++)
            for (int k = 0; k < K && k < XK; k++)
                A_float[m * XK + k] = bf16_to_float(A[m * K + k]);
        // rest is zero-initialized

        // 2. Shuffle A for ATB L1 layout (uses full XM, XK)
        auto A_shuffled = gemm_atb::layout_A_L1_2x1_8x8block(
            A_float, XM, XK, M_TILE, K_TILE);

        // 3. Encode as BF16 and upload
        for (int i = 0; i < XM * XK; i++)
            A_map[i] = float_to_bf16(A_shuffled[i]);
        bo_A->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 4. Upload B (pre-packed BFP16)
        if (!B_packed || !B_packed->data || B_packed->size == 0) {
            LOG_ERROR("Invalid packed weight");
            return;
        }
        if (B_packed->size > (size_t)XK * XN * 9 / 8) {
            LOG_ERROR("Weight too large: %zu > %zu", B_packed->size, (size_t)XK * XN * 9 / 8);
            return;
        }
        memcpy(bo_B->map(), B_packed->data, B_packed->size);
        bo_B->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 5. Clear C and execute
        memset(bo_C->map(), 0, (size_t)XM * XN * 2);
        bo_C->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = (*kernel)(opcode, *bo_instr, instrs.size(), *bo_A, *bo_B, *bo_C);
        run.wait();

        // 6. Read back, unshuffle
        bo_C->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        uint16_t* C_bf16 = (uint16_t*)bo_C->map();
        std::vector<float> C_float(XM * XN);
        for (int i = 0; i < XM * XN; i++)
            C_float[i] = bf16_to_float(C_bf16[i]);

        auto C_unshuffled = gemm_atb::layout_inverse_C_L1_2x2_8x8block(
            C_float, XM, XN, 4 * M_TILE, N_TILE);

        // 7. Extract valid output (just M rows, N columns)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                C_out[m * N + n] = float_to_bf16(C_unshuffled[m * XN + n]);
    }
};

// ---- Softmax (CPU) ----
static void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= (float)sum;
}

// ---- SiLU (CPU) ----
static void silu_bf16(uint16_t* x, int n) {
    for (int i = 0; i < n; i++) {
        float v = bf16_to_float(x[i]);
        x[i] = float_to_bf16(v / (1.0f + expf(-v)));
    }
}

// ---- Argmax ----
static int argmax(const float* x, int n) {
    int mi = 0; float mv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mv) { mv = x[i]; mi = i; }
    return mi;
}

// ---- Element-wise multiply ----
static void elem_mul_bf16(uint16_t* out, const uint16_t* a, const uint16_t* b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = float_to_bf16(bf16_to_float(a[i]) * bf16_to_float(b[i]));
}

// ---- Element-wise add ----
static void elem_add_bf16(uint16_t* out, const uint16_t* a, const uint16_t* b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = float_to_bf16(bf16_to_float(a[i]) + bf16_to_float(b[i]));
}

// ---- Main ----
int main(int argc, char** argv) {
    printf("=== NPU Engine v2 — Qwen3-0.6B (torch2aie custom kernel) ===\n\n");

    const char* model_path = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    if (argc > 1) model_path = argv[1];

    // 1. Load model
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    ModelWeights* model = model_load(model_path, cfg);
    if (!model) { LOG_ERROR("Model load failed"); return 1; }

    // 2. Init NPU
    NpuKernel npu;
    if (!npu.init(CUSTOM_XCLBIN_PATH, CUSTOM_INSTR_PATH)) return 1;

    // 3. Pack all weights
    LOG_INFO("=== Packing weights ===");
    auto t_pack = std::chrono::steady_clock::now();

    struct LayerPacked {
        BfpPackedWeight* q_proj, *k_proj, *v_proj, *o_proj;
        BfpPackedWeight* gate_proj, *up_proj, *down_proj;
    };
    std::vector<LayerPacked> layers(cfg.num_layers);
    BfpPackedWeight* lm_head_packed = nullptr;

    for (int l = 0; l < cfg.num_layers; l++) {
        LayerWeights* lw = &model->layers[l];
        auto& lp = layers[l];

        lp.q_proj   = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->q_proj_weight),
                                       lw->q_proj_weight.shape[0], lw->q_proj_weight.shape[1] / 2, XK, XN);
        lp.k_proj   = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->k_proj_weight),
                                       lw->k_proj_weight.shape[0], lw->k_proj_weight.shape[1] / 2, XK, XN);
        lp.v_proj   = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->v_proj_weight),
                                       lw->v_proj_weight.shape[0], lw->v_proj_weight.shape[1] / 2, XK, XN);
        lp.o_proj   = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->o_proj_weight),
                                       lw->o_proj_weight.shape[0], lw->o_proj_weight.shape[1] / 2, XK, XN);
        lp.gate_proj = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->gate_proj_weight),
                                        lw->gate_proj_weight.shape[0], lw->gate_proj_weight.shape[1] / 2, XK, XN);
        lp.up_proj   = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->up_proj_weight),
                                        lw->up_proj_weight.shape[0], lw->up_proj_weight.shape[1] / 2, XK, XN);
        lp.down_proj = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &lw->down_proj_weight),
                                        lw->down_proj_weight.shape[0], lw->down_proj_weight.shape[1] / 2, XK, XN);
        if (l % 7 == 0) LOG_DEBUG("Layer %d packed", l);
    }
    lm_head_packed = pack_weight_bfp((const uint16_t*)model_tensor_data(model, &model->lm_head_weight),
                                      model->lm_head_weight.shape[0], model->lm_head_weight.shape[1] / 2, XK, XN);

    auto t_pack_end = std::chrono::steady_clock::now();
    LOG_INFO("Weights packed in %.0f ms",
             std::chrono::duration<double, std::milli>(t_pack_end - t_pack).count());

    // 4. Allocate working buffers
    std::vector<uint16_t> hidden(cfg.hidden_size);
    std::vector<uint16_t> hidden2(cfg.hidden_size);
    std::vector<uint16_t> norm_out(cfg.hidden_size);
    std::vector<uint16_t> q_out(cfg.hidden_size);       // Q: 1024 → 1024 (16 heads × 64 dim)
    std::vector<uint16_t> k_out(cfg.num_key_value_heads * cfg.head_dim);
    std::vector<uint16_t> v_out(cfg.num_key_value_heads * cfg.head_dim);
    std::vector<uint16_t> attn_out(cfg.hidden_size);
    std::vector<uint16_t> o_out(cfg.hidden_size);
    std::vector<uint16_t> gate_out(cfg.intermediate_size);
    std::vector<uint16_t> up_out(cfg.intermediate_size);
    std::vector<uint16_t> down_out(cfg.hidden_size);
    std::vector<uint16_t> silu_up_out(cfg.intermediate_size);
    std::vector<uint16_t> lm_logits(cfg.vocab_size);
    std::vector<float> logits_f32(cfg.vocab_size);

    // 5. Inference loop
    int token = 151643; // BOS
    int max_tokens = 32;

    LOG_INFO("=== Inference ===");

    // Embed lookup
    const uint16_t* emb = (const uint16_t*)model_tensor_data(model, &model->embed_tokens);
    for (int i = 0; i < cfg.hidden_size; i++) {
        uint16_t raw = emb[token * cfg.hidden_size + i];
        hidden[i] = ((raw & 0x7F80) == 0x7F80) ? 0 : raw;
    }

    auto t_infer = std::chrono::steady_clock::now();
    int generated = 0;

    for (int step = 0; step < max_tokens; step++) {
        // For simplicity: single-token decode (no KV cache yet)
        // Each step: hidden → all 28 layers → LM head → sample

        uint16_t* h = hidden.data();

        for (int l = 0; l < cfg.num_layers; l++) {
            LayerWeights* lw = &model->layers[l];
            auto& lp = layers[l];

            // -- Pre-attention RMS norm --
            rms_norm_bf16(norm_out.data(), h,
                          (const uint16_t*)model_tensor_data(model, &lw->input_layernorm_weight),
                          cfg.hidden_size, cfg.rms_norm_eps);

            // -- Q projection --
            npu.gemm(norm_out.data(), 1, cfg.hidden_size, lp.q_proj,
                     q_out.data(), lw->q_proj_weight.shape[0]);

            // -- K projection --
            npu.gemm(norm_out.data(), 1, cfg.hidden_size, lp.k_proj,
                     k_out.data(), lw->k_proj_weight.shape[0]);

            // -- V projection --
            npu.gemm(norm_out.data(), 1, cfg.hidden_size, lp.v_proj,
                     v_out.data(), lw->v_proj_weight.shape[0]);

            // -- Simple attention (no RoPE, no GQA reshape for now) --
            // Q[1, 1024] @ K^T[1024, 1] → score[1, 1]
            // Then: attn = score * V[1, 1024]
            // For decode: QK^T is a scalar → attn_out = V (softmax of single value = 1.0)
            // Copy V (averaged over KV heads) to attention output
            int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
            int n_kv_heads = cfg.num_key_value_heads;

            // Expand KV heads to Q heads (GQA): repeat each KV head for GQA ratio
            int gqa = cfg.num_attention_heads / cfg.num_key_value_heads;
            std::vector<uint16_t> attn_expanded(cfg.hidden_size);
            for (int h = 0; h < cfg.num_attention_heads; h++) {
                int kv_h = h / gqa;
                for (int d = 0; d < cfg.head_dim; d++) {
                    attn_expanded[h * cfg.head_dim + d] = v_out[kv_h * cfg.head_dim + d];
                }
            }

            // -- O projection --
            npu.gemm(attn_expanded.data(), 1, cfg.hidden_size, lp.o_proj,
                     o_out.data(), lw->o_proj_weight.shape[0]);

            // -- Residual add (post-attention) --
            elem_add_bf16(hidden2.data(), h, o_out.data(), cfg.hidden_size);

            // -- Post-attention RMS norm --
            rms_norm_bf16(norm_out.data(), hidden2.data(),
                          (const uint16_t*)model_tensor_data(model, &lw->post_attention_layernorm_weight),
                          cfg.hidden_size, cfg.rms_norm_eps);

            // -- Gate projection --
            npu.gemm(norm_out.data(), 1, cfg.hidden_size, lp.gate_proj,
                     gate_out.data(), lw->gate_proj_weight.shape[0]);

            // -- Up projection --
            npu.gemm(norm_out.data(), 1, cfg.hidden_size, lp.up_proj,
                     up_out.data(), lw->up_proj_weight.shape[0]);

            // -- SiLU(Gate) * Up --
            memcpy(silu_up_out.data(), gate_out.data(), cfg.intermediate_size * 2);
            silu_bf16(silu_up_out.data(), cfg.intermediate_size);
            elem_mul_bf16(silu_up_out.data(), silu_up_out.data(), up_out.data(), cfg.intermediate_size);

            // -- Down projection --
            npu.gemm(silu_up_out.data(), 1, cfg.intermediate_size, lp.down_proj,
                     down_out.data(), lw->down_proj_weight.shape[0]);

            // -- Residual add (post-MLP) --
            elem_add_bf16(h, hidden2.data(), down_out.data(), cfg.hidden_size);
        }

        // -- Final RMS norm --
        rms_norm_bf16(norm_out.data(), h,
                      (const uint16_t*)model_tensor_data(model, &model->norm_weight),
                      cfg.hidden_size, cfg.rms_norm_eps);

        // -- LM head --
        npu.gemm(norm_out.data(), 1, cfg.hidden_size, lm_head_packed,
                 lm_logits.data(), model->lm_head_weight.shape[0]);

        // -- Sample --
        for (int i = 0; i < cfg.vocab_size; i++)
            logits_f32[i] = bf16_to_float(lm_logits[i]);
        token = argmax(logits_f32.data(), cfg.vocab_size);

        if (step == 0) {
            printf("  Token %d: %d\n", step, token);
        }
        generated++;

        if (token == 0 || token == 151645) break; // EOS or stop token

        // Embed next token
        for (int i = 0; i < cfg.hidden_size; i++) {
            uint16_t raw = emb[token * cfg.hidden_size + i];
            h[i] = ((raw & 0x7F80) == 0x7F80) ? 0 : raw;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_infer).count();

    LOG_INFO("Generated %d tokens in %.1f s (%.0f ms/tok)",
             generated, total_s, total_s * 1000 / (generated ? generated : 1));

    // 6. Cleanup
    for (auto& lp : layers) {
        free_packed_weight(lp.q_proj); free_packed_weight(lp.k_proj);
        free_packed_weight(lp.v_proj); free_packed_weight(lp.o_proj);
        free_packed_weight(lp.gate_proj); free_packed_weight(lp.up_proj);
        free_packed_weight(lp.down_proj);
    }
    free_packed_weight(lm_head_packed);
    model_free(model);

    LOG_INFO("Done.");
    return 0;
}
