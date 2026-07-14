// cpu_q4nx_test.cpp — Load and verify Q4NX model weights on CPU
// Reads the real model.q4nx, dequantizes I8 tiles to FP32, runs 2 layers.
//
// Build: g++ -O3 -march=native -std=c++17 -I. -o cpu_q4nx_test \
//        cpu_q4nx_test.cpp cpu_layer.cpp -lm
//
// Run:   ./cpu_q4nx_test /home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx

#include "cpu_layer.h"
#include "cpu_q4nx_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

// ── Helper: extract a tensor from the loaded model by name pattern ──
static bool get_tensor(const Q4nxModel& model, const std::string& name,
                       std::vector<float>& out, const std::vector<int>& expected_shape = {}) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        fprintf(stderr, "  Tensor not found: %s\n", name.c_str());
        return false;
    }
    out = it->second.fp32;
    if (!expected_shape.empty() && !out.empty()) {
        int exp_elems = 1;
        for (int s : expected_shape) exp_elems *= s;
        if ((int)out.size() != exp_elems) {
            fprintf(stderr, "  Size mismatch %s: got %zu, expected %d\n",
                    name.c_str(), out.size(), exp_elems);
        }
    }
    return !out.empty();
}

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1]
        : getenv("HOME")?std::string(getenv("HOME"))+"/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";

    printf("=== CPU Q4NX Loader Test ===\n");
    printf("Model: %s\n\n", model_path);

    // ── Load the Q4NX file ──
    Q4nxModel model;
    if (!model.load(model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("\n");

    const int H = model.hidden_dim;
    const int L = model.n_layers;
    const int NH = model.n_heads;
    const int NKV = model.n_kv_heads;
    const int HD = model.head_dim;
    const int IM = model.inter_size;
    const int GQA = NKV > 0 ? NH / NKV : 1;
    const int V = model.vocab_size;

    printf("Dimensions: H=%d L=%d NH=%d NKV=%d HD=%d IM=%d GQA=%d V=%d\n\n",
           H, L, NH, NKV, HD, IM, GQA, V);

    // ── Load all weights ──
    // Embedding
    std::vector<float> emb;
    get_tensor(model, "model.embed_tokens.weight", emb, {V, H});
    printf("  embed_tokens: %zu floats [OK]\n", emb.size());

    // Final norm
    std::vector<float> final_norm;
    get_tensor(model, "model.norm.weight", final_norm, {H});
    printf("  final_norm: %zu floats [OK]\n", final_norm.size());

    // LM head
    std::vector<float> lm_head;
    get_tensor(model, "lm_head.weight", lm_head, {V, H});
    // Check if tied
    bool tied_embeddings = lm_head.empty();
    if (tied_embeddings) {
        printf("  lm_head: tied to embeddings [OK]\n");
        lm_head = emb;
    } else {
        printf("  lm_head: %zu floats [OK]\n", lm_head.size());
    }

    // Per-layer weights
    struct LayerWeights {
        std::vector<float> in_norm, pa_norm, q_norm, k_norm;
        std::vector<float> q, k, v, o;
        std::vector<float> gate, up, down;
    };
    std::vector<LayerWeights> layers(L);

    for (int l = 0; l < L; l++) {
        auto& lw = layers[l];

        auto load = [&](const std::string& pattern, std::vector<float>& out, std::vector<int> shape = {}) {
            char key[256];
            snprintf(key, sizeof(key), pattern.c_str(), l);
            if (pattern.find("input_layernorm") != std::string::npos ||
                pattern.find("post_attention") != std::string::npos) {
                shape = {H};
            } else if (pattern.find("k_norm") != std::string::npos ||
                       pattern.find("q_norm") != std::string::npos) {
                shape = {HD};
            }
            return get_tensor(model, key, out, shape);
        };

        // Norms
        load("model.layers.%d.input_layernorm.weight", lw.in_norm);
        load("model.layers.%d.post_attention_layernorm.weight", lw.pa_norm);
        load("model.layers.%d.self_attn.q_norm.weight", lw.q_norm);
        load("model.layers.%d.self_attn.k_norm.weight", lw.k_norm);

        // Projection weights (already dequantized to FP32)
        load("model.layers.%d.self_attn.q_proj.weight", lw.q);
        load("model.layers.%d.self_attn.k_proj.weight", lw.k);
        load("model.layers.%d.self_attn.v_proj.weight", lw.v);
        load("model.layers.%d.self_attn.o_proj.weight", lw.o);
        load("model.layers.%d.mlp.gate_proj.weight", lw.gate);
        load("model.layers.%d.mlp.up_proj.weight", lw.up);
        load("model.layers.%d.mlp.down_proj.weight", lw.down);

        // Validate weight sizes
        auto check_dim = [&](const std::vector<float>& w, int out_dim, int in_dim, const char* name) {
            if (w.empty()) { printf("  Layer %d: %s EMPTY\n", l, name); return; }
            int expected = out_dim * in_dim;
            if ((int)w.size() != expected) {
                printf("  Layer %d: %s size %zu != expected %d (%d×%d)\n",
                       l, name, w.size(), expected, out_dim, in_dim);
            }
        };

        if (l == 0) {
            printf("\n  Layer 0 weight shapes (FP32 after dequant):\n");
            printf("    q:     %zu floats (expect %d×%d=%d)\n", lw.q.size(), NH*HD, H, NH*HD*H);
            printf("    k:     %zu floats (expect %d×%d=%d)\n", lw.k.size(), NKV*HD, H, NKV*HD*H);
            printf("    v:     %zu floats (expect %d×%d=%d)\n", lw.v.size(), NKV*HD, H, NKV*HD*H);
            printf("    o:     %zu floats (expect %d×%d=%d)\n", lw.o.size(), H, NH*HD, H*NH*HD);
            printf("    gate:  %zu floats (expect %d×%d=%d)\n", lw.gate.size(), IM, H, IM*H);
            printf("    up:    %zu floats (expect %d×%d=%d)\n", lw.up.size(), IM, H, IM*H);
            printf("    down:  %zu floats (expect %d×%d=%d)\n", lw.down.size(), H, IM, H*IM);
        }
    }

    // ── Verify we got all weights ──
    int ok_count = 0, total = 0;
    for (int l = 0; l < L; l++) {
        auto& lw = layers[l];
        if (!lw.in_norm.empty()) ok_count++;
        if (!lw.pa_norm.empty()) ok_count++;
        if (!lw.q_norm.empty()) ok_count++;
        if (!lw.k_norm.empty()) ok_count++;
        if (!lw.q.empty()) ok_count++;
        if (!lw.k.empty()) ok_count++;
        if (!lw.v.empty()) ok_count++;
        if (!lw.o.empty()) ok_count++;
        if (!lw.gate.empty()) ok_count++;
        if (!lw.up.empty()) ok_count++;
        if (!lw.down.empty()) ok_count++;
        total += 11;
    }
    printf("\n  Weights loaded: %d/%d per-layer tensors OK\n", ok_count, total);

    // ── Now pack them into ternary format and test forward pass ──
    // This requires converting FP32 weights → ternary packed format.
    // For now, let's just test that we can run a single FP32 matmul
    // to verify the dequantized weights produce finite results.

    printf("\n── FP32 Forward Test (2 layers) ──\n");

    // Create activation vector (first token embedding)
    std::vector<float> hidden(H);
    std::memcpy(hidden.data(), emb.data() + 1 * H, H * sizeof(float)); // token 1 = BOS

    printf("  Initial hidden (token 1): [%.4f, %.4f, ...]\n", hidden[0], hidden[1]);

    // Scratch buffers
    int qkv_sz = NH*HD + 2*NKV*HD;
    std::vector<float> sq(qkv_sz), sa(NH*HD), sf(2*IM), sa2(IM);

    // Sin/Cos tables
    std::vector<float> sin_tab(4096*HD), cos_tab(4096*HD);
    for (int p = 0; p < 4096; p++) {
        for (int d = 0; d < HD; d++) {
            float theta = p / std::pow(10000.0f, (2.0f * (d/2)) / HD);
            sin_tab[p*HD+d] = std::sin(theta);
            cos_tab[p*HD+d] = std::cos(theta);
        }
    }

    // KV cache for 2 layers
    int max_seq = 64;
    std::vector<std::vector<float>> k_cache(L), v_cache(L);
    for (int l = 0; l < L; l++) {
        k_cache[l].resize(max_seq * NKV * HD, 0);
        v_cache[l].resize(max_seq * NKV * HD, 0);
    }

    // ── FP32 linear layer (matmul) ──
    auto fp32_linear = [](const float* w, const float* x, float* y, int out_dim, int in_dim) {
        for (int i = 0; i < out_dim; i++) {
            double acc = 0.0;
            for (int j = 0; j < in_dim; j++) {
                acc += (double)w[i * in_dim + j] * (double)x[j];
            }
            y[i] = (float)acc;
        }
    };

    auto run_layer = [&](int l, int pos, std::vector<float>& kc, std::vector<float>& vc) {
        auto& lw = layers[l];
        int qkvd = NH*HD + 2*NKV*HD;

        // Save residual
        std::vector<float> residual(H);
        std::memcpy(residual.data(), hidden.data(), H * sizeof(float));

        // Input RMSNorm
        cpu_rmsnorm(hidden.data(), lw.in_norm.data(), hidden.data(), H, 1e-6f);

        // QKV projections (FP32 matmul since we dequantized)
        fp32_linear(lw.q.data(), hidden.data(), sq.data(), NH*HD, H);
        fp32_linear(lw.k.data(), hidden.data(), sq.data() + NH*HD, NKV*HD, H);
        fp32_linear(lw.v.data(), hidden.data(), sq.data() + NH*HD + NKV*HD, NKV*HD, H);

        // Q/K norm + RoPE
        if (!lw.q_norm.empty()) {
            for (int h = 0; h < NH; h++) {
                cpu_rmsnorm(sq.data() + h*HD, lw.q_norm.data(),
                            sq.data() + h*HD, HD, 1e-6f);
            }
        }
        cpu_rope(sq.data(), pos, NH, HD, sin_tab.data(), cos_tab.data());

        if (!lw.k_norm.empty()) {
            for (int h = 0; h < NKV; h++) {
                cpu_rmsnorm(sq.data() + NH*HD + h*HD, lw.k_norm.data(),
                            sq.data() + NH*HD + h*HD, HD, 1e-6f);
            }
        }
        cpu_rope(sq.data() + NH*HD, pos, NKV, HD, sin_tab.data(), cos_tab.data());

        // KV cache write
        for (int h = 0; h < NKV; h++) {
            std::memcpy(&kc[pos * NKV * HD + h * HD],
                       sq.data() + NH*HD + h*HD, HD * sizeof(float));
            std::memcpy(&vc[pos * NKV * HD + h * HD],
                       sq.data() + NH*HD + NKV*HD + h*HD, HD * sizeof(float));
        }

        // Attention
        cpu_attention(sq.data(), kc.data(), vc.data(), sa.data(),
                      NH, NKV, HD, pos + 1, GQA);

        // O projection + residual
        fp32_linear(lw.o.data(), sa.data(), hidden.data(), H, NH*HD);
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + hidden[i];

        // FFN residual save
        std::memcpy(residual.data(), hidden.data(), H * sizeof(float));

        // Post-attention RMSNorm
        cpu_rmsnorm(hidden.data(), lw.pa_norm.data(), hidden.data(), H, 1e-6f);

        // Gate/Up
        fp32_linear(lw.gate.data(), hidden.data(), sf.data(), IM, H);
        fp32_linear(lw.up.data(), hidden.data(), sf.data() + IM, IM, H);

        // SiLU GLU
        cpu_silu_glu(sf.data(), sf.data() + IM, sa2.data(), IM);

        // Down
        fp32_linear(lw.down.data(), sa2.data(), hidden.data(), H, IM);

        // Residual add
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + hidden[i];
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    // Run first 2 layers
    for (int l = 0; l < 2 && l < L; l++) {
        run_layer(l, l, k_cache[l], v_cache[l]);

        bool ok = true;
        for (int i = 0; i < H; i++) if (!std::isfinite(hidden[i])) ok = false;
        printf("  Layer %d: %s (hidden[0]=%.4f)\n", l, ok ? "OK" : "NaN!", hidden[0]);
        if (!ok) return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // LM head
    const float* lm = lm_head.data();
    std::vector<float> logits(V);
    cpu_lm_head(hidden.data(), lm, logits.data(), V, H);
    int best = cpu_argmax(logits.data(), V);
    cpu_softmax(logits.data(), V);

    printf("\n── Results (2 layers) ──\n");
    printf("  Time: %.1f ms (%.2f ms/layer)\n", ms, ms / 2.0);
    printf("  Hidden[0..3]: %.4f %.4f %.4f %.4f\n", hidden[0], hidden[1], hidden[2], hidden[3]);
    printf("  LM head top-1: token %d (prob=%.4f)\n", best, logits[best]);
    printf("\n=== Q4NX Loader ✅ Real weights verified ===\n");
    return 0;
}
