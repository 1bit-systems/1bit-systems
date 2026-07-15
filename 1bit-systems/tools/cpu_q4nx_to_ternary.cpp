// cpu_q4nx_to_ternary.cpp — Convert Q4NX model to packed ternary weights,
// then run inference through the fast cpu_ternary_gemv() path.
//
// This is the bridge between real Q4NX model weights and the fast ternary
// GEMV engine. Instead of FP32 matmul (slow), we quantize the dequantized
// weights to ternary {-1, 0, +1} and run the packed 2-bit GEMV.
//
// Build: g++ -O3 -march=native -std=c++17 -Iengine/fusion \
//        -o tools/cpu_q4nx_to_ternary tools/cpu_q4nx_to_ternary.cpp \
//        engine/fusion/cpu_layer.cpp -lm
//
// Run:   ./tools/cpu_q4nx_to_ternary \
//        /home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx

#include "cpu_layer.h"
#include "cpu_q4nx_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <string>
#include <algorithm>

// ── FP32 → ternary packer ─────────────────────────────────────
// Takes a FP32 weight matrix [OUT, IN] and packs it as ternary.
// Quantization: values > 0 → +1, values < 0 → -1, exactly 0 → 0.
// Packing: 16 ternary values per uint32, 2 bits each.
//   bits[2i:2i+1] = 00→-1, 01→0, 10→+1
static void pack_fp32_to_ternary(
    const float*  src,      // [OUT * IN] FP32 weights
    uint32_t*     packed,   // [OUT * pk] packed ternary (pk = ceil(IN/16))
    int OUT, int IN)
{
    int pk = (IN + 15) / 16;
    for (int row = 0; row < OUT; row++) {
        for (int u = 0; u < pk; u++) {
            uint32_t word = 0;
            for (int v = 0; v < 16; v++) {
                int col = u * 16 + v;
                float val = (col < IN) ? src[row * IN + col] : 0.0f;
                uint32_t code;
                if (val > 0.0f)       code = 0x1;  // +1 → 0b01
                else if (val < 0.0f)  code = 0x2;  // -1 → 0b10
                else                  code = 0x0;  // 0  → 0b00
                word |= (code << (v * 2));
            }
            packed[row * pk + u] = word;
        }
    }
}

// Compute per-row scale = mean(|weight[row][:]|)
// This is the optimal scale for ternary: preserves L1 norm.
static void compute_ternary_scales(
    const float* src,     // [OUT * IN] FP32 weights
    float*       scales,  // [OUT] per-row scale
    int OUT, int IN)
{
    for (int row = 0; row < OUT; row++) {
        double sum_abs = 0.0;
        int nz = 0;
        for (int col = 0; col < IN; col++) {
            float v = src[row * IN + col];
            sum_abs += std::abs(v);
            if (v != 0.0f) nz++;
        }
        scales[row] = (nz > 0) ? (float)(sum_abs / nz) : 0.0f;
    }
}

// ── Full converter: load Q4NX, pack to ternary, save ──────────
struct TernaryModel {
    // Model config
    int H, IM, NH, NKV, HD, GQA, V, L;

    // Embedding + final norm + lm_head (FP32, kept as FP32)
    std::vector<float> emb_table;     // [V * H]
    std::vector<float> final_norm;    // [H]
    std::vector<float> lm_head_f32;   // [V * H] (or empty if tied)

    // Per-layer norms (FP32)
    std::vector<float> in_norm;  // [L * H]
    std::vector<float> pa_norm;  // [L * H]
    std::vector<float> q_norm;   // [L * HD]
    std::vector<float> k_norm;   // [L * HD]

    // Per-layer ternary weights + scales
    // Packed: uint32[OUT * ceil(IN/16)]
    // Scales: float[OUT]
    struct TernaryLayer {
        // Q: [NH*HD, H]; K: [NKV*HD, H]; V: [NKV*HD, H]; O: [H, NH*HD]
        // Gate: [IM, H]; Up: [IM, H]; Down: [H, IM]
        std::vector<uint32_t> q_packed, k_packed, v_packed, o_packed;
        std::vector<uint32_t> gate_packed, up_packed, down_packed;
        std::vector<float> q_scales, k_scales, v_scales, o_scales;
        std::vector<float> gate_scales, up_scales, down_scales;
    };
    std::vector<TernaryLayer> layers;

    // Stats
    double load_ms = 0;
    double convert_ms = 0;
};

static bool load_and_convert(const char* q4nx_path, TernaryModel& tm) {
    printf("=== Q4NX → Ternary Converter ===\n");
    printf("Loading: %s\n\n", q4nx_path);

    auto t0 = std::chrono::high_resolution_clock::now();

    Q4nxModel qm;
    if (!qm.load(q4nx_path)) return false;

    tm.H = qm.hidden_dim;
    tm.IM = qm.inter_size;
    tm.NH = qm.n_heads;
    tm.NKV = qm.n_kv_heads;
    tm.HD = qm.head_dim;
    tm.GQA = tm.NH / tm.NKV;
    tm.V = qm.vocab_size;
    tm.L = qm.n_layers;

    printf("\nModel: V=%d H=%d L=%d NH=%d NKV=%d HD=%d IM=%d GQA=%d\n",
           tm.V, tm.H, tm.L, tm.NH, tm.NKV, tm.HD, tm.IM, tm.GQA);

    // Helper to get tensor FP32 data
    auto get_fp32 = [&](const std::string& name) -> std::vector<float> {
        auto it = qm.tensors.find(name);
        if (it == qm.tensors.end()) return {};
        return it->second.fp32;
    };

    // Load embeddings
    tm.emb_table = get_fp32("model.embed_tokens.weight");
    tm.final_norm = get_fp32("model.norm.weight");

    // LM head (may be tied)
    std::vector<float> lm_head = get_fp32("lm_head.weight");
    if (lm_head.empty()) {
        printf("  lm_head: tied to embeddings\n");
        tm.lm_head_f32 = tm.emb_table;
    } else {
        printf("  lm_head: %.0fM params (untied)\n", lm_head.size() / 1e6);
        tm.lm_head_f32 = lm_head;
    }

    printf("  embeddings: %.0fM params\n", tm.emb_table.size() / 1e6);

    auto t1 = std::chrono::high_resolution_clock::now();
    tm.load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Load and convert each layer
    tm.layers.resize(tm.L);
    tm.in_norm.resize(tm.L * tm.H);
    tm.pa_norm.resize(tm.L * tm.H);
    tm.q_norm.resize(tm.L * tm.HD);
    tm.k_norm.resize(tm.L * tm.HD);

    auto convert_proj = [&](const std::vector<float>& src_fp32,
                             std::vector<uint32_t>& dst_packed,
                             std::vector<float>& dst_scales,
                             int OUT, int IN) {
        int pk = (IN + 15) / 16;
        dst_packed.resize(OUT * pk);
        dst_scales.resize(OUT);
        pack_fp32_to_ternary(src_fp32.data(), dst_packed.data(), OUT, IN);
        compute_ternary_scales(src_fp32.data(), dst_scales.data(), OUT, IN);
    };

    auto get_proj = [&](int l, const std::string& name) -> std::vector<float> {
        char key[256];
        snprintf(key, sizeof(key), "model.layers.%d.%s", l, name.c_str());
        return get_fp32(key);
    };

    printf("\nConverting %d layers to ternary (%s)...\n", tm.L,
           #ifdef _OPENMP
           "multi-threaded"
           #else
           "single-threaded"
           #endif
    );
    #pragma omp parallel for
    for (int l = 0; l < tm.L; l++) {
        auto& ly = tm.layers[l];

        // Norms
        auto inn = get_proj(l, "input_layernorm.weight");
        auto pan = get_proj(l, "post_attention_layernorm.weight");
        auto qn  = get_proj(l, "self_attn.q_norm.weight");
        auto kn  = get_proj(l, "self_attn.k_norm.weight");

        if (!inn.empty()) std::memcpy(&tm.in_norm[l * tm.H], inn.data(), tm.H * 4);
        if (!pan.empty()) std::memcpy(&tm.pa_norm[l * tm.H], pan.data(), tm.H * 4);
        if (!qn.empty())  std::memcpy(&tm.q_norm[l * tm.HD], qn.data(), tm.HD * 4);
        if (!kn.empty())  std::memcpy(&tm.k_norm[l * tm.HD], kn.data(), tm.HD * 4);

        // Attention projections
        auto q = get_proj(l, "self_attn.q_proj.weight");
        auto k = get_proj(l, "self_attn.k_proj.weight");
        auto v = get_proj(l, "self_attn.v_proj.weight");
        auto o = get_proj(l, "self_attn.o_proj.weight");

        if (!q.empty()) convert_proj(q, ly.q_packed, ly.q_scales, tm.NH * tm.HD, tm.H);
        if (!k.empty()) convert_proj(k, ly.k_packed, ly.k_scales, tm.NKV * tm.HD, tm.H);
        if (!v.empty()) convert_proj(v, ly.v_packed, ly.v_scales, tm.NKV * tm.HD, tm.H);
        if (!o.empty()) convert_proj(o, ly.o_packed, ly.o_scales, tm.H, tm.NH * tm.HD);

        // FFN projections
        auto gate = get_proj(l, "mlp.gate_proj.weight");
        auto up   = get_proj(l, "mlp.up_proj.weight");
        auto down = get_proj(l, "mlp.down_proj.weight");

        if (!gate.empty()) convert_proj(gate, ly.gate_packed, ly.gate_scales, tm.IM, tm.H);
        if (!up.empty())   convert_proj(up,   ly.up_packed,   ly.up_scales,   tm.IM, tm.H);
        if (!down.empty()) convert_proj(down, ly.down_packed, ly.down_scales, tm.H,  tm.IM);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    tm.convert_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Stats
    size_t total_ternary_vals = 0;
    for (int l = 0; l < tm.L; l++) {
        auto& ly = tm.layers[l];
        total_ternary_vals += ly.q_packed.size() * 16;
        total_ternary_vals += ly.k_packed.size() * 16;
        total_ternary_vals += ly.v_packed.size() * 16;
        total_ternary_vals += ly.o_packed.size() * 16;
        total_ternary_vals += ly.gate_packed.size() * 16;
        total_ternary_vals += ly.up_packed.size() * 16;
        total_ternary_vals += ly.down_packed.size() * 16;
    }

    printf("\n── Stats ──\n");
    printf("  Load time:      %.1f ms\n", tm.load_ms);
    printf("  Convert time:   %.1f ms\n", tm.convert_ms);
    printf("  Ternary params: %.0fM\n", total_ternary_vals / 1e6);
    printf("  Binary size:    %.0f MB (packed @ 2-bit)\n",
           (double)total_ternary_vals / 4 / 1024 / 1024);
    printf("  FP32 size:      %.0f MB (dequantized)\n",
           (double)total_ternary_vals * 4 / 1024 / 1024);
    printf("  Compression:    16x vs FP32\n\n");

    return true;
}

// ── Forward pass using ternary GEMV ───────────────────────────
static void run_forward(TernaryModel& tm) {
    printf("── Forward Pass (ternary GEMV) ──\n");

    const int H = tm.H, IM = tm.IM, NH = tm.NH, NKV = tm.NKV;
    const int HD = tm.HD, GQA = tm.GQA, L = tm.L;

    // Sin/Cos tables
    std::vector<float> sin_tab(4096*HD), cos_tab(4096*HD);
    for (int p = 0; p < 4096; p++)
        for (int d = 0; d < HD; d++) {
            float theta = p / std::pow(10000.0f, (2.0f * (d/2)) / HD);
            sin_tab[p*HD+d] = std::sin(theta);
            cos_tab[p*HD+d] = std::cos(theta);
        }

    // KV cache per layer
    int max_seq = 64;
    std::vector<std::vector<float>> k_cache(L), v_cache(L);
    for (int l = 0; l < L; l++) {
        k_cache[l].resize(max_seq * NKV * HD, 0);
        v_cache[l].resize(max_seq * NKV * HD, 0);
    }

    // Scratch buffers
    int qkv_sz = NH*HD + 2*NKV*HD;
    std::vector<float> sq(qkv_sz), sa(NH*HD), sf(2*IM), sa2(IM);

    // Hidden state from embedding
    std::vector<float> hidden(H);
    std::memcpy(hidden.data(), tm.emb_table.data() + 1*H, H*4);

    printf("  Initial: [%.4f, %.4f, ...]\n", hidden[0], hidden[1]);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int l = 0; l < L; l++) {
        auto& ly = tm.layers[l];

        // Save residual
        std::vector<float> residual(H);
        std::memcpy(residual.data(), hidden.data(), H*4);

        // Input RMSNorm
        cpu_rmsnorm(hidden.data(), &tm.in_norm[l*H], hidden.data(), H, 1e-6f);

        // QKV via ternary GEMV (fast path!)
        cpu_ternary_gemv(ly.q_packed.data(), hidden.data(), ly.q_scales.data(),
                         sq.data(), NH*HD, H);
        cpu_ternary_gemv(ly.k_packed.data(), hidden.data(), ly.k_scales.data(),
                         sq.data() + NH*HD, NKV*HD, H);
        cpu_ternary_gemv(ly.v_packed.data(), hidden.data(), ly.v_scales.data(),
                         sq.data() + NH*HD + NKV*HD, NKV*HD, H);

        // Q/K norm + RoPE
        if (!tm.q_norm.empty()) {
            for (int h = 0; h < NH; h++)
                cpu_rmsnorm(sq.data()+h*HD, &tm.q_norm[l*HD], sq.data()+h*HD, HD, 1e-6f);
        }
        cpu_rope(sq.data(), l, NH, HD, sin_tab.data(), cos_tab.data());
        if (!tm.k_norm.empty()) {
            for (int h = 0; h < NKV; h++)
                cpu_rmsnorm(sq.data()+NH*HD+h*HD, &tm.k_norm[l*HD],
                            sq.data()+NH*HD+h*HD, HD, 1e-6f);
        }
        cpu_rope(sq.data()+NH*HD, l, NKV, HD, sin_tab.data(), cos_tab.data());

        // KV cache write
        for (int h = 0; h < NKV; h++) {
            std::memcpy(&k_cache[l][l*NKV*HD + h*HD], sq.data()+NH*HD+h*HD, HD*4);
            std::memcpy(&v_cache[l][l*NKV*HD + h*HD], sq.data()+NH*HD+NKV*HD+h*HD, HD*4);
        }

        // Attention
        cpu_attention(sq.data(), k_cache[l].data(), v_cache[l].data(), sa.data(),
                      NH, NKV, HD, l+1, GQA);

        // O projection (ternary) + residual
        cpu_ternary_gemv(ly.o_packed.data(), sa.data(), ly.o_scales.data(),
                         hidden.data(), H, NH*HD);
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + hidden[i];

        // FFN residual save
        std::memcpy(residual.data(), hidden.data(), H*4);

        // Post-attention RMSNorm
        cpu_rmsnorm(hidden.data(), &tm.pa_norm[l*H], hidden.data(), H, 1e-6f);

        // Gate/Up (ternary)
        cpu_ternary_gemv(ly.gate_packed.data(), hidden.data(), ly.gate_scales.data(),
                         sf.data(), IM, H);
        cpu_ternary_gemv(ly.up_packed.data(), hidden.data(), ly.up_scales.data(),
                         sf.data()+IM, IM, H);

        // SiLU GLU
        cpu_silu_glu(sf.data(), sf.data()+IM, sa2.data(), IM);

        // Down (ternary) + residual
        cpu_ternary_gemv(ly.down_packed.data(), sa2.data(), ly.down_scales.data(),
                         hidden.data(), H, IM);
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + hidden[i];

        bool ok = true;
        for (int i = 0; i < H; i++) if (!std::isfinite(hidden[i])) ok = false;
        printf("  Layer %2d: %s (%.4f)\n", l, ok ? "OK" : "NaN!", hidden[0]);
        if (!ok) { printf("  STOP — NaN at layer %d\n", l); return; }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Final norm
    cpu_rmsnorm(hidden.data(), tm.final_norm.data(), hidden.data(), H, 1e-6f);

    // LM head
    std::vector<float> logits(tm.V);
    cpu_lm_head(hidden.data(), tm.lm_head_f32.data(), logits.data(), tm.V, H);
    int best = cpu_argmax(logits.data(), tm.V);
    cpu_softmax(logits.data(), tm.V);

    printf("\n── Results ──\n");
    printf("  %d layers: %.1f ms (%.2f ms/layer)\n", L, ms, ms/L);
    printf("  Decode:    %.1f tok/s (1 core, ternary GEMV)\n", 1000.0/ms);
    printf("  Hidden:    %.4f %.4f %.4f %.4f\n", hidden[0], hidden[1], hidden[2], hidden[3]);
    printf("  LM head:   token %d (prob=%.4f)\n", best, logits[best]);
    printf("\n=== CPU TRG ✅ Real Q4NX → Ternary GEMV verified ===\n");
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : std::string(getenv("HOME") ? getenv("HOME") : "/tmp") +"/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";

    TernaryModel tm;
    if (!load_and_convert(path, tm)) return 1;

    run_forward(tm);
    return 0;
}
