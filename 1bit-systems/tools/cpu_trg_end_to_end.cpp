// tools/cpu_trg_end_to_end.cpp — Full CPU ternary inference smoke test
#include "cpu_layer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>

static const int H = 1536, IM = 4096, NH = 12, NKV = 2, HD = 128;
static const int GQA = NH / NKV;
static const int NV = 100, L = 28, MSL = 256;
static const float EPS = 1e-6f;

static float rand_f(std::mt19937& rng, float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

struct PackedBuf {
    std::vector<uint32_t> data;
    // For per-layer access:
    std::vector<int> row_stride; // uint32 per row for each weight matrix
    std::vector<int> n_rows;     // number of rows for each weight matrix
};

int main() {
    printf("=== CPU TRG End-to-End Smoke Test ===\n");
    printf("  H=%d IM=%d NH=%d NKV=%d HD=%d L=%d\n\n", H, IM, NH, NKV, HD, L);
    
    std::mt19937 rng(42);
    
    // ── Allocate flat weight buffers ──
    // For each layer: Q(K) + K(K) + V(K) + O(K) + gate(K) + up(K) + down(K)
    // where X(K) means N_rows * (K/16) uint32 + N_rows floats
    
    int pk_h  = H / 16;      // uint32 per row for K=H
    int pk_ah = (NH*HD) / 16; // uint32 per row for K=NH*HD (attention input to O)
    int pk_im = IM / 16;     // uint32 per row for K=IM (down input)
    
    auto make_sizes = [&](int l) -> std::vector<int> {
        // Returns {q_u32, k_u32, v_u32, o_u32, g_u32, u_u32, d_u32,
        //          q_s,   k_s,   v_s,   o_s,   g_s,   u_s,   d_s}
        std::vector<int> s = {
            NH*HD * pk_h,     // q packed
            NKV*HD * pk_h,    // k packed
            NKV*HD * pk_h,    // v packed
            H * pk_ah,        // o packed
            IM * pk_h,        // gate packed
            IM * pk_h,        // up packed
            H * pk_im,        // down packed
            NH*HD, NKV*HD, NKV*HD, H, IM, IM, H  // scales
        };
        return s;
    };
    
    // Calculate total sizes
    std::vector<int> sizes = make_sizes(0);
    int total_packed = 0, total_scales = 0;
    for (int i = 0; i < 7; i++) total_packed += sizes[i] * L;
    for (int i = 7; i < 14; i++) total_scales += sizes[i] * L;
    
    std::vector<uint32_t> packed(total_packed);
    std::vector<float> scales(total_scales);
    
    // Fill with balanced ternary
    for (auto& w : packed) {
        uint32_t word = 0;
        for (int b = 0; b < 16; b++) {
            int val = rng() % 3;
            uint32_t code = (val == 0) ? 2 : (val == 1) ? 0 : 1;  // 2→-1, 0→skip, 1→+1
            word |= (code << (b * 2));
        }
        w = word;
    }
    for (auto& s : scales) s = rand_f(rng, 0.0005f, 0.005f);
    
    // Norm weights (close to 1)
    auto make_norm = [&](int N) {
        std::vector<float> v(N * L);
        for (auto& x : v) x = rand_f(rng, 0.95f, 1.05f);
        return v;
    };
    auto in_norm = make_norm(H);
    auto pa_norm = make_norm(H);
    auto q_norm  = make_norm(HD);
    auto k_norm  = make_norm(HD);
    
    std::vector<float> final_norm(H);
    for (auto& x : final_norm) x = rand_f(rng, 0.95f, 1.05f);
    
    // Embedding
    std::vector<float> emb(NV * H);
    for (auto& x : emb) x = rand_f(rng, -0.01f, 0.01f);
    
    // RoPE
    std::vector<float> sin_tab(MSL * HD), cos_tab(MSL * HD);
    for (int p = 0; p < MSL; p++) {
        for (int d = 0; d < HD; d++) {
            float theta = p / std::pow(10000.0f, (2.0f * (d / 2)) / HD);
            sin_tab[p * HD + d] = std::sin(theta);
            cos_tab[p * HD + d] = std::cos(theta);
        }
    }
    
    // KV cache
    std::vector<std::vector<float>> k_cache(L, std::vector<float>(MSL * NKV * HD, 0));
    std::vector<std::vector<float>> v_cache(L, std::vector<float>(MSL * NKV * HD, 0));
    
    // Scratch
    int qkv_sz = NH*HD + 2*NKV*HD;
    std::vector<float> sq(qkv_sz), sa(NH*HD), sf(2*IM), sa2(IM);
    
    // Hidden state
    std::vector<float> hidden(H);
    for (auto& x : hidden) x = rand_f(rng, -0.01f, 0.01f);
    
    // ── Helper: get layer weight pointers ──
    auto layer_ptr = [&](int l, int& p_off, int& s_off) {
        p_off = 0; s_off = 0;
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < 7; j++) p_off += sizes[j];
            for (int j = 7; j < 14; j++) s_off += sizes[j];
        }
    };
    
    printf("Initial hidden state: ");
    for (int i = 0; i < 4; i++) printf("%+.6f ", hidden[i]);
    printf("...\n");
    
    // ── Layer loop ──
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int l = 0; l < L; l++) {
        int p_off, s_off;
        layer_ptr(l, p_off, s_off);
        
        // Build per-layer weight pointers
        auto pw = [&](int off) { return packed.data() + p_off + off; };
        auto sw = [&](int off) { return scales.data() + s_off + off; };
        
        const uint32_t* qp = pw(0);
        const uint32_t* kp = pw(sizes[0]);
        const uint32_t* vp = pw(sizes[0] + sizes[1]);
        const uint32_t* op = pw(sizes[0] + sizes[1] + sizes[2]);
        const uint32_t* gp = pw(sizes[0] + sizes[1] + sizes[2] + sizes[3]);
        const uint32_t* up = pw(sizes[0] + sizes[1] + sizes[2] + sizes[3] + sizes[4]);
        const uint32_t* dp = pw(sizes[0] + sizes[1] + sizes[2] + sizes[3] + sizes[4] + sizes[5]);
        
        const float* qs = sw(0);
        const float* ks = sw(sizes[7]);
        const float* vs = sw(sizes[7] + sizes[8]);
        const float* os = sw(sizes[7] + sizes[8] + sizes[9]);
        const float* gs = sw(sizes[7] + sizes[8] + sizes[9] + sizes[10]);
        const float* us = sw(sizes[7] + sizes[8] + sizes[9] + sizes[10] + sizes[11]);
        const float* ds = sw(sizes[7] + sizes[8] + sizes[9] + sizes[10] + sizes[11] + sizes[12]);
        
        const float* inn = in_norm.data() + l * H;
        const float* pan = pa_norm.data() + l * H;
        const float* qn  = q_norm.data() + l * HD;
        const float* kn  = k_norm.data() + l * HD;
        
        printf("  Layer %2d: ", l);
        fflush(stdout);
        
        int rc = cpu_layer_forward_qwen3(
            hidden.data(), sq.data(), sa.data(), sf.data(), sa2.data(),
            inn, qn, kn, pan,
            (l == L-1) ? final_norm.data() : nullptr,
            qp, qs, kp, ks, vp, vs, op, os,
            gp, gs, up, us, dp, ds,
            k_cache[l].data(), v_cache[l].data(),
            H, IM, NH, NKV, HD, GQA, l,
            sin_tab.data(), cos_tab.data(), EPS
        );
        
        if (rc != 0) { printf("FAILED rc=%d\n", rc); return 1; }
        
        bool ok = true;
        for (int i = 0; i < H; i++) if (!std::isfinite(hidden[i])) ok = false;
        printf("%s\n", ok ? "OK" : "NaN!");
        if (!ok) { printf("  hidden[0]=%f\n", hidden[0]); return 1; }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // LM head
    std::vector<float> logits(NV);
    cpu_lm_head(hidden.data(), emb.data(), logits.data(), NV, H);
    int best = cpu_argmax(logits.data(), NV);
    cpu_softmax(logits.data(), NV);
    
    printf("\nAfter %d layers:\n", L);
    printf("  hidden[0..3]: %.4f %.4f %.4f %.4f\n", hidden[0], hidden[1], hidden[2], hidden[3]);
    printf("  LM head argmax: %d (prob=%.4f)\n", best, logits[best]);
    
    printf("\n── Timing ──\n");
    printf("  %d layers: %.2f ms\n", L, ms);
    printf("  %.2f ms/layer\n", ms / L);
    printf("  Est. decode (1 core, scalar): %.1f tok/s\n", 1000.0 / ms);
    printf("  Est. decode (AVX2):           ~%.0f tok/s\n", 1000.0 / (ms / 3.0));
    printf("  Est. decode (AVX-512):        ~%.0f tok/s\n", 1000.0 / (ms / 6.0));
    
    printf("\n=== CPU TRG ✅ End-to-end PASS ===\n");
    return 0;
}
