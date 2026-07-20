/** onebp_weight_loader.cpp — Load 1BP weights into the NPU engine.
 *
 *  Replaces the Q4NX-specific weight loading section in
 *  npu_engine_universal.cpp when running with -DONEBP_SUPPORT.
 *
 *  Instead of the Q4NX pipeline:
 *    mmap → JSON offsets → dequant_i8_to_float_ex → transpose_pack → packB
 *
 *  This does:
 *    OnebpModel::get_tensor_f32(name) → transpose_pack → packB
 *
 *  Including for norm weights, LM head, and embeddings.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include "onebp_format.h"

// ─── Forward declarations from engine ─────────────────────────────
// These are expected to exist in the including scope
// (I8Ctx, transpose_pack, packB, etc.)

// ─── Load all weights from a 1BP model into the I8 NPU contexts ──
//
// Called during engine init when is_onebp is true.
// Replaces the entire Q4NX weight loading block.
//
// Args:
//   onebp_model  — opened 1BP model (must be valid)
//   NC           — number of layers
//   H, NH, NKV, HD, IM  — model dimensions
//   cq, co, cg, cd      — I8 GEMM contexts (QKV, O, GU, D)
//   qsc, osc, gsc, dsc  — per-layer scale outputs
//   in_n, pa_n, qn_w, kn_w, fin_v  — norm weight outputs
//   emb_f32, lm_head_f32             — embedding / LM head outputs
//
bool onebp_load_weights(
    OnebpModel&           model,
    int                   NC, int H, int NH, int NKV, int HD, int IM,
    I8Ctx&                cq, I8Ctx& co, I8Ctx& cg, I8Ctx& cd,
    std::vector<float>&   qsc, std::vector<float>& osc,
    std::vector<float>&   gsc, std::vector<float>& dsc,
    std::vector<std::vector<float>>& in_n,
    std::vector<std::vector<float>>& pa_n,
    std::vector<std::vector<float>>& qn_w,
    std::vector<std::vector<float>>& kn_w,
    std::vector<float>&   fin_v,
    std::vector<float>&   emb_f32,
    std::vector<float>&   lm_head_f32
) {
    fprintf(stderr, "Loading 1BP weights...\n");
    auto t0 = std::chrono::steady_clock::now();
    
    int QOUT = NH * HD;        // Q projection out
    int KVOUT = NKV * HD;      // K/V projection out
    int qkv_total = QOUT + KVOUT + KVOUT;  // Q+K+V total
    int OOUT = H;              // O projection out
    int OIN = NH * HD;         // O projection in
    int GUOUT = IM;            // Gate/Up out
    int DOUT = H;              // Down out
    int DIN = IM;              // Down in
    
    // Per-layer scale buffers (for I8 quantization)
    qsc.resize(NC);
    osc.resize(NC);
    gsc.resize(NC);
    dsc.resize(NC);
    
    // Norm weight buffers
    in_n.assign(NC, std::vector<float>(H));
    pa_n.assign(NC, std::vector<float>(H));
    qn_w.assign(NC, std::vector<float>(HD));
    kn_w.assign(NC, std::vector<float>(HD));
    fin_v.assign(H, 0);
    
    // Helper: load a float32 tensor by name from the 1BP model
    auto load_f32 = [&](const char* name) -> std::vector<float> {
        std::vector<float> buf;
        model.get_tensor_f32(name, buf);
        return buf;
    };
    
    // Helper: build layer tensor name (GGUF naming)
    auto tn = [](int l, const char* t) -> std::string {
        char b[128];
        snprintf(b, 128, "blk.%d.%s", l, t);
        return b;
    };
    
    // ── 1. Load token embeddings ──
    {
        auto emb = load_f32("token_embd.weight");
        if (!emb.empty()) {
            emb_f32 = emb;
            fprintf(stderr, "  embeddings: %zu elements\n", emb.size());
        } else {
            fprintf(stderr, "  WARN: token_embd.weight not found\n");
        }
    }
    
    // ── 2. Load LM head ──
    {
        auto lm = load_f32("output.weight");
        if (lm.empty()) lm = load_f32("token_embd.weight");  // tied embeddings
        if (!lm.empty()) {
            lm_head_f32.assign(lm.begin(), lm.end());
            fprintf(stderr, "  lm_head: %zu elements\n", lm.size());
        }
    }
    
    // ── 3. Load output norm ──
    {
        auto norm = load_f32("output_norm.weight");
        if (!norm.empty()) {
            fin_v.assign(norm.begin(), norm.end());
            fprintf(stderr, "  output_norm: %zu elements\n", norm.size());
        }
    }
    
    // ── 4. Load per-layer weights ──
    for (int l = 0; l < NC; l++) {
        // Norm weights
        auto in_w = load_f32(tn(l, "attn_norm.weight").c_str());
        if (!in_w.empty()) in_n[l].assign(in_w.begin(), in_w.end());
        
        auto pa_w = load_f32(tn(l, "ffn_norm.weight").c_str());
        if (!pa_w.empty()) pa_n[l].assign(pa_w.begin(), pa_w.end());
        
        auto qn_w_t = load_f32(tn(l, "attn_q_norm.weight").c_str());
        if (!qn_w_t.empty()) qn_w[l].assign(qn_w_t.begin(), qn_w_t.end());
        
        auto kn_w_t = load_f32(tn(l, "attn_k_norm.weight").c_str());
        if (!kn_w_t.empty()) kn_w[l].assign(kn_w_t.begin(), kn_w_t.end());
        
        // ── QKV weights ──
        auto qw = load_f32(tn(l, "attn_q.weight").c_str());
        auto kw = load_f32(tn(l, "attn_k.weight").c_str());
        auto vw = load_f32(tn(l, "attn_v.weight").c_str());
        
        if (!qw.empty() && !kw.empty() && !vw.empty()) {
            int t = qkv_total;
            std::vector<float> w((size_t)H * t);
            transpose_pack(qw.data(), QOUT, H, w.data(), t, 0);
            transpose_pack(kw.data(), KVOUT, H, w.data(), t, QOUT);
            transpose_pack(vw.data(), KVOUT, H, w.data(), t, QOUT + KVOUT);
            cq.packB(l, w.data(), H, t, qsc[l]);
        }
        
        // ── O weights ──
        auto ow = load_f32(tn(l, "attn_output.weight").c_str());
        if (!ow.empty()) {
            std::vector<float> wo((size_t)OIN * OOUT);
            transpose_pack(ow.data(), OOUT, OIN, wo.data(), OOUT, 0);
            co.packB(l, wo.data(), OIN, OOUT, osc[l]);
        }
        
        // ── Gate + Up weights ──
        auto gw = load_f32(tn(l, "ffn_gate.weight").c_str());
        auto uw = load_f32(tn(l, "ffn_up.weight").c_str());
        
        if (!gw.empty() && !uw.empty()) {
            int t2 = GUOUT + GUOUT;
            std::vector<float> w2((size_t)H * t2);
            transpose_pack(gw.data(), GUOUT, H, w2.data(), t2, 0);
            transpose_pack(uw.data(), GUOUT, H, w2.data(), t2, GUOUT);
            cg.packB(l, w2.data(), H, t2, gsc[l]);
        }
        
        // ── Down weights ──
        auto dw = load_f32(tn(l, "ffn_down.weight").c_str());
        if (!dw.empty()) {
            std::vector<float> wd((size_t)DIN * DOUT);
            transpose_pack(dw.data(), DOUT, DIN, wd.data(), DOUT, 0);
            cd.packB(l, wd.data(), DIN, DOUT, dsc[l]);
        }
        
        if (l == 0 || l == NC - 1 || l % 20 == 0)
            fprintf(stderr, "  layer %d/%d\n", l + 1, NC);
    }
    
    // ── 5. Override LM head if tied ──
    if (lm_head_f32.empty() && !emb_f32.empty()) {
        lm_head_f32 = emb_f32;
        fprintf(stderr, "  lm_head: using tied embeddings\n");
    }
    
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "  1BP weights loaded in %.1f sec\n", sec);
    
    return true;
}
