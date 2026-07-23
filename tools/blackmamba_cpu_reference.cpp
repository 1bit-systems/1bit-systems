// blackmamba_cpu_reference.cpp — scalar CPU reference forward pass for
// BlackMamba (Zyphra) GGUF files (one-time bootstrap conversion from BF16).
//
// BlackMamba alternates two pure block types per layer (never both in one
// layer — the HF checkpoint's "mamba_moe_layers" pattern is "r","8","r","8"...):
//   "r" (regular): x = x + mamba1_ssm(rmsnorm(x))
//   "8" (MoE):     x = x + top1_moe_swiglu(rmsnorm(x))
// Followed by: logits = tied_embedding @ rmsnorm(x)
//
// This is a correctness-first, unoptimized scalar reference — not the
// project's fast path (no Q4NX tiling, no GPU kernels, single-token
// recurrent SSM update rather than a parallel scan). It exists to get a
// real, honest baseline number and a coherence check (matches the same
// "not exact-token-match, but not degenerately stuck" bar used by
// tests/test_hip_generic_attention.cpp) before any faster path exists.
//
// Usage: blackmamba_cpu_reference <model.gguf> [n_tokens]

#include "gguf_reader.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

static float silu(float x) { return x / (1.0f + std::expf(-x)); }
static float softplus(float x) { return x > 20.0f ? x : std::log1pf(std::expf(x)); }

static void rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                     std::vector<float>& out, float eps = 1e-5f) {
    double ss = 0.0;
    for (float v : x) ss += (double)v * v;
    float inv = 1.0f / std::sqrt((float)(ss / x.size()) + eps);
    out.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) out[i] = x[i] * inv * w[i];
}

// y = W @ x for a PyTorch nn.Linear-style weight (original shape (out,in))
// that the Python converter wrote as `W.T` -- numpy .flatten() on a
// transposed array respects the *logical* (in,out) shape, giving flat
// layout W_flat[i*out_dim + o] = W_original[o,i]. Verified empirically
// against a known small matrix (a plausible-but-wrong [out,in] reading of
// this same data produces numerically different, wrong output -- this
// isn't a hypothetical, it silently produced non-degenerate-looking but
// numerically incorrect logits for the first several runs of this tool).
static void matvec(const std::vector<float>& W, int out_dim, int in_dim,
                    const std::vector<float>& x, std::vector<float>& y) {
    y.assign(out_dim, 0.0);
    std::vector<double> acc(out_dim, 0.0);
    for (int i = 0; i < in_dim; i++) {
        const float* row = &W[(size_t)i * out_dim];
        float xi = x[i];
        for (int o = 0; o < out_dim; o++) acc[o] += (double)row[o] * xi;
    }
    for (int o = 0; o < out_dim; o++) y[o] = (float)acc[o];
}

// y = W @ x for a weight written WITHOUT a .T in the converter (only
// token_embd.weight / the tied LM head): original numpy shape (out_dim,
// in_dim) flattened directly, so it's already row-major [out_dim, in_dim].
static void matvec_untransposed(const std::vector<float>& W, int out_dim, int in_dim,
                                 const std::vector<float>& x, std::vector<float>& y) {
    y.assign(out_dim, 0.0f);
    for (int o = 0; o < out_dim; o++) {
        const float* row = &W[(size_t)o * in_dim];
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) acc += (double)row[i] * x[i];
        y[o] = (float)acc;
    }
}

struct MambaLayer {
    std::vector<float> in_proj, conv_w, conv_b, x_proj, dt_w, dt_b, A_log, D, out_proj, norm_w;
    int d_inner = 0, d_state = 0, dt_rank = 0, d_conv = 0;
    // Persistent state
    std::vector<float> ssm_state;      // [d_inner, d_state]
    std::vector<float> conv_state;     // [d_inner, d_conv-1] ring buffer, oldest-first
};

struct MoeLayer {
    std::vector<float> router_w, router_b, norm_w;
    std::vector<std::vector<float>> fc1, fc2;  // per expert
    int n_experts = 0, ffn_hidden = 0;
};

struct Layer {
    bool is_mamba = false;
    MambaLayer mamba;
    MoeLayer moe;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_tokens]\n", argv[0]);
        return 1;
    }
    std::string path = argv[1];
    int n_tokens = argc > 2 ? std::atoi(argv[2]) : 10;

    GgufReader r;
    if (!r.open(path)) { fprintf(stderr, "FAIL: could not open %s\n", path.c_str()); return 1; }

    uint32_t H = 0, L = 0, V = 0, d_state = 0, d_conv = 0;
    r.get_u32("mamba.embedding_length", H);
    r.get_u32("mamba.block_count", L);
    r.get_u32("mamba.vocab_size", V);
    r.get_u32("mamba.ssm.state_size", d_state);
    r.get_u32("mamba.ssm.conv_kernel", d_conv);
    if (!H || !L || !V) { fprintf(stderr, "FAIL: could not read model config\n"); return 1; }
    printf("Model: H=%u L=%u V=%u d_state=%u d_conv=%u\n", H, L, V, d_state, d_conv);

    std::vector<float> embed, final_norm;
    size_t n;
    if (!r.get_tensor_f32("token_embd.weight", embed, &n)) { fprintf(stderr, "FAIL: no token_embd.weight\n"); return 1; }
    r.get_tensor_f32("output_norm.weight", final_norm);

    std::vector<Layer> layers(L);
    for (uint32_t l = 0; l < L; l++) {
        char pfx[64]; snprintf(pfx, sizeof(pfx), "blk.%u.", l);
        std::string p(pfx);
        auto& lay = layers[l];
        std::vector<float> tmp;
        if (r.has_tensor(p + "ssm_in.weight")) {
            lay.is_mamba = true;
            auto& m = lay.mamba;
            m.d_state = d_state; m.d_conv = d_conv;
            r.get_tensor_f32(p + "attn_norm.weight", m.norm_w);
            r.get_tensor_f32(p + "ssm_in.weight", m.in_proj);
            m.d_inner = (int)(m.in_proj.size() / H) / 2;
            r.get_tensor_f32(p + "ssm_conv1d.weight", m.conv_w);
            r.get_tensor_f32(p + "ssm_conv1d.bias", m.conv_b);
            r.get_tensor_f32(p + "ssm_x.weight", m.x_proj);
            m.dt_rank = (int)(m.x_proj.size() / m.d_inner) - 2 * d_state;
            r.get_tensor_f32(p + "ssm_dt.weight", m.dt_w);
            r.get_tensor_f32(p + "ssm_dt.bias", m.dt_b);
            r.get_tensor_f32(p + "ssm_a", m.A_log);
            r.get_tensor_f32(p + "ssm_d", m.D);
            r.get_tensor_f32(p + "ssm_out.weight", m.out_proj);
            m.ssm_state.assign((size_t)m.d_inner * d_state, 0.0f);
            m.conv_state.assign((size_t)m.d_inner * (d_conv - 1), 0.0f);
        } else if (r.has_tensor(p + "ffn_gate.weight")) {
            lay.is_mamba = false;
            auto& e = lay.moe;
            r.get_tensor_f32(p + "attn_norm.weight", e.norm_w);
            r.get_tensor_f32(p + "ffn_gate.weight", e.router_w);
            r.get_tensor_f32(p + "ffn_gate.bias", e.router_b);  // may be absent on older exports
            e.n_experts = (int)(e.router_w.size() / H);
            e.fc1.resize(e.n_experts); e.fc2.resize(e.n_experts);
            for (int x = 0; x < e.n_experts; x++) {
                char ep[96]; snprintf(ep, sizeof(ep), "blk.%u.ffn_expert.%d.weight_1", l, x);
                r.get_tensor_f32(ep, e.fc1[x]);
                snprintf(ep, sizeof(ep), "blk.%u.ffn_expert.%d.weight_2", l, x);
                r.get_tensor_f32(ep, e.fc2[x]);
            }
            e.ffn_hidden = (int)(e.fc1[0].size() / H) / 2;  // fused gate+up
        } else {
            fprintf(stderr, "FAIL: layer %u has neither ssm_in.weight nor ffn_gate.weight\n", l);
            return 1;
        }
    }
    printf("Loaded %u layers.\n", L);

    bool dbg = std::getenv("BM_DEBUG_STEPS") != nullptr;
    auto pv = [&](const char* label, const std::vector<float>& v) {
        if (!dbg) return;
        fprintf(stderr, "%s: ", label);
        for (int k = 0; k < 5 && k < (int)v.size(); k++) fprintf(stderr, "%.6f ", v[k]);
        fprintf(stderr, "\n");
    };
    auto mamba_step = [&](MambaLayer& m, const std::vector<float>& x_in, std::vector<float>& out) {
        std::vector<float> xn;
        rmsnorm(x_in, m.norm_w, xn);
        pv("xn", xn);
        std::vector<float> xz;
        matvec(m.in_proj, 2 * m.d_inner, (int)H, xn, xz);
        std::vector<float> x(xz.begin(), xz.begin() + m.d_inner);
        std::vector<float> z(xz.begin() + m.d_inner, xz.end());
        pv("x(pre-conv)", x);
        pv("z", z);

        // Depthwise causal conv1d (kernel=d_conv), per-channel, using ring state.
        std::vector<float> xc(m.d_inner);
        for (int c = 0; c < m.d_inner; c++) {
            double acc = m.conv_b.empty() ? 0.0 : m.conv_b[c];
            for (int k = 0; k < m.d_conv - 1; k++)
                acc += (double)m.conv_state[(size_t)c * (m.d_conv - 1) + k] * m.conv_w[(size_t)c * m.d_conv + k];
            acc += (double)x[c] * m.conv_w[(size_t)c * m.d_conv + (m.d_conv - 1)];
            xc[c] = silu((float)acc);
            // shift ring buffer
            for (int k = 0; k < m.d_conv - 2; k++)
                m.conv_state[(size_t)c * (m.d_conv - 1) + k] = m.conv_state[(size_t)c * (m.d_conv - 1) + k + 1];
            if (m.d_conv > 1) m.conv_state[(size_t)c * (m.d_conv - 1) + (m.d_conv - 2)] = x[c];
        }

        pv("xc(post-conv)", xc);

        std::vector<float> proj;
        matvec(m.x_proj, m.dt_rank + 2 * (int)m.d_state, m.d_inner, xc, proj);
        std::vector<float> dt_low(proj.begin(), proj.begin() + m.dt_rank);
        std::vector<float> Bv(proj.begin() + m.dt_rank, proj.begin() + m.dt_rank + m.d_state);
        std::vector<float> Cv(proj.begin() + m.dt_rank + m.d_state, proj.end());
        pv("dt_low", dt_low);
        pv("Bv", Bv);
        pv("Cv", Cv);

        std::vector<float> dt;
        matvec(m.dt_w, m.d_inner, m.dt_rank, dt_low, dt);
        for (int i = 0; i < m.d_inner; i++) dt[i] = softplus(dt[i] + m.dt_b[i]);
        pv("dt", dt);

        out.assign(m.d_inner, 0.0f);
        for (int i = 0; i < m.d_inner; i++) {
            double y = 0.0;
            for (int s = 0; s < (int)m.d_state; s++) {
                // ssm_a was written as A_log.T.flatten() by the converter
                // (original shape (d_inner, d_state)) -- .T makes it
                // logically (d_state, d_inner) before flatten, so the flat
                // layout is state-major: flat[s*d_inner + i], not [i*d_state+s].
                float A = -std::expf(m.A_log[(size_t)s * m.d_inner + i]);
                float dA = std::expf(dt[i] * A);
                float dB = dt[i] * Bv[s];
                float& h = m.ssm_state[(size_t)i * m.d_state + s];
                h = dA * h + dB * xc[i];
                y += (double)Cv[s] * h;
            }
            y += (double)m.D[i] * xc[i];
            out[i] = (float)y * silu(z[i]);
        }
        std::vector<float> final_out;
        matvec(m.out_proj, (int)H, m.d_inner, out, final_out);
        out = final_out;
    };

    auto moe_step = [&](MoeLayer& e, const std::vector<float>& x_in, std::vector<float>& out) {
        std::vector<float> xn;
        rmsnorm(x_in, e.norm_w, xn);
        std::vector<float> logits;
        matvec(e.router_w, e.n_experts, (int)H, xn, logits);
        if (!e.router_b.empty()) for (int i = 0; i < e.n_experts; i++) logits[i] += e.router_b[i];
        // BlackMamba's default routing_mode is "sinkhorn" (switch_mlp.py):
        // argmax is over sigmoid(logits), and the winning expert's output is
        // scaled by that sigmoid value directly -- NOT a softmax probability
        // over all experts. sigmoid is monotonic so argmax is unaffected,
        // but the gate weight formula is different (verified against
        // github.com/Zyphra/BlackMamba's switch_mlp.py SwitchMLP.forward).
        int top1 = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        float gate_w = 1.0f / (1.0f + std::expf(-logits[top1]));

        std::vector<float> fc1_out;
        matvec(e.fc1[top1], 2 * e.ffn_hidden, (int)H, xn, fc1_out);
        std::vector<float> act(e.ffn_hidden);
        for (int i = 0; i < e.ffn_hidden; i++)
            act[i] = silu(fc1_out[i]) * fc1_out[e.ffn_hidden + i];  // [gate; up]
        matvec(e.fc2[top1], (int)H, e.ffn_hidden, act, out);
        for (auto& v : out) v *= gate_w;
    };

    bool debug_layer0 = std::getenv("BM_DEBUG_LAYER0") != nullptr;
    auto forward = [&](int token_id, std::vector<float>& logits) {
        std::vector<float> x(H);
        for (uint32_t i = 0; i < H; i++) x[i] = embed[(size_t)token_id * H + i];
        int li = 0;
        for (auto& lay : layers) {
            std::vector<float> delta;
            if (lay.is_mamba) mamba_step(lay.mamba, x, delta);
            else moe_step(lay.moe, x, delta);
            if (debug_layer0 && li == 0) {
                fprintf(stderr, "layer0 output (first 8): ");
                for (int k = 0; k < 8; k++) fprintf(stderr, "%.6f ", delta[k]);
                double nrm = 0; for (float v : delta) nrm += (double)v * v;
                fprintf(stderr, "\nlayer0 output norm: %.6f\n", std::sqrt(nrm));
                std::exit(0);
            }
            for (uint32_t i = 0; i < H; i++) x[i] += delta[i];
            li++;
        }
        std::vector<float> xn;
        rmsnorm(x, final_norm, xn);
        matvec_untransposed(embed, (int)V, (int)H, xn, logits);
    };

    // Fixed arbitrary prompt tokens (semantics don't matter -- only that the
    // pipeline produces varied, non-degenerate output across positions, same
    // bar as tests/test_hip_generic_attention.cpp).
    std::vector<int> prompt = {1212, 318, 257, 1332, 13};
    std::vector<int> predicted;
    auto t0 = std::chrono::steady_clock::now();
    int last = prompt[0];
    for (size_t i = 0; i < prompt.size(); i++) {
        std::vector<float> logits;
        forward(prompt[i], logits);
        last = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    predicted.push_back(last);
    for (int i = 1; i < n_tokens; i++) {
        std::vector<float> logits;
        forward(last, logits);
        last = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
        predicted.push_back(last);
    }
    auto t1 = std::chrono::steady_clock::now();
    double decode_s = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_tok = 1000.0 * decode_s / n_tokens;

    printf("Predicted tokens: ");
    for (int t : predicted) printf("%d ", t);
    printf("\n");
    std::vector<int> uniq(predicted.begin(), predicted.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    bool degenerate = uniq.size() <= 1 && n_tokens > 2;
    printf("Unique tokens: %zu / %d  %s\n", uniq.size(), n_tokens, degenerate ? "DEGENERATE (stuck)" : "varied (not stuck)");
    printf("Decode: %.1f ms/tok, %.2f tok/s (scalar CPU reference, unoptimized, single-threaded)\n",
           ms_per_tok, 1000.0 / ms_per_tok);
    return degenerate ? 1 : 0;
}
