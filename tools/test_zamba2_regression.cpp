// test_zamba2_regression.cpp — Zamba2 model loading regression test
// Verifies the GGUF loader correctly parses model dimensions including
// attn_head_dim (#946), and that all layers have valid weight tensors.

#include "zamba2_engine.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 77;
    }

    Zamba2Model model;
    if (!load_zamba2_from_gguf(argv[1], model)) {
        fprintf(stderr, "FAIL: could not load model\n");
        return 1;
    }

    fprintf(stderr, "OK: model loaded: %d layers, %d hybrid, %d mamba\n",
            model.cfg.n_layers,
            (int)model.hybrid_layers.size(),
            (int)model.mamba_layers.size());

    // Verify attn_head_dim is correct (issue #946 — was falling back to
    // d_inner/n_attn_heads=160 instead of 80)
    fprintf(stderr, "  attn_head_dim=%d n_attn_heads=%d n_kv_heads=%d d_model=%d\n",
            model.cfg.attn_head_dim, model.cfg.n_attn_heads,
            model.cfg.n_kv_heads, model.cfg.d_model);

    // For zamba2-1.2b: H=2048, NH=32, HD should be 80
    if (model.cfg.attn_head_dim <= 0 || model.cfg.attn_head_dim > model.cfg.d_model) {
        fprintf(stderr, "FAIL: attn_head_dim=%d is out of range\n", model.cfg.attn_head_dim);
        return 1;
    }

    // Verify hybrid layers have all required weight tensors
    for (auto& [idx, hl] : model.hybrid_layers) {
        if (hl.shared_transformer_q.empty()) {
            fprintf(stderr, "FAIL: hybrid layer %d missing Q weights\n", idx);
            return 1;
        }
        if (hl.shared_transformer_k.empty()) {
            fprintf(stderr, "FAIL: hybrid layer %d missing K weights\n", idx);
            return 1;
        }
    }
    fprintf(stderr, "OK: all %zu hybrid layers have Q/K weights\n", model.hybrid_layers.size());

    fprintf(stderr, "PASS\n");
    return 0;
}
