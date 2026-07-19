// test_hip_generic_attention.cpp — regression test for the HIP backend's
// generic (non-Zaya) attention path.
//
// Guards against the exact bug class fixed in commits b88cce8a6/046d24c8c:
// the generic path used to compute Q/K/V and throw them away (no RoPE, no
// KV cache, no attention at all), and load_gguf_model() had several bugs
// (wrong tensor-naming convention, wrong tensor offsets, a metadata-parsing
// desync) that meant no GGUF model's per-layer weights had ever actually
// loaded. Both failure modes produce the same observable symptom: greedy
// decoding gets stuck predicting the same token (usually 0) at every
// position, regardless of the input. That's what this test actually checks
// for — not exact-token-match (which would be brittle across hardware/driver
// floating-point non-determinism), but "the model isn't degenerately stuck."
//
// Build: cmake --build build --target test_hip_generic_attention -j8
// Run:   ./build/test_hip_generic_attention <model.gguf>
//        (or via ctest with -DGGUF_TEST_MODEL=/path/to/model.gguf)

#include "backend.h"
#include "model_discovery.h"
#include <cstdio>
#include <set>
#include <vector>

std::vector<InferenceBackend*> detect_backends_hip();

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model.gguf>\n"
            "  Pass a path (or run via ctest with GGUF_TEST_MODEL set).\n",
            argv[0]);
        return 77;
    }
    const std::string path = argv[1];

    ModelConfig cfg;
    if (!read_gguf_header(path, cfg)) {
        fprintf(stderr, "FAIL: could not read GGUF header from %s\n", path.c_str());
        return 1;
    }
    cfg.max_seq_len = 32;

    auto backends = detect_backends_hip();
    if (backends.empty()) { fprintf(stderr, "no HIP backend compiled in\n"); return 77; }
    InferenceBackend* be = backends[0];
    if (!be->is_available()) { fprintf(stderr, "HIP backend not available (no GPU)\n"); return 77; }
    if (!be->load_model(cfg)) {
        fprintf(stderr, "FAIL: load_model failed for %s\n", path.c_str());
        return 1;
    }

    // Fixed, arbitrary prompt tokens (valid ids in any BPE vocab of
    // reasonable size — semantics don't matter, only that the pipeline
    // produces varied, sane output across positions).
    const int prompt[] = {9707, 11, 358, 1079, 264};
    std::vector<int> predicted;
    int pos = 0;
    for (int t : prompt) {
        int next = be->forward(t, pos++);
        if (next < 0 || next >= cfg.vocab_size) {
            fprintf(stderr, "FAIL: predicted token %d out of vocab range [0, %d)\n", next, cfg.vocab_size);
            return 1;
        }
        predicted.push_back(next);
    }
    // Continue decoding a few more steps, greedy, feeding predictions back in.
    int last = predicted.back();
    for (int i = 0; i < 10; i++) {
        int next = be->forward(last, pos++);
        if (next < 0 || next >= cfg.vocab_size) {
            fprintf(stderr, "FAIL: predicted token %d out of vocab range [0, %d)\n", next, cfg.vocab_size);
            return 1;
        }
        predicted.push_back(next);
        last = next;
    }

    // The actual regression check: a working model produces varied tokens
    // across 15 decode steps. A broken attention/loading path collapses to
    // the same token (observed: always 0) at every single position.
    std::set<int> distinct(predicted.begin(), predicted.end());
    fprintf(stderr, "predicted %zu tokens, %zu distinct\n", predicted.size(), distinct.size());
    if (distinct.size() <= 1) {
        fprintf(stderr, "FAIL: degenerate output — every prediction was the same token (%d). "
                        "This is the exact symptom of attention being skipped or weights failing "
                        "to load silently.\n", *distinct.begin());
        return 1;
    }

    printf("OK: %zu distinct tokens across %zu decode steps\n", distinct.size(), predicted.size());
    return 0;
}
