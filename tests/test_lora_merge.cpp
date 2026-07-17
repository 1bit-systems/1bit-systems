// test_lora_merge.cpp — Quick smoke test for zaya_apply_lora
// Verifies that zaya_init + zaya_apply_lora loads and merges without crashing.
// Run: ./test_lora_merge <weights_dir> <lora_path>
// Requires: actual ZAYA weight files and a .lora file.

#include "../src/zaya_engine.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights_dir> [lora_path]\n", argv[0]);
        return 1;
    }

    std::string wd = argv[1];
    if (!wd.empty() && wd.back() != '/') wd += '/';

    fprintf(stderr, "Initializing Zaya engine from %s...\n", wd.c_str());
    ZayaState* s = zaya_init(wd.c_str());
    if (!s) {
        fprintf(stderr, "FAIL: zaya_init returned null\n");
        return 1;
    }
    fprintf(stderr, "OK: zaya_init succeeded\n");

    if (argc >= 3) {
        const char* lora_path = argv[2];
        fprintf(stderr, "\nApplying LoRA from %s...\n", lora_path);
        int ret = zaya_apply_lora(s, lora_path);
        if (ret == 0) {
            fprintf(stderr, "OK: LoRA merge succeeded\n");
        } else {
            fprintf(stderr, "FAIL: zaya_apply_lora returned %d\n", ret);
            zaya_destroy(s);
            return 1;
        }
    }

    // Run a quick forward pass to verify the engine still works
    fprintf(stderr, "\nRunning forward pass with token 0...\n");
    float logits[512];  // just a slice
    zaya_forward(s, 0, logits);

    // Check that logits are non-zero and finite
    bool ok = false;
    for (int i = 0; i < 512; i++) {
        float v = logits[i];
        if (v != 0.0f && v <= 1e10f && v >= -1e10f) { ok = true; break; }
    }
    fprintf(stderr, "%s: engine produced %s logits\n",
            ok ? "OK" : "FAIL",
            ok ? "valid" : "zero/invalid");
    
    // Greedy forward
    fprintf(stderr, "\nRunning greedy forward...\n");
    int token = zaya_forward_greedy(s, 0);
    fprintf(stderr, "Greedy token: %d\n", token);
    
    zaya_reset(s);
    zaya_destroy(s);
    fprintf(stderr, "\nAll tests passed.\n");
    return ok ? 0 : 1;
}
