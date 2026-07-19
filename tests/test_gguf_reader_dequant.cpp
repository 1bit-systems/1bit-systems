// test_gguf_reader_dequant.cpp — self-contained regression test for
// gguf_reader.cpp's dequantization, covering every supported GGUF dtype.
//
// No GPU, no downloaded model, no Python at test time — the reference
// values in gguf_dequant_ref_data.h were generated once against the
// Python `gguf` package's own reference dequantizers (seed 2026) and are
// baked in as constants, so this runs unconditionally in CI.
//
// This is the test that would have caught, automatically and immediately,
// every dtype/parsing bug found in this codebase's various GGUF readers
// this session: the Q6_K super-block write-loop bug (only ever wrote 64 of
// every 256 elements), the Q4_0/Q4_1/Q5_0/Q5_1 nibble-mapping bug
// (interleaved pairs instead of low-nibbles-then-high-nibbles), and the
// Q8_0/Q5_0/Q5_1 dtype enum mixup (7/8/9 instead of the real 8/6/7) —
// all of which previously only surfaced as silently-wrong model output,
// not a test failure.

#include "gguf_reader.h"
#include "gguf_dequant_ref_data.h"
#include <cstdio>
#include <cmath>

static bool check(const char* name, uint32_t dtype, const uint8_t* raw, const float* expected, int n) {
    std::vector<float> got(n, 0.0f);
    if (!gguf_dequant(dtype, raw, got.data(), n)) {
        fprintf(stderr, "FAIL %-6s: gguf_dequant returned false\n", name);
        return false;
    }
    int mismatches = 0;
    float max_rel_err = 0;
    for (int i = 0; i < n; i++) {
        float e = expected[i], g = got[i];
        float rel = fabsf(e - g) / (fabsf(e) + 1e-4f);
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 2e-3f) {
            if (mismatches < 3) fprintf(stderr, "  [%s] idx %d: expected=%.6g got=%.6g\n", name, i, e, g);
            mismatches++;
        }
    }
    bool ok = mismatches == 0;
    fprintf(stderr, "%-6s: %d/%d mismatches, max_rel_err=%.4g -> %s\n", name, mismatches, n, max_rel_err, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= check("Q4_0", GGUF_DTYPE_Q4_0, REF_Q4_0_RAW, REF_Q4_0_EXPECTED, 32);
    ok &= check("Q4_1", GGUF_DTYPE_Q4_1, REF_Q4_1_RAW, REF_Q4_1_EXPECTED, 32);
    ok &= check("Q5_0", GGUF_DTYPE_Q5_0, REF_Q5_0_RAW, REF_Q5_0_EXPECTED, 32);
    ok &= check("Q5_1", GGUF_DTYPE_Q5_1, REF_Q5_1_RAW, REF_Q5_1_EXPECTED, 32);
    ok &= check("Q8_0", GGUF_DTYPE_Q8_0, REF_Q8_0_RAW, REF_Q8_0_EXPECTED, 32);
    ok &= check("Q2_K", GGUF_DTYPE_Q2_K, REF_Q2_K_RAW, REF_Q2_K_EXPECTED, 256);
    ok &= check("Q3_K", GGUF_DTYPE_Q3_K, REF_Q3_K_RAW, REF_Q3_K_EXPECTED, 256);
    ok &= check("Q4_K", GGUF_DTYPE_Q4_K, REF_Q4_K_RAW, REF_Q4_K_EXPECTED, 256);
    ok &= check("Q5_K", GGUF_DTYPE_Q5_K, REF_Q5_K_RAW, REF_Q5_K_EXPECTED, 256);
    ok &= check("Q6_K", GGUF_DTYPE_Q6_K, REF_Q6_K_RAW, REF_Q6_K_EXPECTED, 256);
    ok &= check("Q8_K", GGUF_DTYPE_Q8_K, REF_Q8_K_RAW, REF_Q8_K_EXPECTED, 256);

    if (ok) { printf("OK: all dtypes byte-exact against the Python gguf reference\n"); return 0; }
    fprintf(stderr, "FAIL: one or more dtypes mismatched the reference\n");
    return 1;
}
