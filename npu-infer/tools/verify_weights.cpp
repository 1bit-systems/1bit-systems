/**
 * verify_weights.cpp
 * Compares one layer of Q4NX dequant weights against HF safetensors weights.
 *
 * Usage: ./verify_weights <layer>
 *
 * Reads:
 *   /tmp/qwen3_raw/layer_N/gu.F32  (HF reference)
 *   Via dequant_i8_to_float from model.q4nx (GPU engine)
 *
 * Outputs: Stats comparison, max absolute diff.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

extern "C" float* dequant_i8_to_float(const uint8_t*, int, int*, int*);

static const char* MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const int H = 1024, IM = 3072;

static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz); fread(buf.data(), 1, sz, f); fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    int layer = 0;
    if (argc > 1) layer = atoi(argv[1]);

    // Load HF reference
    char path[256];
    snprintf(path, 256, "/tmp/qwen3_raw/layer_%d/gu.F32", layer);
    auto hf_raw = read_file(path);
    if (hf_raw.empty()) { fprintf(stderr, "Cannot read %s\n", path); return 1; }
    const float* hf_gu = (const float*)hf_raw.data(); // (H * 6144) in B[k*N+col] layout
    printf("HF GU: %zu elements\n", hf_raw.size()/4);

    // We need gate part (first IM=3072 columns) in B[k*IM+i] layout for comparison
    // hf_gu[k*6144 + i] = gate_proj weight connecting hidden[k] to gate_out[i]
    // For the Q4NX comparison, we want the gate weights in the GEMM-useful layout:
    //   hf_gate[k][i] = hf_gu[k * 6144 + i]  for i=0..3071

    // Load Q4NX via dequant
    auto q4nx = read_file(MODEL);
    uint64_t hdr_len; memcpy(&hdr_len, q4nx.data(), 8);
    uint64_t ds = 8 + hdr_len;
    const char* js = (const char*)(q4nx.data() + 8);

    // Find gate_proj offset for layer
    char key[128];
    snprintf(key, 128, "model.layers.%d.mlp.gate_proj.weight", layer);
    auto foff = [&](const char* n) -> uint64_t {
        const char* p = js; const char* e = js + hdr_len; size_t nl = strlen(n);
        while (p < e) {
            auto q = (const char*)memmem(p, e-p, n, nl);
            if (!q) return 0;
            if (*(q-1)=='"' && *(q+nl)=='"') {
                auto o = strstr(q, "\"data_offsets\"");
                if (o) { auto b = strchr(o, '['); if(b) return strtoull(b+1,NULL,10); }
            }
            p = q + 1;
        }
        return 0;
    };
    uint64_t gp_off = foff(key);
    printf("Q4NX gate_proj offset: %llu\n", (unsigned long long)gp_off);

    int gr, gc;
    float* gw = dequant_i8_to_float(q4nx.data() + ds + gp_off, 384, &gr, &gc);
    printf("Q4NX gate dequant: (%d, %d)\n", gr, gc);

    // gw is (IM=3072, H=1024) in dequant's (out_feat, in_feat) layout
    // To compare with HF which is transposed (H, IM):
    // hf_gate[k][i] = hf_gu[k * 6144 + i]
    // q4nx_gate[k][i] = gw[i * H + k]  (transpose dequant)
    // So compare: hf_gu[k * 6144 + i] vs gw[i * H + k]

    double max_diff = 0, sum_diff = 0, sum_sq = 0;
    int n_diffs = 0, n_gt_1 = 0, n_gt_01 = 0;
    for (int k = 0; k < H; k++) {
        for (int i = 0; i < IM; i++) {
            float hf_val = hf_gu[(size_t)k * 6144 + i];
            float q4_val = gw[(size_t)i * H + k];
            double diff = fabs((double)hf_val - (double)q4_val);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
            sum_sq += diff * diff;
            n_diffs++;
            if (diff > 1.0) n_gt_1++;
            if (diff > 0.1) n_gt_01++;
        }
    }

    printf("\n=== GU Weight Comparison (layer %d) ===\n", layer);
    printf("Total elements: %d\n", n_diffs);
    printf("Max absolute diff: %.6f\n", max_diff);
    printf("Mean absolute diff: %.6f\n", sum_diff / n_diffs);
    printf("RMS diff: %.6f\n", sqrt(sum_sq / n_diffs));
    printf("Elements with diff > 0.1: %d (%.2f%%)\n", n_gt_01, 100.0 * n_gt_01 / n_diffs);
    printf("Elements with diff > 1.0: %d (%.2f%%)\n", n_gt_1, 100.0 * n_gt_1 / n_diffs);

    // Also compare input_norm and other 1D vectors
    printf("\n=== Input norm comparison ===\n");
    {
        auto hf_in_raw = read_file((std::string("/tmp/qwen3_raw/layer_") + std::to_string(layer) + "/in_norm.F32").c_str());
        if (!hf_in_raw.empty()) {
            const float* hf_in = (const float*)hf_in_raw.data();
            // Load Q4NX input norm
            snprintf(key, 128, "model.layers.%d.input_layernorm.weight", layer);
            uint64_t in_off = foff(key);
            const uint16_t* bf16 = (const uint16_t*)(q4nx.data() + ds + in_off);
            float max_diff_n = 0;
            for (int i = 0; i < H; i++) {
                union { uint32_t u; float f; } conv;
                conv.u = (uint32_t)bf16[i] << 16;
                double d = fabs((double)hf_in[i] - (double)conv.f);
                if (d > max_diff_n) max_diff_n = d;
            }
            printf("Max diff for input_norm: %.8f\n", max_diff_n);
        }
    }

    // Compare first few gate values
    printf("\nFirst 16 HF gate values (HF layout, i=0..15):\n  ");
    for (int i = 0; i < 16; i++) printf("%.6f ", hf_gu[(size_t)0*6144+i]);
    printf("\nFirst 16 Q4NX gate values (transposed to match):\n  ");
    for (int i = 0; i < 16; i++) printf("%.6f ", gw[(size_t)i*H+0]);
    printf("\n");

    // HF gate mean and negative ratio
    int neg_hf = 0; double sum_hf = 0;
    for (int i = 0; i < IM; i++) {
        for (int k = 0; k < H; k++) {
            float v = hf_gu[(size_t)k * 6144 + i];
            sum_hf += v;
            if (v < 0) neg_hf++;
        }
    }
    printf("\nHF gate: mean=%.6f, neg=%d/%d (%.1f%%)\n",
           sum_hf/(H*IM), neg_hf, H*IM, 100.0*neg_hf/(H*IM));
    printf("(Compare: Q4NX gate was 94%% negative, all-negative GEMM output)\n");

    free(gw);
    return 0;
}
