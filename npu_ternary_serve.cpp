/**
 * npu_ternary_serve — stdin/stdout JSON server for native ternary NPU inference
 *
 * Wraps NpuTernaryTarget with a simple line-based JSON protocol.
 * The daemon spawns this as a subprocess and communicates via pipe.
 *
 * Protocol (one JSON object per line):
 *   Input:  {"tokens": [int...], "max_new_tokens": int}
 *   Output: {"tokens": [int...], "finished": bool, "error": str?}
 *
 * Build:
 *   g++ -std=c++23 -O2 -o npu_ternary_serve npu_ternary_serve.cpp \
 *       -I/usr/include/xrt -I/opt/xilinx/xrt/include -I. \
 *       -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -fopenmp -lm -luuid
 *
 * Usage:
 *   ./npu_ternary_serve <model.q4nx> <xclbin_dir>
 */

#include "spec-decode/engine/npu_ternary_target.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>

// ── Minimal JSON parser (no dependencies) ───────────────

static std::string json_get_str(const char* js, size_t jl, const char* key) {
    // Find "key": then extract string value
    std::string kq = "\"" + std::string(key) + "\"";
    auto p = strstr(js, kq.c_str());
    if (!p) return "";
    p += kq.size();
    while (*p == ':' || *p == ' ') p++;
    if (*p != '"') return "";  // not a string
    p++;
    auto start = p;
    while (*p && *p != '"') p++;
    return std::string(start, p - start);
}

static int json_get_int(const char* js, size_t jl, const char* key) {
    std::string kq = "\"" + std::string(key) + "\"";
    auto p = strstr(js, kq.c_str());
    if (!p) return -1;
    p += kq.size();
    while (*p == ':' || *p == ' ') p++;
    return atoi(p);
}

static std::vector<int32_t> json_get_int_array(const char* js, size_t jl, const char* key) {
    std::vector<int32_t> result;
    std::string kq = "\"" + std::string(key) + "\"";
    auto p = strstr(js, kq.c_str());
    if (!p) return result;
    p += kq.size();
    while (*p == ':' || *p == ' ' || *p == '[') p++;
    while (*p && *p != ']') {
        char* end;
        long val = strtol(p, &end, 10);
        if (end == p) break;
        result.push_back((int32_t)val);
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    return result;
}

// ── Main ────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.q4nx> <xclbin_dir>\n", argv[0]);
        fprintf(stderr, "Reads JSON from stdin, writes JSON to stdout.\n");
        return 1;
    }

    const char* model_path = argv[1];
    const char* xclbin_dir = argv[2];

    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered for line protocol

    // Target layers for draft feature extraction
    int32_t target_layers[] = {4, 10, 18, 26};
    int num_target = 4;

    fprintf(stderr, "[serve] Loading model: %s\n", model_path);
    fprintf(stderr, "[serve] Xclbin dir:   %s\n", xclbin_dir);

    NpuTernaryTarget target(model_path, xclbin_dir, target_layers, num_target);

    fprintf(stderr, "[serve] Ready — waiting for requests...\n");

    // ── Request loop ────────────────────────────
    char line[65536];
    std::vector<float> logits(151936);
    std::vector<float> hidden(1024);

    while (fgets(line, sizeof(line), stdin)) {
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (ll == 0) continue;

        auto tokens = json_get_int_array(line, ll, "tokens");
        int max_new = json_get_int(line, ll, "max_new_tokens");
        if (max_new <= 0) max_new = 64;

        if (tokens.empty()) {
            printf("{\"error\":\"no tokens in request\"}\n");
            continue;
        }

        std::vector<int32_t> output_tokens;
        bool finished = false;

        // Run forward on the input token(s)
        // For a single token: embed, run layers, get logits, sample argmax
        int32_t token = tokens.back();  // use the last token
        target.forward(&token, 1, logits.data(), hidden.data());

        // Greedy decode: pick argmax
        float best_score = -1e30f;
        int best_token = 0;
        for (int i = 0; i < 151936; i++) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best_token = i;
            }
        }

        output_tokens.push_back(best_token);

        // Build JSON response
        printf("{\"tokens\":[");
        for (size_t i = 0; i < output_tokens.size(); i++) {
            if (i) printf(",");
            printf("%d", output_tokens[i]);
        }
        printf("],\"finished\":%s}\n", finished ? "true" : "false");
    }

    fprintf(stderr, "[serve] stdin closed, exiting.\n");
    return 0;
}
