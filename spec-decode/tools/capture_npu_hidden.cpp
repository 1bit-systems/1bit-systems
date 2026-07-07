/**
 * capture_npu_hidden.cpp — Capture NPU-generated hidden states for draft training.
 *
 * Reads pre-tokenized inputs (from prepare_npu_inputs.py), runs NPU forward on
 * each, and dumps per-layer hidden states to a flat binary file.
 *
 * Input binary format (from prepare_npu_inputs.py):
 *   [num_examples: i32]
 *   For each example:
 *     [input_len: i32]
 *     [input_ids: input_len × i32]
 *
 * Output binary format (for build_cache_from_capture.py):
 *   [num_examples: i32]
 *   For each example:
 *     [input_len: i32]
 *     [input_ids: input_len × i32]
 *     [num_layers: i32]  (= 5)
 *     [hidden_size: i32] (= 1024)
 *     [features: 5 × 1024 × f32]  (row-major: [layer_idx * hidden_size + feature])
 *
 * Build:
 *   g++ -std=c++23 -O3 -o build/capture_npu_hidden tools/capture_npu_hidden.cpp \
 *     /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *     -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -I.
 *
 * Run:
 *   sudo ./build/capture_npu_hidden <input_bin> <output_bin> [max_examples]
 *
 * Example:
 *   # Step 1: Prepare inputs (Python, uses real Qwen3 tokenizer)
 *   python3 tools/prepare_npu_inputs.py \
 *     train_data_10k/perfectblend_train_regen.jsonl \
 *     npu_inputs.bin --max 200
 *
 *   # Step 2: Capture NPU hidden states
 *   ./build/capture_npu_hidden npu_inputs.bin npu_hidden_cache.bin
 *
 *   # Step 3: Build .pt cache
 *   python3 tools/build_cache_from_capture.py npu_hidden_cache.bin target_cache_npu.pt
 *
 *   # Step 4: Train on NPU data
 *   python3 train_from_cache.py
 */

#include "../engine/npu_target_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Config (matches NPUQwen3Target's constants)
// ---------------------------------------------------------------------------
static constexpr int H = 1024;          // hidden size
static constexpr int NC = 28;           // num layers
static constexpr int NV = 151936;       // vocab size
static constexpr int NUM_TARGET_LAYERS = 5;
static constexpr int TARGET_LAYER_IDS[NUM_TARGET_LAYERS] = {1, 6, 12, 18, 24};

static const char* kModelPath  = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir  = "/home/bcloud/npu-sandbox/npu-infer/build/int8";

// ---------------------------------------------------------------------------
// Read binary input file
// ---------------------------------------------------------------------------
struct InputExample {
    std::vector<int32_t> tokens;
};

static std::vector<InputExample> read_inputs(const char* path, int max_examples) {
    std::vector<InputExample> examples;
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen inputs"); return examples; }

    int32_t num;
    if (fread(&num, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return examples;
    }

    if (max_examples > 0 && num > max_examples) num = max_examples;

    for (int32_t i = 0; i < num; i++) {
        int32_t len;
        if (fread(&len, sizeof(int32_t), 1, f) != 1) break;
        if (len <= 0 || len > 4096) break;

        InputExample ex;
        ex.tokens.resize(len);
        if (fread(ex.tokens.data(), sizeof(int32_t), len, f) != (size_t)len) break;
        examples.push_back(std::move(ex));
    }

    fclose(f);
    return examples;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_bin> <output_bin> [max_examples]\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  input_bin:   Pre-tokenized input binary (from prepare_npu_inputs.py)\n");
        fprintf(stderr, "  output_bin:  Output binary with hidden states\n");
        fprintf(stderr, "  max_examples: Max examples to process (default: all)\n");
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int max_examples = (argc > 3) ? atoi(argv[3]) : 0;

    printf("═══ NPU Hidden State Capture ═══\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("\n");

    // Read pre-tokenized inputs
    printf("Reading pre-tokenized inputs...\n");
    auto examples = read_inputs(input_path, max_examples);
    if (examples.empty()) {
        fprintf(stderr, "ERROR: No valid inputs read\n");
        return 1;
    }
    printf("  Read %zu examples\n", examples.size());

    // Load NPU target model
    printf("Loading NPU target model...\n");
    auto t0 = std::chrono::steady_clock::now();

    NPUQwen3Target target(kModelPath, kXclbinDir, TARGET_LAYER_IDS, NUM_TARGET_LAYERS);

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  Loaded in %.0f ms\n\n", load_ms);

    // Buffers
    std::vector<float> logits(NV);
    std::vector<float> all_hidden((size_t)NC * H);

    // Open output file
    FILE* out_fp = fopen(output_path, "wb");
    if (!out_fp) { perror("fopen output"); return 1; }

    // Write placeholder header
    int32_t num_examples = (int32_t)examples.size();
    fwrite(&num_examples, sizeof(int32_t), 1, out_fp);

    int success = 0;
    int errors = 0;
    auto capture_start = std::chrono::steady_clock::now();

    for (size_t idx = 0; idx < examples.size(); idx++) {
        const auto& tokens = examples[idx].tokens;
        int32_t seq_len = (int32_t)tokens.size();

        // Run NPU forward
        target.forward(tokens.data(), seq_len, logits.data(), all_hidden.data());

        // Extract target layers' hidden states
        std::vector<float> features((size_t)NUM_TARGET_LAYERS * H);
        for (int li = 0; li < NUM_TARGET_LAYERS; li++) {
            int layer_id = TARGET_LAYER_IDS[li];
            memcpy(&features[(size_t)li * H], &all_hidden[(size_t)layer_id * H], (size_t)H * sizeof(float));
        }

        // Write output: input_len, input_ids, num_layers, hidden_size, features
        fwrite(&seq_len, sizeof(int32_t), 1, out_fp);
        fwrite(tokens.data(), sizeof(int32_t), seq_len, out_fp);

        int32_t nl = NUM_TARGET_LAYERS;
        int32_t hs = H;
        fwrite(&nl, sizeof(int32_t), 1, out_fp);
        fwrite(&hs, sizeof(int32_t), 1, out_fp);
        fwrite(features.data(), sizeof(float), features.size(), out_fp);

        success++;

        if ((idx + 1) % 10 == 0) {
            double elapsed = std::chrono::duration<double, std::chrono::seconds::period>(
                std::chrono::steady_clock::now() - capture_start).count();
            printf("  Progress: %zu/%zu (%.0f s, %.1f ex/min)\n",
                   idx + 1, examples.size(), elapsed,
                   (double)(idx + 1) / (elapsed / 60.0));
        }
    }

    fclose(out_fp);

    struct stat st;
    stat(output_path, &st);

    double total_s = std::chrono::duration<double, std::chrono::seconds::period>(
        std::chrono::steady_clock::now() - capture_start).count();

    printf("\n═══ Capture Complete ═══\n");
    printf("  Success:  %d / %zu examples\n", success, examples.size());
    printf("  Errors:   %d\n", errors);
    printf("  File:     %s (%.1f MB)\n", output_path, st.st_size / (1024.0 * 1024.0));
    printf("  Time:     %.0f s (%.1f min, %.1f ex/min)\n", total_s, total_s / 60.0,
           (double)success / (total_s / 60.0));

    return success > 0 ? 0 : 1;
}
