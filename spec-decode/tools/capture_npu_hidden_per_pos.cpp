/**
 * capture_npu_hidden_per_pos.cpp — Capture NPU per-position hidden states for draft training.
 *
 * Unlike capture_npu_hidden.cpp (which only captures the LAST position per example),
 * this captures ALL token positions' hidden states from the NPU target layers.
 *
 * This is essential for training the draft model on NPU-generated data: the draft
 * needs to see every position's features, not just the final one.
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
 *     [num_positions: i32]  (= input_len, but explicit for forward compat)
 *     [features: num_positions × num_layers × hidden_size × f32]
 *       Position-major: [pos0_l0_h0..pos0_l0_hH-1, pos0_l1_h0.., ..., posN_l4_hH-1]
 *
 * Build:
 *   g++ -std=c++23 -O3 -o build/capture_npu_hidden_per_pos tools/capture_npu_hidden_per_pos.cpp \
 *     /home/bcloud/engine/npu/build/dequant_q4nx.o \
 *     -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl -I.
 *
 * Run:
 *   sudo ./build/capture_npu_hidden_per_pos <input_bin> <output_bin> [max_examples]
 *
 * Example:
 *   python3 tools/prepare_npu_inputs.py \
 *     spec-decode/train_data_10k/perfectblend_train_regen.jsonl \
 *     spec-decode/npu_inputs.bin --max 50000
 *   sudo ./build/capture_npu_hidden_per_pos spec-decode/npu_inputs.bin spec-decode/npu_hidden_per_pos.bin
 *   python3 tools/build_cache_from_capture.py spec-decode/npu_hidden_per_pos.bin spec-decode/target_cache_npu_per_pos.pt
 */

#include "../engine/npu_target_model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Config (matches NPUQwen3Target's constants)
// ---------------------------------------------------------------------------
static constexpr int H = 1024;
static constexpr int NC = 28;
static constexpr int NV = 151936;
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
    if (fread(&num, sizeof(int32_t), 1, f) != 1) { fclose(f); return examples; }
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
        return 1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int max_examples = (argc > 3) ? atoi(argv[3]) : 0;

    printf("═══ NPU Per-Position Hidden State Capture ═══\n");
    printf("Input:  %s\n", input_path);
    printf("Output: %s\n", output_path);
    printf("\n");

    // Read pre-tokenized inputs
    printf("Reading pre-tokenized inputs...\n");
    auto examples = read_inputs(input_path, max_examples);
    if (examples.empty()) { fprintf(stderr, "ERROR: No valid inputs read\n"); return 1; }
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

    int32_t num_examples = (int32_t)examples.size();
    fwrite(&num_examples, sizeof(int32_t), 1, out_fp);

    int success = 0;
    auto capture_start = std::chrono::steady_clock::now();
    size_t total_positions = 0;

    for (size_t idx = 0; idx < examples.size(); idx++) {
        const auto& tokens = examples[idx].tokens;
        int32_t seq_len = (int32_t)tokens.size();

        // Run NPU forward
        target.forward(tokens.data(), seq_len, logits.data(), all_hidden.data());

        // Extract per-position hidden states from target layers
        // get_layer_hidden_positions returns position-major: [pos0*l0..l4, pos1*l0..l4, ...]
        std::vector<float> features((size_t)seq_len * NUM_TARGET_LAYERS * H);
        target.get_layer_hidden_positions(
            TARGET_LAYER_IDS, NUM_TARGET_LAYERS, seq_len, features.data());

        // Write output: input_len, input_ids, num_layers, hidden_size, num_positions, features
        fwrite(&seq_len, sizeof(int32_t), 1, out_fp);
        fwrite(tokens.data(), sizeof(int32_t), seq_len, out_fp);

        int32_t nl = NUM_TARGET_LAYERS;
        int32_t hs = H;
        fwrite(&nl, sizeof(int32_t), 1, out_fp);
        fwrite(&hs, sizeof(int32_t), 1, out_fp);
        fwrite(&seq_len, sizeof(int32_t), 1, out_fp);  // num_positions
        fwrite(features.data(), sizeof(float), features.size(), out_fp);

        total_positions += seq_len;
        success++;

        if ((idx + 1) % 10 == 0) {
            double elapsed = std::chrono::duration<double, std::chrono::seconds::period>(
                std::chrono::steady_clock::now() - capture_start).count();
            printf("  Progress: %zu/%zu (%.0f s, %.1f ex/min, %.0f pos/s)\n",
                   idx + 1, examples.size(), elapsed,
                   (double)(idx + 1) / (elapsed / 60.0),
                   (double)total_positions / elapsed);
        }
    }

    fclose(out_fp);

    struct stat st;
    stat(output_path, &st);

    double total_s = std::chrono::duration<double, std::chrono::seconds::period>(
        std::chrono::steady_clock::now() - capture_start).count();

    printf("\n═══ Capture Complete ═══\n");
    printf("  Examples:   %d\n", success);
    printf("  Positions:  %zu total (avg %.1f per example)\n",
           total_positions, (double)total_positions / success);
    printf("  File:       %s (%.1f MB)\n", output_path, st.st_size / (1024.0 * 1024.0));
    printf("  Time:       %.0f s (%.1f min, %.1f pos/s)\n",
           total_s, total_s / 60.0, (double)total_positions / total_s);

    return success > 0 ? 0 : 1;
}
