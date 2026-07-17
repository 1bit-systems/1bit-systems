// test_lora_parse.cpp — Test .lora file parsing and delta computation
// Standalone test that doesn't need GPU or ZAYA weights.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

struct LoraModule {
    int mod_id;
    int rank, in_dim, out_dim;
    std::vector<float> A, B;
    std::string name() const {
        const char* names[] = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};
        return (mod_id >= 0 && mod_id < 7) ? names[mod_id] : "unknown";
    }
};

struct LoraLayer { std::vector<LoraModule> modules; };

static std::vector<LoraLayer> read_lora(const char* path, float& scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return {}; }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "LORA") { fprintf(stderr, "Bad magic\n"); return {}; }
    uint32_t num_layers;
    f.read((char*)&num_layers, 4);
    f.read((char*)&scale, 4);
    fprintf(stderr, "File: %s\n", path);
    fprintf(stderr, "Layers: %u, Scale: %.4f\n", num_layers, scale);
    std::vector<LoraLayer> layers(num_layers);
    int total = 0;
    size_t total_a_bytes = 0, total_b_bytes = 0;
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t num_mod;
        f.read((char*)&num_mod, 4);
        layers[l].modules.resize(num_mod);
        for (uint32_t m = 0; m < num_mod; m++) {
            LoraModule& mod = layers[l].modules[m];
            f.read((char*)&mod.mod_id, 4);
            f.read((char*)&mod.rank, 4);
            f.read((char*)&mod.in_dim, 4);
            f.read((char*)&mod.out_dim, 4);
            mod.A.resize((size_t)mod.rank * mod.in_dim);
            mod.B.resize((size_t)mod.out_dim * mod.rank);
            f.read((char*)mod.A.data(), mod.A.size() * 4);
            f.read((char*)mod.B.data(), mod.B.size() * 4);
            total_a_bytes += mod.A.size() * 4;
            total_b_bytes += mod.B.size() * 4;
            total++;
            if (l < 3) {
                fprintf(stderr, "  Layer %d: %s rank=%d in=%d out=%d\n",
                        l, mod.name().c_str(), mod.rank, mod.in_dim, mod.out_dim);
            }
        }
    }
    fprintf(stderr, "Total modules: %d\n", total);
    fprintf(stderr, "A matrices: %.1f MB, B matrices: %.1f MB\n",
            total_a_bytes / 1e6, total_b_bytes / 1e6);
    return layers;
}

static std::vector<float> compute_delta(const LoraModule& mod, float scale) {
    std::vector<float> delta((size_t)mod.out_dim * mod.in_dim, 0.0f);
    for (int o = 0; o < mod.out_dim; o++) {
        for (int r = 0; r < mod.rank; r++) {
            float br = mod.B[(size_t)o * mod.rank + r] * scale;
            if (br == 0.0f) continue;
            const float* A_row = mod.A.data() + (size_t)r * mod.in_dim;
            float* delta_row = delta.data() + (size_t)o * mod.in_dim;
            for (int i = 0; i < mod.in_dim; i++) {
                delta_row[i] += br * A_row[i];
            }
        }
    }
    return delta;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <lora_path>\n", argv[0]);
        return 1;
    }

    float scale;
    auto layers = read_lora(argv[1], scale);
    if (layers.empty()) return 1;

    // Compute delta for first layer, first module, verify it's non-trivial
    auto& mod = layers[0].modules[0];
    fprintf(stderr, "\nComputing delta for Layer 0, %s (rank=%d, %dx%d)...\n",
            mod.name().c_str(), mod.rank, mod.out_dim, mod.in_dim);
    
    auto delta = compute_delta(mod, scale);
    
    // Stats
    double sum = 0, sum2 = 0, max_abs = 0;
    for (size_t i = 0; i < delta.size(); i++) {
        float v = delta[i];
        sum += v;
        sum2 += v * v;
        if (fabs(v) > max_abs) max_abs = fabs(v);
    }
    double mean = sum / delta.size();
    double stddev = sqrt(sum2 / delta.size() - mean * mean);
    
    fprintf(stderr, "Delta stats: mean=%.6f, std=%.6f, max_abs=%.6f, size=%zu\n",
            mean, stddev, max_abs, delta.size());
    
    if (max_abs > 0) {
        fprintf(stderr, "\n✓ LoRA file parsed and delta computed successfully\n");
        return 0;
    } else {
        fprintf(stderr, "\n✗ Delta is all zeros (something wrong)\n");
        return 1;
    }
}
