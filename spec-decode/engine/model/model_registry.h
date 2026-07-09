#pragma once
// Model Registry — Generic model interface + registry for multi-architecture support.
//
// Supports registering new model architectures at runtime via a factory pattern.
// New models implement ModelInterface and register via REGISTER_MODEL("name", factory).
//
// Currently supported:
//   - Qwen3-0.6B  (default)
//   - Qwen3-1.7B  (extensible)
//   - LLaMA-3.2-1B (extensible)
//
// Each model defines:
//   - Architecture config (heads, layers, dims, etc.)
//   - Weight loading from various formats (GGUF, Safetensors, raw)
//   - Forward pass implementation using kernel library
//   - KV cache management
//   - Speculative decoding hooks

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <span>
#include <expected>
#include <cstring>
#include <algorithm>
#include <bit>

namespace specdecode::model {

// ─── Model Architecture Configuration ───────────────────────────────────────

struct ModelConfig {
    // Identity
    std::string name;
    std::string family;           // "qwen3", "llama", etc.
    int64_t parameter_count = 0;

    // Architecture
    int32_t hidden_size = 1024;
    int32_t num_layers = 28;
    int32_t num_heads = 16;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    int32_t inter_dim = 3072;     // FFN intermediate (SwiGLU: gate+up dim)
    int32_t vocab_size = 151936;  // Qwen3 tokenizer
    int32_t max_seq_len = 4096;
    int32_t gqa_ratio = 2;       // num_heads / num_kv_heads

    // Normalization
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string rope_scaling_type; // "none", "linear", "dynamic", "yarn"

    // Quantization
    bool use_int8_weights = true;
    bool use_int8_kv_cache = false;

    // Speculative decoding
    bool supports_spec_decode = true;
    int32_t num_draft_layers = 5;
    int32_t num_target_features = 5; // Hidden states fed to draft
    std::vector<int32_t> target_layer_ids; // Layers to extract for draft

    // Helpers
    int32_t qkv_proj_dim() const noexcept { return (num_heads + 2 * num_kv_heads) * head_dim; }
    int32_t ffn_proj_dim() const noexcept { return 2 * inter_dim; } // gate + up
};

// ─── Weight Store ───────────────────────────────────────────────────────────
//
// Generic tensor storage. Tensors are named (e.g., "model.layers.0.self_attn.q_proj.weight")
// and stored as flat float32 arrays.

class WeightStore {
public:
    using TensorMap = std::unordered_map<std::string, std::vector<float>>;

    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    std::span<const float> get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return {};
        return it->second;
    }

    std::span<float> get_mut(const std::string& name) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return {};
        return it->second;
    }

    void put(std::string name, std::vector<float> data) {
        tensors_[std::move(name)] = std::move(data);
    }

    // Get tensor shape from name suffix
    std::pair<int32_t, int32_t> shape(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return {0, 0};
        // Flat tensors: infer shape from size
        return {(int32_t)it->second.size(), 1};
    }

    size_t size() const noexcept { return tensors_.size(); }

    // Merge another weight store into this one
    void merge(const WeightStore& other) {
        for (auto& [k, v] : other.tensors_) {
            tensors_[k] = v;
        }
    }

    auto begin() const { return tensors_.begin(); }
    auto end() const { return tensors_.end(); }

private:
    TensorMap tensors_;
};

// ─── Forward Pass Output ────────────────────────────────────────────────────

struct ForwardOutput {
    std::vector<float> logits;         // [vocab_size]
    std::vector<float> last_hidden;    // [hidden_size]
    std::vector<float> all_hidden;     // [num_layers * hidden_size] (last position per layer)

    // Per-position hidden states (for prefill)
    std::vector<float> all_positions_hidden; // [seq_len * num_layers * hidden_size]
    bool has_all_positions = false;
};

// ─── Model Interface ────────────────────────────────────────────────────────

class ModelInterface {
public:
    virtual ~ModelInterface() = default;

    // Model identity
    virtual const ModelConfig& config() const = 0;
    virtual const std::string& name() const = 0;

    // Weight management
    virtual bool load_weights(const std::string& path) = 0;
    virtual bool weights_loaded() const = 0;
    virtual const WeightStore& weights() const = 0;

    // Forward: prefill (seq_len tokens, no prior context)
    virtual ForwardOutput forward(
        std::span<const int32_t> input_ids,
        bool return_all_hidden = true,
        bool return_all_positions = false
    ) = 0;

    // Forward with KV cache continuation (decode)
    virtual ForwardOutput forward_with_kv(
        std::span<const int32_t> input_ids,
        int32_t past_len,
        bool return_all_hidden = true,
        bool return_all_positions = false
    ) = 0;

    // KV cache management
    virtual void clear_kv_cache() = 0;
    virtual void commit_kv_cache(int32_t start_pos, int32_t n_accept) = 0;
    virtual int32_t kv_cache_length() const = 0;

    // Layer hidden states (for speculative decoding draft features)
    virtual void get_layer_hidden(
        std::span<const float> all_hidden,
        std::span<const int32_t> target_layer_ids,
        std::span<float> out
    ) = 0;

    // INT8 quantization: apply INT8 weights (updates internal quantized matrices)
    virtual bool apply_int8_quantization() { return false; }
};

// ─── Model Factory ──────────────────────────────────────────────────────────

using ModelFactory = std::function<std::unique_ptr<ModelInterface>(const ModelConfig&)>;

class ModelRegistry {
public:
    static ModelRegistry& instance() {
        static ModelRegistry registry;
        return registry;
    }

    // Register a model family
    void register_factory(const std::string& family, ModelFactory factory) {
        factories_[family] = std::move(factory);
    }

    // Create a model by family name (auto-detects config from weight file if possible)
    std::unique_ptr<ModelInterface> create(
        const std::string& family,
        const ModelConfig& cfg
    ) {
        auto it = factories_.find(family);
        if (it == factories_.end()) return nullptr;
        return it->second(cfg);
    }

    // List registered families
    std::vector<std::string> registered_families() const {
        std::vector<std::string> families;
        for (auto& [k, v] : factories_)
            families.push_back(k);
        return families;
    }

    // Check if a family is registered
    bool is_registered(const std::string& family) const {
        return factories_.find(family) != factories_.end();
    }

    // Detect model family from weight file header
    static std::string detect_family(const std::string& weight_path);

private:
    ModelRegistry() = default;
    std::unordered_map<std::string, ModelFactory> factories_;
};

// ─── Model Detection ────────────────────────────────────────────────────────

inline std::string ModelRegistry::detect_family(const std::string& weight_path) {
    // Try to read first few bytes to detect format
    // GGUF: magic bytes 'GGUF'
    // Safetensors: JSON header starting with '{'
    // Raw: check for model identifier in filename
    std::string lower_path = weight_path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower_path.find("qwen") != std::string::npos) return "qwen3";
    if (lower_path.find("llama") != std::string::npos) return "llama";
    if (lower_path.find("mistral") != std::string::npos) return "mistral";
    if (lower_path.find("gemma") != std::string::npos) return "gemma";
    if (lower_path.find("deepseek") != std::string::npos) return "deepseek";

    // Check file extension
    if (lower_path.ends_with(".gguf")) {
        // Read magic
        FILE* f = fopen(weight_path.c_str(), "rb");
        if (f) {
            char magic[4];
            if (fread(magic, 1, 4, f) == 4) {
                fclose(f);
                if (memcmp(magic, "GGUF", 4) == 0) return "gguf-generic";
            } else {
                fclose(f);
            }
        }
    }

    return "unknown";
}

// ─── Register Model Helper ──────────────────────────────────────────────────

#define REGISTER_MODEL(family, factory_fn) \
    namespace { \
        struct Register_##family { \
            Register_##family() { \
                ModelRegistry::instance().register_factory(#family, factory_fn); \
            } \
        } reg_##family##_; \
    }

// ─── Qwen3 Model Configuration ──────────────────────────────────────────────

struct Qwen3Config {
    static ModelConfig config_0_6B() {
        ModelConfig cfg;
        cfg.name = "Qwen3-0.6B";
        cfg.family = "qwen3";
        cfg.parameter_count = 619000000;
        cfg.hidden_size = 1024;
        cfg.num_layers = 28;
        cfg.num_heads = 16;
        cfg.num_kv_heads = 8;
        cfg.head_dim = 128;
        cfg.inter_dim = 3072;
        cfg.vocab_size = 151936;
        cfg.max_seq_len = 4096;
        cfg.gqa_ratio = 2;
        cfg.target_layer_ids = {1, 6, 12, 18, 24};
        return cfg;
    }

    static ModelConfig config_1_7B() {
        ModelConfig cfg;
        cfg.name = "Qwen3-1.7B";
        cfg.family = "qwen3";
        cfg.parameter_count = 1700000000;
        cfg.hidden_size = 2048;
        cfg.num_layers = 28;
        cfg.num_heads = 16;
        cfg.num_kv_heads = 8;
        cfg.head_dim = 128;
        cfg.inter_dim = 8192;
        cfg.vocab_size = 151936;
        cfg.max_seq_len = 8192;
        cfg.gqa_ratio = 2;
        cfg.target_layer_ids = {1, 6, 12, 18, 24};
        return cfg;
    }
};

struct LLaMAConfig {
    static ModelConfig config_3B() {
        ModelConfig cfg;
        cfg.name = "LLaMA-3.2-3B";
        cfg.family = "llama";
        cfg.parameter_count = 3200000000;
        cfg.hidden_size = 3072;
        cfg.num_layers = 28;
        cfg.num_heads = 24;
        cfg.num_kv_heads = 8;
        cfg.head_dim = 128;
        cfg.inter_dim = 8192;
        cfg.vocab_size = 128256;
        cfg.max_seq_len = 8192;
        cfg.gqa_ratio = 3;
        cfg.rope_theta = 500000.0f;
        cfg.target_layer_ids = {1, 6, 12, 18, 24};
        return cfg;
    }
};

} // namespace specdecode::model
