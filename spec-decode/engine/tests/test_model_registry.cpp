// Test suite for Model Registry
#include "model/model_registry.h"
#include <cstdio>
#include <cassert>

using namespace specdecode::model;

int test_qwen3_config() {
    printf("=== Qwen3-0.6B Config ===\n");

    auto cfg = Qwen3Config::config_0_6B();

    assert(cfg.hidden_size == 1024);
    assert(cfg.num_layers == 28);
    assert(cfg.num_heads == 16);
    assert(cfg.num_kv_heads == 8);
    assert(cfg.head_dim == 128);
    assert(cfg.inter_dim == 3072);
    assert(cfg.vocab_size == 151936);
    assert(cfg.gqa_ratio == 2);
    assert(cfg.max_seq_len == 4096);
    assert(cfg.target_layer_ids.size() == 5);

    printf("  Model: %s (%ld params)\n", cfg.name.c_str(), (long)cfg.parameter_count);
    printf("  Hidden=%d, Layers=%d, Heads=%d, KV=%d, HeadDim=%d\n",
           cfg.hidden_size, cfg.num_layers, cfg.num_heads,
           cfg.num_kv_heads, cfg.head_dim);
    printf("  InterDim=%d, Vocab=%d\n", cfg.inter_dim, cfg.vocab_size);
    printf("  QKV proj dim: %d\n", cfg.qkv_proj_dim());
    printf("  FFN proj dim: %d\n", cfg.ffn_proj_dim());

    assert(cfg.qkv_proj_dim() == (16 + 2 * 8) * 128);  // NH + 2*NKV * head_dim
    assert(cfg.ffn_proj_dim() == 2 * 3072);  // gate + up

    printf("  PASS\n");
    return 0;
}

int test_llama_config() {
    printf("=== LLaMA-3.2-3B Config ===\n");

    auto cfg = LLaMAConfig::config_3B();

    assert(cfg.hidden_size == 3072);
    assert(cfg.num_layers == 28);
    assert(cfg.num_kv_heads == 8);
    assert(cfg.head_dim == 128);
    assert(cfg.rope_theta == 500000.0f);
    assert(cfg.vocab_size == 128256);

    printf("  Model: %s (%ld params)\n", cfg.name.c_str(), (long)cfg.parameter_count);
    printf("  Hidden=%d, Heads=%d, KV=%d, InterDim=%d, Vocab=%d\n",
           cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads,
           cfg.inter_dim, cfg.vocab_size);

    printf("  PASS\n");
    return 0;
}

int test_weight_store() {
    printf("=== Weight Store ===\n");

    WeightStore store;

    assert(store.size() == 0);

    // Put a tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    store.put("test.weight", data);

    assert(store.size() == 1);
    assert(store.has("test.weight"));
    assert(!store.has("missing.weight"));

    auto loaded = store.get("test.weight");
    assert(loaded.size() == 3);
    assert(loaded[0] == 1.0f);
    assert(loaded[2] == 3.0f);

    // Mutate
    auto mut = store.get_mut("test.weight");
    mut[0] = 42.0f;
    auto reloaded = store.get("test.weight");
    assert(reloaded[0] == 42.0f);

    // Merge
    WeightStore other;
    other.put("other.weight", {4.0f, 5.0f});
    store.merge(other);
    assert(store.size() == 2);
    assert(store.has("other.weight"));

    printf("  PASS\n");
    return 0;
}

// Mock model for testing the registry
class TestModel : public ModelInterface {
public:
    TestModel(const ModelConfig& cfg) : cfg_(cfg) {
        store_.put("dummy", {1.0f});
    }

    const ModelConfig& config() const override { return cfg_; }
    const std::string& name() const override { return cfg_.name; }
    bool load_weights(const std::string&) override { loaded_ = true; return true; }
    bool weights_loaded() const override { return loaded_; }
    const WeightStore& weights() const override { return store_; }

    ForwardOutput forward(std::span<const int32_t>, bool, bool) override {
        ForwardOutput out;
        out.logits.resize(cfg_.vocab_size, 0.0f);
        out.last_hidden.resize(cfg_.hidden_size, 0.0f);
        return out;
    }

    ForwardOutput forward_with_kv(std::span<const int32_t>, int32_t, bool, bool) override {
        return forward({}, false, false);
    }

    void clear_kv_cache() override {}
    void commit_kv_cache(int32_t, int32_t) override {}
    int32_t kv_cache_length() const override { return 0; }

    void get_layer_hidden(std::span<const float>, std::span<const int32_t>, std::span<float>) override {}

private:
    ModelConfig cfg_;
    WeightStore store_;
    bool loaded_ = false;
};

int test_model_registry() {
    printf("=== Model Registry ===\n");

    auto& registry = ModelRegistry::instance();

    // Register test model
    registry.register_factory("test", [](const ModelConfig& cfg) {
        return std::make_unique<TestModel>(cfg);
    });

    auto families = registry.registered_families();
    printf("  Registered families: ");
    for (auto& f : families) printf("%s ", f.c_str());
    printf("\n");

    assert(registry.is_registered("test"));
    assert(!registry.is_registered("nonexistent"));

    // Create model
    ModelConfig cfg;
    cfg.name = "test-model";
    cfg.hidden_size = 128;
    cfg.vocab_size = 1000;

    auto model = registry.create("test", cfg);
    assert(model != nullptr);
    assert(model->name() == "test-model");
    assert(model->config().hidden_size == 128);

    // Verify forward works
    std::vector<int32_t> input = {1, 2, 3};
    auto out = model->forward(input, false, false);
    assert(out.logits.size() == 1000);
    assert(out.last_hidden.size() == 128);

    // Detection
    std::string detected = ModelRegistry::detect_family("/path/to/qwen3-0.6b.gguf");
    printf("  Detected (qwen): %s\n", detected.c_str());
    assert(detected == "qwen3");

    detected = ModelRegistry::detect_family("/path/to/llama-3.2-3b.safetensors");
    printf("  Detected (llama): %s\n", detected.c_str());
    assert(detected == "llama");

    printf("  PASS\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_qwen3_config();
    failures += test_llama_config();
    failures += test_weight_store();
    failures += test_model_registry();

    printf("\n=== Model Registry Tests: %d failures ===\n", failures);
    return failures;
}
