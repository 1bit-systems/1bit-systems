// lora.h — GGUF LoRA adapter hot-loader for 1BP models
//
// LoRA (Low-Rank Adaptation): W' = W + B*A * (alpha/rank)
// Each adapter is a GGUF file containing {tensor_name}.lora_a and {tensor_name}.lora_b
// weights. Supports hot-loading multiple adapters at runtime.
//
// Format: llama.cpp GGUF LoRA convention
//   tensor names: "blk.{layer}.attn_q.weight.lora_a", etc.
//   metadata:  "general.name" (adapter name)
//              "adapter.lora.rank" (uint32)
//              "adapter.lora.alpha" (float)

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

#include "gguf_reader.h"

// ─── LoRA adapter weights for a single target tensor ──────────────
struct LoraTensorPair {
    // A: [rank, cols], B: [rows, rank]  (standard LoRA order)
    std::vector<float> lora_a;  // rank × cols
    std::vector<float> lora_b;  // rows × rank
    int rank;
    int rows;
    int cols;
};

// ─── A loaded LoRA adapter ────────────────────────────────────────
struct LoraAdapter {
    std::string name;
    int rank;
    float alpha;
    std::string path;  // source GGUF path
    
    // target tensor name -> LoRA weights
    // "blk.0.attn_q.weight" -> pair
    std::unordered_map<std::string, LoraTensorPair> tensors;
    
    // Merge scale pre-computed: scale = alpha / rank
    float scale() const { return alpha / (float)rank; }
};

// ─── LoRA manager: hot-load, unload, apply ───────────────────────
class LoraManager {
public:
    LoraManager();
    ~LoraManager();

    // ── Lifecycle ──
    /// Load a LoRA adapter from a GGUF file.
    /// Returns adapter index on success, -1 on failure.
    int load_adapter(const std::string& gguf_path);
    
    /// Unload an adapter by index.
    bool unload_adapter(int idx);
    
    /// Unload an adapter by name.
    bool unload_adapter(const std::string& name);
    
    /// Clear all loaded adapters.
    void clear_all();
    
    /// Get list of loaded adapter names.
    std::vector<std::string> loaded_adapters() const;

    // ── Application ──
    /// Apply all active LoRAs to a weight matrix in-place.
    /// weight: [rows × cols] row-major float buffer
    /// tensor_name: e.g. "blk.0.attn_q.weight"
    /// rows, cols: dimensions of the weight matrix
    void apply(float* weight, const std::string& tensor_name, int rows, int cols);
    
    /// Apply a single specific adapter to a weight matrix.
    void apply_adapter(float* weight, const std::string& tensor_name,
                       int rows, int cols, int adapter_idx);
    
    /// Check if any loaded adapter targets this tensor.
    bool has_adapter_for(const std::string& tensor_name) const;

    // ── Enable/disable ──
    void enable_adapter(int idx, bool enabled = true);
    void enable_adapter(const std::string& name, bool enabled = true);
    bool adapter_enabled(int idx) const;
    
    int adapter_count() const { return (int)adapters_.size(); }

private:
    struct LoadedAdapter {
        LoraAdapter adapter;
        bool enabled = true;
    };
    
    std::vector<LoadedAdapter> adapters_;
    mutable std::mutex mtx_;
    
    // Parse adapter metadata from GGUF header
    bool parse_metadata(GgufReader& reader, LoraAdapter& out);
    
    // Load tensor weights from GGUF
    bool load_tensor(GgufReader& reader, const std::string& tensor_name,
                     LoraTensorPair& out);
};

// ─── Convenience: merge LoRA into a model's weights permanently ───
// Reads the base GGUF, applies LoRA adapters, writes merged GGUF.
// Returns true on success.
bool lora_merge_gguf(const std::string& base_gguf_path,
                     const std::vector<std::string>& lora_paths,
                     const std::string& output_gguf_path);
