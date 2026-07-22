#pragma once
/// C++ Vulkan wrappers for GPU inference engine.
/// Port of ZINC's vulkan/*.zig to C++.
/// Loads pre-compiled SPIR-V shaders from share/zinc/shaders/
/// and dispatches compute operations for transformer inference.
///
/// Requires: Vulkan 1.3 (for VK_KHR_dynamic_rendering, timeline semaphores),
///            VK_KHR_shader_float16_int8 (for fp16 compute)
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <map>
#include <memory>
#include <optional>

// ─── Convenience error checking ──────────────────────────────────
#define VK_CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    throw std::runtime_error("Vulkan error"); }} while(0)

// ─── Physical device selection ────────────────────────────────────
struct GpuDeviceInfo {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    uint32_t compute_queue_family = UINT32_MAX;
    uint32_t transfer_queue_family = UINT32_MAX;
    std::string name;
    uint64_t total_vram = 0;
    uint64_t free_vram = 0;  // queried at init
    bool has_fp16 = false;
    bool has_int8 = false;
    bool has_int64_atomics = false;
    bool has_subgroup_ballot = false;
    bool has_subgroup_shuffle = false;
};

// ─── Shader module cache ──────────────────────────────────────────
// Loads .spv files from disk and caches VkShaderModule handles.
class ShaderCache {
public:
    ShaderCache(VkDevice device) : device_(device) {}
    ~ShaderCache();
    
    VkShaderModule load(const std::string& name);  // e.g. "dmmv_q4k"
    VkShaderModule load_path(const std::string& path);  // full path

private:
    VkDevice device_;
    std::map<std::string, VkShaderModule> cache_;
};

// ─── Pipeline cache (compute pipelines from shaders) ─────────────
class ComputePipelineCache {
public:
    ComputePipelineCache(VkDevice device) : device_(device), shaders_(device) {}
    ~ComputePipelineCache();

    VkPipeline get(const std::string& shader_name,
                   const std::vector<VkSpecializationMapEntry>& spec_constants = {},
                   const void* spec_data = nullptr, size_t spec_data_size = 0);

private:
    VkDevice device_;
    ShaderCache shaders_;
    std::map<std::string, VkPipeline> cache_;
    // Pipeline layout shared by all compute pipelines (push constants + descriptor set)
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout_ = VK_NULL_HANDLE;
    bool layout_inited_ = false;
    void ensure_layout();
};

// ─── Buffer object (device or host-visible) ──────────────────────
class GpuBuffer {
public:
    GpuBuffer() = default;
    GpuBuffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags mem_flags, VkSharingMode sharing = VK_SHARING_MODE_EXCLUSIVE);
    ~GpuBuffer() { destroy(); }
    
    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    void* map(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void unmap();
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void destroy();

    VkBuffer buffer() const { return buffer_; }
    VkDeviceMemory memory() const { return memory_; }
    VkDeviceSize size() const { return size_; }
    explicit operator bool() const { return buffer_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
};

// ─── Command pool / buffer helpers ───────────────────────────────
class CommandPool {
public:
    CommandPool(VkDevice device, uint32_t queue_family);
    ~CommandPool();
    
    CommandPool(CommandPool&&) = default;
    CommandPool& operator=(CommandPool&&) = default;
    CommandPool(const CommandPool&) = delete;

    VkCommandBuffer begin_once();  // begin a one-time command buffer
    void submit(VkCommandBuffer cmd, VkQueue queue,
                VkFence fence = VK_NULL_HANDLE);
    void submit_and_wait(VkCommandBuffer cmd, VkQueue queue);

    VkCommandPool pool() const { return pool_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
};

// ─── ZINC inference engine (Vulkan backend) ──────────────────────
// This is the C++ equivalent of ZINC's main.zig + compute/forward.zig
// + vulkan/*.zig + model/*.zig + scheduler/*.zig rolled together.
//
// Architecture:
//   1. Init: create Vulkan instance, select GPU, load shaders
//   2. Load model: parse GGUF, upload weights to GPU
//   3. Infer: for each token, dispatch layers through compute shaders
//
// Shader dispatch follows the same pattern as ZINC's forward.zig:
//   bind descriptor set → push constants → dispatch compute → barrier

class ZincEngine {
public:
    ZincEngine() = default;
    ~ZincEngine() { destroy(); }

    // No copy
    ZincEngine(const ZincEngine&) = delete;
    ZincEngine& operator=(const ZincEngine&) = delete;
    ZincEngine(ZincEngine&&) = default;
    ZincEngine& operator=(ZincEngine&&) = default;

    // ── Lifecycle ──
    /// Initialize Vulkan and select GPU.
    /// @param shader_dir  Path to directory containing .spv files
    /// @param device_idx   GPU device index (-1 = auto-select)
    void init(const std::string& shader_dir, int device_idx = -1);

    /// Load a GGUF model and upload weights to GPU.
    bool load_model(const std::string& gguf_path);

    /// Destroy all resources.
    void destroy();

    // ── Inference ──
    /// Reset KV cache and position for a new sequence.
    void reset();

    /// Generate one token from the model.
    /// @param token_id  Input token ID
    /// @return Next token ID, or -1 on failure
    int generate(int token_id);

    // ── Accessors ──
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queue_family() const { return queue_family_; }
    CommandPool* cmd_pool() { return cmd_pool_.get(); }
    ComputePipelineCache* pipeline_cache() { return pipeline_cache_.get(); }

private:
    // Vulkan state
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = UINT32_MAX;
    std::unique_ptr<CommandPool> cmd_pool_;
    std::unique_ptr<ShaderCache> shader_cache_;
    std::unique_ptr<ComputePipelineCache> pipeline_cache_;
    std::string shader_dir_;

    // Model state
    struct ModelState {
        int n_layers = 0;
        int hidden = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int inter_dim = 0;
        int vocab = 0;
        int max_seq_len = 4096;
        float rope_theta = 10000.0f;
        float rms_eps = 1e-6f;
        
        // GPU buffers for weights (one per layer)
        struct LayerWeights {
            GpuBuffer wq, wk, wv, wo;      // attention projections
            GpuBuffer w1, w2, w3;           // FFN projections (gate, up, down)
            GpuBuffer rms_attn, rms_ffn;    // RMS norm weights
            GpuBuffer bq, bk, bv;           // optional biases
        };
        std::vector<LayerWeights> layers;
        
        // Shared weights
        GpuBuffer embed;      // token embeddings [vocab, hidden]
        GpuBuffer final_norm; // final RMS norm [hidden]
        GpuBuffer lm_head;    // optional [vocab, hidden] or null for tied
        
        // KV cache
        GpuBuffer k_cache[128];  // [layers][max_seq, n_kv, hd]
        GpuBuffer v_cache[128];
        int current_pos = 0;
    } model_;
};
