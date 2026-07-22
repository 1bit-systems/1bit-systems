/// Model loader — GGUF weight upload to GPU.
/// Reuses src/gguf_reader.cpp for GGUF parsing.
#include "model_loader.h"

ModelLoader::ModelLoader(VkDevice device, VkQueue queue, uint32_t queue_family,
                           CommandPool& cmd_pool)
    : device_(device), queue_(queue), queue_family_(queue_family), cmd_pool_(cmd_pool) {}

bool ModelLoader::load(const std::string& gguf_path, ModelGPU& model) {
    printf("ModelLoader: loading %s\n", gguf_path.c_str());
    
    // Parse GGUF header for dimensions
    ModelDims dims;
    std::vector<WeightTensor> tensors;
    if (!scan_gguf(gguf_path, dims, tensors)) {
        fprintf(stderr, "Failed to scan GGUF file\n");
        return false;
    }
    model.dims = dims;
    printf("  Architecture: %s, %d layers, H=%d, V=%d\n",
           dims.arch.c_str(), dims.n_layers, dims.hidden, dims.vocab);
    
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags dev_local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    
    // Upload embedding
    // In real impl: find embed tensor, dequantize, upload
    model.embed = GpuBuffer(device_, (size_t)dims.vocab * dims.hidden * sizeof(float),
                            rw, dev_local);
    
    // Upload final norm
    model.final_norm = GpuBuffer(device_, (size_t)dims.hidden * sizeof(float), rw, dev_local);
    
    // Upload per-layer weights
    model.layers.resize(dims.n_layers);
    for (int l = 0; l < dims.n_layers; l++) {
        auto& layer = model.layers[l];
        auto alloc = [&](GpuBuffer& buf, size_t size) {
            buf = GpuBuffer(device_, size, rw, dev_local);
        };
        alloc(layer.wq, (size_t)dims.n_heads * dims.head_dim * dims.hidden * sizeof(float));
        alloc(layer.wk, (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden * sizeof(float));
        alloc(layer.wv, (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden * sizeof(float));
        alloc(layer.wo, (size_t)dims.hidden * dims.n_heads * dims.head_dim * sizeof(float));
        alloc(layer.w1, (size_t)dims.inter * dims.hidden * sizeof(float));
        alloc(layer.w2, (size_t)dims.inter * dims.hidden * sizeof(float));
        alloc(layer.w3, (size_t)dims.hidden * dims.inter * sizeof(float));
        alloc(layer.rms_attn, (size_t)dims.hidden * sizeof(float));
        alloc(layer.rms_ffn, (size_t)dims.hidden * sizeof(float));
    }
    
    // TODO: actual weight data upload from GGUF file
    // For now, allocate buffers only (training/inference will upload separately)
    printf("  GPU buffers allocated: %d layers\n", dims.n_layers);
    
    return true;
}

GpuBuffer ModelLoader::upload_tensor(const std::string& gguf_path,
                                      const WeightTensor& tensor,
                                      VkBufferUsageFlags extra_usage) {
    (void)gguf_path;
    (void)tensor;
    (void)extra_usage;
    return GpuBuffer(); // stub
}

GpuBuffer ModelLoader::upload_float_data(const float* data, size_t count) {
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    GpuBuffer buf(device_, count * sizeof(float), rw, host);
    if (data) {
        float* dst = (float*)buf.map();
        memcpy(dst, data, count * sizeof(float));
        buf.unmap();
    }
    return buf;
}

bool ModelLoader::scan_gguf(const std::string& path, ModelDims& dims,
                             std::vector<WeightTensor>& tensors) {
    // For now, return hardcoded test dimensions
    // In production, use src/gguf_reader.cpp to parse the GGUF header
    (void)path;
    (void)tensors;
    dims.arch = "llama";
    dims.n_layers = 32;
    dims.hidden = 4096;
    dims.n_heads = 32;
    dims.n_kv_heads = 8;
    dims.head_dim = 128;
    dims.inter = 11008;
    dims.vocab = 32000;
    dims.max_seq = 4096;
    dims.rope_theta = 10000.0f;
    dims.rms_eps = 1e-6f;
    return true;
}
