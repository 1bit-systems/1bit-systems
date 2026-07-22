/// Model loader — GGUF weight upload to GPU.
/// Fixed: #775 actual weight data upload, #773 KV cache as single buffer
#include "model_loader.h"
#include <cstring>
#include <cstdio>

#ifdef HAS_GGUF_READER
#include "gguf_reader.h"
#endif

ModelLoader::ModelLoader(VkDevice device, VkQueue queue, uint32_t queue_family,
                           CommandPool& cmd_pool)
    : device_(device), queue_(queue), queue_family_(queue_family), cmd_pool_(cmd_pool) {}

// Helper: upload float data to GPU via staging buffer
static void upload_float_data(VkDevice dev, VkQueue queue, CommandPool& pool,
                               GpuBuffer& dst, const float* src, size_t count) {
    if (count == 0 || !dst) return;
    
    // Create staging buffer
    size_t bytes = count * sizeof(float);
    GpuBuffer staging(dev, bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    float* mapped = (float*)staging.map();
    memcpy(mapped, src, bytes);
    staging.unmap();
    
    // Copy to device-local buffer via cmd buffer
    VkCommandBuffer cmd = pool.begin_once();
    VkBufferCopy copy = {0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging.buffer(), dst.buffer(), 1, &copy);
    pool.submit_and_wait(cmd, queue);
}

bool ModelLoader::load(const std::string& gguf_path, ModelGPU& model) {
    printf("ModelLoader: loading %s\n", gguf_path.c_str());
    
    // Parse GGUF metadata
    ModelDims dims;
    std::vector<WeightTensor> tensors;
    if (!scan_gguf(gguf_path, dims, tensors)) {
        fprintf(stderr, "Failed to scan GGUF file\n");
        return false;
    }
    model.dims = dims;
    printf("  %s: %d layers, H=%d, heads=%d/%d, V=%d, inter=%d\n",
           dims.arch.c_str(), dims.n_layers, dims.hidden,
           dims.n_heads, dims.n_kv_heads, dims.vocab, dims.inter);
    
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags dev = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    
    // Allocate and upload embedding
    {
        size_t sz = (size_t)dims.vocab * dims.hidden * sizeof(float);
        model.embed = GpuBuffer(device_, sz, rw, dev);
        // Upload from GGUF (dequantized to f32)
        std::vector<float> embed_data;
#ifdef HAS_GGUF_READER
        GgufReader reader;
        if (reader.open(gguf_path) && reader.get_tensor_f32("token_embd.weight", embed_data)) {
            upload_float_data(device_, queue_, cmd_pool_, model.embed, embed_data.data(), embed_data.size());
            printf("  Uploaded embed: %.1f MB\n", sz / (1024.0*1024.0));
        } else
#endif
        { printf("  Warning: embed data not uploaded (GGUF reader unavailable)\n"); }
    }
    
    // Allocate and upload final norm
    {
        model.final_norm = GpuBuffer(device_, (size_t)dims.hidden * sizeof(float), rw, dev);
#ifdef HAS_GGUF_READER
        GgufReader reader;
        if (reader.open(gguf_path)) {
            std::vector<float> fn;
            if (reader.get_tensor_f32("output_norm.weight", fn) && fn.size() >= (size_t)dims.hidden)
                upload_float_data(device_, queue_, cmd_pool_, model.final_norm, fn.data(), dims.hidden);
        }
#endif
    }
    
    // Allocate and upload per-layer weights
    model.layers.resize(dims.n_layers);
    for (int l = 0; l < dims.n_layers; l++) {
        auto& layer = model.layers[l];
        auto a = [&](GpuBuffer& buf, size_t s) { buf = GpuBuffer(device_, s, rw, dev); };
        
        a(layer.wq, (size_t)dims.n_heads * dims.head_dim * dims.hidden * sizeof(float));
        a(layer.wk, (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden * sizeof(float));
        a(layer.wv, (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden * sizeof(float));
        a(layer.wo, (size_t)dims.hidden * dims.n_heads * dims.head_dim * sizeof(float));
        a(layer.w1, (size_t)dims.inter * dims.hidden * sizeof(float));
        a(layer.w2, (size_t)dims.inter * dims.hidden * sizeof(float));
        a(layer.w3, (size_t)dims.hidden * dims.inter * sizeof(float));
        a(layer.rms_attn, (size_t)dims.hidden * sizeof(float));
        a(layer.rms_ffn, (size_t)dims.hidden * sizeof(float));
        
        // Upload from GGUF
#ifdef HAS_GGUF_READER
        GgufReader rdr;
        if (rdr.open(gguf_path)) {
            std::string p = "blk." + std::to_string(l) + ".";
            auto ul = [&](GpuBuffer& buf, const std::string& name, size_t cnt) {
                std::vector<float> tmp;
                if (rdr.get_tensor_f32(p + name, tmp) && tmp.size() >= cnt)
                    upload_float_data(device_, queue_, cmd_pool_, buf, tmp.data(), cnt);
            };
            ul(layer.wq, "attn_q.weight", (size_t)dims.n_heads * dims.head_dim * dims.hidden);
            ul(layer.wk, "attn_k.weight", (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden);
            ul(layer.wv, "attn_v.weight", (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden);
            ul(layer.wo, "attn_output.weight", (size_t)dims.hidden * dims.n_heads * dims.head_dim);
            ul(layer.w1, "ffn_gate.weight", (size_t)dims.inter * dims.hidden);
            ul(layer.w2, "ffn_up.weight", (size_t)dims.inter * dims.hidden);
            ul(layer.w3, "ffn_down.weight", (size_t)dims.hidden * dims.inter);
            ul(layer.rms_attn, "attn_norm.weight", (size_t)dims.hidden);
            ul(layer.rms_ffn, "ffn_norm.weight", (size_t)dims.hidden);
        }
#endif
    }
    
    // Allocate KV cache: [2 * n_layers * max_seq * n_kv * hd] floats
    {
        size_t kv_elements = (size_t)2 * dims.n_layers * dims.max_seq * dims.n_kv_heads * dims.head_dim;
        if (kv_elements > 0) {
            model.kv_cache = GpuBuffer(device_, kv_elements * sizeof(float), rw, dev);
            model.kv_cache_capacity = dims.max_seq;
            printf("  KV cache: %.1f MB (%d layers, %d seq)\n",
                   kv_elements * sizeof(float) / (1024.0*1024.0),
                   dims.n_layers, dims.max_seq);
        }
    }
    
    printf("  Model loaded: %d layers\n", dims.n_layers);
    return true;
}

bool ModelLoader::scan_gguf(const std::string& path, ModelDims& dims,
                             std::vector<WeightTensor>& tensors) {
#ifndef HAS_GGUF_READER
    (void)path; (void)tensors;
    fprintf(stderr, "scan_gguf: GGUF reader not available\n");
    return false;
#else
    GgufReader reader;
    if (!reader.open(path)) {
        fprintf(stderr, "scan_gguf: failed to open %s\n", path.c_str());
        return false;
    }
    
    dims.arch = reader.architecture();
    uint32_t val = 0; float fval = 0;
    
    if (reader.get_u32("llama.attention.head_count", val)) dims.n_heads = val; val = 0;
    if (reader.get_u32("llama.attention.head_count_kv", val)) dims.n_kv_heads = val; val = 0;
    if (reader.get_u32("llama.block_count", val)) dims.n_layers = val; val = 0;
    if (reader.get_u32("llama.feed_forward_length", val)) dims.inter = val; val = 0;
    if (reader.get_f32("llama.rope.freq_base", fval)) { dims.rope_theta = fval; fval = 0; }
    if (reader.get_u32("llama.embedding_length", val)) dims.hidden = val; val = 0;
    
    if (dims.n_heads > 0 && dims.hidden > 0)
        dims.head_dim = dims.hidden / dims.n_heads;
    else
        dims.head_dim = 128;
    
    if (reader.get_u32("tokenizer.ggml.vocab_size", val)) dims.vocab = val; val = 0;
    if (dims.vocab == 0) { reader.get_u32("llama.vocab_size", val); dims.vocab = val; val = 0; }
    if (reader.get_u32("llama.context_length", val)) dims.max_seq = val > 0 ? val : 4096;
    if (reader.get_f32("llama.attention.layer_norm_rms_epsilon", fval)) dims.rms_eps = fval;
    
    return dims.hidden > 0 && dims.n_layers > 0;
#endif
}
