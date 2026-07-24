/// Model loader — GGUF weight upload to GPU.
/// Fixed: #775 actual weight data upload, #773 KV cache as single buffer, #782 open GGUF once
#include "model_loader.h"
#include <cfloat>
#include <cstring>
#include <cmath>
#include <cstdio>

#ifdef HAS_GGUF_READER
#include "gguf_reader.h"

// Helper: upload float data to GPU via staging buffer
static void upload_float_data(VkDevice dev, VkQueue queue, CommandPool& pool,
                               GpuBuffer& dst, const float* src, size_t count) {
    if (count == 0 || !dst) return;
    size_t bytes = count * sizeof(float);
    GpuBuffer staging(dev, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    float* mapped = (float*)staging.map();
    memcpy(mapped, src, bytes);
    staging.unmap();
    if (bytes > 0) {
        VkCommandBuffer cmd = pool.begin_once();
        VkBufferCopy copy = {0, 0, bytes};
        vkCmdCopyBuffer(cmd, staging.buffer(), dst.buffer(), 1, &copy);
        pool.submit_and_wait(cmd, queue);
    }
}

// Helper: upload tensor from GgufReader to GPU
static void upload_tensor(GgufReader& reader, const std::string& name,
                           GpuBuffer& dst, size_t expected_floats,
                           VkDevice dev, VkQueue queue, CommandPool& pool) {
    std::vector<float> tmp;
    // Upload whatever the tensor actually provides, up to the expected size.
    // Some models (e.g. ZR1) store a token_embd.weight with fewer rows than
    // vocab_size (padding vocab entries have no embedding); requiring an exact
    // >= match silently skipped the upload and left the buffer zero, which
    // collapsed the whole forward pass to zero logits.
    if (reader.get_tensor_f32(name, tmp) && !tmp.empty()) {
        size_t n = std::min(tmp.size(), expected_floats);
        if (getenv("ZINC_DEBUG_UPLOAD"))
            fprintf(stderr, "[zinc] upload %s: got=%zu need=%zu -> %zu first=%g\n",
                    name.c_str(), tmp.size(), expected_floats, n, tmp[0]);
        upload_float_data(dev, queue, pool, dst, tmp.data(), n);
    }
}

// Q8_0 quantization: block size 32, each block = fp16 scale + 32 int8 values = 34 bytes
static size_t q8_0_quantize(const float* f32, int8_t* q8, int count) {
    int blocks = (count + 31) / 32;
    size_t out_size = (size_t)blocks * 34;
    if (!q8) return out_size;  // query size only
    for (int b = 0; b < blocks; b++) {
        int start = b * 32;
        int end = (start + 32 < count) ? start + 32 : count;
        float amax = 0.0f;
        for (int i = start; i < end; i++) { float a = fabsf(f32[i]); if (a > amax) amax = a; }
        int8_t* block = q8 + b * 34;
        float scale = amax / 127.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        uint32_t scale_bits; memcpy(&scale_bits, &scale, 4);
        uint16_t fp16 = (uint16_t)(scale_bits >> 16);
        memcpy(block, &fp16, 2);
        float inv = 1.0f / scale;
        for (int i = start; i < end; i++)
            block[2 + (i - start)] = (int8_t)(f32[i] * inv + (f32[i] >= 0 ? 0.5f : -0.5f));
    }
    return out_size;
}

// Upload+quantize tensor to Q8_0 format
static void upload_tensor_q8(GgufReader& reader, const std::string& name,
                              GpuBuffer& dst, size_t expected_floats,
                              VkDevice dev, VkQueue queue, CommandPool& pool) {
    std::vector<float> tmp;
    if (!reader.get_tensor_f32(name, tmp) || tmp.size() < expected_floats) return;
    
    size_t q8_bytes = q8_0_quantize(tmp.data(), nullptr, expected_floats);
    std::vector<int8_t> q8_buf(q8_bytes);
    q8_0_quantize(tmp.data(), q8_buf.data(), expected_floats);
    
    // Upload to GPU
    GpuBuffer staging(dev, q8_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(staging.map(), q8_buf.data(), q8_bytes);
    staging.unmap();
    
    VkCommandBuffer cmd = pool.begin_once();
    VkBufferCopy copy = {0, 0, q8_bytes};
    vkCmdCopyBuffer(cmd, staging.buffer(), dst.buffer(), 1, &copy);
    pool.submit_and_wait(cmd, queue);
}

// Q4_K quantization: 256 elements per block, 144 bytes
// Layout: fp16 d, fp16 dmin, 12 bytes 6-bit scales, 128 bytes 4-bit values
static size_t q4k_quantize(const float* f32, uint8_t* q4k, int count) {
    const int BS = 256;
    int nb = (count + BS - 1) / BS;
    size_t out_size = (size_t)nb * 144;
    if (!q4k) return out_size;
    
    for (int b = 0; b < nb; b++) {
        int base = b * BS;
        uint8_t* block = q4k + b * 144;
        
        // Find max and min
        float amax = -FLT_MAX, amin = FLT_MAX;
        for (int j = 0; j < BS && base + j < count; j++) {
            float v = f32[base + j];
            if (v > amax) amax = v;
            if (v < amin) amin = v;
        }
        
        // 4-bit nibbles hold 0..15, so the step must span the range in 15
        // steps, not 255. The old /255 made ~94% of values clamp to the same
        // nibble (dense weights were destroyed). Dequant is nib*d + dmin.
        float d = (amax - amin) / 15.0f;
        float dmin = amin;
        if (d < 1e-12f) d = 1e-12f;
        
        // Store d and dmin as fp16 (crude truncation)
        uint32_t db, dmb; memcpy(&db, &d, 4); memcpy(&dmb, &dmin, 4);
        *(uint16_t*)(block + 0) = (uint16_t)(db >> 16);
        *(uint16_t*)(block + 2) = (uint16_t)(dmb >> 16);
        
        // Quantize values to 4-bit nibbles
        uint8_t* qs = block + 16;  // after scales
        for (int j = 0; j < BS && base + j < count; j++) {
            float v = (f32[base + j] - dmin) / d;
            int qi = (int)(v + 0.5f);
            if (qi < 0) qi = 0;
            if (qi > 15) qi = 15;
            if (j % 2 == 0)
                qs[j / 2] = (uint8_t)(qi & 0xF);
            else
                qs[j / 2] |= (uint8_t)(qi << 4);
        }
        
        // Simple per-subblock scales (6-bit, packed into 12 bytes)
        // For a minimal implementation, use constant scale per superblock
        // Production: 8 × 6-bit scales for 8 sub-blocks
        uint8_t* scales = block + 4;
        memset(scales, 0, 12);
        for (int s = 0; s < 8; s++) {
            int s_start = s * 32;
            float smax = -FLT_MAX;
            for (int j = s_start; j < s_start + 32 && base + j < count; j++) {
                float a = fabsf(f32[base + j]); if (a > smax) smax = a;
            }
            int si = (int)(smax / d + 0.5f);
            if (si < 0) si = 0;
            if (si > 63) si = 63;
            // Pack 6-bit: first 4 scales in low 24 bits, next 4 in high 24 bits
            if (s < 4) scales[s] = (uint8_t)((scales[s] & 0xC0) | si);
            else scales[s + 4] = (uint8_t)((scales[s + 4] & 0xC0) | si);
        }
    }
    return out_size;
}

// Upload+quantize tensor to Q4_K format
static void upload_tensor_q4k(GgufReader& reader, const std::string& name,
                               GpuBuffer& dst, size_t expected_floats,
                               VkDevice dev, VkQueue queue, CommandPool& pool) {
    std::vector<float> tmp;
    if (!reader.get_tensor_f32(name, tmp) || tmp.size() < expected_floats) return;
    
    size_t q4k_bytes = q4k_quantize(tmp.data(), nullptr, expected_floats);
    std::vector<uint8_t> q4k_buf(q4k_bytes);
    q4k_quantize(tmp.data(), q4k_buf.data(), expected_floats);
    
    GpuBuffer staging(dev, q4k_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(staging.map(), q4k_buf.data(), q4k_bytes);
    staging.unmap();
    
    VkCommandBuffer cmd = pool.begin_once();
    VkBufferCopy copy = {0, 0, q4k_bytes};
    vkCmdCopyBuffer(cmd, staging.buffer(), dst.buffer(), 1, &copy);
    pool.submit_and_wait(cmd, queue);
}
#endif

ModelLoader::ModelLoader(VkDevice device, VkQueue queue, uint32_t queue_family,
                           CommandPool& cmd_pool)
    : device_(device), queue_(queue), queue_family_(queue_family), cmd_pool_(cmd_pool) {}

bool ModelLoader::load(const std::string& gguf_path, ModelGPU& model) {
    printf("ModelLoader: loading %s\n", gguf_path.c_str());
    
    ModelDims dims;
    std::vector<WeightTensor> tensors;
    if (!scan_gguf(gguf_path, dims, tensors)) {
        fprintf(stderr, "Failed to scan GGUF file\n");
        return false;
    }
    model.dims = dims;
    printf("  %s: %d layers, H=%d, heads=%d/%d, V=%d, inter=%d, hd=%d, theta=%.1f, eps=%g\n",
           dims.arch.c_str(), dims.n_layers, dims.hidden,
           dims.n_heads, dims.n_kv_heads, dims.vocab, dims.inter,
           dims.head_dim, dims.rope_theta, dims.rms_eps);
    
    bool have_reader = false;
#ifdef HAS_GGUF_READER
    GgufReader reader;
    have_reader = reader.open(gguf_path);
    if (!have_reader) fprintf(stderr, "  Warning: GGUF reader failed to open\n");
#else
    (void)have_reader;
#endif
    
    VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags dev = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    
    // Embedding
    {
        size_t sz = (size_t)dims.vocab * dims.hidden * sizeof(float);
        model.embed = GpuBuffer(device_, sz, rw, dev);
#ifdef HAS_GGUF_READER
        if (have_reader) upload_tensor(reader, "token_embd.weight", model.embed,
                (size_t)dims.vocab * dims.hidden, device_, queue_, cmd_pool_);
#endif
    }
    
    // Final norm
    {
        model.final_norm = GpuBuffer(device_, (size_t)dims.hidden * sizeof(float), rw, dev);
#ifdef HAS_GGUF_READER
        if (have_reader) upload_tensor(reader, "output_norm.weight", model.final_norm,
                dims.hidden, device_, queue_, cmd_pool_);
#endif
    }

#ifdef HAS_GGUF_READER
    // Untied lm_head (output.weight), quantized to Q4_K. Absent -> tied embeddings.
    if (have_reader) {
        std::vector<float> ow;
        size_t need = (size_t)dims.vocab * dims.hidden;
        if (reader.get_tensor_f32("output.weight", ow) && !ow.empty()) {
            ow.resize(need, 0.0f);  // pad any missing vocab rows
            model.lm_head = GpuBuffer(device_, need * sizeof(float), rw, dev);
            upload_float_data(device_, queue_, cmd_pool_, model.lm_head, ow.data(), need);
            model.has_lm_head = true;
            if (getenv("ZINC_DEBUG"))
                fprintf(stderr, "[zinc] lm_head loaded: rows_present=%zu need=%zu first=[%g %g %g]\n",
                        ow.size() == need ? need : ow.size(), need, ow[0], ow[1], ow[2]);
        }
    }
#endif
    
    // Per-layer weights
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
        
#ifdef HAS_GGUF_READER
        // Allocate fused QKV buffer (Q + K + V concatenated, Q4_K format)
        {
            size_t q_rows = dims.n_heads * dims.head_dim;
            size_t kv_rows = dims.n_kv_heads * dims.head_dim;
            size_t total_rows = q_rows + 2 * kv_rows;
            // F32 fused weights [total_rows, hidden] — no requantization, so the
            // ZINC forward pass uses the exact values cpu_generic does.
            layer.qkv_fused = GpuBuffer(device_, total_rows * dims.hidden * sizeof(float), rw, dev);
        }
        // Allocate fused gate+up buffer (gate + up concatenated), F32
        {
            size_t total_rows = (size_t)2 * dims.inter;
            layer.gate_up_fused = GpuBuffer(device_, total_rows * dims.hidden * sizeof(float), rw, dev);
        }
        
        if (have_reader) {
            std::string p = "blk." + std::to_string(l) + ".";
            upload_tensor_q4k(reader, p + "attn_q.weight", layer.wq,
                (size_t)dims.n_heads * dims.head_dim * dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor_q4k(reader, p + "attn_k.weight", layer.wk,
                (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor_q4k(reader, p + "attn_v.weight", layer.wv,
                (size_t)dims.n_kv_heads * dims.head_dim * dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor(reader, p + "attn_output.weight", layer.wo,
                (size_t)dims.hidden * dims.n_heads * dims.head_dim, device_, queue_, cmd_pool_);
            upload_tensor_q4k(reader, p + "ffn_gate.weight", layer.w1,
                (size_t)dims.inter * dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor_q4k(reader, p + "ffn_up.weight", layer.w2,
                (size_t)dims.inter * dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor(reader, p + "ffn_down.weight", layer.w3,
                (size_t)dims.hidden * dims.inter, device_, queue_, cmd_pool_);
            
            // Fuse gate+up weights into single buffer for fused gate_up shader
            if (have_reader && layer.gate_up_fused) {
                std::vector<float> g_tmp, u_tmp;
                if (reader.get_tensor_f32(p + "ffn_gate.weight", g_tmp) && g_tmp.size() >= (size_t)dims.inter * dims.hidden &&
                    reader.get_tensor_f32(p + "ffn_up.weight", u_tmp) && u_tmp.size() >= (size_t)dims.inter * dims.hidden) {
                    std::vector<float> fused(g_tmp.size() + u_tmp.size());
                    memcpy(fused.data(), g_tmp.data(), g_tmp.size() * 4);
                    memcpy(fused.data() + g_tmp.size(), u_tmp.data(), u_tmp.size() * 4);
                    upload_float_data(device_, queue_, cmd_pool_, layer.gate_up_fused, fused.data(), fused.size());
                }
            }
            
            // Fuse Q, K, V weights into single buffer for fused QKV shader
            if (have_reader && layer.qkv_fused) {
                size_t q_rows = dims.n_heads * dims.head_dim;
                size_t kv_rows = dims.n_kv_heads * dims.head_dim;
                std::vector<float> q_tmp, k_tmp, v_tmp;
                if (reader.get_tensor_f32(p + "attn_q.weight", q_tmp) && q_tmp.size() >= (size_t)q_rows * dims.hidden &&
                    reader.get_tensor_f32(p + "attn_k.weight", k_tmp) && k_tmp.size() >= (size_t)kv_rows * dims.hidden &&
                    reader.get_tensor_f32(p + "attn_v.weight", v_tmp) && v_tmp.size() >= (size_t)kv_rows * dims.hidden) {
                    
                    std::vector<float> fused(q_tmp.size() + k_tmp.size() + v_tmp.size());
                    memcpy(fused.data(), q_tmp.data(), q_tmp.size() * 4);
                    memcpy(fused.data() + q_tmp.size(), k_tmp.data(), k_tmp.size() * 4);
                    memcpy(fused.data() + q_tmp.size() + k_tmp.size(), v_tmp.data(), v_tmp.size() * 4);
                    upload_float_data(device_, queue_, cmd_pool_, layer.qkv_fused, fused.data(), fused.size());
                }
            }
            upload_tensor(reader, p + "attn_norm.weight", layer.rms_attn,
                dims.hidden, device_, queue_, cmd_pool_);
            upload_tensor(reader, p + "ffn_norm.weight", layer.rms_ffn,
                dims.hidden, device_, queue_, cmd_pool_);
            // Fused QKV bias (Qwen2/ZR1): attn_q/k/v.bias concatenated, F32,
            // matching the qkv buffer layout [Q | K | V].
            {
                size_t q_rows = (size_t)dims.n_heads * dims.head_dim;
                size_t kv_rows = (size_t)dims.n_kv_heads * dims.head_dim;
                std::vector<float> bq_, bk_, bv_;
                if (reader.get_tensor_f32(p + "attn_q.bias", bq_) && bq_.size() >= q_rows &&
                    reader.get_tensor_f32(p + "attn_k.bias", bk_) && bk_.size() >= kv_rows &&
                    reader.get_tensor_f32(p + "attn_v.bias", bv_) && bv_.size() >= kv_rows) {
                    std::vector<float> fb(q_rows + 2 * kv_rows);
                    memcpy(fb.data(), bq_.data(), q_rows * 4);
                    memcpy(fb.data() + q_rows, bk_.data(), kv_rows * 4);
                    memcpy(fb.data() + q_rows + kv_rows, bv_.data(), kv_rows * 4);
                    layer.qkv_bias_fused = GpuBuffer(device_, fb.size() * sizeof(float), rw, dev);
                    upload_float_data(device_, queue_, cmd_pool_, layer.qkv_bias_fused, fb.data(), fb.size());
                    layer.has_bias = true;
                }
            }
        }
#endif
    }
    
    // KV cache
    {
        size_t kv_elements = (size_t)2 * dims.n_layers * dims.max_seq * dims.n_kv_heads * dims.head_dim;
        if (kv_elements > 0) {
            model.kv_cache = GpuBuffer(device_, kv_elements * sizeof(float), rw, dev);
            model.kv_cache_capacity = dims.max_seq;
            printf("  KV cache: %.1f MB (%d layers, %d seq)\n",
                   kv_elements * sizeof(float) / (1024.0*1024.0), dims.n_layers, dims.max_seq);
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
    
    // Architecture-aware key lookup (fix #785)
    auto arch_u32 = [&](const std::string& key, uint32_t& out) -> bool {
        std::string arch_key = dims.arch + "." + key;
        if (reader.get_u32(arch_key, out)) return true;
        return reader.get_u32(key, out);
    };
    auto arch_f32 = [&](const std::string& key, float& out) -> bool {
        std::string arch_key = dims.arch + "." + key;
        if (reader.get_f32(arch_key, out)) return true;
        return reader.get_f32(key, out);
    };
    
    uint32_t val = 0; float fval = 0;
    if (arch_u32("attention.head_count", val)) dims.n_heads = val; val = 0;
    if (arch_u32("attention.head_count_kv", val)) dims.n_kv_heads = val; val = 0;
    if (arch_u32("block_count", val)) dims.n_layers = val; val = 0;
    if (arch_u32("feed_forward_length", val)) dims.inter = val; val = 0;
    if (arch_f32("rope.freq_base", fval)) dims.rope_theta = fval;
    if (arch_u32("embedding_length", val)) dims.hidden = val; val = 0;
    if (dims.n_heads > 0 && dims.hidden > 0) dims.head_dim = dims.hidden / dims.n_heads;
    else dims.head_dim = 128;
    if (reader.get_u32("tokenizer.ggml.vocab_size", val)) dims.vocab = val; val = 0;
    if (dims.vocab == 0) { arch_u32("vocab_size", val); dims.vocab = val; }
    // Fallback: common vocab sizes for known architectures
    if (dims.vocab == 0) {
        if (dims.arch == "qwen3" || dims.arch == "qwen2") dims.vocab = 152064;
        else if (dims.arch == "llama") dims.vocab = 32000;
        else if (dims.arch == "gemma2") dims.vocab = 256000;
        else dims.vocab = 32000;
    }
    if (arch_u32("context_length", val)) dims.max_seq = val > 0 ? val : 4096;
    // Cap the KV-cache context so the buffer stays within maxStorageBufferRange
    // (4 GiB on RADV). At 131072 the V region alone starts at ~3.7 GiB, past the
    // bindable range, so cached V reads returned zero and attention collapsed.
    if (dims.max_seq > 8192) dims.max_seq = 8192;
    if (arch_f32("attention.layer_norm_rms_epsilon", fval)) dims.rms_eps = fval;
    printf("  GGUF: %s, %d layers, H=%d, heads=%d/%d, V=%d\n",
           dims.arch.c_str(), dims.n_layers, dims.hidden,
           dims.n_heads, dims.n_kv_heads, dims.vocab);
    return dims.hidden > 0 && dims.n_layers > 0;
#endif
}
