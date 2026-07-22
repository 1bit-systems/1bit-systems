/// ZINC C++ — GPU inference engine (Vulkan backend)
/// Port of ZINC (engine/gpu/) from Zig to C++.
///
/// One binary: loads GGUF models, dispatches compute shaders on Vulkan,
/// serves an OpenAI-compatible HTTP API.
///
/// Build: cmake -B build && cmake --build build
/// Run:   ./build/zinc_cpp --model model.gguf --port 8080
#include "vulkan_wrapper.h"
#include "compute_engine.h"
#include "model_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <getopt.h>
#include <signal.h>
#include <atomic>

// ═══════════════════════════════════════════════════════════════════
//  Signal handler
// ═══════════════════════════════════════════════════════════════════
static std::atomic<bool> keep_running{true};
static void handle_sigint(int) { keep_running = false; }

// ═══════════════════════════════════════════════════════════════════
//  Inference engine — orchestrates model layers on GPU
// ═══════════════════════════════════════════════════════════════════
struct InferenceEngine {
    ComputeEngine* compute = nullptr;
    ModelGPU* model = nullptr;
    int pos = 0;
    
    // Per-layer scratch buffers on GPU
    GpuBuffer hidden;      // [hidden] current hidden state
    GpuBuffer residual;    // [hidden] residual connection
    GpuBuffer qkv;         // [n_heads * head_dim + 2 * n_kv * head_dim]
    GpuBuffer attn_out;    // [n_heads * head_dim]
    GpuBuffer gate_up;     // [2 * inter]
    GpuBuffer silu_buf;    // [inter]
    GpuBuffer logits;      // [vocab]
    GpuBuffer argmax_buf;  // [1] for argmax result

    bool init(ComputeEngine& ce, ModelGPU& m) {
        compute = &ce;
        model = &m;
        
        auto& d = m.dims;
        VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        
        // Allocate scratch buffers
        auto alloc = [&](GpuBuffer& buf, size_t size, const char* name) {
            if (size == 0) return;
            buf = GpuBuffer(compute->device(), size, rw,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            printf("  Scratch %s: %.1f MB\n", name, size / (1024.0 * 1024.0));
        };
        
        alloc(hidden,     d.hidden * sizeof(float),            "hidden");
        alloc(residual,   d.hidden * sizeof(float),            "residual");
        alloc(qkv,        (d.n_heads * d.head_dim + d.n_kv_heads * d.head_dim * 2) * sizeof(float), "qkv");
        alloc(attn_out,   d.n_heads * d.head_dim * sizeof(float), "attn_out");
        alloc(gate_up,    d.inter * 2 * sizeof(float),         "gate_up");
        alloc(silu_buf,   d.inter * sizeof(float),             "silu_buf");
        alloc(logits,     d.vocab * sizeof(float),             "logits");
        alloc(argmax_buf, sizeof(int),                         "argmax");
        
        printf("  Inference engine ready: %d layers, H=%d\n", d.n_layers, d.hidden);
        return true;
    }
    
    void reset() { pos = 0; }
    
    int generate(int token_id) {
        if (!compute || !model) return -1;
        auto& d = model->dims;
        
        // 1. Embedding lookup
        compute->embed_lookup(hidden.buffer(), model->embed.buffer(), token_id, d.hidden);
        
        // 2. Process each layer
        for (int l = 0; l < d.n_layers; l++) {
            auto& layer = model->layers[l];
            
            // Save residual
            compute->dispatch("copy_buffer", {.M = (uint32_t)d.hidden},
                              hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
            
            // RMSNorm → QKV
            compute->rms_norm(hidden.buffer(), layer.rms_attn.buffer(), d.hidden, d.rms_eps);
            
            // Q projection: q = hidden @ Wq^T
            compute->gemv(qkv.buffer(), hidden.buffer(), layer.wq.buffer(),
                           d.n_heads * d.head_dim, 1, d.hidden);
            
            // K projection: k = hidden @ Wk^T
            compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.wk.buffer(),
                           d.n_kv_heads * d.head_dim, 1, d.hidden);
            
            // V projection: v = hidden @ Wv^T  
            compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.wv.buffer(),
                           d.n_kv_heads * d.head_dim, 1, d.hidden);
            
            // RoPE on Q and K
            compute->rope(qkv.buffer(), VK_NULL_HANDLE, d.head_dim, pos,
                          d.n_heads, d.n_kv_heads, d.rope_theta);
            
            // Flash attention
            compute->flash_attn(qkv.buffer(),
                                model->k_cache.buffer(), model->v_cache.buffer(),
                                attn_out.buffer(), pos + 1,
                                d.n_heads, d.n_kv_heads, d.head_dim,
                                d.n_heads / d.n_kv_heads);
            
            // O projection: hidden = attn_out @ Wo^T
            compute->gemv(hidden.buffer(), attn_out.buffer(), layer.wo.buffer(),
                           d.hidden, 1, d.n_heads * d.head_dim);
            
            // Residual add: hidden += residual
            // (implemented as a shader that adds two buffers)
            compute->dispatch("add_residual",
                              {.M = (uint32_t)d.hidden},
                              hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
            
            // FFN: RMSNorm → gate/up → SiLU → down → residual
            compute->rms_norm(hidden.buffer(), layer.rms_ffn.buffer(), d.hidden, d.rms_eps);
            
            // Gate + Up projections
            compute->gemv(gate_up.buffer(), hidden.buffer(), layer.w1.buffer(),
                           d.inter, 1, d.hidden);
            compute->gemv(VK_NULL_HANDLE, hidden.buffer(), layer.w2.buffer(),
                           d.inter, 1, d.hidden);
            
            // SiLU(gate) * up
            compute->silu_mul(silu_buf.buffer(), gate_up.buffer(), VK_NULL_HANDLE, d.inter);
            
            // Down projection: hidden = silu_buf @ W3^T
            compute->gemv(hidden.buffer(), silu_buf.buffer(), layer.w3.buffer(),
                           d.hidden, 1, d.inter);
            
            // Residual add
            compute->dispatch("add_residual",
                              {.M = (uint32_t)d.hidden},
                              hidden.buffer(), residual.buffer(), VK_NULL_HANDLE, 1);
        }
        
        // 3. Final RMSNorm
        compute->rms_norm(hidden.buffer(), model->final_norm.buffer(), d.hidden, d.rms_eps);
        
        // 4. LM head: logits = hidden @ embed^T (tied embeddings)
        compute->gemv(logits.buffer(), hidden.buffer(),
                       model->tied_embed ? model->embed.buffer() : model->lm_head.buffer(),
                       d.vocab, 1, d.hidden);
        
        pos++;
        
        // 5. Argmax
        return compute->argmax(logits.buffer(), d.vocab);
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    std::string model_path;
    int port = 8080;
    int device_idx = -1;

    // Parse args
    static struct option opts[] = {
        {"model",  required_argument, nullptr, 'm'},
        {"port",   required_argument, nullptr, 'p'},
        {"device", required_argument, nullptr, 'd'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:d:", opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 'd': device_idx = atoi(optarg); break;
        }
    }

    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║    ZINC C++ — GPU Inference Engine       ║\n");
    printf("║    Vulkan backend, GGUF models           ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s --model model.gguf [--port 8080] [--device 0]\n", argv[0]);
        return 1;
    }

    // ── Init Vulkan ──
    ZincEngine engine;
    try {
        engine.init("shaders", device_idx);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to init Vulkan: %s\n", e.what());
        return 1;
    }

    // ── Load model ──
    printf("\n── Loading Model ──\n");
    ModelLoader loader(engine.device(), engine.queue(),
                        engine.queue_family(), *engine.cmd_pool());
    ModelGPU model;
    if (!loader.load(model_path, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Init compute engine ──
    printf("\n── Init Compute ──\n");
    ComputeEngine compute(engine.device(), engine.queue(),
                          engine.queue_family(), *engine.cmd_pool(),
                          *engine.pipeline_cache());
    InferenceEngine infer;
    infer.init(compute, model);

    // ── Quick benchmark ──
    printf("\n── Benchmark (10 tokens) ──\n");
    infer.reset();
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 0;  // BOS
    for (int i = 0; i < 10; i++) {
        tok = infer.generate(tok);
        if (tok < 0) break;
    }
    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("  %.1f ms/tok (%.1f tok/s)\n", ms / 10.0f, 10.0f / (ms / 1000.0f));

    printf("\n── Ready. Press Ctrl+C to stop. ──\n\n");

    // ── Main loop (idle, shuts down on signal) ──
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\nShutting down...\n");
    return 0;
}
