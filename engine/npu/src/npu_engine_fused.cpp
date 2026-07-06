/** NPU Engine — Fused Layer. One xclbin call per layer (QKV→attn→O→GU→SiLU→D).
 *  Uses design_full_layer.xclbin with per-position instruction files.
 *  Target: 182 tok/s at M=32 batch.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// BF16 helpers
static inline uint16_t f2bf(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    return (uint16_t)((b + 0x8000) >> 16);
}
static inline float bf16f(uint16_t v) {
    uint32_t b = (uint32_t)v << 16; float r;
    memcpy(&r, &b, 4); return r;
}

// Load a binary file
static std::vector<char> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return {}; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> data(sz); f.read(data.data(), sz);
    return data;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        printf("Usage: %s model.q4nx [decode_tokens]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int num_decode = (argc > 2) ? atoi(argv[2]) : 32;
    
    // --- Paths ---
    const char* xclbin_root = getenv("FUSED_XCLBIN_DIR");
    if (!xclbin_root) xclbin_root = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
    
    const char* weights_dir = getenv("FUSED_WEIGHTS_DIR");
    if (!weights_dir) weights_dir = "/home/bcloud/npu-sandbox/npu-infer/build/int8";
    
    // --- Load xclbin ---
    printf("Loading fused xclbin...\n");
    std::string xclbin_path = std::string(xclbin_root) + "/design.xclbin";
    auto xclbin_data = load_file(xclbin_path);
    if (xclbin_data.empty()) { fprintf(stderr, "Failed to load xclbin\n"); return 1; }
    
    xrt::device dev(0);
    xrt::xclbin xclbin(xclbin_data);
    dev.register_xclbin(xclbin);
    xrt::kernel kernel(dev, xclbin.get_uuid(), "MLIR_AIE");
    
    int dg = kernel.group_id(3); // HOST DRAM bank
    int ig = kernel.group_id(1); // SRAM bank for instructions
    
    // --- Pre-load instruction files ---
    printf("Loading instruction files...\n");
    std::vector<xrt::bo> instr_bos;
    std::vector<int> instr_positions;
    // Generic fallback
    auto generic_data = load_file(std::string(xclbin_root) + "/design.bin");
    xrt::bo generic_bo(dev, generic_data.size(), xrt::bo::flags::cacheable, ig);
    memcpy(generic_bo.map(), generic_data.data(), generic_data.size());
    generic_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    printf("  Loaded design.bin (%zu bytes, generic fallback)\n", generic_data.size());
    
    // Try to load all 128 position-specific files
    instr_bos.resize(128);
    for (int pos = 0; pos < 128; pos++) {
        char fname[256];
        snprintf(fname, 256, "%s/design-token127-to-token%d.bin", xclbin_root, pos);
        auto data = load_file(fname);
        if (!data.empty()) {
            xrt::bo bo(dev, data.size(), xrt::bo::flags::cacheable, ig);
            memcpy(bo.map(), data.data(), data.size());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            instr_bos[pos] = std::move(bo);
        }
        // If position-specific file doesn't exist, generic_bo will be used via fallback below
    }
    printf("  Position-specific files: %zu\n", 
           std::count_if(instr_bos.begin(), instr_bos.end(), [](auto& b){ return b; }));
    // Helper to get instruction BO for a position
    auto get_instr = [&](int pos) -> xrt::bo& {
        if (pos >= 0 && pos < 128 && instr_bos[pos]) return instr_bos[pos];
        return generic_bo;
    };
    
    // --- Pre-load all 28 layer weight BOs ---
    printf("Loading weight files...\n");
    int num_layers = 28;
    std::vector<xrt::bo> weight_bos(num_layers);
    size_t weight_bytes = 0;
    for (int l = 0; l < num_layers; l++) {
        char fname[256];
        snprintf(fname, 256, "%s/fused_weights_l%d.bin", weights_dir, l);
        auto data = load_file(fname);
        if (data.empty()) { fprintf(stderr, "Missing weights for layer %d\n", l); return 1; }
        
        xrt::bo bo(dev, data.size(), xrt::bo::flags::host_only, dg);
        memcpy(bo.map(), data.data(), data.size());
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        weight_bos[l] = std::move(bo);
        weight_bytes = data.size();
    }
    printf("  %d weight files loaded (%zu MB each)\n", num_layers, weight_bytes / 1048576);
    
    // --- Create persistent BOs for KV cache, hidden, output ---
    printf("Creating data BOs...\n");
    xrt::bo bKCache(dev, 8192*4, xrt::bo::flags::host_only, dg);
    xrt::bo bVCache(dev, 8192*4, xrt::bo::flags::host_only, dg);
    xrt::bo bHidden(dev, 512*4, xrt::bo::flags::host_only, dg);
    xrt::bo bOutput(dev, 512*4, xrt::bo::flags::host_only, dg);
    
    // Initialize KV cache to zeros
    memset(bKCache.map(), 0, 8192*4);
    memset(bVCache.map(), 0, 8192*4);
    bKCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bVCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Embedding table (for input token → hidden state)
    // For this initial version: load the model and use random embeddings
    auto model_data = load_file(model_path);
    printf("Model: %s (%zu bytes)\n", model_path, model_data.size());
    
    // Prefill with a fixed prompt
    int npt = 9;
    int prompt_tokens[] = {151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198};
    
    // For the fusion engine, we use a simple approach:
    // Each fused layer call reads hidden state from BO, writes to output BO
    // For prefill: run layers sequentially, feeding output back as input
    
    // --- Prefill ---
    printf("\n=== Prefill %d tokens ===\n", npt);
    auto t0 = std::chrono::steady_clock::now();
    
    // Initialize hidden state from embed table (simplified: use constant)
    uint16_t* hdata = (uint16_t*)bHidden.map();
    for (int i = 0; i < 1024; i++) hdata[i] = f2bf(0.01f);
    bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Prefill: for each prompt token, run all layers, updating KV cache
    for (int pi = 0; pi < npt; pi++) {
        // Set hidden state from embedding (use constant for now, real impl would lookup embed table)
        uint16_t* hd = (uint16_t*)bHidden.map();
        for (int i = 0; i < 1024; i++) hd[i] = f2bf(0.01f * (1 + (pi + i) % 10));
        bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Run all 28 layers at position pi
        for (int l = 0; l < num_layers; l++) {
            auto& ibo2 = get_instr(pi);
            auto run = kernel((uint64_t)3, ibo2, (uint32_t)ibo2.size(),
                             bKCache, bVCache, weight_bos[l], bOutput, bHidden);
            run.wait();
            
            // Output becomes input for next layer
            memcpy(bHidden.map(), bOutput.map(), 512*4);
            bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
    }
    
    double prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  Prefill: %.0fms (%.0f ms/tok)\n", prefill_ms, prefill_ms / npt);
    
    // --- Decode loop ---
    printf("\n=== Decode %d tokens ===\n", num_decode);
    auto t1 = std::chrono::steady_clock::now();
    
    for (int step = 0; step < num_decode; step++) {
        int pos = npt + step;  // current KV cache position
        
        auto ts = std::chrono::steady_clock::now();
        
        // Set hidden state from embedding (constant for now)
        uint16_t* hd = (uint16_t*)bHidden.map();
        for (int i = 0; i < 1024; i++) hd[i] = f2bf(0.01f * (1 + (pos + step + i) % 10));
        bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        // Run all 28 layers at position pos
        for (int l = 0; l < num_layers; l++) {
            auto& ibo2 = get_instr(pos);
            auto run = kernel((uint64_t)3, ibo2, (uint32_t)ibo2.size(),
                             bKCache, bVCache, weight_bos[l], bOutput, bHidden);
            run.wait();
            
            // Output becomes input for next layer
            memcpy(bHidden.map(), bOutput.map(), 512*4);
            bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        
        double step_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ts).count();
        
        if (step < 5 || step % 10 == 0 || step == num_decode - 1) {
            printf("  [%d] %.1fms (%.0f ms/tok)\n", step, step_ms, step_ms);
        }
    }
    
    double total_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t1).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) ===\n",
           total_s * 1000 / num_decode, num_decode / total_s);
    
    return 0;
}
