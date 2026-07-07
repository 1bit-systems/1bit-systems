/**
 * debug_fused_lmhead.cpp — Debug the fused target's LM head output.
 * Build & run as before.
 */
#define FUSED_DBG
#include "engine/npu_fused_target.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

static const char* kModelPath  = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
static const char* kXclbinDir  = "/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127";
static const char* kWeightsDir = "/home/bcloud/npu-sandbox/npu-infer/build/int8/capref";
static constexpr int32_t kTargetLayerIds[] = {1, 6, 12, 18, 24};

// Direct test: create the target and check every step
#include "engine/npu_fused_target.h"
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int H = 1024, NV = 151936;

    // Load the model metadata for LM head reference
    npu_fused_detail::Q4NXModel model;
    if (!model.load(kModelPath)) { fprintf(stderr, "Model load failed\n"); return 1; }
    
    printf("Model: vocab=%d H=%d layers=%d\n", model.vocab_size, H, model.num_layers);
    printf("final_norm[0..4]: %.4f %.4f %.4f %.4f %.4f\n",
           model.final_norm_w[0], model.final_norm_w[1],
           model.final_norm_w[2], model.final_norm_w[3],
           model.final_norm_w[4]);
    
    // Get embedding for test token
    int32_t test_token = 151643;  // <|im_start|>
    const uint16_t* emb_row = model.embed_table + (size_t)test_token * H;
    printf("\nEmbedding for token %d (first 5 BF16): ", test_token);
    for (int i = 0; i < 5; i++) printf("0x%04x(%.4f) ", emb_row[i], npu_fused_detail::bf16f(emb_row[i]));
    printf("\n");

    // ── Direct XRT test ──
    printf("\n=== Direct XRT Test ===\n");
    
    // Load xclbin
    auto xclbin_data = npu_fused_detail::load_binary(std::string(kXclbinDir) + "/design.xclbin");
    printf("xclbin: %zu bytes\n", xclbin_data.size());
    
    auto generic_data = npu_fused_detail::load_binary(std::string(kXclbinDir) + "/design.bin");
    printf("generic instr: %zu bytes\n", generic_data.size());
    
    auto wref_data = npu_fused_detail::load_binary(std::string(kWeightsDir) + "/wref_l0.bin");
    printf("wref_l0: %zu bytes\n", wref_data.size());

    auto rt_data = npu_fused_detail::load_binary(std::string(kWeightsDir) + "/rope_table.bin");
    printf("rope_table: %zu bytes\n", rt_data.size());
    
    // Setup device
    xrt::device dev(0);
    xrt::xclbin xclbin(xclbin_data);
    dev.register_xclbin(xclbin);
    xrt::hw_context hctx(dev, xclbin.get_uuid());
    xrt::kernel kernel(hctx, "MLIR_AIE");
    
    int ig = kernel.group_id(1);
    int kg = kernel.group_id(3);
    int vg = kernel.group_id(4);
    int wg = kernel.group_id(5);
    int og = kernel.group_id(6);
    int hg = kernel.group_id(7);
    printf("group_ids: 1=%d 3=%d 4=%d 5=%d 6=%d 7=%d\n", ig, kg, vg, wg, og, hg);
    
    // Load instruction BO
    xrt::bo generic_bo(dev, generic_data.size(), xrt::bo::flags::cacheable, ig);
    memcpy(generic_bo.map(), generic_data.data(), generic_data.size());
    generic_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Load per-position instruction for pos=0
    auto pos0_data = npu_fused_detail::load_binary(
        std::string(kXclbinDir) + "/design-token127-to-token0.bin");
    printf("instr pos0: %zu bytes\n", pos0_data.size());
    xrt::bo instr_bo(dev, pos0_data.size(), xrt::bo::flags::cacheable, ig);
    memcpy(instr_bo.map(), pos0_data.data(), pos0_data.size());
    instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Load weight BO for layer 0
    xrt::bo weight_bo(dev, wref_data.size(), xrt::bo::flags::host_only, wg);
    memcpy(weight_bo.map(), wref_data.data(), wref_data.size());
    weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Patch RoPE at position 0
    const size_t ROPE_COSSIN_DWORDS = 64;
    const size_t ROPE_COSSIN_DWORD_OFFSET = 1152;
    const int32_t* rope_table = (const int32_t*)rt_data.data();
    int32_t* wmap = (int32_t*)weight_bo.map();
    memcpy(wmap + ROPE_COSSIN_DWORD_OFFSET, rope_table, ROPE_COSSIN_DWORDS * 4);
    weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                   ROPE_COSSIN_DWORDS * 4, ROPE_COSSIN_DWORD_OFFSET * 4);
    
    // Create KV caches (256KB each)
    const size_t KV_BYTES = 256 * 1024;
    xrt::bo kCache(dev, KV_BYTES, xrt::bo::flags::host_only, kg);
    xrt::bo vCache(dev, KV_BYTES, xrt::bo::flags::host_only, vg);
    memset(kCache.map(), 0, KV_BYTES);
    memset(vCache.map(), 0, KV_BYTES);
    kCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    vCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Create hidden and output BOs
    xrt::bo bHidden(dev, H * 2, xrt::bo::flags::host_only, hg);
    xrt::bo bOutput(dev, H * 2, xrt::bo::flags::host_only, og);
    
    // Load embedding input
    memcpy(bHidden.map(), emb_row, H * 2);
    bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    printf("\nRunning kernel...\n");
    
    // Run
    auto run = kernel(3, instr_bo, (uint32_t)1723,
                     kCache, vCache, weight_bo,
                     bOutput, bHidden);
    run.wait();
    
    // Read output
    bOutput.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t* out_data = (uint16_t*)bOutput.map();
    
    // Check output
    int nonzero = 0;
    float sum = 0;
    float first_vals[10];
    for (int i = 0; i < H; i++) {
        float val = npu_fused_detail::bf16f(out_data[i]);
        if (val != 0) nonzero++;
        sum += val;
        if (i < 10) first_vals[i] = val;
    }
    
    printf("Output from fused kernel (layer 0, pos 0):\n");
    printf("  nonzero: %d/%d\n", nonzero, H);
    printf("  mean: %.4f\n", sum / H);
    printf("  first 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", first_vals[i]);
    printf("\n");
    
    if (nonzero == 0) {
        printf("\n⚠️ ALL ZEROS — fused kernel is not producing output!\n");
        printf("This means the xclbin/kernel/BO setup is wrong.\n");
        
        // Check BO sizes
        printf("\nBO map sizes:\n");
        printf("  instr_bo: %zu\n", instr_bo.size());
        printf("  weight_bo: %zu\n", weight_bo.size());
        printf("  kCache: %zu\n", kCache.size());
        printf("  vCache: %zu\n", vCache.size());
        printf("  bHidden: %zu\n", bHidden.size());
        printf("  bOutput: %zu\n", bOutput.size());
    }
    
    // ── Multi-layer direct test (no reload) ──
    printf("\n=== Direct XRT Multi-Layer Test ===\n");
    // Run layers 0-4 directly, chaining output to input, NO reload
    for (int layer = 0; layer < 5; layer++) {
        // Load weight for this layer
        char wpath[256];
        snprintf(wpath, 256, "%s/wref_l%d.bin", kWeightsDir, layer);
        auto wdata = npu_fused_detail::load_binary(wpath);
        xrt::bo lweight_bo(dev, wdata.size(), xrt::bo::flags::host_only, wg);
        memcpy(lweight_bo.map(), wdata.data(), wdata.size());
        // Patch RoPE at position 0
        int32_t* lwmap = (int32_t*)lweight_bo.map();
        memcpy(lwmap + ROPE_COSSIN_DWORD_OFFSET, rope_table, ROPE_COSSIN_DWORDS * 4);
        lweight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE,
                       ROPE_COSSIN_DWORDS * 4, ROPE_COSSIN_DWORD_OFFSET * 4);
        
        // Layer KV caches (separate per layer)
        xrt::bo lkCache(dev, KV_BYTES, xrt::bo::flags::host_only, kg);
        xrt::bo lvCache(dev, KV_BYTES, xrt::bo::flags::host_only, vg);
        memset(lkCache.map(), 0, KV_BYTES);
        memset(lvCache.map(), 0, KV_BYTES);
        lkCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        lvCache.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        auto lrun = kernel(3, instr_bo, (uint32_t)1723,
                          lkCache, lvCache, lweight_bo,
                          bOutput, bHidden);
        lrun.wait();
        bOutput.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        
        uint16_t* lod = (uint16_t*)bOutput.map();
        int lnz = 0; float lsum = 0;
        bool has_nan = false;
        for (int i = 0; i < H; i++) {
            float v = npu_fused_detail::bf16f(lod[i]);
            if (std::isnan(v)) has_nan = true;
            if (v != 0) lnz++;
            lsum += v;
        }
        printf("  Layer %d: nonzero=%d/%d mean=%.4f NaN=%s\n",
               layer, lnz, H, lsum / H, has_nan ? "YES⚠️" : "no");
        
        // Copy output to input for next layer
        memcpy(bHidden.map(), bOutput.map(), H * 2);
        bHidden.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        if (has_nan) {
            printf("    → NaN detected at layer %d, stopping\n", layer);
            break;
        }
    }
    
    // Now test with the FULL NpuFusedTarget
    printf("\n=== Full NpuFusedTarget Test ===\n");
    NpuFusedTarget target(kModelPath, kXclbinDir, kWeightsDir,
                           kTargetLayerIds, 5);
    
    printf("\nRunning target.forward(token=%d)...\n", test_token);
    std::vector<float> logits(NV);
    std::vector<float> hidden(28 * H);
    
    target.forward(&test_token, 1, logits.data(), hidden.data());
    
    // Check hidden state values
    float h_min = hidden[0], h_max = hidden[0], h_sum = 0;
    int h_nz = 0;
    for (int i = 0; i < 28 * H; i++) {
        if (hidden[i] < h_min) h_min = hidden[i];
        if (hidden[i] > h_max) h_max = hidden[i];
        h_sum += hidden[i];
        if (hidden[i] != 0) h_nz++;
    }
    printf("Hidden state stats: min=%.4f max=%.4f mean=%.4f nonzero=%d/%d\n",
           h_min, h_max, h_sum / (28 * H), h_nz, 28 * H);
    
    float l_min = logits[0], l_max = logits[0];
    int argmax = 0, l_nz = 0;
    for (int v = 0; v < NV; v++) {
        if (logits[v] < l_min) l_min = logits[v];
        if (logits[v] > l_max) { l_max = logits[v]; argmax = v; }
        if (logits[v] != 0) l_nz++;
    }
    printf("Logit stats: min=%.4f max=%.4f argmax=%d nonzero=%d/%d\n",
           l_min, l_max, argmax, l_nz, NV);
    
    return 0;
}
