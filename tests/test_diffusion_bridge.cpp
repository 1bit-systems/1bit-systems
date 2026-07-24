// test_diffusion_bridge.cpp — Smoke test for the Diffusion bridge
// Tests: create, destroy, load nonexistent model, health check

#include "diffusion_bridge.h"
#include <cstdio>
#include <cassert>
#include <string>

int main() {
    printf("test_diffusion_bridge: starting...\n");
    
    // 1. Create engine (singleton)
    auto& engine = diffusion_engine();
    assert(!engine.is_loaded());
    printf("  ✅ DiffusionEngine created, not loaded\n");
    
    // 2. Load nonexistent model — should fail gracefully
    bool ok = engine.load_model("/nonexistent/model.gguf");
    assert(!ok);
    assert(!engine.is_loaded());
    printf("  ✅ Load nonexistent model fails gracefully\n");
    
    // 3. Load with empty path
    ok = engine.load_model("");
    assert(!ok);
    printf("  ✅ Load empty path fails gracefully\n");
    
    // 4. Unload when nothing loaded — should not crash
    engine.unload_model();
    assert(!engine.is_loaded());
    printf("  ✅ Unload on empty engine (no-op)\n");
    
    // 5. Generate txt2img with no model — should return empty result
    DiffusionParams params;
    params.prompt = "test";
    auto result = engine.txt2img(params);
    assert(result.data.empty());
    assert(result.width == 0);
    printf("  ✅ txt2img on empty engine returns empty result\n");
    
    // 6. supports_video when not loaded
    assert(!engine.supports_video());
    printf("  ✅ supports_video false when not loaded\n");
    
    // 7. Load upscaler with nonexistent path
    ok = engine.load_upscaler("/nonexistent/upscaler.pth");
    assert(!ok);  // should fail
    printf("  ✅ Load nonexistent upscaler fails gracefully\n");
    
    // 8. Upscale with no upscaler loaded
    std::vector<uint8_t> dummy_rgb(64*64*3, 128);
    auto up_result = engine.upscale(dummy_rgb.data(), 64, 64, 2);
    assert(up_result.data.empty());
    printf("  ✅ Upscale without upscaler returns empty\n");
    
    // 9. Load/clear LoRA with no model
    engine.load_lora("/nonexistent/lora.safetensors", 1.0f);
    engine.clear_loras();
    printf("  ✅ LoRA ops on empty engine (no-op)\n");
    
    // 10. Second unload (should be safe)
    engine.unload_model();
    printf("  ✅ Double unload (safe)\n");
    
    printf("\n=== ALL DIFFUSION BRIDGE SMOKE TESTS PASSED ===\n");
    return 0;
}
