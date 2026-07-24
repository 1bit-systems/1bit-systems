// test_lora_runtime.cpp — Smoke test for the LoRA runtime
// Tests: creation, empty state, load/unload from nonexistent file (error path)

#include "lora.h"
#include <cstdio>
#include <cassert>
#include <string>
#include <vector>

int main() {
    printf("test_lora_runtime: starting...\n");
    
    // 1. Create manager
    LoraManager mgr;
    assert(mgr.adapter_count() == 0);
    printf("  ✅ LoraManager created, count=%d\n", mgr.adapter_count());
    
    // 2. Empty adapter list
    auto names = mgr.loaded_adapters();
    assert(names.empty());
    printf("  ✅ Empty adapter list\n");
    
    // 3. Load nonexistent GGUF — should fail gracefully
    int idx = mgr.load_adapter("/nonexistent/path/lora.gguf");
    assert(idx == -1);
    printf("  ✅ Load nonexistent fails gracefully (idx=%d)\n", idx);
    
    // 4. Unload invalid index — should fail gracefully
    bool ok = mgr.unload_adapter(0);
    assert(!ok);
    printf("  ✅ Unload invalid index fails gracefully\n");
    
    // 5. Unload nonexistent name — should fail gracefully
    ok = mgr.unload_adapter("nonexistent");
    assert(!ok);
    printf("  ✅ Unload nonexistent name fails gracefully\n");
    
    // 6. Clear all on empty manager
    mgr.clear_all();
    printf("  ✅ Clear all on empty manager\n");
    
    // 7. Apply on empty weights — should not crash
    std::vector<float> weights(16, 1.0f);
    mgr.apply(weights.data(), "test_tensor", 4, 4);
    printf("  ✅ Apply on empty manager (no-op)\n");
    
    // 8. has_adapter_for on empty
    assert(!mgr.has_adapter_for("test_tensor"));
    printf("  ✅ has_adapter_for false when empty\n");
    
    // 9. Enable/disable on empty
    mgr.enable_adapter(0, true);   // no-op, should not crash
    mgr.enable_adapter("x", true); // no-op, should not crash
    printf("  ✅ Enable/disable on empty manager (no-op)\n");
    
    // 10. Load adapter with invalid GGUF (path exists but not GGUF)
    // Use a temp file that's not a valid GGUF
    {
        FILE* f = fopen("/tmp/test_invalid.gguf", "wb");
        assert(f);
        fwrite("NOTAGGUFFILE", 1, 12, f);
        fclose(f);
    }
    idx = mgr.load_adapter("/tmp/test_invalid.gguf");
    assert(idx == -1);
    printf("  ✅ Load invalid GGUF fails gracefully\n");
    
    printf("\n=== ALL LORA RUNTIME SMOKE TESTS PASSED ===\n");
    return 0;
}
