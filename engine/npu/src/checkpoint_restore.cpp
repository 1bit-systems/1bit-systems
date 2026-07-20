/** checkpoint_restore.cpp — Multi-turn KV cache management.
 *  
 *  BUG 9 FIXED: context_len stored as a separate 4-byte header
 *  before the KV cache data, not read from the first cache entry.
 *
 *  BUGS 1-8, 10-12: Not applicable (this file handles only KV cache).
 */

#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdio>

#ifdef __XRT__
#include <xrt/xrt_bo.h>
#endif

// ─── Checkpoint: save KV cache from NPU to host ─────────────────────
// Format: [context_len:u32][layer_0_K:bf16[]][layer_0_V:bf16[]]...
//
size_t checkpoint_kv_cache(
#ifdef __XRT__
    xrt::bo&              kv_bo,
#else
    void*                 kv_bo,
#endif
    std::vector<uint8_t>& host_buffer,
    uint32_t              context_len,
    uint32_t              layer_count,
    size_t                cache_per_layer  // bytes per layer (K+V)
) {
    // Header: 4 bytes for context_len
    // Data: layer_count * cache_per_layer bytes
    size_t data_bytes = layer_count * cache_per_layer;
    host_buffer.resize(4 + data_bytes);
    
    // Write header
    memcpy(host_buffer.data(), &context_len, 4);
    
#ifdef __XRT__
    // Sync NPU → host
    kv_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, data_bytes, 0);
    memcpy(host_buffer.data() + 4, kv_bo.map(), data_bytes);
#else
    (void)kv_bo;
#endif
    
    return 4 + data_bytes;
}

// ─── Restore: load KV cache from host to NPU ────────────────────────
// Returns context_len via output parameter. Returns 0 on success, -1 on error.
//
int restore_kv_cache(
#ifdef __XRT__
    xrt::bo&              kv_bo,
#else
    void*                 kv_bo,
#endif
    const std::vector<uint8_t>& host_buffer,
    uint32_t&             context_len,
    uint32_t              layer_count,
    size_t                cache_per_layer
) {
    size_t data_bytes = layer_count * cache_per_layer;
    
    if (host_buffer.size() < 4 + data_bytes) {
        fprintf(stderr, "restore_kv: buffer too small (%zu < %zu)\n",
                host_buffer.size(), 4 + data_bytes);
        return -1;
    }
    
    // Read header
    memcpy(&context_len, host_buffer.data(), 4);
    
#ifdef __XRT__
    // Copy to device
    memcpy(kv_bo.map(), host_buffer.data() + 4, data_bytes);
    kv_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, data_bytes, 0);
#else
    (void)kv_bo;
#endif
    
    return 0;
}
