# NPU+GPU Direct Wiring Architecture

**Strix Halo**: NPU (XDNA 2) + GPU (Radeon 8060S) share unified memory.
**Goal**: Zero-copy pipeline — NPU GEMM outputs feed directly into GPU compute shaders without host-RAM staging.

## Current Bottleneck

```
NPU xclbin → int16 BO → XRT sync → host RAM → float convert → CPU attention/SiLU → quantize → XRT sync → NPU xclbin
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^
                                                      Every intermediate op copies through host memory
```

## Target Architecture

```
NPU xclbin → [dma-buf fd] → Vulkan external memory → GPU compute shader → [dma-buf fd] → NPU xclbin
                                 (zero copy, same physical memory)
```

## Implementation Plan

### Phase 1: XRT dma-buf Export
XRT `bo::export_buffer()` exports a BO as a dma-buf file descriptor.
```
xrt::bo npu_output = xrt::bo(device, size, XRT_BO_FLAGS_HOST_ONLY, group_id);
int dma_buf_fd = npu_output.export_buffer();  // fd for the buffer
```

### Phase 2: Vulkan dma-buf Import
Vulkan `VK_EXT_external_memory_dma_buf` imports a dma-buf fd.
```
VkExternalMemoryBufferCreateInfo ext = {
    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
};
VkImportMemoryFdInfoEXT import = {
    .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_EXT,
    .fd = dma_buf_fd  // from XRT export
};
vk::DeviceMemory = allocMemory(..., &import);
vk::Buffer = createBuffer(device, size, usage, handleTypes: DMA_BUF);
bindBufferMemory(buffer, memory, 0);
```

### Phase 3: Compute Shaders (replacing CPU ops)
Move these to Vulkan compute shaders (GLSL/SPIR-V):
1. **Attention** — accepts Q, K, V from NPU, produces attention output
2. **RoPE** — rotary position embedding (or fuse into QKV)
3. **SiLU activation** — gate + element-wise multiply
4. **RMSNorm** — layer normalization
5. **LM head** — dot product over full vocab (ZINC already has `dmmv_stq1_0.comp`)

### Phase 4: NPU Write / GPU Read Synchronization
Use XRT completion + Vulkan semaphore timeline for ordering:
```
NPU kernel → run.wait() → GPU timeline semaphore signal → GPU compute → GPU timeline semaphore signal → NPU read
```

## Key APIs

| API | Purpose |
|-----|---------|
| `xrt::bo::export_buffer()` | Export XRT BO as dma-buf fd |
| `VK_EXT_external_memory_dma_buf` | Import dma-buf into Vulkan |
| `VK_KHR_external_memory_fd` | FD-based external memory |
| `VK_KHR_timeline_semaphore` | Cross-device synchronization |
| ZINC `src/compute/dmmv.zig` | Existing GPU compute dispatch (reference) |
| ZINC `src/shaders/dmmv_stq1_0.comp` | Existing GPU GLSL shader (reference) |

## Risks

- XRT + Vulkan interop not tested on this hardware — need proof-of-concept dma-buf round-trip
- Timeline semaphore across NPU/GPU may need AMD-specific extensions
- NPU BOs are `XRT_BO_FLAGS_HOST_ONLY` — may not support dma-buf export (try `XRT_BO_FLAGS_CACHEABLE` or device-only)
- GPU compute shaders for attention need GLSL compilation for Radeon 8060S (gfx1151)
