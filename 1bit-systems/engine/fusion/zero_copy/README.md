# Zero-Copy NPU↔GPU↔Host Fusion Substrate

## Status: ⚡ PROVEN on Strix Halo (gfx1151 + XDNA 2)

This directory contains the **correct, empirically-verified zero-copy substrate** for NPU+GPU fused inference on Strix Halo. Every previous "fused" implementation was aspirational (lied in its headers, never compiled, IO_PAGE_FAULT'd, or ran GPU-only).

---

## What It Ships (Steps 1–4)

| # | Step | Deliverable | Status |
|---|------|-------------|--------|
| **2** | Zero-copy handoff | `SharedBO` — NPU-owned, GPU-imported, host-coherent. All three alias the same physical pages. No memcpy between them. | ✅ **PROVEN** on hardware (3/3 runs, zero IO_PAGE_FAULTs) |
| **1** | NPU unblock | Root-cause analysis of the `npu_engine_universal.cpp` boot EINVAL + `xclbin_health` validation tool. | ✅ **Debunked** "too many contexts" myth; pinpointed `Invalid num_col N` |
| **3** | NPU-FFN ∥ GPU-attn pipeline | `PipelineOverlap` — 2-slot double-buffered pipeline skeleton on SharedBO. Dummy callbacks exercise the pattern. | ✅ **Skeleton runs** (95.79ms vs 120ms sequential with 40 layers) |
| **4** | Cleanup | `gpu_npu_bridge.cpp` header fixed (no more lying about zero-copy); dead `import_to_hip/import_to_xrt` removed; this README. | ✅ **Fixed** |

---

## Architecture

```
              SharedBO (NPU-owned XRT HOST_ONLY BO)
              ┌──────────────────────────────────────┐
              │          physical pages               │
              └──┬──────────────────────────────┬─────┘
                 │                              │
                 ▼                              ▼
       XRT mmap (CPU)                  dma-buf fd
       → host_ptr                       → exported by NPU → imported by GPU
       (coherent, system RAM)           → HIP hipHostRegister() (test)
                                        → Vulkan VK_KHR_external_memory_fd
                                          (production — only API that works
                                           on ROCm 7.2.4, which lacks HIP
                                           DmaBuf external memory)
                 │                              │
                 └──────────┬───────────────────┘
                            ▼
                   ONE set of pages, three views.
                   No memcpy, no staging buffer, no DMA.
```

**Direction rule (critical):** The NPU must **own** the allocation. The `gpu_npu_bridge.cpp` approach (GPU allocates GTT, imports into NPU via XRT dma-buf) causes `AMD-Vi IO_PAGE_FAULT` because `amdxdna`'s dma-buf import path doesn't wire up the NPU's IOMMU domain correctly. When the NPU owns the allocation (XRT HOST_ONLY BO), its IOMMU domain already covers the pages, and the mature `amdgpu` import path handles the GPU side correctly.

---

## Key Findings That Overturn Previous Assumptions

### 1. "5 hw_contexts collide" → WRONG. Real cause: `Invalid num_col N`

The state-of-the-stack doc (2026-07-14) hypothesized that `npu_engine_universal.cpp`'s boot SIGABRT came from 4-5 simultaneous `xrt::hw_context`s exhausting the AIE column/tile budget. **This is incorrect.**

- **Probe**: 8 identical xclbins → 8 OK. 5 distinct xclbins → 3 OK, 2 FAIL. The failing ones (`final_12col_test.xclbin`, `final_40col_v2.xclbin`) fail EVEN FIRST/ALONE.
- **dmesg smoking gun**: `[drm] *ERROR* aie2_hwctx_col_list: Invalid num_col 12`
- **Real cause**: The engine's xclbins request a number of AIE columns the driver's column allocator rejects (12 and 40 both fail; small-column bf16 designs succeed at ≤8). The 40-column target the state doc flagged as "contested" is genuinely **blocked by the shipping driver**.

**The fix**: Use xclbins whose `num_col` the driver accepts. The `xclbin_health` tool validates this at startup so the engine diagnoses (not SIGABRTs) bad xclbins.

### 2. `hipExternalMemoryHandleTypeDmaBuf` → DOES NOT EXIST in ROCm 7.2.4

The `gpu_npu_bridge.cpp` code that used `hipImportExternalMemory` with `hipExternalMemoryHandleTypeDmaBuf` never compiled. ROCm 7.2.4's HIP lacks that enum value. The only Linux handle type is `OpaqueFd` (inter-ROCm internal), which doesn't accept cross-device dma-buf fds from other drivers.

**The production GPU import path must be Vulkan** (`VK_KHR_external_memory_fd`), matching `engine/fusion/gpu_attn.zig` and the stub in `interop.zig`.

### 3. Zero-copy IS possible and PROVEN

`hipHostRegister(npu_mmap_ptr) + hipHostGetDevicePointer()` gives a GPU device pointer aliasing the NPU's pages on this APU. The test proves: GPU writes → CPU reads with NO D2H copy = match. Three runs every time, zero IO_PAGE_FAULTs.

---

## File Inventory

| File | Purpose |
|------|---------|
| `shared_bo.h/.cpp` | NPU-owned zero-copy buffer. XRT HOST_ONLY BO + dma-buf fd export. No HIP dep. |
| `test_zero_copy.cpp` | Airtight zero-copy proof. Uses `hipHostRegister` to validate the memory model. |
| `pipeline_overlap.h/.cpp` | 2-slot double-buffered pipeline skeleton (NPU∥GPU). Injected callbacks for testability. |
| `test_pipeline.cpp` | Pipeline demo with dummy timings. |
| `xclbin_health.cpp` | Validate any xclbin against the running driver. Detects `Invalid num_col` rejection. |
| `probe_contexts.cpp` | Probe max concurrent `hw_context`s (empirically disproves the "5 contexts collide" myth). |
| `probe_multi_xclbin.cpp` | Probe mixing distinct xclbins (tests what the real engine does). |
| `Makefile` | Build everything. Needs XRT + ROCm 7.2.4. |

---

## Next Steps for Production

1. **Vulkan dma-buf import** (replaces the `hipHostRegister` test idiom): wire `xrt::bo::export_buffer()` fd → `VK_KHR_external_memory_fd` in `interop.zig`. This gives the production GPU path (matching `gpu_attn.zig`).
2. **Replace pipeline dummy callbacks**: inject real HIP-attention kernel launches and real XRT-FFN kernel launches into the 2-slot pipeline.
3. **xclbin column-count**: build/reuse xclbins with `num_col` ≤8 (what the driver accepts) for the 1bit engine, or build a generic instruction-driven xclbin (like FastFlowLM's `mvm_i8`) that handles all shapes from one context.
4. **Integrate `xclbin_health`** into the NPU engine startup as a graceful validation gate before any `CREATE_HWCTX`.
