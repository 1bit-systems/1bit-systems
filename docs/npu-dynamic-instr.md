> **📜 Historical reference** — This document describes the NPU reverse-engineering process comparing against FastFlowLM's dynamic instruction generator. The NPU engine now generates its own instruction sequences natively via `npu_engine_universal`. The FLM comparison is kept for provenance.
>
# Dynamic NPU Instruction Generation — Matching FLM's Architecture

## Overview

FLM (FastFlowLM) generates NPU instructions **dynamically at runtime** using `npu_sequence` → `xrt::ext::kernel`. Your engine currently uses **pre-compiled instruction files** with `xrt::kernel`. This is the fundamental architecture gap.

## Architecture Comparison

### Your current engine (pre-compiled):
```
xrt::kernel k = xrt::kernel(hc, "MLIR_AIE");
k.run(3, instr_bo, ninstrs, act_bo, weight_bo, out_bo);
  ↕
Pre-compiled .insts files → tied to specific xclbin shapes
Cannot adapt to different M, K, N at runtime
BO addresses baked into instructions at compile time
```

### FLM (dynamic):
```
npu_sequence seq;
Gemm::generate_seq(&seq, M, K, N, ...);  // ← generates commands for ANY shape
seq.cmds2seq();                              // ← compiles to instructions
npu_app app = mgr.create_app();           // ← wraps as xrt::ext::kernel
app.run(act_bo, weight_bo, out_bo);       // ← DDR_PATCH patches BO addrs
  ↕
Instructions generated per-shape at runtime
BO addresses patched via DDR_PATCH commands
Supports any M, K, N the xclbin can handle
```

## What We Built

### `include/npu_instr.hpp` — The instruction system
- `npu_sequence` — holds commands, compiles to uint32_t[] instructions
- `npu_dma_block_cmd` — DMA buffer descriptor (data movement between DDR and tiles)
- `npu_ddr_cmd` — DDR address patching (replaces BO addresses in instructions)
- `npu_write_cmd` — Register writes + queue push (trigger DMA chains)
- `npu_issue_token_cmd` — DMA completion signaling
- `npu_wait_cmd` — Synchronization barrier
- `generate_gemm_seq_full()` — Generates complete GEMM instruction sequence
- `generate_gemm_seq_patches()` — Patches pre-compiled instructions with BO addresses

### `include/npu_app.hpp` — The application layer
- `npu_app` — wraps `npu_sequence` + `xrt::ext::kernel`
- `npu_app_manager` — wraps one xclbin + hw_context, creates apps
- `npu_xclbin_manager` — top-level manager, deduplicates xclbins

## Integration Steps

### Step 1: Replace xrt::kernel with npu_app in your engine

**Before (npu_engine_cb.cpp):**
```cpp
xrt::kernel k_q(*hc_q, "MLIR_AIE");
auto r = k_q(3, *bI_q, (unsigned)ins_q.size(), *bA_q, *layerB[l], *bC_q);
r.wait();
```

**After:**
```cpp
// During init:
npu_xclbin_manager xclbin_mgr(npu_device::npu2, &dev);
npu_app_manager* qkv_mgr = xclbin_mgr.register_xclbin("final_i8_QKV_v.xclbin");
npu_app qkv_app = qkv_mgr->create_app();

// Per-layer:
// 1. Pack weights into BO (same as before)
// 2. Generate instruction sequence for this specific M, K, N
auto* seq = qkv_app.seq();
seq->clear();
gemm_config cfg = { .M = XM, .K = H, .N = 4096, .tile_M = 128, .tile_K = 64, .tile_N = 128, .num_cols_used = 8, .arg_act = 3, .arg_weight = 4, .arg_out = 5 };
generate_gemm_seq_patches(*seq, cfg, qkv_instrs.data(), qkv_instrs.size(), patch_locs);
seq->cmds2seq();
// 3. Run
qkv_app.run(*bA, *layerB[l], *bC);
```

### Step 2: Generate DDR_PATCH locations from your .insts files

The key insight from FLM: your pre-compiled `.insts` files contain DMA block writes (BDs) at fixed positions. Each BD's address field needs to be patched with the actual BO address. The `npu_ddr_cmd` does this.

To find patch locations, parse your .insts file looking for:
```
0x00000001  ← XAIE_IO_BLOCKWRITE
0x00000000
0x....1D000  ← BD register address at (row, col, bd_id)
0x....
...
```

Each BLOCKWRITE BD needs a corresponding DDR_PATCH. The `arg_idx` tells the hardware which BO argument provides the address.

### Step 3: Generate full dynamic sequences (FLM-compatible)

For full dynamic instruction generation (bypassing .insts files entirely):

```cpp
gemm_config cfg = {
    .M = 128,
    .K = 1024,
    .N = 4096,
    .num_cols_used = 8,
    .arg_act = 3,
    .arg_weight = 4,
    .arg_out = 5
};

seq->clear();
generate_gemm_seq_full(*seq, cfg, 0, 0, 0);
seq->cmds2seq();
qkv_app.run(act_bo, weight_bo, out_bo);
```

### Step 4: Add to CMakeLists.txt

```cmake
# No new libraries needed — header-only
# But need to link aiebu for ELF generation:
target_link_libraries(npu_engine PRIVATE aiebu xrt_coreutil)
```

## Expected Benefits

| Aspect | Before (pre-compiled) | After (dynamic) |
|--------|----------------------|-----------------|
| Shape flexibility | Fixed M=128, K, N | Any M, K, N |
| BO patching | None (baked at compile) | DDR_PATCH at runtime |
| Multiple models | Separate .insts per shape | One sequence generator |
| FLM compatibility | None | Same architecture |
| Coherence gap | xclbin numerics diverge | Can match FLM's BD layout |
| Init time | Fast (load pre-compiled) | Slower (generate per shape) |

## Matching FLM's Exact Numerics

The remaining coherence issues are in the **xclbin kernels themselves**, not the instruction generation. But using dynamic instruction generation gives you:

1. **Exact same DMA protocol** as FLM (same BD layout, same chaining, same DDR_PATCH)
2. **Control over weight placement** in tile SRAM (FLM pre-packs weights into tile SRAM before compute)
3. **Ability to add debug/trace commands** to verify BO contents at each stage

To fully match FLM:
1. Use FLM's xclbins with your instruction generator (the BO captures show the exact BD format)
2. Or rebuild your custom xclbins to match FLM's weight-stationary protocol (weights in tile SRAM, not kernel args)

## Files Created

```
include/npu_instr.hpp  — NPU instruction/sequence system (8 cmd types + sequence)
include/npu_app.hpp    — NPU app manager (app, app_manager, xclbin_manager)
tests/test_dynamic_instr.cpp  — Example/test of dynamic instruction generation
```
