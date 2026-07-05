# FLM Integration — Fused NPU Engine

## Overview

The C++ NPU engine (npu_engine_cb) runs at **4.7 tok/s** on the XDNA 2 NPU. To reach **10–12 tok/s**, we need to reduce XRT kernel launches from **112 per token** (4 GEMMs × 28 layers) to **28 per token** (1 fused launch per layer).

FLM (FastFlowLM) achieves 94.7 tok/s by using a **fused `layer.xclbin`** that runs all 4 GEMMs per layer in a single NPU instruction stream. FLM's `libgemm.so` + `libqwen3_npu.so` generate these instruction sequences.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  npu_engine_cb (C++ engine, 4.7 tok/s)                     │
│                                                             │
│  For each layer:                                             │
│    CPU: quantize → sync A → launch QKV → wait → dequant     │
│    CPU: attention (score + V blend)                         │
│    CPU: quantize → sync A → launch O → wait → dequant       │
│    CPU: residual + LN                                       │
│    CPU: quantize → sync A → launch GU → wait → dequant      │
│    CPU: SiLU                                                │
│    CPU: quantize → sync A → launch D → wait → dequant       │
│    CPU: residual                                            │
│                                                             │
│  Problem: 112 XRT launches/token @ ~1.5ms each = 168ms NPU  │
│  Solution: 1 launch/layer = 28 launches = ~42ms NPU         │
└─────────────────────────────────────────────────────────────┘
```

## What's Built (5 Engines)

### 1. `engine/npu/src/npu_engine_cb.cpp` — Production (✅ Working)

Current production engine. 4.7 tok/s with all optimizations applied:

| Optimization | Before | After | Gain |
|---|---|---|---|
| AVX-512 LM head (float32 dot product) | 92 ms | **2.1 ms** | 43× |
| Remove redundant `layerB[l]->sync()` | ~50 ms | **0** | ∞ |
| Remove `memset(Am, 0, ...)` | ~2 ms | **0** | ∞ |
| Double-buffered async quantize/launch | — | Ready | — |
| Per-GEMM timing instrumentation | — | ✓ | — |

### 2. `engine/npu/src/npu_engine_fused.cpp` — FLM Instruction Gen (✅ Gen, ❌ Submit)

Generates NPU instruction sequences using FLM's `libgemm.so` + `libqwen3_npu.so`:

```
Pipeline: gen_dequant → send_x → move_weights → generate_seq → cmds2seq
```

Per-layer instruction counts (M=1 padded to 512):
- QKV: 5,592 words (128×1024→4096 GEMM)
- O:   8,872 words (128×2048→1024 GEMM)
- GU:  5,592 words (128×1024→6144 GEMM)  
- D:  12,152 words (128×3072→1024 GEMM)
- **Total: 32,208 words/layer**

Submit to `layer.xclbin` fails because instruction DDR offsets don't match our BO allocations.

### 3. `engine/npu/src/npu_engine_v4.cpp` — Full API (✅ Architecture)

Calls FLM's exported constructors directly via dlsym:

```cpp
npu_xclbin_manager mgr;            // from libqwen3_npu.so
mgr.register_xclbin("layer.xclbin");
qwen3_npu model(LM_Config{...}, &mgr, 0);  // loads weights at correct DDR offsets
auto instrs = model.gen_layer_seq(&seq, l); // fused per-layer instructions
kern(instrs, mgr.get_bo(), ...);            // submit
```

Blocked by: `npu_xclbin_manager` constructor lives in `/usr/bin/flm` (not exported from shared libs).

### 4. `engine/npu/src/npu_engine_direct.cpp` — app_manager (✅ Architecture)

Uses FLM's weak `npu_app_manager::npu_app_manager()` constructor directly:

```cpp
npu_app_manager mgr(npu_device, xrt::device*, xclbin_path, true);
// NPU initialized, BOs created
```

The `npu_app_manager` constructor IS a weak symbol and can be called. But passing it as `npu_xclbin_manager*` to `qwen3_npu::Impl::Impl()` segfaults — class layouts differ by an unknown offset.

### 5. `engine/npu/src/flm_bo_capture.cpp` — LD_PRELOAD (✅ Architecture)

Overrides weak FLM symbols to capture BOs at runtime:

```cpp
// Override npu_app_manager::npu_app_manager (weak → strong)
// Override npu_xclbin_manager::register_xclbin (weak → strong)  
// After FLM loads weights, BO handles are captured globally
```

Blocked by: overridden weak symbols can't call their originals (ELF symbol resolution replaces weak with strong globally).

## The Fix: One-Line Patch in FLM

The cleanest solution: add an export hook in FLM's source code at the point where `qwen3_npu::Impl::Impl()` finishes loading weights.

### Patch for `/usr/bin/flm` source (or equivalent):

```cpp
// In qwen3_npu::Impl::Impl(), after load_weights() succeeds:
// Expose the npu_xclbin_manager's BO handles as global symbols

// Add to FLM's model loading code (after line calling load_weights):
extern "C" {
    // Structure to pass BO info to external engines
    struct __attribute__((packed)) NpuEngineBOs {
        void* bo_input;      // input activation BO  
        void* bo_output;     // output activation BO
        void* bo_weights;    // weight BO
        void* bo_scratch;    // scratch BO
        void* bo_instr;      // instruction BO
        size_t bo_input_size;
        size_t bo_output_size;
        size_t bo_weights_size;
    };
    
    // Global pointer set after model construction
    NpuEngineBOs* g_npu_engine_bos __attribute__((weak)) = nullptr;
}

// In qwen3_npu::Impl::Impl(), after internal BOs are created:
g_npu_engine_bos = new NpuEngineBOs{
    .bo_input  = input_bo.impl_ptr,
    .bo_output = output_bo.impl_ptr,
    .bo_weights = weights_bo.impl_ptr,
    .bo_scratch = scratch_bo.impl_ptr,
    .bo_instr  = instr_bo.impl_ptr,
    .bo_input_size = input_size,
    .bo_output_size = output_size,
    .bo_weights_size = weights_size,
};
```

### Using the patch from `npu_engine_fused.cpp`:

```cpp
extern "C" NpuEngineBOs* g_npu_engine_bos;

void run_fused() {
    // Wait for FLM to load weights
    while (!g_npu_engine_bos) sleep_ms(100);
    
    // Wrap the BO handles as xrt::bo objects
    xrt::bo bo_input(dev, g_npu_engine_bos->bo_input);
    xrt::bo bo_weights(dev, g_npu_engine_bos->bo_weights);
    xrt::bo bo_output(dev, g_npu_engine_bos->bo_output);
    
    // Generate fused instruction sequence
    auto instrs = gen_fused_layer_seq(flm, l);
    
    // Submit once per layer
    kern(instrs, bo_input, bo_weights, bo_output, ...);
}
```

## DDR Offset Layout (Reverse-Engineered)

| Region | DDR Offset Range | Content |
|---|---|---|
| Control | 0x0000–0x200000 | Tile config, BD entries |
| Tile 0x2 weights | 0x2000000–0x2501000 | Q4NX weight copies |
| Tile 0x4 weights | 0x4000000–0x4501000 | |
| Tile 0x6 weights | 0x6201000–0x6501000 | |
| Tile 0x8 weights | 0x801d000–0x8501000 | |
| Tile 0xa weights | 0xa201000–0xa501000 | |
| Tile 0xc weights | 0xc01d000–0xc501000 | |
| Tile 0xe weights | 0xd000000–0xe501000 | |

Model file weight offsets: 0x18598000–0x18e34000 (Q4NX packed).
FLM reads from file at these offsets, dequantizes, and writes to DDR at the tile regions above.

## Key Findings

1. **FLM weights are I8, not Q4** — the model file stores INT8 weights (not Q4 packed). `data_offsets` × shape sizes match raw I8 perfectly.

2. **FLM instruction generators need M=512 padding** — even for M=1 decode, the generator internally pads to 512. This replicates weights across 8 tile columns.

3. **BO bank separation** — `layer.xclbin` requires instruction BO in DDR bank 1 (group_id=65537) and data BOs in bank 0 (group_id=65536). Attempting to share a single BO crashes.

4. **5 BO args** — `layer.xclbin` kernel takes 5 buffer arguments (bo0-bo4) + 1 instruction buffer. All 5 data args should point to the same physical BO (fused addressing).

## Performance Target

| Engine | Launch/token | Est. ms/tok | Est. tok/s | Status |
|---|---|---|---|---|
| v3 (current) | 112 | 212 | **4.7** | ✅ Production |
| v4 fused (with FLM patch) | 28 | ~80-100 | **10-12** | 🔧 Needs BO wiring |
| FLM proprietary | 1 | 10.6 | **94.7** | Proprietary |
