# MLX IRON XDNA 2 NPU Backend — Scope

**Goal**: Add an NPU device backend to MLX so `mlx.core.default_device(Device(DeviceType::npu))` dispatches array operations to the XDNA 2 NPU.

---

## What MLX Provides

MLX's `Device` enum currently has two types:
```cpp
enum class DeviceType { cpu, gpu };
```

Backends live under `mlx/backend/<name>/` and implement ~50 array operations:
```
mlx/backend/
├── cpu/      # Reference implementations (portable C++)
├── metal/    # Apple GPU (Metal shaders)
├── rocm/     # AMD GPU (ROCm/HIP)
├── cuda/     # NVIDIA GPU (CUDA kernels)
└── gpu/      # Shared GPU infrastructure
```

Each backend implements a set of operation classes (`matmul.cpp`, `binary.cpp`, `copy.cpp`, etc.). Operations not implemented in the backend fall through to the CPU via [`no_gpu`](https://github.com/ml-explore/mlx/tree/main/mlx/backend/no_gpu) stubs.

---

## What an NPU Backend Would Need

### Phase 1: Skeleton (2-3 days)

| Task | Files | What it involves |
|------|-------|-----------------|
| Add `DeviceType::npu` | `mlx/device.h`, `mlx/device.cpp` | New enum value, `default_device()` handling, `device_info()` |
| Create backend stub | `mlx/backend/npu/CMakeLists.txt` | Build system, link `xrt_coreutil` + `OpenMP` |
| Copy `no_gpu` stubs | `mlx/backend/npu/` | ~50 empty operation stubs that forward to CPU |
| NPU device init | `mlx/backend/npu/device.cpp` | `xrt::device(0)` open, xclbin preload |
| Test: `set_default_device(Device(DeviceType::npu))` passes | `tests/` | Device exists, doesn't crash |

**Deliverable**: MLX recognizes the NPU as a device, all operations fall through to CPU.

### Phase 2: GEMM (1 week)

The NPU only does one thing well: **INT8 GEMM**. Everything else stays on CPU.

| Task | What it involves |
|------|-----------------|
| Wire `matmul` to NPU | `mlx/backend/npu/matmul.cpp` — detect supported shapes, quantize f32→INT8, call XRT GEMM, dequantize result |
| Reuse `I8Ctx` from NPU engine | Import `platform.h`, `dequant_q4nx.c`, `model_config.h` — they're MIT-licensed, portable C/C++ |
| Shape adapter | MLX matmul is general (any M×K @ K×N). NPU xclbins are fixed-tile. Need a tiling loop: break large matmuls into xclbin-sized tiles |
| Quantization integration | MLX does f32 internally. Each matmul call needs: quantize f32→INT8 → NPU dispatch → dequant INT8→f32 |
| Fallback for unsupported shapes | If shape doesn't fit xclbin tile limits, forward to CPU backend |

**Deliverable**: MLX dispatches matmul to NPU. ~50-200 tok/s on supported models.

### Phase 3: Integration (3-5 days)

| Task | What it involves |
|------|-----------------|
| Model loading | Q4NX format parser → MLX array weights → Lemon MLX Engine can load NPU-compatible weights |
| KV cache on NPU | NPU GEMM for attention QKV projections (what the C++ engine already does) |
| Quantized matmul | MLX has `quantized_matmul` op — wire it to NPU natively (skip f32 roundtrip) |
| Memory management | NPU buffer allocation via `xrt::bo` → MLX allocator |
| OpenMP integration | NPU dispatch is blocking (`r.wait()`). Lemon MLX Engine's async pipeline needs careful threading |

**Deliverable**: Lemon MLX Engine runs models on NPU with coherent output (leverages MLX's existing architecture support for 50+ models).

### Phase 4: Polish (ongoing)

| Task | Why |
|------|-----|
| xclbin auto-selection | Different models need different xclbin shapes. Auto-detect from model config |
| Multi-batch | M=32 batch decode (same as C++ engine) |
| Mixed precision | f16 activations, INT8 weights, NPU GEMM |
| Performance tuning | Tile size tuning, DMA overlap, async dispatch |

---

## Architecture

```
MLX Python/C++ API
        │
        ▼
  MLX Dispatch (array.cpp)
        │
        ├── Backend::DeviceType::cpu  ──→ mlx/backend/cpu/  (always available)
        ├── Backend::DeviceType::gpu  ──→ mlx/backend/metal/ or rocm/
        └── Backend::DeviceType::npu  ──→ mlx/backend/npu/  (NEW)
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  npu/matmul  │──→ quantize f32→INT8
                                        │  npu/copy    │──→ xrt::bo::sync
                                        │  npu/binary  │──→ CPU fallback
                                        │  npu/...     │──→ CPU fallback
                                        └──────┬───────┘
                                               │
                                               ▼
                                        XRT (xrt_coreutil.dll / .so)
                                               │
                                               ▼
                                        XDNA 2 NPU
```

**Critical rule**: The NPU only does GEMM. Everything else (layernorm, softmax, RoPE, elementwise ops) falls through to CPU. This matches the pattern of the existing C++ engine.

---

## What Already Exists (Reusable)

| Component | Location | State |
|-----------|----------|-------|
| XRT C++ API | `engine/npu/src/platform.h` | Portable, _WIN32 ready |
| INT8 quantize/dequant | `engine/npu/src/dequant_q4nx.c` | Pure C, reusable |
| xclbin dispatch | `engine/npu/src/npu_engine_universal.cpp` (I8Ctx) | MIT licensed, drop-in |
| xclbin generation | `engine/npu/xclbins/n1_core_i8_v2.py` | Python/IRON, generates INT8 GEMM xclbins |
| Model config parser | `engine/npu/src/model_config.h` | Q4NX → dimensions |
| Lemon MLX Engine | `/home/bcloud/lemon-mlx-engine/` | 50+ model architectures, HF integration, HTTP server |
| MLX ROCm fork | `/home/bcloud/mlx-rocm-local/` | MLX + AMD GPU backend — reference for adding backends |

---

## What Needs to Be Written

| File | Lines | What it does |
|------|-------|-------------|
| `mlx/backend/npu/matmul.cpp` | ~300 | Quantize → dispatch → dequantize. Core loop. |
| `mlx/backend/npu/device.cpp` | ~100 | XRT init, device open, teardown |
| `mlx/backend/npu/allocator.cpp` | ~100 | `xrt::bo` wrapper as MLX allocator |
| `mlx/backend/npu/CMakeLists.txt` | ~30 | Build config, link XRT |
| `mlx/device.h` patch | +3 lines | `DeviceType::npu` |
| `mlx/device.cpp` patch | +10 lines | `default_device()`, `device_info()`, `is_available()` |
| `mlx/backend/npu/*.cpp` stubs | ~50×10 lines | Stubs forwarding to CPU for non-GEMM ops |

**New code total**: ~1,500-2,000 lines C++.

---

## Effort Estimate

| Phase | Time | Risk | Depends On |
|-------|------|------|-----------|
| 1. Skeleton | 2-3 days | 🟢 Low | MLX builds on Linux |
| 2. GEMM | 1 week | 🟡 Medium | XRT working on target |
| 3. Integration | 3-5 days | 🟡 Medium | Lemon MLX Engine + NPU |
| 4. Polish | ongoing | 🟢 Low | User feedback |

**Total to MVP**: ~2-3 weeks for a single person familiar with both MLX internals and the NPU engine.

---

## Why This Matters

Currently the NPU engine supports **5 model families** (hardcoded xclbin dimensions). An MLX backend inherits **50+ model architectures** from the Lemon MLX Engine — Llama, Qwen, Gemma, Phi, DeepSeek, Mistral, etc. — without writing any model-specific C++. You just need xclbins that match each model's hidden/intermediate sizes.

The NPU engine is fast but narrow. MLX would make it **fast and broad**.
