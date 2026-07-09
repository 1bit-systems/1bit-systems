# NPU Engine Architecture

**One person reverse-engineered AMD's proprietary NPU stack in 4 days.**  
This doc explains how the open-source replacement worked.

> **Retired (commit `cd232a091`)**: the standalone `engine/npu/` C++ engine
> described below no longer exists in this repo — superseded by the
> `spec-decode/` stack (FLM target + DSpark draft engine). This doc is kept as
> a technical reference for how the NPU dispatch model worked, not as a
> description of the current codebase. For current, accurate engine status see
> [docs/wiki/performance.md](wiki/performance.md) (source of truth): the fused
> and v12 engines discussed here were **raw throughput only, never coherent**
> — production NPU inference runs via the FLM proxy (94 tok/s, coherent).

---

## Overview

The NPU engine is a **C++23 INT8 inference engine** for LLMs on AMD XDNA 2 NPUs (Strix Halo). It loads Q4NX format model files, sends GEMM operations to NPU compute tiles via pre-compiled xclbin kernels, and handles everything else (attention, embeddings, LM head, RoPE) in software on the CPU.

```
┌─────────────────────────────────────────────────────┐
│                   engine/npu/                       │
│                                                     │
│   model.q4nx ──→ parse_q4nx_header() ──→ ModelConfig│
│                      │                              │
│                      ▼                              │
│   dequant_q4nx.c ──→ f32 weights ──→ packB() ──→ BO │
│                      │                              │
│                      ▼                              │
│   xclbin files ──→ xrt::kernel() ──→ go() ──→ GEMM  │
│                      │                              │
│                      ▼                              │
│   attn_omp()  ←  NPU output  ←  sync BO from device │
│   rmsnorm()                                        │
│   lm_head()                                        │
│   next_token()                                      │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow (Step by Step)

### Phase 1: Model Loading

**Q4NX format** is a simple container:
```
┌─────────┬──────────────┬──────────────────────────┐
│ 8 bytes │ JSON header  │ Raw weight data          │
│ hdr_sz  │ (hdr_sz)     │ (rest of file)           │
└─────────┴──────────────┴──────────────────────────┘
```

1. **`mmap`** the entire model file (`platform_open_read` + `platform_mmap`)
2. Read 8-byte header size → extract JSON header at offset 8
3. `parse_q4nx_header()` scans JSON to extract model dimensions:
   - `H` (hidden size) from `embed_tokens.weight` shape
   - `NC` (layer count) by scanning `model.layers.N.*` patterns
   - `NH`, `NKV`, `HD` from tile row counts in Q4NX weight metadata
   - `IM` (intermediate size) from `gate_proj.weight` tile rows
   - `NV` (vocabulary size) from `embed_tokens.weight` shape
   - Architecture flags: `has_q_norm`, `has_k_norm`, `has_rope_freqs_file`, `gu_split`

4. Parse per-layer weight offsets via `jo()` — JSON key scan for each weight:
   - `model.layers.N.self_attn.q_proj.weight` (Q projection)
   - `model.layers.N.self_attn.k_proj.weight` (K projection)
   - `model.layers.N.self_attn.v_proj.weight` (V projection)
   - `model.layers.N.self_attn.o_proj.weight` (O projection)
   - `model.layers.N.mlp.gate_proj.weight` (Gate)
   - `model.layers.N.mlp.up_proj.weight` (Up)
   - `model.layers.N.mlp.down_proj.weight` (Down)

### Phase 2: Weight Dequantization + Packing

Q4NX stores weights as **quantized INT8 tiles** with per-tile scale factors.

1. `dequant_i8_to_float_ex()` (in `dequant_q4nx.c`) = reads Q4NX tile format → outputs f32 row-major weights
   - Input: raw bytes + tile dimensions
   - Output: float array of size `out_features × in_features`
   - Dequantizes INT8 values using per-tile scale factors

2. `transpose_pack()` — transposes from PyTorch convention (out_features, in_features) to GEMM convention (in_features, out_features)

3. `I8Ctx::packB()` — quantizes f32 weights back to INT8 with per-tensor scale, copies to NPU BO, syncs to device

### Phase 3: NPU GEMM Dispatch

The NPU doesn't run a single "model" — it runs individual GEMM operations via **xclbin kernels**.

Each engine has **4 I8Ctx contexts** (one per xclbin):

| Context | xclbin | Operation | Shape |
|---------|--------|-----------|-------|
| `cq` | `final_i8_QKV_v.xclbin` | QKV projection | A[M, H] @ B[H, NH*HD + 2*NKV*HD] |
| `co` | `final_i8_O_v.xclbin` | Attention output | A[M, NH*HD] @ B[NH*HD, H] |
| `cg` | `final_i8_GU_v.xclbin` | Gate+Up (fused) | A[M, H] @ B[H, 2*IM] |
| `cd` | `final_i8_D_v.xclbin` | Down projection | A[M, IM] @ B[IM, H] |

For models with `gu_split=true` (IM > 7168), gate and up are separate xclbins.

#### The `I8Ctx::go()` call:

```
go(layer, activations, M, K, activation_scale, weight_scale, output, N):
  1. Quantize activations f32 → INT8 (per-tensor dynamic scale)
  2. Copy activations to BO A (xrt::bo::sync TO_DEVICE)
  3. Copy layer weights to BO B (already on device from packB())
  4. Dispatch: (*kernel)(3, instruction_BO, ins_count, BO_A, BO_B, BO_C)
  5. r.wait() — blocks until NPU completes
  6. Copy results from BO C (xrt::bo::sync FROM_DEVICE)
  7. Dequantize INT8 results → f32 using combined scale
```

### Phase 4: Per-Layer Compute (CPU)

After each GEMM, the CPU handles the non-GEMM operations:

```
for each layer:
    for each token in batch (M=32):
        RMSNorm(hidden)                          ← rn_c()
        apply RoPE(hidden)                       ← ra()
    
    QKV = cq.go(layer, hidden)                   ← NPU GEMM
    attn_output = attention(Q, K, V, past_KV)    ← attn_omp() on CPU
    hidden += attn_output                        ← residual
    RMSNorm(hidden)                              ← rn_c()
    gate = cg.go(layer, hidden)                  ← NPU GEMM
    hidden = cd.go(layer, gate * up)             ← NPU GEMM (SiLU gating)
    hidden += residual                           ← residual
```

**Attention** (`attn_omp()`) runs on CPU via OpenMP:
- Scaled dot-product: `softmax(Q @ K^T / sqrt(HD)) @ V`
- Parallelized over attention heads (`#pragma omp parallel for`)
- Each head vectorized with `#pragma omp simd`

**LM Head** (final projection to vocabulary):
- Embedding lookup: f32 from pre-converted buffer
- OpenMP-reduced dot product with weight matrix
- Returns token probabilities

---

## Key Files

### Engine Variants (Historical Record)

The `/src/` directory contains 20+ engine variants — this is the evolution from v2 through v12, preserved as a development diary. Only **two are actively maintained**:

| File | Lines | Purpose |
|------|-------|---------|
| **`npu_engine_universal.cpp`** | ~800 | **Production engine** — auto-detects all 5 model families, M=32 batch, OpenMP attn+LM. Built per-model via `-DMODEL_xxx` flag. |
| **`npu_engine_all.cpp`** | ~600 | All-models-in-one — detects model at runtime from Q4NX header (no per-model build needed). Same speed as universal. |

The rest (v2–v13, batch, cb, i8, mt, profile, server, spec) are historical snapshots — they compile and work but aren't maintained.

### Support Files

| File | What it does |
|------|--------------|
| **`dequant_q4nx.c`** | Q4NX weight dequantizer. Pure C, platform-independent. Reads tile-based INT8 format → f32. |
| **`model_config.h`** | `ModelConfig` struct + `parse_q4nx_header()`. Scans JSON to auto-detect model dimensions. |
| **`npu_dims.h`** | Hardcoded dimension constants (used by v2–v9 pre-auto-detect variants). |
| **`platform.h`** | **Portability layer** — unified API across Linux/Win32 for POSIX calls (mmap, open, fstat, memmem). |
| **`platform_mmap.cpp`** | CMake compilation unit for platform.h. |
| **`kv_quant.h`** / **`kv_quant.cpp`** | KV cache quantization utilities. |

### Build System

| File | Purpose |
|------|---------|
| **`build_npu.sh`** | Shell script. Compiles dequantizer + builds all 5 model variants via `-DMODEL_xxx`. |
| **`CMakeLists.txt`** | Cross-platform CMake (Linux g++ + Windows MSVC). Links `xrt_coreutil` + OpenMP. |

### Kernels & xclbins

| File | What it is |
|------|-----------|
| **`kernel/edge_attention.cc`** | Chess C++ — NPU attention kernel. Compiles to AIE tile code via `aiecc`. **Not used** — CPU OpenMP is faster for context < 128. |
| **`xclbins/n1_core_i8_v2.py`** | MLIR generator for INT8 GEMM xclbins. Generates `final_i8_QKV_v.xclbin` etc. via the IRON Python API. |
| **`build/`** | Pre-compiled xclbins + instruction files. Checked into repo. |

### External: Daemon

| File | What it does |
|------|-------------|
| **`daemon/npu-gpu-cpud.cpp`** | C++23 HTTP proxy daemon (port 9090). Routes to FLM for production. OpenAI-compatible API. Stripe checkout, order management, email notifications. Zero Python. Build with `g++ -std=c++23 -O3 daemon/npu-gpu-cpud.cpp -lpthread`. Stripe support: add `-lcurl`. |

---

## The Three Backends

The project runs **three inference backends** — only one is truly open-source:

### 1. Fused Layer Engine (Production) — 291 tok/s
```
     Client ←→ Daemon ←→ npu_engine_fused ←→ XRT ←→ NPU
                     (fused xclbin, one call/layer)
```
- All code was in `engine/npu/src/npu_engine_fused.cpp` (retired, see notice above)
- 291 tok/s on Qwen3-0.6B (3.4 ms/tok), 81 KB binary
- One xclbin call per transformer layer (QKV→attention→O→GU→SiLU→D on NPU)
- No CPU attention — entire layer runs on NPU
- **Raw throughput only — output was never coherent** (per `docs/wiki/performance.md`)

### 2. C++ v12 Engine (Fallback) — 28–97 tok/s
```
     Client ←→ Daemon ←→ npu_engine ←→ XRT ←→ NPU
                                        xclbins
```
- All code was in `engine/npu/` (retired, see notice above)
- 97 tok/s on Qwen3-0.6B (v12 single-model)
- 28 tok/s with all 5 models auto-detected
- Raw throughput only — the AIE micro-tiling/GEMM bug was fixed July 5, but
  decode output was still not coherent text (per `docs/wiki/performance.md`)

### 3. FLM Proxy (Fallback v2) — 94 tok/s
```
     ┌─────────┐     HTTP      ┌──────────┐
     │ Client  │ ←── port ──→  │ Daemon   │──→ flm (proprietary AMD binary)
     └─────────┘    9090       │ Python   │     └── port 52625
                               └──────────┘
```
- Proxies to AMD's proprietary `flm` binary (FastFlowLM)
- Used when fused and v12 are unavailable
- Engine code location: **not in this repo** (AMD proprietary)

### 4. GPU Engine (ZINC) — 22 tok/s
```
     Client ←→ ZnS server ←→ Vulkan compute ←→ iGPU
                              (GGUF format)
```
- Code was in `engine/gpu/` (retired, see notice above; GPU inference is now
  ZINC/Vulkan, outside this repo's engine tree)
- Written in Zig, Vulkan compute shaders
- Bonsai-1.7B-F16 at 22 tok/s, 99.6% bandwidth utilization
- Coherent output — this one works

---

## Adding a New Model Family

The engine auto-detects model dimensions from the Q4NX header. To add a new model:

### Step 1: Check if it auto-detects

Place the `.q4nx` file in the models directory and run:
```bash
./npu_engine_all model.q4nx 1
```

The engine will:
1. Parse the JSON header
2. Derive `H`, `NC`, `NH`, `NKV`, `HD`, `IM`, `NV` from tile metadata
3. Print detected dimensions
4. Attempt dispatch

If it works, you're done. The engine handles any model within xclbin dimension limits.

### Step 2: If auto-detect fails

The xclbins have fixed tile dimensions. Each xclbin's GEMM shape is determined at compile time (inside `n1_core_i8_v2.py`). If your model's hidden size or intermediate size doesn't match, you need new xclbins:

1. Edit `n1_core_i8_v2.py` — set `M`, `K`, `N` for your model's dimensions
2. Run through the MLIR-AIE toolchain → new `.xclbin` + `.txt` instruction files
3. Add a `MODEL_xxx` section in `model_config.h` for any architecture flags (q_norm, gu_split, rope_theta)

### Step 3: Build + verify

```bash
cd engine/npu
g++ -std=c++23 -O3 -I src -I /usr/include \
    -DMODEL_xxx -o build/npu_engine_xxx \
    src/npu_engine_universal.cpp build/dequant_q4nx.o \
    -lxrt_coreutil -lxrt_core -luuid -lm -ldl
./build/npu_engine_xxx model.q4nx 9
```

---

## NPU Dispatch Model

Understanding the NPU programming model is key to working on the engine:

### The xclbin

An xclbin is a **pre-compiled NPU kernel binary**. It contains:
- AIE tile machine code (for the 32 AIE2P tiles)
- Tile interconnect configuration (how data flows between tiles)
- DMA patterns (how data enters and leaves the tile array)

The xclbin is compiled via the AMD MLIR-AIE toolchain — once compiled, it's a fixed binary.

### The instruction buffer

Each xclbin comes with an **instruction text file** (`.txt`). This is a sequence of 32-bit words that configure:
- Which tiles participate in the GEMM
- Input/output buffer addresses per tile
- Loop counts (for tiled GEMM across large matrices)

The instruction buffer is loaded into a `xrt::bo` and passed to every kernel invocation.

### The I8Ctx lifecycle

```
init(device, xclbin_path, instruction_path, group_id):
  1. fopen/fread instruction file → ins[] vector
  2. xrt::xclbin(xclbin_path) → load kernel binary
  3. device.register_xclbin(xclbin) → make it available
  4. xrt::hw_context(device, uuid) → create execution context
  5. xrt::kernel(ctxt, "MLIR_AIE") → get kernel handle
  6. xrt::bo(device, size, flags, group_id(1)) → instruction BO
  7. memcpy + sync → upload instructions
  8. xrt::bo(device, M*K, flags, group_id(3)) → activation BO
  9. xrt::bo(device, M*N*2, flags, group_id(5)) → output BO
  10. For each layer:
      xrt::bo(device, K*N, flags, group_id(gid_B)) → weight BO

go(layer, activations, M, K, a_scale, b_scale, output, N):
  1. Quantize activations, copy to BO A, sync to device
  2. Sync weight BO to device
  3. kernel(3, ins_BO, ins_count, BO_A, BO_B, BO_C)
  4. r.wait() — blocks until NPU finishes
  5. Sync BO C from device
  6. Dequantize: output = int16_result * a_scale * b_scale
```

### Why M=32 matters

The NPU has high dispatch overhead — ~1,334 μs per `r.wait()`. By batching M=32 tokens together, this overhead is amortized across all 32 tokens: 1,334 μs / 32 = 42 μs/token. Without batching (M=1), the overhead adds 1.3 ms to every token, making the engine ~10× slower.

---

## Platform Portability

The `platform.h` header provides a unified API across platforms:

| Linux API | Windows Equivalent | Source Location |
|-----------|-------------------|-----------------|
| `open(path, O_RDONLY)` | `CreateFileW(path, GENERIC_READ, ...)` | `platform.h` — `platform_open_read()` |
| `close(fd)` | `CloseHandle(fd)` | `platform.h` — `platform_close()` |
| `fstat(fd, &st)` | `GetFileSizeEx(fd, &size)` | `platform.h` — `platform_fstat()` |
| `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)` | `CreateFileMapping()` + `MapViewOfFileEx()` | `platform.h` — `platform_mmap()` |
| `munmap(addr, len)` | `UnmapViewOfFile(addr)` | `platform.h` — `platform_munmap()` |
| `memmem(h, hl, n, nl)` | Custom scan | `platform.h` — `platform_memmem()` |
| `xrt::device` | `xrt_coreutil.dll` (same API) | `platform.h` includes XRT headers |
| `#pragma omp` | MSVC `/openmp` | Compiler flag in CMakeLists.txt |

To add a new platform (e.g., macOS):
```cpp
#elif defined(__APPLE__)
// macOS has POSIX APIs — mmap, memmem, etc. all available
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
using PlatformFileHandle = int;
// ... passthrough same as Linux
```

---

## Engine Evolution

| Version | Date | Speed | Breakthrough |
|---------|------|-------|-------------|
| v2 | Jun 28 | ~2000 ms/tok | First working BFP16 decode |
| v3 CB | Jul 1 | 244 ms/tok | K-interleaving bug fix |
| v6 | Jul 2 | 50 ms/tok | Batch-4 + OpenMP LM head |
| v7 probe | Jul 2 | — | µs profiling: ioctl=9µs, r.wait=1334µs |
| v8 | Jul 2 | 27 ms/tok | M=8 batch decode |
| v9 | Jul 2 | 16 ms/tok | M=16 batch decode |
| v12 | Jul 2 | 10 ms/tok | M=32 + OpenMP attention |
| ALL | Jul 2 | 36/62/93/100/127 ms/tok | 5 models, auto-detect |
| **Universal** | Jul 4 | same | Model-agnostic via Q4NX header parse |

Each version is preserved in `/src/` as individual `.cpp` files — the evolution is fully auditable.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Fused layer (production) | **291 tok/s** (3.4 ms/tok) |
| C++ v12 (fallback) | **97 tok/s** (10 ms/tok) ✅ coherent |
| FLM proxy (fallback v2) | **94 tok/s** (10.6 ms/tok) |
| C++ ALL (5 models) | **28 tok/s** (36 ms/tok) |
| NPU raw GEMM | **55.7 TFLOPS** (INT8, XDNA 2) |
| Dispatch overhead | 1,334 µs per kernel call |
| Binary size | **81 KB** |
| Models supported | 5 (Qwen3-0.6B/8B, Qwen3-VL-4B, Llama-3.1-8B, Gemma4-E2B) |
| Batch size | M=32 (M=1–128 configurable) |

---

## Quick Reference: Where to Find Things

| I want to... | Go to... |
|-------------|----------|
| Work on the current NPU/spec-decode stack | `spec-decode/` (retired `engine/npu/` files above are historical reference only) |
| See current, honest benchmark status | `docs/wiki/performance.md` (source of truth) |
| Benchmark the engine | `docs/wiki/performance.md` |
| See the journey | `docs/journey.md` |
| Check build prerequisites | `docs/building.md` |

---

*Engine built by one person in 4 days. Open source. MIT licensed. Zero Python.*
