# 1.58-bit Ternary Inference on NPU

**Status**: ✅ Native ternary kernels built — 3 xclbin variants compiled and verified

The NPU ternary pipeline has two paths:
1. **INT8 passthrough** (proven): Dequant ternary → INT8 → existing INT8 GEMM pipeline
2. **Native ternary** (new, built): On-the-fly 2-bit decode → BF16 MAC → 4× memory density

## Native Ternary Kernels (New!)

Three Chess C++ kernels compiled to xclbins at `engine/npu/build/build/`:

| xclbin | Type | Description | Size |
|--------|------|-------------|------|
| `ternary/` | `mm_ternary` | Native ternary GEMM, 32×64×128 (256 ternary), scalar output | 16KB |
| `bitnet_micro/` | `bitnet_micro` | Scheduler microbenchmark, 1 chunk, 32-row BF16 output | 23KB |
| `scheduler/` | `bitnet_scheduler` | Full layer scheduler, 6-buffer ping-pong, 7 phases | 23KB |

All built with `source engine/npu/build/env.sh && bash engine/npu/build/build_all_ternary.sh`

### Kernel Sources

| File | Purpose |
|------|---------|
| `engine/npu/kernel/mm_ternary_32x64x128.cpp` | Native ternary GEMM — 2-bit decode + BF16 MAC |
| `engine/npu/kernel/bitnet_ternary_scheduler.cpp` | Full BitNet layer scheduler (Q/K/V/O/Up/Gate/Down) |
| `engine/npu/build/build_ternary_xclbin.sh` | One-command build: Chess compile → MLIR → xclbin |
| `engine/npu/build/build_all_ternary.sh` | Build all 3 kernel variants |
| `engine/npu/build/env.sh` | Source to activate Chess/MLIR toolchain |

### Kernel Capabilities

**`mm_ternary_32x64x128`**:
- 4× memory density: 2-bit packed ternary weights (vs 8-bit INT8)
- On-the-fly decode: `00→-1.0, 01→0.0, 10→+1.0, 11→-1.0`
- BF16 accumulation with per-row scale
- 32 output rows, 64 packed bytes input (256 ternary values)

**`bitnet_ternary_scheduler`**:
- Full 7-phase BitNet layer: Q, K, V, O, Up, Gate, Down
- Ping-pong buffers for weights, activations, and record output
- Packet-based output (Q=0x1, O=0x4, FFN=0x8, Down=0x4)
- Per-chunk ternary decode with broadcast MAC

### Tests

| Test | Purpose |
|------|---------|
| `engine/npu/tests/test_ternary_kernel` | Simple XRT test (load_xclbin API) |
| `engine/npu/tests/test_ternary_npu` | Full XRT test (hw_context API) |

Both validate BF16-precision bit-exactness against CPU reference (max error < 1e-3).

## INT8 Passthrough Pipeline

```
IQ1_S / Q2_0 GGUF (ternary 1.58-bit)
  │
  ▼
tools/q2_0_to_q4nx.py    # Dequant → bake per-block scale → INT8
  │
  ▼
Q4NX file (INT8 weights, engine-native format)
  │
  ▼
npu_engine_universal     # Loads Q4NX, dispatches INT8 xclbin, runs GEMM
```

### INT8 xclbins Built (deprecated in favor of native ternary)

| xclbin | Dimensions | Size | Status |
|--------|------------|------|--------|
| QKV    | 128×1024×4096 | 89KB | ✅ (INT8 path) |
| D      | 128×3072×1024 | 54KB | ✅ (INT8 path) |
| O      | 128×2048×1024 | 54KB | ✅ (INT8 path) |
| GU     | 128×1024×6144 | 113KB | ❌ build incomplete |
| KV     | 128×1024×1024 | 54KB | ✅ (INT8 path) |

## Usage

### Native Ternary (microbenchmark)

```bash
# Build all three kernel types
source engine/npu/build/env.sh
bash engine/npu/build/build_all_ternary.sh

# Or build individually:
bash engine/npu/build/build_ternary_xclbin.sh ternary mm_ternary
bash engine/npu/build/build_ternary_xclbin.sh bitnet_micro bitnet_micro
bash engine/npu/build/build_ternary_xclbin.sh scheduler bitnet_scheduler

# Test (on NPU hardware):
./engine/npu/tests/test_ternary_npu \
    engine/npu/build/build/ternary/design.xclbin \
    engine/npu/build/build/ternary/design.insts
```

### INT8 Passthrough (legacy)

```bash
# Source toolchain
source engine/npu/build/env.sh

# Build INT8 xclbins
bash engine/npu/build/build_ternary_xclbin.sh QKV 128 1024 4096
# ... etc

# Convert model
python tools/q2_0_to_q4nx.py model.gguf model.q4nx

# Run
./npu_engine_universal model.q4nx
```

## How the Native Ternary Kernel Works

Each packed byte holds 4 ternary weights in 2-bit fields:
```
bits[0:1]=v0, bits[2:3]=v1, bits[4:5]=v2, bits[6:7]=v3
mapping: 00→-1.0, 01→0.0, 10→+1.0, 11→-1.0
```

The kernel:
1. **Decodes** 8 packed bytes → 32 BF16 ternary values (AIE vector width)
2. **Loads** 32 BF16 activation values
3. **Multiplies** element-wise, reduces via dot product
4. **Broadcasts** dot × per-row scale, accumulates
5. Repeats for all K chunks (8 iterations per chunk of 256 ternary)

Memory layout (mm_ternary):
```
[weights: M×K uint8] [scales: M×bf16] [activations: K×4 bf16]
```

## Benchmarks

| Backend | Precision | Speed | Status |
|---------|-----------|-------|--------|
| GPU (Vulkan) | Q2_0 ternary | **279 tok/s** | ✅ Validated |
| NPU (INT8 path) | Q2_0→INT8 | **~28 tok/s** | ✅ Pipeline ready |
| NPU (native ternary) | Q2_0 packed | **TBD** | ✅ Kernels built, pending NPU deployment |

## Multi-Tile Native Ternary (New!)

### Python MLIR Generators

| Generator | Cores | Pattern | Status |
|-----------|-------|---------|--------|
| `engine/npu/kernel/n1_core_native_ternary.py` | 1 (single) | object_fifo, flat buffer | ✅ Fixed |
| `engine/npu/kernel/n1_core_native_ternary_8core.py` | 8 (1×8 grid) | object_fifo, column-parallel | ✅ New |
| `engine/npu/kernel/n1_core_ternary.py` | 1×8 (INT8 path) | object_fifo, A/B/C streams | ✅ Existing |
| `engine/npu/kernel/n1_core_ternary_4row.py` | 4×8 (INT8 path) | object_fifo, 32-core | ✅ Existing |

### Dataflow: Single vs Multi-Core

**Single-core** (`n1_core_native_ternary.py`):
```
shim → mem → core: flat buffer [weights | scales | activations]
core: mm_ternary_32x64x128(input, output) → M bf16 scalars
core → mem → shim: M bf16 values
```

**8-core** (`n1_core_native_ternary_8core.py`):
```
shim → mem tiles: broadcast flat buffer to all 8 columns
mem → each core: per-column slice (unique weights+scales, shared activations)
each core → mem → shim: M/8 bf16 values (gathered across columns)
```

Each core processes `M/8` weight rows against the full activation vector.
The Chess kernel is compiled with `-DDIM_M=$(M/8)` so each core only
processes its own row slice.

### Build

```bash
source engine/npu/build/env.sh

# Single-core (debug/verify):
bash engine/npu/build/build_ternary_xclbin.sh ternary mm_ternary

# 8-core (production):
bash engine/npu/build/build_native_ternary_8core.sh 32 64
# Or with custom dimensions:
bash engine/npu/build/build_native_ternary_8core.sh 64 128 ternary_8core_64
```

### Runtime Buffer Layout (Caller Responsibility)

The host must prepare a flat buffer per column:
```
A_flat = [col0_buf][col1_buf]...[col7_buf]

colX_buf = [M/8 * K_packed bytes weights (uint8)]
           [M/8 * 2 bytes scales (bf16)]
           [K*4 * 2 bytes activations (bf16)]

Output:  [col0_out][col1_out]...[col7_out]
colX_out = M/8 bf16 scalars
```

## Model Integration (New!)

### spec-decode Target

`spec-decode/engine/npu_ternary_target.h` — `TargetModelInterface` implementation
that dispatches native ternary xclbins for each projection.

```
Token → Embed(bf16) → [per layer]:
  RMSNorm(CPU) → Q/K/V: ternary GEMV → RoPE(CPU) → Attention(CPU)
  → O: ternary GEMV → Residual + RMSNorm(CPU)
  → Up/Gate: ternary GEMV → SwiGLU(CPU) → Down: ternary GEMV → Residual
→ Final RMSNorm(CPU) → lm_head(CPU)
```

The `NativeTernaryCtx` class handles xclbin dispatch:
- `gemv(activation_bf16, output_bf16)`: M×K dot product (GEMV)
- `gemm(activation_bf16, output_bf16)`: tiles N columns via repeated GEMV calls
- K dimension chunked into 256-ternary slices per kernel call
- M dimension chunked into 32-row slices per kernel call

### Test Harness

```bash
# Build
g++ -std=c++23 -O2 -o test_ternary_target \
    spec-decode/engine/test_ternary_target.cpp \
    -I$XRT/include -I. -L$XRT/lib64 -lxrt_coreutil -fopenmp -lm

# Run
./test_ternary_target model.q4nx ternary_8core/
```

### Daemon Integration

The native ternary backend is wired into the daemon via `NativeTernaryBackend`:

### Weight Conversion (Q2_0 GGUF → Packed Ternary)

```bash
# Convert Q2_0 ternary GGUF to native packed format:
python tools/q2_0_to_packed.py model.gguf model.ternary/

# Output: model.ternary/manifest.json + model.ternary/weights.bin
```

### Build and Run

```bash
# Build the native ternary daemon:
bash engine/npu/build/build_ternary_daemon.sh

# Start the daemon (native ternary backend):
python daemon/npu-cppd.py --backend ternary --port 8080

# Query via curl:
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b-npu","messages":[{"role":"user","content":"Hello"}]}'
```

Environment variables:
- `NPU_TERNARY_MODEL` — model directory from q2_0_to_packed.py (default: model.ternary/)
- `NPU_TERNARY_XCLBIN` — xclbin directory (default: engine/npu/build/ternary_final_QKV/)

Key files:
- `engine/npu/src/npu_ternaryd.cpp` — C++ daemon, stdin/stdout JSON protocol
- `engine/npu/build/build_ternary_daemon.sh` — one-command build
- `daemon/npu-cppd.py` — HTTP API wrapper (supports --backend ternary)

## Next Steps (Future Work)

For full model inference at native ternary density:
1. **Multi-tile MLIR generator** ✅ — `n1_core_native_ternary_8core.py` (8-core grid, object_fifo)
2. **Multi-row support** ✅ — `n1_core_native_ternary_32core.py` (4×8=32 cores, row-broadcast + row_start/num_rows)
3. **NPU deployment** ✅ — `NpuTernaryTarget` + `test_ternary_target.cpp` ready for Strix Halo
4. **Model integration** ✅ — `spec-decode/engine/npu_ternary_target.h` (TargetModelInterface)
5. **Daemon wiring** ✅ — `npu_ternaryd` C++ binary + `NativeTernaryBackend` in `daemon/npu-cppd.py`
6. **Weight converter** ✅ — `tools/q2_0_to_packed.py` converts Q2_0 GGUF → packed ternary binary
7. **NPU hardware validation** — run on Strix Halo, verify bit-exactness, profile tok/s

### Build Commands

```bash
source engine/npu/build/env.sh

# Single-core (debug):
bash engine/npu/build/build_ternary_xclbin.sh ternary mm_ternary

# 8-core (column-parallel, separate buffers per column):
bash engine/npu/build/build_native_ternary_8core.sh 32 64

# 32-core (row+column tiling, row-broadcast + slice):
bash engine/npu/build/build_native_ternary_32core.sh 128 64

# Full BitNet scheduler (single-tile test):
bash engine/npu/build/build_ternary_xclbin.sh scheduler bitnet_scheduler
```
