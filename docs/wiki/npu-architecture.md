# NPU Engine Architecture — Knowledge Base

> Auto-generated from reverse engineering sessions. Last updated: 2026-07-12.

## Engine Stack

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Client (OpenAI-compatible API)                     │
│   curl :9090/v1/chat/completions                            │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Daemon (daemon/npu-gpu-cpud, 115KB C++)            │
│   Proxies to FLM, adds x-device metadata, Stripe support    │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Fused Engine (engine/fusion/, 13MB Zig)            │
│   8 dispatch policies, HTTP server, FLM proxy, unit tests   │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: NPU Backend                                        │
│   ├── FLM (79 tok/s, production, coherent)                  │
│   ├── Universal Engine (46→~55 tok/s, custom, pipelined)    │
│   ├── Fused Engine (WIP, 1 launch/layer vs 4)              │
│   └── GPU Zinc (ternary, needs GGUF model)                  │
└─────────────────────────────────────────────────────────────┘
```

## XCLBIN Architecture

### Simple GEMM (4 xclbins per layer — universal engine)
```
Kernel: MLIR_AIE (arg_index 1=SRAM, 3-7=HOST)
Args:   run(3), instr_bo, instr_count, A(act), B(weight), C(out)
Sizes:  12KB-113KB per xclbin
Format: INT8 activations, INT8 weights (column-mature BO layout)
Tiles:  M=128, K=variable, N=variable, mt=128, kt=64, nt=128
Build:  torch2aie/examples/gemm_asymmetric_tile_buffering
Status: ✅ Working (0% error verified per-kernel with 16MB BOs)
```

### Fused Layer (1 xclbin per layer — target)
```
Kernel: MLIR_AIE (same arg layout)
Args:   run(3), instr_bo, count, KCache, VCache, Weights, Output, Hidden
Size:   416KB per xclbin
Format: BF16 hidden state, BF16 pre-packed weights (65MB/layer)
Instructions: 1723 words, token-transition format (token127→tokenN)
Build:  torch2aie/examples/qwen3-decode-layer (design.py + run_full_layer.py)
RTP:    Run-time parameter patching for token position
Status: ⚠️ Runs at 37ms/layer — edge kernel backpressure bottleneck
```

### FLM (production — 4 xclbins per model)
```
Kernels: attn.xclbin, dequant.xclbin, layer.xclbin, mm.xclbin
Sizes:  317KB, 114KB, 450KB, 507KB
Format: Internal FLM format, C++ API via libqwen3_npu.so
Path:   /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/
Status: ✅ Production (79 tok/s coherent)
```

## Universal Engine Optimization Log

### Baseline (v12, 2026-07-10)
- 4 fully-serial xclbin launches per layer
- Each `go()` call: quantize → sync A → launch → wait → sync C → dequantize
- **46 tok/s** (BS=128), **34 tok/s** (BS=32)

### Commit 1: Async go() refactor (6853fe35)
Split `go()` into three sub-operations:
```
quantize_async()   — CPU quantize (no DMA)
sync_and_launch()  — sync A + submit kernel
dequantize()       — wait + sync C + dequantize
```
Enables overlapping quantize for kernel N+1 with NPU execution of kernel N.

### Commit 2: Parallel O+GU launch (ebdb5c6e)
Co (O projection) and Cg (gate/up GEMM) are data-independent:
- Co needs at_b from CPU attention
- Cg needs h_b (original residual, available from layer start)

Previously serial: `co → dequant → residual → rn_c → cg`
Now parallel: `co → wait → launch cg → [dequant co + residual + rn_c] overlaps cg NPU`

### Commit 3: Split sync_A from launch (98db4569)
Three-phase parallel launch:
```
Phase 1: cg.sync_A()      — DMA sync (MM2S) runs WHILE co on NPU
Phase 2: co.wait_kernel()  — minimal NPU completion
Phase 3: cg.launch() + co.sync_back() — submit cg + read co (S2MM) SIMULTANEOUSLY
Phase 4: CPU residual+rn_c — overlaps with cg NPU
Phase 5: cg readback
```

### Per-Layer Pipeline (current)
```
Cq GEMM ──► CPU attn ──► Co GEMM ──► Cg GEMM ──► Cd GEMM
                            │            │
                      sync_A(cg) ───┐   │
                      wait(co) ─┐   │   │
                      launch(cg)├───┘   │
                      readback(co) ─┐   │
                      residual+rn_c │   │
                      readback(cg) ─┘   │
                      readback(cd) ─────┘
```

### Next: Cross-Layer Pipeline
Layer N's Cd output feeds Layer N+1's Cq input. Quantize for N+1's Cq can overlap with N's Cd NPU execution:
```
Layer N:  ... Cd GEMM ──► Cd dequant ──► Layer N+1: Cq quantize
                              │               │
                        sync_A(cq_next) ──┐   │
                        wait(cd) ─┐       │   │
                        launch(cq_next) ──┘   │
                        readback(cd) ─────┐   │
                        residual+rn_c     │   │
                        readback(cq) ─────┘   │
```

## Fused XCLBIN Edge Kernel Analysis

The fused xclbin at `torch2aie/examples/qwen3-decode-layer/` runs but is **37ms/layer**.
The bottleneck is NOT the dataflow (10.8ms with light stubs) — it's the edge kernels.

### Backpressure Breakdown
| Configuration | Time | vs. Isolated | Cumulative |
|---|---|---|---|
| Isolated weight path | 8,031 µs | 1.0× | baseline |
| Light-edge (all stubs) | 10,830 µs | 1.35× | +2.8ms dataflow overhead |
| Light-attention-only | 28,342 µs | 3.53× | **+17.5ms attention** 🏆 |
| Full production | 37,000 µs | 4.61× | +8.7ms other edge kernels |

Attention is **67%** of the edge overhead. Full-vector station + SwiGLU + postprocess add the rest.

### Dataflow
```
DDR ──► Shim ──► MemTile ──► Main16 (Q4NX GEMM)
                               │
                        compact records
                               │
                          c1r1 hub
                           ╱    ╲
                          ╱      ╲
                   postprocess   full-vector
                   QKV (c1r3)    station (c1r2)
                       │              │
                  edge attention   SwiGLU
                       │          (c6r2)
                    KV cache      │
                    writeback  down proj
                                   │
                              output
```

### Edge Kernel Status
- ✅ postprocess QKV: passes oracle (294 lines)
- ✅ full-vector station: passes oracle (262 lines)
- ✅ edge attention: passes oracle (343 lines) — **biggest perf hit**
- ✅ SwiGLU: passes oracle (73 lines)
- ✅ Main16 Q4NX: passes oracle (504 µs isolated)

## Performance Target

| Engine | tok/s | Coherent | Launches/layer | Notes |
|--------|-------|----------|----------------|-------|
| FLM | 79 | ✅ | 1 (fused) | Production target |
| Universal (BS=128) | 46 | ✅ (~16 tok) | 4 | Current, improving |
| Universal (BS=32) | 34 | ✅ (~16 tok) | 4 | Current |
| Fused layer (torch2aie) | ~1 | ✅ | 1 | Edge kernel bottleneck |
| GPU ternary | 279 | ❓ | N/A | No model |

### 6-Step Parallelism Roadmap
| Step | Optimization | Est. tok/s | Status |
|------|------------|------------|--------|
| 1 | Pipelined DMA — async quantize/dequantize overlap | 46→50 | ✅ Done (C++) |
| 2 | O+GU parallel launch — hide readback behind NPU | 50→55 | ✅ Done |
| 3 | Cross-layer pipeline — overlap cd dequant + cq quantize | 55→58 | 🔧 Next |
| 4 | Software pipeline — II=1 inner loop, 4 MACs/cycle | 58→65 | ❌ (kernel change) |
| 5 | Full 32-tile grid | 65→70 | ❌ (xclbin rebuild) |
| 6 | INT8 via Triton-XDNA (2.5× MAC density) | 70→85+ | ❌ (toolchain fix) |

## Model Details — Qwen3-0.6B

| Parameter | Value |
|-----------|-------|
| H (hidden) | 1024 |
| NC (layers) | 28 |
| NH (heads) | 16 |
| NKV (KV heads) | 8 |
| HD (head dim) | 128 |
| IM (intermediate) | 3072 |
| NV (vocab) | 151936 |
| GQA | 2 |
| tie_word_embeddings | true |
| rope_theta | 1000000 |

## XCLBIN Dimensions (Qwen3-0.6B)

| Kernel | M | K | N | kt | nt |
|--------|---|---|---|----|-----|
| QKV | 128 | 1024 | 4096 | 64 | 128 |
| O | 128 | 2048 | 1024 | 64 | 128 |
| GU | 128 | 1024 | 6144 | 64 | 128 |
| D | 128 | 3072 | 1024 | 64 | 128 |

## BO Size Requirements

All xclbins require BOs padded to safe sizes for DMA offset safety:

| BO | Minimum Safe Size | Reason |
|----|-------------------|--------|
| Activation (bA) | 16 MB | DMA accesses beyond exact data |
| Weight (layerB) | max(KD×ND, 16MB) | Per-layer weight data + padding |
| Output (bC) | 16 MB | Kernel writes full tile region |
| Instruction (bI) | 64 KB | SRAM tile instruction buffer |

**Critical**: Build script uses `M=128` with tile `mt=128` — kernels process 128 rows even for M=1 decode. BO must accommodate full 128-row memory region.

## Bugs Fixed (19 commits)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | BO sizes too small | Kernel DMA overflow, crashes | 16MB padding |
| 2 | Norm weights clamped to [-2,2] | Qwen3 weights up to 8.69 → under-normalization | Remove clamp |
| 3 | Fixed activation scale 8.0/127 | Hidden state explosion | Dynamic per-GEMM amax |
| 4 | Separate lm_head for tied embeddings | Quantization mismatch | Use emb_f32 |
| 5 | Missing causal attention max_pos | Batch tokens attend to future | sp+b+1 in attn_omp |
| 6 | Hardcoded top[32] in lm_topk_omp | Stack overflow at BS>32 | vector<K> |

## MLIR-AIE Toolchain

```
Toolchain: ~/torch2aie/toolchain/
  bin/     aiecc.py, aie-opt, xchesscc
  mlir_aie/ Python bindings
  xrt/     XRT headers + libs
  aietools/ AIE compiler tools

Python:   ~/torch2aie/.venv/bin/python (3.12)
Venv:     ~/mlir-aie/.venv/ (3.14, for IRON dev)

Build:    cd ~/torch2aie/examples/qwen3-decode-layer
          make full-build  # fuse full layer into 1 xclbin
```

### MLIR-AIR Paper (arxiv 2510.14871)
Key findings from AMD's open-source spatial compiler stack:
- `air.herd` — spatial partitioning across tiles
- `air.channel` — point-to-point DMA communication
- `air.token` — explicit synchronization
- Matrix multiplication: **78.7% compute efficiency** (≈ hand-optimized MLIR-AIE)
- LLaMA 2 MHA: **2.24× speedup** from kernel fusion (834µs → 373µs)
- **~150 lines** for fused MHA implementation
- Pipeline: Triton → Linalg → MLIR Transform → MLIR-AIR/AIE → XCLBIN

### Known Toolchain Issue
After PC crash (2026-07-12), rebuild broke symlinks in `install/python/`.
Fix: `find build/python -type l -xtype l -delete` then `ninja install`.
Version mismatch between locally-generated `_aie_ops_gen.py` and pip-installed
MLIR C++ bindings causes nanobind `OpResult` rejection in `ObjectFifoCreateOp`.

## FLM Shared Libraries

```
/opt/fastflowlm/lib/
  libqwen3_npu.so    — Qwen3 inference pipeline
  libgemm.so         — Gemm::generate_seq (instruction generator)
  libmha.so          — Multi-head attention
  libq4_npu_eXpress.so — Q4 dequantization
```

## Build Commands

```bash
# NPU engines
bash engine/npu/build_npu.sh              # All 5 variants
                                          # (universal, qwen3_0_6b, qwen3_8b, etc.)

# Fused xclbin (torch2aie)
make -C ~/torch2aie/examples/qwen3-decode-layer full-build
make -C ~/torch2aie/examples/qwen3-decode-layer full-run

# MLIR-AIE (IRON Python experiments)
cd ~/mlir-aie/build && ninja install
source ~/mlir-aie/.venv/bin/activate
export PYTHONPATH=~/mlir-aie/install/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/torch2aie/toolchain/xrt/lib64:$LD_LIBRARY_PATH

# Universal engine test
cd ~/projects/1bit-systems
./engine/npu/build/npu_engine_universal \
    --model ~/weights/qwen3_0.6b.npu \
    --tokens 128

# Driver reload (after crash)
sudo modprobe -r amdxdna && sudo modprobe amdxdna
```

## Experiment Design Space (Experiments 1-11)

| Exp | Focus | Status |
|-----|-------|--------|
| 1-6 | Passthrough, vec-add, BFP16 accuracy | ✅ Historical |
| 7 | Single-tile GEMM (Peano) | ✅ Verified |
| 8 | Multi-tile BF16 GEMM (Worker.grid) | ✅ 82 GFLOPS |
| 9 | Hand-written BFP16 kernel via ExternalFunction | ✅ |
| 10 | Pre-packed BFP16 + column-major B | ✅ |
| **11** | **Pipelined DMA** — separate fill/drain task groups | ✳️ Written, blocked by IRON version mismatch |
| Strix experiments | 31 TFLOPS Chess, INT8 investigation | ✅ Documented |

## Remaining Work

1. **Cross-layer pipeline** — overlap cd dequant + cq quantize between layers
2. **Fused xclbin edge kernel tuning** — attention bottleneck (17.5ms)
3. **Prefill instruction format** — fused xclbin hangs at token0
4. **Per-channel quantization** — eliminate ~2%/layer hidden state growth
5. **GPU ternary** — download/convert ternarized GGUF model for zinc
6. **Zig NPU engine** — fix XRT C API symbol names
7. **INT8 via Triton-XDNA** — unblock MLIR parser `i8` type rejection
