# NPU Engine Architecture — Knowledge Base

> Auto-generated from reverse engineering session. Last updated: 2026-07-07.

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
│   ├── Universal Engine (46 tok/s, custom, coherent-ish)     │
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
Format: INT8 activations, INT8 weights (column-major BO layout)
Tiles:  M=128, K=variable, N=variable, mt=128, kt=64, nt=128
Build:  torch2aie/examples/gemm_asymmetric_tile_buffering
Status: ✅ Working (0% error verified per-kernel with 16MB BOs)
```

### Fused Layer (1 xclbin per layer — 4× faster target)
```
Kernel: MLIR_AIE (same arg layout)
Args:   run(3), instr_bo, count, KCache, VCache, Weights, Output, Hidden
Size:   416KB per xclbin (split) · 436 KB npu_spec_decode unified binary · 9.7 MB token router (Rust)
Format: BF16 hidden state, BF16 pre-packed weights (65MB/layer)
Instructions: 1723 words, token-transition format (token127→tokenN)
Build:  torch2aie/examples/qwen3-decode-layer (design.py + run_full_layer.py)
RTP:    Run-time parameter patching for token position
Status: ⚠️  Kernel loads but hangs — needs prefill instruction format
```

### FLM (production — 4 xclbins per model)
```
Kernels: attn.xclbin, dequant.xclbin, layer.xclbin, mm.xclbin
Sizes:  317KB, 114KB, 450KB, 507KB
Format: Internal FLM format, C++ API via libqwen3_npu.so
Path:   /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/
Status: ✅ Production (79 tok/s coherent)
```

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
| gu_split | false |

## XCLBIN Dimensions (Qwen3-0.6B)

| Kernel | M | K | N | kt | nt |
|--------|---|---|---|----|-----|
| QKV | 128 | 1024 | 4096 | 64 | 128 |
| O | 128 | 2048 | 1024 | 64 | 128 |
| GU | 128 | 1024 | 6144 | 64 | 128 |
| D | 128 | 3072 | 1024 | 64 | 128 |

## Keyring & Credentials

| Service | Location |
|---------|----------|
| Anthropic/Claude | `~/.claude/.credentials.json` |
| DeepSeek | `~/.pi/agent/auth.json` |
| OpenCode/Go | `~/.pi/agent/auth.json` |
| ProtonMail | `~/.pi/agent/mcp.json` |
| Ollama | `~/.pi/agent/models.json` (localhost) |
| GitHub | `~/.ssh/` (SSH key auth) |
| Xilinx/XRT | `~/torch2aie/licenses/` + `~/.flexlmrc` |

## MLIR-AIE Toolchain

```
Toolchain: ~/torch2aie/toolchain/
  bin/     aiecc.py, aie-opt, xchesscc
  mlir_aie/ Python bindings
  xrt/     XRT headers + libs
  aietools/ AIE compiler tools

Python:   ~/torch2aie/.venv/bin/python
PYTHONPATH: ~/torch2aie/toolchain/mlir_aie/python:~/mlir-aie/python

Build:    cd ~/torch2aie/examples/qwen3-decode-layer
          TORCH2AIE_PYTHON=... run_full_layer.py --build-only
          → design.xclbin + design.bin + design-token127-to-tokenN.bin
```

## FLM Shared Libraries

```
/opt/fastflowlm/lib/
  libqwen3_npu.so    — Qwen3 inference pipeline
  libgemm.so         — Gemm::generate_seq (instruction generator)
  libmha.so          — Multi-head attention
  libq4_npu_eXpress.so — Q4 dequantization

Key symbols (C++ mangled):
  qwen3_npu_sequence::gen_layer_seq(npu_sequence*, uint)
  qwen3_npu_sequence::gen_lm_head_seq(npu_sequence*)
  Gemm::generate_seq(npu_sequence*, ...)
  npu_app_manager::npu_app_manager(...) — weak symbol, overridable
```

## LD_PRELOAD BO Capture

```
hook_bo.so intercepts:
  xrt::ext::bo::bo(device&, size_t)    — FLM uses ext::bo, not bo
  xrt::ext::bo::bo(hw_context&, size_t)
  xrt::bo::bo(device&, size_t, group_id)
  xrt::bo::sync(xclBOSyncDirection, ...)
  npu_app_manager::npu_app_manager(...)

Result: 121 BOs captured (128MB, 10MB, 1MB groups)
Log:    /tmp/hook_bo.log
```

## Performance

| Engine | tok/s | Coherent | Launches/layer |
|--------|-------|----------|----------------|
| FLM | 79 | ✅ | 1 (fused) |
| Universal (BS=128) | 46 | ✅ (~16 tok) | 4 |
| Universal (BS=32) | 34 | ✅ (~16 tok) | 4 |
| Fused layer | ∞ (hangs) | - | 1 (target) |
| GPU ternary | 279 | ❓ | N/A (no model) |

## Build Commands

```bash
# Full rebuild
bash daemon/build-daemon.sh          # Daemon (115K/120K)
bash engine/npu/build_npu.sh         # NPU engines (5 variants)
cd engine/fusion && zig build        # Fused engine (13MB)
cd engine/fusion && zig build test   # Unit tests

# Production stack
./scripts/serve.sh                   # FLM + daemon on :9090

# LD_PRELOAD capture
g++ -shared -fPIC -O2 -o hook_bo.so engine/npu/src/hook_bo.cpp -ldl
LD_PRELOAD=./hook_bo.so flm serve qwen3:0.6b
```

## Remaining Work

1. **Fused xclbin integration**: Generate prefill instruction files, fix engine data flow
2. **Per-channel quantization**: Eliminate ~2%/layer hidden state growth
3. **GPU ternary**: Download/convert ternarized GGUF model for zinc
4. **Zig NPU engine**: Fix XRT C API symbol names (xrtXclbinAllocFilename, etc.)
