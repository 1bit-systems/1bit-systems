# 1bit Speculative Decoding

Multi-Token Prediction (MTP) speculative decoding for the **1bit NPU inference stack** (XDNA 2, Qwen3-0.6B, 94 tok/s baseline).

**Target: 141-235 tok/s (1.5-2.5x speedup) with lossless quality.**

## Architecture

```
                    ┌──────────────────────────┐
                    │  Draft Model (Eagle3)     │
                    │  1 transformer layer      │
                    │  ~8.5M params @ FP16      │
                    │  ~34 MB — fits on NPU     │
                    └──────┬───────────┬────────┘
                           │           │
                    trunk_hidden   token_embed
                           │           │
                           ▼           ▼
                    ┌──────────────────────────┐
                    │  eh_proj fusion           │
                    │  RMS Norm → Concat → Proj │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  Self-Attention (16 heads)│
                    │  + KV Cache               │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  SwiGLU FFN               │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │  LM Head → draft logits   │
                    │  + acceptance check       │
                    └──────────────────────────┘
```

## Speculative Decoding = MTP

The core insight: **Multi-Token Prediction (MTP)** is the engine behind speculative decoding. Instead of predicting one token at a time:

- **Without MTP:** Predict 1 token → verify 1 → predict 1 → verify 1 → ...
- **With MTP:** Predict N tokens in parallel → verify all N in one forward pass → accept ~80%

The draft model is tiny (1 layer, ~8.5M params) compared to the target (0.6B). It runs on the same NPU as a fused kernel.

## Repository Structure

```
spec-decode/
├── configs/                          # DeepSpec configs for Qwen3-0.6B
│   ├── eagle3_qwen3_0.6b.py          # ✅ RECOMMENDED — 1 draft layer
│   ├── dflash_qwen3_0.6b.py          # 5 layers, no Markov
│   └── dspark_qwen3_0.6b.py          # 5 layers, full Markov + confidence
├── draft/
│   └── mtp_draft.h                   # C++ Eagle3 draft model (single header)
├── engine/
│   ├── spec_decode.h                 # Speculative decoding orchestrator
│   └── npu_spec_integration.cpp      # NPU (XRT) integration point
├── bench/
│   └── spec_decode_bench.cpp         # Performance benchmark (simulated)
├── tests/
│   ├── test_draft_model.cpp          # Draft model unit tests
│   └── test_verify_loop.cpp          # Verification loop tests
├── train_draft.py                    # Training launcher (uses DeepSpec)
├── eval/
│   └── eval_draft.py                 # Evaluation harness
├── CMakeLists.txt                    # Build system
└── README.md
```

## Quick Start

### 1. Train the Draft Model (requires GPU)

```bash
# Install DeepSpec
cd ~/DeepSpec
pip install -r requirements.txt

# Train Eagle3 draft (1 layer — recommended for NPU)
cd ~/1bit-systems/spec-decode
python train_draft.py --arch eagle3 --epochs 10 --num-gpus 8

# Or train DFlash for higher acceptance (5 layers)
python train_draft.py --arch dflash --epochs 10 --num-gpus 8
```

### 2. Evaluate

```bash
python eval/eval_draft.py \
    --target Qwen/Qwen3-0.6B \
    --draft ./checkpoints/eagle3_step_30000 \
    --tasks gsm8k math500 humaneval
```

### 3. Build NPU Engine

```bash
# Build with NPU support (requires XRT toolchain)
cd ~/1bit-systems/spec-decode
mkdir -p build && cd build
cmake .. -DENABLE_NPU=ON -DXRT_DIR=/home/bcloud/torch2aie/toolchain
make -j$(nproc)

# Run benchmark (simulated target — no NPU needed)
./spec_decode_bench

# Run with actual NPU
./npu_spec_decode /path/to/int8.xclbin 128 512
```

### 4. Integrate with FLM Proxy

The speculative decoding engine was originally designed to hook into the FLM proxy at port 9090 (FLM has since been replaced; the proxy architecture remains for other backends):

```bash
# Option A: Replace server.cjs with spec-decode server
# (see engine/npu_spec_integration.cpp for the API)

# Option B: Add as a middleware layer
# FLM proxy → spec_decode → NPU
```

## Architecture Comparison for 0.6B

| Architecture | Draft Params | Training Cost | Expected Speedup | NPU Fit |
|---|---|---|---|---|
| **Eagle3** | **~8.5M** | **Lowest** | **1.5-2.0x** | **✅ Trivial** |
| DFlash | ~43M | Medium | 1.8-2.5x | ✅ Easy |
| DSpark | ~43M + Markov | Highest | 2.0-2.8x | ⚠️ Tight |

**Recommendation: Start with Eagle3** — the 1-layer draft adds minimal overhead and is the easiest to train and deploy. The Markov/confidence heads in DSpark add complexity that may not justify the marginal acceptance gain on a 0.6B target.

## Research Sources

| Repo | Stars | What It Provides |
|---|---|---|
| [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) | ⭐5948 | Training framework for all 3 architectures + eval benchmarks |
| [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) | ⭐964 | Training + SGLang deployment, online training |
| [Indras-Mirror/llama.cpp-turboq-mtp](https://github.com/Indras-Mirror/llama.cpp-turboq-mtp) | ⭐83 | C++ MTP implementation, tensor sharing, RotorQuant |
| [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen) | ⭐43 | Strix Halo + DFlash reference (your hardware) |

## License

MIT — use freely. Built from research by DeepSeek, LMSYS, Google, and the open-source community.
