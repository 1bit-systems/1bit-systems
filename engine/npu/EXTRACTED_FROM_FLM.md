# Models Extracted from FLM v0.9.45/46 (ROCm Mirror)

Source: https://github.com/ROCm/FastFlowLM (mirror of FastFlowLM/FastFlowLM)
Date: 2026-07-29

## New Model Architectures

### 1. Qwen3.5 Omni (audio+vision+text) — NEW in ROCm mirror
- Files copied to `1bit-systems/src/`:
  - `modeling_qwen3_5_omni.cpp` — text backbone
  - `modeling_qwen3_5_omni_audio.cpp` — audio encoder
  - `modeling_qwen3_5_omni_image.cpp` — vision encoder
- Source: `rocm-fastflowlm/src/common/AutoModel/`

### 2. Qwen3.6-MoE-35B-A3B — 256 experts, 40 layers
- Config pulled from HF: `hidden_size=2048, head_dim=256, 16 heads, 2 KV heads`
- Alternating linear/full attention (full every 4th)
- 262k context length, Q4_K_S quantization
- Expert FFN: `intermediate_size=512` per expert

### 3. All new xclbin shapes calculated
See `build_new_xclbins.sh` — 25 new shapes across 5 models.
Toolchain dependency: needs MLIR Python bindings (`llvm-project` checkout) on dev machine.

## Model Configs Added

`npu_dims.h` now has entries for:
- qwen3_6_moe_35b — MoE dense config (experts handled at runtime)
- qwen3_5_4b — 2560-dim vision model
- gemma4_e4b — 2560-dim gemma4 variant
- phi4_mini_4b — 3072-dim dense model
- nanbeige4_1_3b — 2560-dim, head_dim=80 (unusual)

## Build Steps (on dev machine with full toolchain)

```bash
cd 1bit-systems/engine/npu/generators
source env.sh  # or activate the mlir-aie venv

# 1. Build all new xclbins
./build_new_xclbins.sh

# 2. Build NPU engine variants
./build_npu.sh

# 3. Verify
./xclbins/build_xclbins.sh qwen3.6-moe_35b
```

## What Each Model Gives You

| Model | Arch | Params | Your Engine Benefit |
|-------|------|--------|-------------------|
| Qwen3.6-MoE | MoE, 256 experts | 35B/3B active | MoE support, 262k ctx |
| Qwen3.5-4B | Dense VLM | 4B | Vision + reasoning |
| Gemma4-E4B | Dense omni | 8B | Audio + vision + text |
| Phi4-Mini | Dense | 4B | Full attention model |
| Nanbeige4.1 | Dense | 3B | Reasoning model |
| Qwen3.5 Omni | Omni | ? | Multi-modal C++ impl |

## New C++ Source from ROCm Mirror

```
rocm-fastflowlm/src/common/AutoModel/
├── modeling_qwen3_5_omni.cpp          # Copied
├── modeling_qwen3_5_omni_audio.cpp    # Copied
├── modeling_qwen3_5_omni_image.cpp    # Copied
```

These contain FLM's implementation of Qwen3.5 omni-model inference — useful for understanding their multi-modal routing approach.
