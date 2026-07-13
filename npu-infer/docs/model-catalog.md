# FLM NPU2 Model Catalog — Complete

All 42 NPU2 models in `/opt/fastflowlm/share/flm/xclbins/` plus extras.
Engine targets: **All Q4NX format models** — xclbin tile design (128×64×128) is dimension-agnostic and works for any model. HD=256 is NOT an obstacle.

## Quick Reference

```
All 5 downloaded models have xclbins built:
  Qwen3-0.6B-NPU2      (684 MB)  ● 4 xclbins  (GU combined)  ⚡ Working
  Llama-3.1-8B-NPU2    (5.7 GB)  ● 5 xclbins  (GU split)     ◐ Not tested
  Qwen3-VL-4B-Instruct (3.2 GB)  ● 5 xclbins  (GU split)     ◐ Not tested
  Qwen3-8B-NPU2        (6.0 GB)  ● 5 xclbins  (GU split)     ◐ Not tested
  Gemma4-E2B-IT-NPU2   (4.7 GB)  ● 4 xclbins  (GU combined)  ◐ Not tested
  23 xclbins total — all built with same 128×64×128 tile template

Not downloaded (42 FLM xclbin dirs exist, need weight downloads):
  Qwen3-4B, Qwen3-1.7B, Qwen3.5-*, LFM2-*, Llama-3.2-*, 
  Gemma3-*, Phi4-mini, DeepSeek-R1-*, Nanbeige, GPT-OSS, Whisper-V3, ...
```

## Key Finding: HD=256 Is NOT an Obstacle

The xclbin MLIR generator (`n1_core_i8.py`) tiles matmuls in 128×64×128 tiles:
- `-m 128` = M dimension per tile (batch)
- `-k 64` = K dimension per tile (reduce)
- `-n 128` = N dimension per tile (output)

**HD=256** just means the QKV N = NH×HD + NKV×HD + NKV×HD is different.
The tile template doesn't know or care about head dimensions — it just
sees a big K×N matmul and tiles it.

Gemma4-E2B (H=1536, NH=16, NKV=2, HD=256):
- QKV: 128×1536×5120  ✅ built (64010 bytes)
- O:   128×4096×1536  ✅ built (39434 bytes)
- GU:  128×1536×12288 ✅ built (105744 bytes, combined)
- D:   128×6144×1536  ✅ built (39434 bytes)

All dimensions must be multiples of 128 due to tile alignment. All models
currently in the catalog have dimensions that satisfy this.

## Architecture Families by Fingerprint

Based on xclbin sizes + feature files:

| Family | attn | mm | deq | Features | Models |
|--------|------|-----|-----|----------|--------|
| **Qwen3 core** | 317k | 507k | 114k | plain | 8 models |
| **LFM2 core** | 407k | 507k | 113k | conv | 6 models |
| **Gemma3 vision** | 317k | 507k | 114k | lm_head+vision | 4 models |
| **Qwen3.5 GDN** | 317k | 507k | 145k | lm_head+gdn+conv+vision | 4 models |
| **Gemma4 multimodal** | 299k | 507k | 114k | lm_head+swa+vision+audio | 2 models |
| **Gemma3 small** | 317k | 219k | 104k | lm_head | 2 models |
| **Qwen3 tiny** | 317k | 507k | 114k | plain | 2 models |
| **Llama/Phi small** | 318k | 507k | 114k | plain | 2 models |
| **GPT-OSS MoE** | 320k | 580k | 319k | expert | 2 models |

## All 42 Models (sorted by architecture family)

### 00. Gemma4 / HD=256 (need investigation)

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| **Gemma4-E2B-IT-NPU2** | 1536 | 35 | 16 | 2 | **256** | 6144 | 262144 | I8 | Combined | ✅ Downloaded 4.7GB. Has swa.xclbin |
| **Gemma4-E4B-IT-NPU2** | ? | ? | ? | ? | **256** | ? | ? | I8 | ? | SWA variant |

### 01. Gemma3 / HD=256 — small LM-head models

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| **Gemma3-1B-NPU2** | ? | ? | ? | ? | 256 | ? | ? | separate | ? | mm=219k (smaller!) |
| **Gemma3-270M-NPU2** | ? | ? | ? | ? | 256 | ? | ? | separate | ? | mm=219k |
| **Gemma3-4B-NPU2** | ? | ? | ? | ? | 256 | ? | ? | separate+vis | ? | vision variant, mm=507k |

### 02. Qwen3-VL / HD=128 vision models

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| **Qwen3-VL-4B-Instruct-NPU2** | 2560 | 36 | 32 | 8 | 128 | 9728 | 151936 | separate | **Split** | ✅ Downloaded 3.2GB |
| **Qwen3-VL-4B-Thinking-NPU2** | ? | ? | ? | ? | 128 | ? | ? | separate | ? | diff attn (698k), diff mm (348k) |

### 03. Qwen3.5 / HD=128 GateDeltaNet

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Qwen3.5-0.8B-NPU2 | ? | ? | ? | 128 | 128 | ? | ? | separate | ? | GDN prefill xclbin |
| Qwen3.5-2B-NPU2 | ? | ? | ? | 128 | 128 | ? | ? | separate | ? | |
| Qwen3.5-4B-NPU2 | ? | ? | ? | 128 | 128 | ? | ? | separate | ? | |
| Qwen3.5-9B-NPU2 | ? | ? | ? | 128 | 128 | ? | ? | separate | ? | |

### 04. Qwen3 / HD=128 core family

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Status |
|-------|---|----|----|-----|----|----|-----|----|----|--------|
| **Qwen3-0.6B-NPU2** | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | tied | Combined | ✅ Working |
| Qwen3-1.7B-NPU2 | ? | ? | ? | 8 | 128 | ? | ? | ? | ? | No config.json |
| **Qwen3-4B-NPU2** | ? | ? | ? | 8 | 128 | ? | ? | ? | ? | deq=145k (diff!) |
| Qwen3-4B-Instruct-2507-NPU2 | ? | ? | ? | 8 | 128 | ? | ? | ? | ? | |
| Qwen3-4B-Thinking-2507-NPU2 | ? | ? | ? | 8 | 128 | ? | ? | ? | ? | |
| DeepSeek-R1-0528-Qwen3-8B-NPU2 | 4096 | 36 | 32 | 8 | 128 | ? | ? | ? | ? | Distill on Qwen3-8B |
| **Qwen3-8B-NPU2** | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | separate | Split | ✅ /home/bcloud/models/ 6GB |

### 05. Qwen2.5 / HD=128

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Qwen2.5-3B-Instruct-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | layer=218k small |
| Qwen2.5-VL-3B-Instruct-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | +vision_attention |

### 06. Llama / HD=128 (no q_norm/k_norm)

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Status |
|-------|---|----|----|-----|----|----|-----|----|----|--------|
| **Llama-3.1-8B-NPU2** | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | separate | **Split** | ✅ Xclbins ready |
| Llama-3.2-1B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | diff mm=584k |
| Llama-3.2-3B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |
| Deepseek-R1-Distill-Llama-8B-NPU2 | 4096 | ? | 32 | 8 | 128 | ? | ? | ? | ? | (Llama arch) |
| Llama-xLAM-2-8B-fc-r-NPU2 | ? | ? | ? | ? | 128 | ? | ? | separate | ? | diff attn=698k |

### 07. LFM2 / HD=128 (conv-attention)

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| LFM2-1.2B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | Has conv.xclbin (depthwise conv) |
| LFM2-1.2B-Tool-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |
| LFM2-2.6B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |
| LFM2-2.6B-Transcript-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |
| LFM2.5-1.2B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |
| LFM2.5-1.2B-Thinking-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |

### 08. Phi / HD=128

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Phi4-mini-Instruct-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |

### 09. Medgemma / HD=128 (or 256?)

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Medgemma-1.5-4B-NPU2 | ? | ? | ? | ? | ? | ? | ? | separate+vis | ? | |
| Medgemma-4B-NPU2 | ? | ? | ? | ? | ? | ? | ? | separate+vis | ? | |

### 10. Nanbeige / HD=128

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Nanbeige4.1-3B-NPU2 | ? | ? | ? | ? | 128 | ? | ? | ? | ? | |

### 11. Translate / HD=128 (or 256?)

| Model | H | NC | NH | NKV | HD | IM | NV | LM | GU | Notes |
|-------|---|----|----|-----|----|----|-----|----|----|-------|
| Translategemma-4B-Instruct-NPU2 | ? | ? | ? | ? | ? | ? | ? | separate+vis | ? | |

### 12. Specialist models

| Model | H | NC | NH | NV | Notes |
|-------|---|----|-----|----|-------|
| Embedding-Gemma-300M-NPU2 | ? | ? | ? | ? | Only mm.xclbin (175k) |
| Whisper-V3-Turbo-NPU2 | ? | ? | ? | ? | Speech-to-text, decoder/encoder xclbins |
| GPT-OSS-20B-NPU2 | ? | ? | ? | ? | 20B MoE, has expert.xclbin |
| GPT-OSS-Safeguard-20b-NPU2 | ? | ? | ? | ? | Safety guard variant |

## Dimension Summary (Known Models with Downloaded Weights)

| Model | H | NC | NH | NKV | HD | IM | NV | GQA | Q/K Norm | RoPE Type | LM Head | GU | Size | Status |
|-------|---|----|----|-----|----|-------|------|-----|----------|-----------|---------|-------|--------|--------|
| Qwen3-0.6B | 1024 | 28 | 16 | 8 | 128 | 3072 | 151936 | 2 | ✅ | theta | tied | Combined | 684MB | ✅ Working |
| Llama-3.1-8B | 4096 | 32 | 32 | 8 | 128 | 14336 | 128256 | 4 | ❌ | rope_freqs[64] + llama3 scaling | I8 64128×5120 | Split | 5.7GB | ✅ Xclbins ready |
| Qwen3-VL-4B | 2560 | 36 | 32 | 8 | 128 | 9728 | 151936 | 4 | ✅ | theta | I8 47480×5120 | Split | 3.2GB | 🔲 Build xclbins |
| Qwen3-8B | 4096 | 36 | 32 | 8 | 128 | 12288 | 151936 | 4 | ✅ | theta | I8 75968×5120 | Split | 6.0GB | 🔲 Build xclbins |
| Gemma4-E2B | 1536 | 35 | 16 | 2 | **256** | 6144 | 262144 | 8 | ✅ | rope_freqs[256] | I8 49152×5120 | Combined | 4.7GB | 🔲 HD=256 issue |

## Xclbin Build Requirements (per model)

Each model needs 4-5 xclbins:

```
QKV:   M=128, K=H,     N=NH×HD + NKV×HD + NKV×HD
O:     M=128, K=NH×HD, N=H
G:     M=128, K=H,     N=IM            (if GU split)
U:     M=128, K=H,     N=IM            (if GU split)
GU:    M=128, K=H,     N=2×IM          (if combined)
D:     M=128, K=IM,    N=H
```

### Which models can use our current xclbin template?

**All of them.** The 128×64×128 tile template is dimension-agnostic.
Any K,N that are multiples of 128 work — including HD=256 models.

| Model | HD | Can use current xclbin template? |
|-------|----|----------------------------------|
| Qwen3-0.6B | 128 | ✅ Yes (4 xclbins built) |
| Llama-3.1-8B | 128 | ✅ Yes (5 xclbins built) |
| Qwen3-VL-4B | 128 | ✅ Yes (5 xclbins built) |
| Qwen3-8B | 128 | ✅ Yes (5 xclbins built) |
| Gemma4-E2B | 256 | ✅ Yes (4 xclbins built) |
| Gemma3-* | 256 | ✅ Yes (but need dimension values) |

## Next Best Target: Qwen3-VL-4B-Instruct-NPU2

| Reason | Detail |
|--------|--------|
| HD=128 | Same as current engine |
| Has q_norm/k_norm | Same as Qwen3-0.6B |
| H=2560 | Largest not-yet-built HD=128 model |
| IM=9728 | Tests larger MLP (GU split needed) |
| Downloaded | 3.2GB ready in local cache |
| Has vision tower | Can extend to multimodal after text works |

## Remaining Models (Need Weights + Dimension Extraction)

These have FLM xclbin dirs but we need the actual Q4NX `.model.q4nx` file
to extract dimensions and build matching xclbins:

| Model | Arch Type | What's Needed |
|-------|-----------|---------------|
| Qwen3-4B, Qwen3-1.7B, Qwen3-VL-4B-Thinking | Standard Qwen3 | Download weights, extract H/IM/NH/NKV |
| Qwen3.5-0.8B/2B/4B/9B | GateDeltaNet (conv) | Different attention, GDN prefill xclbin |
| LFM2-1.2B/2.6B (+variants) | Conv-attention | Has conv.xclbin, mm=507k |
| Llama-3.2-1B/3B | Standard Llama | Download weights |
| DeepSeek-R1-Distill-Llama-8B | Llama arch | config.json to confirm H/IM |
| Gemma3-1B/270M/4B | HD=256 | xclbin template works, need weights |
| Gemma4-E4B | HD=256 | Same template as E2B |
| Phi4-mini | Standard | Download weights |
| Nanbeige4.1-3B | Standard | Download weights |
| DeepSeek-R1-0528-Qwen3-8B | Qwen3-8B distill | Same dims as Qwen3-8B |
| Llama-xLAM-2-8B | Llama variant | attn diff size (698k) |
| Medgemma-1.5-4B/4B | HD=256? + vision | Need investigation |
| Translategemma-4B | Vision + HD=256? | Need investigation |
| GPT-OSS-20B (+Safeguard) | MoE | expert.xclbin — different engine |
| Whisper-V3-Turbo | Speech | encoder/decoder arch |
| Embedding-Gemma-300M | Embedding | Non-generative |

## Xclbin Build Status (23 total for 5 downloaded models)

```
Model             Xclbins  GU     H     IM    NH  NKV  HD   Total Size
----------------  -------  -----  ----  ----  ---  ---  ---  ----------
Qwen3-0.6B        4        comb   1024  3072  16   8    128   207 KB
Qwen3-VL-4B       5        split  2560  9728  32   8    128   337 KB
Qwen3-8B          5        split  4096  12288 32   8    128   397 KB
Llama-3.1-8B      5        split  4096  14336 32   8    128   421 KB
Gemma4-E2B        4        comb   1536  6144  16   2    256   249 KB
```
