# All Supported Models

Full list of every model tested with the 1bit.systems engine stack.
*Last updated: 2026-07-05*

---

## NPU (XDNA 2) — 36 models

| Model Tag | Family | Params | Footprint | Capabilities | Installed |
|-----------|--------|--------|-----------|-------------|-----------|
| `qwen3:0.6b` | Qwen3 | 0.6B | 0.66 GB | chat, reasoning | ✅ |
| `qwen3:1.7b` | Qwen3 | 1.7B | 1.6 GB | chat, reasoning | ⬜ |
| `qwen3:4b` | Qwen3 | 4B | 3.1 GB | chat, reasoning, tool-calling | ⬜ |
| `qwen3:8b` | Qwen3 | 8B | 5.6 GB | chat, reasoning, tool-calling | ⬜ |
| `qwen3-it:4b` | Qwen3 Instruct | 4B | 3.1 GB | tool-calling | ⬜ |
| `qwen3-tk:4b` | Qwen3 Thinking | 4B | 3.1 GB | reasoning, tool-calling | ⬜ |
| `qwen3vl-it:4b` | Qwen3 VL | 4B | 3.9 GB | vision, tool-calling | ⬜ |
| `qwen3.5:0.8b` | Qwen3.5 | 0.8B | 1.4 GB | vision, reasoning | ⬜ |
| `qwen3.5:2b` | Qwen3.5 | 2B | 3.2 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen3.5:4b` | Qwen3.5 | 4B | 5.2 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen3.5:9b` | Qwen3.5 | 9B | 8.94 GB | vision, reasoning, tool-calling | ⬜ |
| `qwen2.5-it:3b` | Qwen2.5 | 3B | 2.5 GB | chat | ⬜ |
| `qwen2.5vl-it:3b` | Qwen2.5 VL | 3B | 3.8 GB | vision | ⬜ |
| `llama3.2:1b` | Llama 3.2 | 1B | 1.3 GB | chat | ✅ |
| `llama3.2:3b` | Llama 3.2 | 3B | 2.7 GB | chat | ⬜ |
| `llama3.1:8b` | Llama 3.1 | 8B | 5.4 GB | chat | ⬜ |
| `gemma3:1b` | Gemma 3 | 1B | 1.2 GB | chat | ⬜ |
| `gemma3:4b` | Gemma 3 | 4B | 4.5 GB | vision | ⬜ |
| `gemma4-it:e2b` | Gemma 4 | 5B | 6.0 GB | audio, vision, reasoning, tool-calling, ASR | ⬜ |
| `gemma4-it:e4b` | Gemma 4 | 8B | 9.1 GB | audio, vision, reasoning, tool-calling, ASR | ⬜ |
| `deepseek-r1:8b` | DeepSeek-R1 | 8B | 5.4 GB | reasoning | ⬜ |
| `deepseek-r1-0528:8b` | DeepSeek-R1-0528 | 8B | 5.6 GB | reasoning | ⬜ |
| `phi4-mini-it:4b` | Phi-4 Mini | 4B | 3.4 GB | chat | ⬜ |
| `gpt-oss:20b` | GPT-OSS | 20B | 14.0 GB | reasoning | ⬜ |
| `gpt-oss-sg:20b` | GPT-OSS Safeguard | 20B | 14.0 GB | reasoning | ⬜ |
| `lfm2:1.2b` | LFM2 | 1.2B | 0.96 GB | chat | ⬜ |
| `lfm2:2.6b` | LFM2 | 2.6B | 1.8 GB | chat | ⬜ |
| `lfm2-trans:2.6b` | LFM2 Transcript | 2.6B | 1.8 GB | chat | ⬜ |
| `lfm2.5-it:1.2b` | LFM2.5 | 1.2B | 0.96 GB | chat | ⬜ |
| `lfm2.5-tk:1.2b` | LFM2.5 Thinking | 1.2B | 0.96 GB | reasoning | ⬜ |
| `nanbeige4.1:3b` | Nanbeige | 3B | 3.1 GB | reasoning | ⬜ |
| `embed-gemma:300m` | Embedding-Gemma | 300M | 0.62 GB | embeddings | ⬜ |
| `whisper-v3:turbo` | Whisper V3 | 1B | 0.62 GB | ASR, transcription | ⬜ |
| `medgemma:4b` | Med-Gemma | 4B | 4.5 GB | vision (medical) | ⬜ |
| `medgemma1.5:4b` | Med-Gemma 1.5 | 4B | 4.5 GB | vision (medical) | ⬜ |
| `translategemma:4b` | Translate-Gemma | 4B | 4.5 GB | vision (translation) | ⬜ |

### NPU Custom Engines (C++23 · XRT direct)

| Engine | Models | Speed | Precision |
|--------|--------|-------|-----------|
| **ALL** (4-xclbin swap) | Qwen3-0.6B, Llama-3.1-8B, Qwen3-VL-4B, Qwen3-8B, Gemma4-E2B | 28 tok/s | INT8 |
| **v12** (single-model) | Qwen3-0.6B | **97 tok/s** | INT8 |
| **BitNet** (ternary) | BitNet b1.58-2B-4T | ~1 tok/s (8× w/ multi-core) | TQ1_0 ternary |
| **Spec Decode** (Eagle3) | Qwen3-0.6B + draft | **~100+ tok/s** | INT8 + CPU draft |

---

## GPU — 1-bit / Ternary (llama.cpp / ZINC Vulkan) — 7 models

| Model | Format | BPW | Size | Engine | Decode |
|-------|--------|-----|------|--------|--------|
| Qwen2 0.5B | IQ1_S | 1.06 | 296 MB | llama.cpp | **381 tok/s** |
| Qwen3.5-0.8B | Q1_0 | 1.25 | 268 MB | llama.cpp | **312 tok/s** |
| Hy-MT2 1.8B | STQ1_0 (ternary) | 1.3125 | 441 MB | ZINC (Sherry) | **267 tok/s** |
| gemma-2-2b | IQ1_S | 1.06 | 788 MB | llama.cpp | **158 tok/s** |
| gemma3 4B | IQ1_S | 1.06 | 1.05 GB | llama.cpp | **122 tok/s** |
| Nemo 8B | IQ1_S | 1.06 | 1.97 GB | llama.cpp | **79 tok/s** |
| Qwen3.5-9B | Q1_0 | 1.25 | 1.82 GB | llama.cpp | **70 tok/s** |

### GPU — ROCm Custom Kernels (ggml-rocm)

| Model | Format | BPW | Size | Engine | Decode |
|-------|--------|-----|------|--------|--------|
| Bonsai-1.7B | TQ2 (ternary) | 2-bit | 1.6 GB | ROCm HIP (ggml-rocm) | **113 tok/s** |

### GPU — F16 / Standard (ZINC Vulkan)

| Model | Format | Size | Engine | Decode |
|-------|--------|------|--------|--------|
| Bonsai-1.7B | F16 | 3.3 GB | ZINC Vulkan | **22 tok/s** |
| Zaya 1.8B | Q2_0 (AMD-native) | varies | ZINC Vulkan | **~18 tok/s** |
| BitNet b1.58 2B | i2_s (ternary) | 132 MB | ZINC (planned) | _benchmarking_ |

### GPU — Video Diffusion (C++ ggml, CUDA, Vulkan, Metal)

| Model | Type | Backend | Parameters | Features |
|-------|------|---------|-----------|----------|
| Wan2.2 | T2V / I2V | ggml + optional NPU | 1.3B / 14B | Reward + Camera LoRAs |

---

## Multi-modal — 22 Models (Diffusers, via video-lora)

| Model | Type | Backend | LoRA | Notes |
|-------|------|---------|------|-------|
| **Wan2.2** | video | diffusers | Reward + Camera | 1.3B / 14B |
| **LTX-Video** | video | diffusers | IC LoRA | 13B, V2V control |
| **LTX2** | video | diffusers | IC LoRA + HDR | 2B, next-gen |
| **CogVideoX** | video | diffusers | Fun LoRA | 2B / 5B, transformer |
| **HunyuanVideo** | video | diffusers | — | 13B, Tencent flagship |
| **AnimateDiff** | video | diffusers | 1000+ LoRAs | SD1.5 base, LCM |
| **Sana Video** | video | diffusers | — | 2B, linear attention |
| **Mochi** | video | diffusers | — | 10B, Genmo |
| **EasyAnimate** | video | diffusers | ✓ | 3B / 7B / 12B |
| **Cosmos** | video | diffusers | — | 7B, NVIDIA world model |
| **Allegro** | video | diffusers | — | Rhymes AI, custom VAE |
| **Motif Video** | video | diffusers | — | 2B, long-form |
| **ConsisID** | video (I2V) | diffusers | — | Identity-consistent I2V |
| **SkyReels V2** | video | diffusers | — | 14B, 540P / 720P |
| **Stable Audio Open** | audio | diffusers | — | 44.1kHz, up to 47s |
| **AudioLDM2** | audio | diffusers | — | speech, music, SFX |
| **LongCat-AudioDiT** | audio | diffusers | — | TTS, waveform diffusion |
| **Flux.1** | image | diffusers | ✓ | 12B, photorealistic |
| **Flux Schnell** | image | diffusers | ✓ | 12B, 4-step |
| **Flux.2** | image | diffusers | ✓ | 12B, next-gen |
| **SDXL** | image | diffusers | 1000s | 2.6B, largest ecosystem |
| **SD3.5** | image | diffusers | ✓ | latest SD, best quality |
