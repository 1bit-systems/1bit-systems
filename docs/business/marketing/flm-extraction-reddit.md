# r/LocalLLaMA — We extracted AMD's entire closed-source NPU model zoo. 37 models, 209 xclbins, zero config. All open source.

**What we did:**

AMD ships a locked-down NPU inference stack (FastFlowLM) with every Strix Halo APU — ~48 million units this year, 50 TOPS XDNA 2 silicon sitting idle because the proprietary runtime needs a license chain most users never set up. We reverse-engineered the stack in 4 days, then pulled every pre-built model from the ROCm/FastFlowLM v0.9.46 mirror.

The haul: **37 models, 209 pre-compiled NPU xclbins**, plus Qwen3.5 Omni multi-modal C++ source (audio encoder, vision encoder, text backbone — 1,611 lines actual C++, not Python/ONNX).

**How it works:**

Drop a Q4NX file onto our single C++ binary (~207 KB). It reads the Q4NX header, auto-detects architecture and dimensions, picks the right xclbin from `flm_model_map.json` (which maps all 37 model tags), and runs. Zero config files. Zero Python at runtime. Works with the mainline `amdxdna` driver — no proprietary FastFlowLM `.so` files needed.

**What's in the box:**

| Model Family | Xclbins | Notes |
|---|---|---|
| Qwen3 (0.6B–8B) | 4 each | Peano dims ready for 0.6B, 8B |
| Qwen3.5 (0.8B–9B) | 8 each | 4B Peano dims ready |
| Qwen3.6-MoE-35B | 9 | 256 experts, 3B active, 3 Peano INT8 xclbins |
| Qwen3-VL-4B | 6 | Vision-language on NPU |
| Gemma4 (E2B, E4B) | 10 each | Both Peano dims ready |
| Phi4-Mini-4B | 4 | Peano dims ready |
| Nanbeige4.1-3B | 4 | head_dim=80, Peano dims ready |
| Llama3.2 (1B, 3B), Llama3.1-8B | 4 each | |
| DeepSeek-R1 (8B distill) | 4 each | |
| GPT-OSS-20B (MoE), +Safeguard | 6 each | |
| Gemma3 (1B, 4B), MedGemma, TranslateGemma | 5–7 each | |
| LFM2 (1.2B, 2.6B), LFM2.5 | 5 each | |
| Qwen2.5-3B, Qwen2.5-VL-3B | 4–7 each | |
| Embedding-Gemma-300M, Whisper-V3-Turbo | 4–5 each | |
| Bonsai-1.7B | 0 | ternary model, no FLM xclbins |

**Performance (validated on Strix Halo, Radeon 8060S + XDNA 2):**

| Benchmark | tok/s | Backend | Status |
|---|---|---|---|
| Q1 GEMV kernel | 433 | ROCm HIP | validated |
| Fused TQ2 kernel | 420 | ROCm HIP | validated |
| GPU ternary | 318 | Vulkan ZINC | validated |
| BlackMamba 1.5B e2e | 79.4 | ROCm HIP | validated |
| BlackMamba 2.8B e2e | 46.0 | ROCm HIP | validated |
| NPU INT8 GEMM | 0/10000 errors | XDNA 2 via Peano | verified |

**What's still broken (we tell you):**

- 22 of 209 xclbin shapes still need compilation via MLIR-AIE/Peano toolchain. Build scripts are checked in; the compiled binaries aren't.
- Qwen3.5 Omni C++ source is extracted and documented, but audio/vision encoder paths don't connect to our NPU dispatch yet.
- MoE expert routing for Qwen3.6-35B is runtime-only — no dispatch overhead benchmarks yet.
- No end-to-end token-generation benchmarks for the newly added architectures.

**Why this matters:**

The 1bit-systems engine is 18 model architectures, 46+ 1BP models, 4 backends (NPU, GPU HIP, GPU Vulkan, CPU), MIT licensed. One binary. No config. Every Strix Halo owner can now use their NPU without AMD's proprietary stack.

Blog: https://1bit.systems/blog/flm-37-models-extracted
GitHub: https://github.com/bong-water-water-bong/1bit-systems
Full model catalog: https://1bit.systems/wiki/models
Benchmark methodology: https://1bit.systems/wiki/performance
