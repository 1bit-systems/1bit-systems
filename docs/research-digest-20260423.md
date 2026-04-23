# Research digest — 2026-04-23

Five parallel research scans dispatched while Run 5 Sparse-BitNet training holds the H200 NVL pod. All four sub-2-bit lanes (audio, video/diffusion, ASR, NPU) and one competitive scan. Summary: **halo-ai is still alone in the 1.58-bit + native-HIP + C++ + multimodal lane on gfx1151.** The audio + video ternary frontiers are effectively empty — open doors for first-public runnable artifacts.

---

## 1. Audio BitNet / ternary TTS

**Scope**: 1-bit / 1.58-bit / ternary / sub-2-bit TTS / voice / codec / vocoder / music-gen, 2024-01 → 2026-04.

### Real finds
- **BitTTS** (arXiv 2506.03515, Kawamura et al., LY Corp Japan, Interspeech 2025) — the only 1.58-bit TTS paper. JETS + HiFi-GAN quantized ternary {-1,0,+1} + int8 "weight indexing". 83% size reduction, MOS ≥ 4-bit baseline. **No code. No weights.** Demo page only.
- **ENERZAi EZWhisper 1.58-bit** (industry, Embedded Vision Summit 2025) — Whisper-small QAT'd to 1.58-bit, ~70 MB on Synaptics/Renesas MCUs. Closed commercial, proprietary Optimium runtime.
- **SparkNet** (arXiv 2406.06634, Svirsky et al., Interspeech 2024) — 1-bit sparse KWS only, narrow task.
- **DNSMOS w/ binary activations** (arXiv 2407.04578) — 1-bit activations + 8-bit weights, narrow (speech-quality prediction).
- **Adaptive binary speaker verification** (arXiv 2406.05359) — no weights/code released.

### Empty channels
- Zero ternary EnCodec / DAC / SoundStream / Mimi codecs.
- Zero ternary HiFi-GAN / BigVGAN vocoders.
- Zero 1.58-bit wav2vec 2.0 / HuBERT / WavLM.
- Zero 1-bit MusicGen / AudioLDM / Stable Audio.
- Zero ternary speech-LM (Moshi / Parler / VoxCPM class).

### Verdict
**Near-empty field.** One academic paper (no code), one closed commercial demo, one narrow KWS result. halo-ai clean-room implementing BitTTS on Qwen3-TTS-0.6B during Run 6 would be the first public runnable 1.58-bit TTS on any GPU — nobody else has shipped this.

---

## 2. Diffusion / DiT / video-gen BitNet

**Scope**: 1-bit / 1.58-bit / ternary DiT, U-Net, VAE, text-encoder, video-gen, 2024-01 → 2026-04.

### Real-with-code (shippable today)
- **TerDiT** (arXiv 2405.14854, Lucky-Lance/TerDiT, MIT) — ternary class-conditional DiT, Large-DiT 4.2B beats FP DiT-XL/2 on ImageNet-512 FID. HQQ-derived 2-bit CUDA kernel only. **Reference only** (wrong task — class-cond, not T2V; wrong kernel — CUDA not HIP). Blueprint for QAT recipe + adaLN-RMSNorm stability trick.
- **RobuQ — W1.58A2 DiT** (arXiv 2509.23582, racoonykc/RobuQ) — first ternary-weight + 2-bit-activation DiT. Hadamard-transform activation quantizer. ImageNet only. **Steal the Hadamard activation-quant technique** for our Sherry lane.
- **BiDM** (W1A1, NeurIPS 2024) + **BinaryDM** (W1A4, ICLR 2025) — Xingyu-Zheng repos, LSUN/CIFAR unconditional, no pretrained weights shipped. Research code only, skip.

### Paper-only (vaporware)
- **1.58-bit FLUX** (arXiv 2412.18653, ByteDance + POSTECH) — 99.5% of 11.9B FLUX params ternarized, claims 7.7× storage / 5.1× memory. **Repo empty 16+ months, weights never released.** ComfyUI community gave up.
- **BitsFusion — 1.99-bit SD1.5 UNet** (NeurIPS 2024, Snap Research) — SD1.5 UNet 1.72 GB → 219 MB, beats baseline 54% in human eval. **Repo "under company review", README only, weights never shipped.**

### Negative results
- No ternary Wan / Hunyuan-Video / OpenSora / CogVideoX / PixArt / Sana anywhere.
- Video DiT quant floor = W4A4 (DVD-Quant, Mar 2026).
- Zero sub-2-bit GGUF diffusion weights on HF (QuantStack / city96 / leejet floor = Q2_K).

### Verdict
**Ternary video-gen is greenfield.** Nobody has shipped ternary T2V on any GPU. Wan 2.2 TI2V-5B at 1.58-bit ≈ 1 GB (vs 10 GB FP16 — text encoder UMT5-XXL + VAE stay FP16). Stacking TerDiT's adaLN-RMSNorm stability fix + RobuQ's Hadamard activation quant + our HIP ternary GEMV kernel = viable first-mover paper. Aligns with 1bit.systems brand better than LLM-ternary does.

---

## 3. 1-bit ASR

**Scope**: 1-bit / 1.58-bit / ternary ASR / STT / Conformer / wav2vec / HuBERT, 2024-01 → 2026-04.

### Real hits
- **AsadIsmail/whisper-{large-v3,medium,small}-ternary** (HF, Apache-2.0, 2026-04-14-17) — decoder-only ternary ("tritplane3"), encoder stays FP16. 1.8× compression (944 MB vs 3 GB, large). No published WER, only "exactly matches FP16 on a sample." Code repo 404s. **PTQ not retrain**, shippable status unclear.
- **arXiv 2505.21245 — "Towards One-bit ASR"** (CUHK + Tsinghua, May 2026) — Conformer not Whisper. **1-bit test-clean 2.79% WER vs 2.55% fp32** (statistically lossless). 16.6× compression. **No code, no weights.**
- **ENERZAi EZWhisper** — closed commercial, Qualcomm QCS6490 only.

### Empty channels
- Zero ternary Whisper with actual code + WER claims.
- Zero ternary wav2vec2 / HuBERT / Conformer weights anywhere.
- Zero streaming / real-time 1-bit ASR.
- No Samsung / Qualcomm / Apple public sub-2-bit ASR research.

### Verdict
**Pass.** Field ~6 months behind LLM ternary. whisper.cpp q5 on sliger B580 wins pragmatically today. Watch-only: if `Asad-Ismail/ternary-quant` repo materializes with packed weights + kernel signature compatible with our BitLinear HIP GEMV, it's the only candidate for port. Revisit Q3 2026.

---

## 4. AMD XDNA2 NPU Linux — Strix Halo ship-gate status

**Scope**: past 30 days (2026-03-23 → 2026-04-23).

### Driver
- `amd/xdna-driver` main: `c8d1727` DRAM work buffer for AIE4, `a87f856` NPU busy-time via DRM fdinfo. AIE4 / npu5 (17f0_11) support present since March. No new firmware drop; `1.7.1` already in-tree.
- Prior memory note about an `amd-ipu-staging` branch was wrong — AIE4 landed on `main` directly.

### Userspace
- `amd/Triton-XDNA` (MIT, Python+MLIR) — `AMD_TRITON_NPU_TARGET` env var (PR #56, 2026-04-20). Still no ternary lowering.
- `amd/IRON` — Phoenix (npu1) only, not npu5.
- `amd/RyzenAI-SW v1.7.1` (2026-03-27) — STX/KRK only, **STX-H still excluded from supported-SKU list**. Issue `#366` unanswered since Feb 2026.
- `ryzenai.docs.amd.com/en/latest/linux.html` (v1.7.1, Apr 2026) lists device ID `1022:17f0` as STX/KRK only.
- **Ubuntu 26.04 LTS (2026-04-23)** ships Linux 7.0 with amdxdna kernel driver — **driver yes, no userspace Vitis EP**.

### ONNX Runtime VitisAI EP
- No Linux STX-H bring-up. `MatMulNBits N=2` kernel still "planned." Can't run BitNet-1.58 even if EP shipped.

### Community userspace
- **FastFlowLM + Lemonade** (2026-03-11) — lists Max 300-series as Linux Supported. Closed-source binary, **Q4 only, no ternary/BitNet**. Fails Rule A — can't host BitNet.
- Hand-patched forks (ChrisJR035/xdna-cerberus, Peterc3-dev/amdxdna-strix-fix) — single-user, not supportable.

### Phoronix / LWN / bench
- No dedicated STX-H NPU Linux article in window. Zero npu5 community benchmarks on Linux.

### Forward signal
- **Jeremy Fowers (AMD) DM 2026-04-23**: "NPU is being worked on internally." Only positive signal, not installable.

### Verdict
**Ship-gate still closed.** No public AMD or community path lets us run BitNet-1.58 on npu5 Linux today. Revisit on Ryzen AI 1.8 release or AMD blog post naming STX-H Linux.

---

## 5. Competitive Strix Halo AI stack — 90-day scan

### The dominant stack
- **kyuz0/amd-strix-halo-toolboxes** — 1,363 stars, Python/Dockerfile, Fedora 43 Toolbx/Distrobox containers over ROCm 7.2.1 + kernel 6.18.4. Endorsed by `strixhalo-homelab.d7.wtf` wiki as canonical. Sibling repos cover LLM (llama.cpp + vLLM) / image / video / voice / fine-tune — **same five lanes as halo-ai but containerized wrappers over upstream stock kernels.** Zero BitNet, zero 1.58-bit, no custom HIP.

### AMD-maintained
- **lemonade-sdk/lemonade** — 3,648★, Apache-2.0, C++, today's release v10.2.0. LLM-only, llama.cpp/ROCm wrappers, FastFlowLM for NPU. No 1.58-bit. (We ship the `lemon-mlx` backend PR into this stack.)
- **AMDResearch/intellikit** — 11★, MIT, kernel profiling + rocprof MCP servers. Integration opportunity, not competitor.

### Adjacent
- `lychee-technology/llama-cpp-for-strix-halo`, `pablo-ross/strix-halo-gmktec-evo-x2`, `MaxusAI/ryzen-ai-max-rocm-ollama-testbench`, `hogeheer499-commits/strix-halo-guide`, `schutzpunkt/strix-halo-ai-stack`, `IgnatBeresnev/comfyui-gfx1151`, `SometimesCake/amd-strix-halo-whisper-toolbox` — all llama.cpp / ComfyUI / Ollama wrappers.
- **glovepost/wmma_ops** — 1★, C++. Only other custom gfx1151 WMMA kernel work. Narrow scope, no model stack.
- **oresk/lemonade-npu-toolbox** — containerized FastFlowLM + lemonade NPU. Windows-only NPU still.

### Tetramatrix ecosystem (Chi Hoang)
- Zero gfx1151-tagged repos pushed since 2026-01-23. Still fork-of-known-thing pattern. Not competing here.

### Chinese OEM stacks
- Bosgame/GMKtec/FEVM/NIMO ship Windows 11 + proprietary "FlowyAIPC 4.0" marketing blob. No Chinese open-source gfx1151 stack found.

### Verdict
**halo-ai is still alone in 1.58-bit + native-HIP + C++ + 5-lane-multimodal + systemd-service space on gfx1151.**

Positioning for launch: do NOT fight kyuz0 on containerized llama.cpp — he has won that lane decisively. Lead with BitNet-1.58 + native HIP kernels + multimodal-in-one-install + C++-first. Cite kyuz0 respectfully; intellikit is an integration target, not a threat. Ship-gate still holds for NPU.

---

## Immediate roadmap implications

| Lane | Greenfield? | Priority |
|---|---|---|
| Ternary TTS (Qwen3-TTS 0.6B) | yes, clean-room BitTTS | **Run 6 post-Run-5** |
| Ternary video (Wan 2.2 TI2V-5B) | yes, first public | **Run 7 candidate** |
| Ternary LLM | no (we already ship) | shipping |
| Ternary ASR | no (field behind, not worth) | skip, whisper.cpp q5 stays |
| Ternary image (SDXL / FLUX) | vaporware competition | steal TerDiT + RobuQ recipes |
| NPU lane | blocked (ship-gate) | Q3 2026 revisit |

Ternary video-gen is the narrative-strongest lane for 1bit.systems — first public, clean license (Wan 2.2 Apache-2.0), visibly novel.

---

*Digest compiled from 4 parallel general-purpose agents + 1 direct GH scan, 2026-04-23. Run 5 Sparse-BitNet 2:4 retrain holds the H200 NVL pod; all agent work was portable off-pod.*
