# Install — what AMD says vs what actually works

> Reference doc for the cross-distro / cross-stack picture. If you're on
> Arch / CachyOS and just want this repo running, [`README.md`](../README.md#install--arch--cachyos)
> has the one-command path. This file is the "which install route do I
> actually want, and what trade-off am I taking" map.

AMD's docs say "Ubuntu 24.04 only." That's accurate for **one** of two
AMD-built stacks on Strix Halo. We tested all three install routes; they
don't give you the same thing. Pick the one that matches what you need.

## The two stacks

| Stack | Runs | License | Distribution |
|---|---|---|---|
| **Ryzen AI 1.7.1 EP** (closed) | AMD's 200+ AWQ/OGA HF checkpoints (`huggingface.co/amd/*_rai_1.7.1_npu_*`) | Closed | `.deb`, Ubuntu 24.04 only, behind `account.amd.com` login |
| **Lemonade Server + FastFlowLM** (open) | FLM's curated catalog (qwen3, gemma3, phi-4-mini, lfm2, llama-3.2, deepseek-r1, gpt-oss) | MIT + Apache | AUR (Arch / CachyOS), PPA (Ubuntu), Linux build everywhere |

The closed EP runs models the open stack can't. The open stack runs on
distros the closed EP can't. **Neither is a superset.**

## Three install paths

### Path A — Ubuntu 24.04 native (both stacks)

AMD's `.deb` bundles install cleanly. The
`xrt_plugin-amdxdna 2.21.260102.53` kernel module DKMS-builds against
Ubuntu's stock kernel. Closed EP runs the AWQ/OGA models. Lemonade + FLM
also works alongside.

**Trade-off:** you have to run Ubuntu.

### Path B — Distrobox on a non-Ubuntu host

Looks tempting. **Mostly doesn't work for the closed EP.** Tested on
CachyOS host (kernel `7.0.2-1-cachyos`) → Ubuntu 24.04 distrobox container
with NPU passthrough:

- ✅ `xrt-smi examine` sees the NPU through `/dev/accel/accel0`
- ✅ AMD's `quicktest.py` (basic ONNX via `VitisAIExecutionProvider`) passes
- ❌ `onnxruntime-genai-ryzenai` LLM init fails with `Generic Failure`

Cause: the strict `RyzenAI` provider checks something the AMD-shipped
kernel module exposes that the in-tree `amdxdna` doesn't. Distrobox
passes the device through; it cannot give the container its own kernel
module.

**Trade-off:** distrobox gets you 80 % of the way and falls down at the
actual LLM workload. If you want the closed EP, you're booting Ubuntu
24.04 native.

### Path C — Native non-Ubuntu (Arch / CachyOS / Fedora / …)

Lemonade + FastFlowLM works directly on the host. No container, no DKMS
for closed-EP modules. The in-tree `amdxdna` (kernel 7+) or
`xrt-amdxdna` DKMS (kernel 6.x) is enough. **This is what this repo
installs** ([`README.md` → Install](../README.md#install--arch--cachyos)).
All four lanes light up:

| Lane | Backend | Recipe |
|---|---|---|
| NPU (XDNA 2) | FastFlowLM | `flm:npu` |
| iGPU (gfx1151) — Vulkan | llama.cpp | `llamacpp:vulkan` |
| iGPU (gfx1151) — ROCm | llama.cpp | `llamacpp:rocm` |
| CPU (Zen 5) | llama.cpp | `llamacpp:cpu` |

**Trade-off:** no closed EP. AMD's 200+ AWQ/OGA checkpoints are
downloadable but unrunnable on the NPU. FLM's catalog overlaps on the
popular architectures.

## Receipts (Path C, kernel 7.0.2-1-cachyos)

| Model | Backend | Lane | Decode t/s | Prefill t/s | TTFT |
|---|---|---|---:|---:|---:|
| `LFM2-1.2B-GGUF` | `llamacpp:rocm` | iGPU | 216.3 | 1366.5 | ~0 |
| `qwen3-0.6b-FLM` | `flm:npu` | NPU | 95.4 | 76.0 | 460 ms |
| `qwen3-1.7b-FLM` | `flm:npu` | NPU | 41.8 | 53.7 | 577 ms |
| `deepseek-r1-8b-FLM` | `flm:npu` | NPU | 11.3 | 14.7 | 1430 ms |

iGPU is faster than NPU on these sizes. The NPU lane's value isn't peak
tg — it's offloading the iGPU for ROCm bigger-model serving, lower power
for background inference, and a unified `:13305` API where the request
picks the silicon per model name.

Full sweeps:
[`benchmarks/RESULTS-stack-2026-04-28.md`](../benchmarks/RESULTS-stack-2026-04-28.md),
[`benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md`](../benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md).

## What's still missing on Linux

- **AWQ/OGA on non-Ubuntu** — needs an `xrt`-based EP equivalent of the
  closed `.so`. Doesn't exist publicly.
- **Brevitas in `optimum-amd`** — "Coming soon" since 2024.
- **Diffusion + CLIP on NPU under Linux** — Windows-only.
- **BitNet on NPU** — research-tier; not shipped on any consumer NPU
  yet.

See [`docs/npu-roadmap.md`](npu-roadmap.md) for the longer-form gap list
and [`docs/reddit-npu-gate.md`](reddit-npu-gate.md) for the original
investigation that mapped the closed vs open distinction.
