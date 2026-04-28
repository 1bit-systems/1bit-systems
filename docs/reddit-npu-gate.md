⚠️ **UPDATE (2026-04-28). Most of this post is wrong. Two separate corrections; both are mine to own. Leaving the post up with strikethroughs because the failure mode is more interesting than the original argument, and I owe clean retractions more than a clean delete.**

**Correction #1 — I conflated two AMD stacks that are not the same stack.** I drew confident conclusions about the wrong one and skipped the one that actually answers the question I was asking.

The two stacks:

1. **Ryzen AI 1.7.1 deployment toolkit** — closed-source, DirectML-bound. *I extracted this from the public NuGet (`RyzenAI_Deployment.1.7.1.nupkg`)*, which contains only Windows DLLs (`onnxruntime_providers_ryzenai.dll`, `onnxruntime_vitisai_ep.dll`, `dyn_dispatch_core.dll`, etc). The NuGet observation is accurate. The conclusion I drew from it — *"therefore there is no Linux Ryzen AI EP at all"* — was wrong (see Correction #2 below).

2. **Lemonade Server** — a *separate* AMD-built, MIT-licensed, open-source project. Linux build exists. Has its own backend recipe system (`llamacpp:rocm`, `llamacpp:vulkan`, `flm:npu`, `kokoro:cpu`, `whispercpp:*`, `sd-cpp:*`). Officially documents Linux NPU support via FastFlowLM at **lemonade-server.ai/flm_npu_linux.html**, dated **2026-03-11**, **co-authored by the Lemonade team and the FastFlowLM team**.

I investigated stack #1 thoroughly. I never looked at stack #2. I assumed "AMD on Linux = no NPU LLM" because *the deployment NuGet I tore apart* shipped no Linux artifacts. **Lemonade is the open-stack answer to the LLM-on-NPU-on-Linux question, and it's been working since March 2026.**

**Correction #2 — AMD's *closed* Ryzen AI EP also exists for Linux. I just couldn't see it from the NuGet because it's distributed separately, behind an account login.**

The official page **`ryzenai.docs.amd.com/en/latest/llm_linux.html`** (last updated **2026-04-19**) walks through running LLMs on the NPU under Linux using `libonnxruntime_providers_ryzenai.so` — the *Linux* equivalent of the `.dll` I extracted. Distribution constraints:

- Bundle: `ryzen_ai-1.7.1.tgz` + `RAI_1.7.1_Linux_NPU_XRT.zip`, both downloaded from `account.amd.com` (AMD account login required).
- Packaged as `.deb`, target Ubuntu 24.04 LTS (Python 3.12, kernel ≥ 6.10). Not on Arch / not on PyPI; AMD also runs `pypi.amd.com/ryzenai_llm/1.7.1/linux/simple/` for the `model-generate` Python piece.
- Cited bench in their docs: **Phi-3.5-mini-3.8B at ~17.6 tok/s decode, 864 tok/s prefill** on the NPU through this stack.
- Loads AMD's own 200+ AWQ/OGA HF checkpoints (`huggingface.co/amd/*_rai_1.7.1_npu_*`) — the artifacts that Lemonade + FLM *can't* run today.

So my original "zero Linux runtime exists" claim was wrong on two counts: Lemonade + FLM is the **open** Linux LLM-on-NPU path (Correction #1), and AMD's **closed** Ryzen AI EP also has a Linux distribution (Correction #2) — it's just .deb-only, auth-walled, and not in any public package manager. The NuGet I extracted *is* Windows-only; what I missed is that AMD ships a parallel Linux bundle through a different channel.

**Correction #2.5 — "Ubuntu 24.04 only" is not a packaging restriction. It's a kernel-ABI restriction.** I tested this. I installed the `.deb` stack on a CachyOS box via Distrobox (Ubuntu 24.04 container with NPU pass-through), got `xrt-smi examine` to see the NPU just fine, ran AMD's `quicktest.py` (basic ONNX inference on NPU through `VitisAIExecutionProvider`) and it passed cleanly — *"Test Finished"*, session initialized, NPU responded. So far so good. Then I tried the actual LLM flow (`onnxruntime-genai-ryzenai` loading `Phi-3.5-mini-instruct_rai_1.7.1_npu_4K`) and got the *least helpful possible error message*:

```
[E:onnxruntime:, inference_session.cc:2544 operator()] Exception during initialization: Generic Failure
```

Closed `.so`, no string. Quicktest passes; LLM init fails. The difference: the strict `RyzenAI` provider used for LLMs requires the AMD-shipped kernel module (`xrt_plugin-amdxdna 2.21.260102.53.release`, distributed in the same `.deb` bundle as the userspace), and that module's DKMS build fails inside the container because there are no kernel headers for `7.0.2-1-cachyos` to compile against. The host's in-kernel `amdxdna` handles the userspace handshake (so `xrt-smi` works, quicktest works), but the strict EP path checks something the AMD-shipped module exposes that the in-kernel one doesn't, and rejects with no diagnostic.

**Practical implication:** AMD's Linux Ryzen AI flow runs on **Ubuntu 24.04 with AMD's amdxdna .ko loaded** — full stop. Distrobox-on-Arch isn't enough. Native Arch / Fedora / etc. would need to either (a) build AMD's amdxdna against the running kernel via DKMS, replacing the in-kernel one, or (b) boot Ubuntu 24.04 natively. The "Ubuntu only" framing in the docs is technically a *kernel ABI* gate, not a packaging convenience.

This makes **Lemonade + FastFlowLM the only Linux NPU LLM path that works on non-Ubuntu distributions today.** Not because AMD ignored Linux — they shipped the EP — but because the closed EP's kernel-module pinning makes anything outside Ubuntu 24.04 a port project. (FLM, by contrast, is happy with whatever in-kernel `amdxdna` your distro provides.)

What pushed me into the wrong conclusion: when I tried to use Lemonade's `flm:npu` recipe on Linux, it reported `update_required: Backend update is required before use` and the auto-installer raised *"FLM auto-install is only supported on Windows."* I read that and stopped. The actual cause: a **strict-equality version-pin** in `/usr/share/lemonade-server/resources/backend_versions.json` — the manifest pins `flm.npu = v0.9.38`, the Arch AUR package ships `v0.9.39`. Any newer/older patch version flips the recipe to `update_required` even when FLM is fully installed and validates green. One `sed` to bump the pin and the recipe reads `installed`, the same OpenAI-compat API on `:13305` now routes to NPU or iGPU per model, and benches confirm both lanes serve LLMs end-to-end (numbers at the bottom).

If you read the original post and updated your priors based on it: please update them again. The closed-EP critique is fine; the "Linux NPU LLM on Strix Halo doesn't exist" framing is wrong.

Strikethroughs below mark the falsified claims.

---

TITLE OPTIONS (pick one):

A) ~~AMD's Ryzen AI NPU LLM stack is structurally Windows-only. I extracted the 1.7.1 toolkit. Here's why.~~ → AMD's *Ryzen AI 1.7.1* stack is Windows-only — but that's not the only AMD stack, and I missed Lemonade.

B) ~~Receipts: Why FastFlowLM is the only Linux NPU LLM runtime on Strix Halo (the AMD/HF model zoo, the Windows wall, what's actually open)~~ → Receipts retracted: Lemonade + FastFlowLM is the *officially documented* Linux NPU LLM stack, not a community workaround.

C) ~~Linux + Ryzen AI: The model artifacts are public. The runtime is gated by DirectML. Deep dive with extracted DLL list.~~ → Linux + Ryzen AI: the *AWQ/OGA EP* is gated by DirectML. The LLM-on-NPU path on Linux is Lemonade + FLM, fully shipped.

---

SUGGESTED SUBREDDIT: r/LocalLLaMA (cross-post candidates: r/AMD, r/ROCm, r/Amd_Tech)

---

POST BODY:

I have a Strix Halo (Ryzen AI MAX+ 395) box and have been running 1-bit / sub-2-bit LLMs locally. The GPU lane (ROCm + Vulkan llama.cpp) is great. The NPU lane is more interesting — and more frustrating. So I tore apart AMD's Ryzen AI 1.7.1 release to figure out why.

~~**TL;DR:** AMD has 200+ Strix-Halo-targeted LLM checkpoints publicly available on HuggingFace. The runtime to execute them on the NPU is structurally Windows-only — not because of packaging, but because it's built tightly on top of Microsoft DirectML + DirectX 12. A Linux port is research-tier work, not a packaging fix.~~

**TL;DR (corrected):** Ryzen AI 1.7.1's *deployment EP* is structurally Windows-only and built on DirectML. **Lemonade Server is a separate AMD-built MIT stack** with documented Linux NPU support via FastFlowLM. I conflated them. The Linux LLM-on-NPU lane works today via `pacman -S xrt xrt-plugin-amdxdna fastflowlm` + Lemonade Linux's `flm:npu` recipe, with one packaging gotcha (a version-pin equality bug in Lemonade's `backend_versions.json`).

## Same hardware, two parallel universes

| Capability (on the *exact same* Strix Halo box) | Windows | Linux |
|---|---|---|
| ~~AMD official NPU LLM runtime~~ | ~~✅ Bundled (DirectML + Ryzen AI EP)~~ | ~~❌ Doesn't exist~~ |
| ~~Ryzen AI 1.7.1 EP (`onnxruntime_providers_ryzenai`)~~ ~~❌ No Linux `.so`~~ | ✅ Bundled, DirectML-bound | ⚠️ `libonnxruntime_providers_ryzenai.so` *does* ship for Linux — separate bundle (`ryzen_ai-1.7.1.tgz`) at `account.amd.com`, Ubuntu 24.04 / `.deb` only |
| **Lemonade Server (separate AMD/MIT stack)** | **✅ MSI installer** | **✅ AUR/PPA + Linux-native `flm:npu` recipe (officially documented)** |
| ~~Lemonade Server with `ryzenai-llm`~~ | ~~✅ One-click GUI installer~~ | ~~❌ `update_required: Backend update required` — auto-installer "only supported on Windows"~~ |
| Lemonade `ryzenai-llm:npu` recipe | ✅ Windows-only | ❌ Windows-only by design (uses the closed EP) |
| Lemonade `flm:npu` recipe | n/a (Windows uses ryzenai-llm) | ✅ Linux, manual install via `pacman -S fastflowlm` (auto-install is Windows-only by convention) |
| ~~AMD's 200+ NPU LLM checkpoints~~ | ~~✅ Loadable~~ | ~~⚠️  Downloadable, no runtime to execute them~~ |
| AMD's 200+ NPU LLM checkpoints (UINT4-AWQ, OGA hybrid) | ✅ Loadable via Ryzen AI EP | ⚠️ No Linux runtime for the AWQ/OGA artifacts. **FLM ships its own AMD-aligned model collection** (qwen3, gemma3, phi4-mini, lfm2, llama3.2, deepseek-r1, gpt-oss) that runs on Linux NPU. |
| Quark quantizer workflow + conda env | ✅ Documented, working | ❌ READMEs literally say "activate the Ryzen AI 1.7.1 conda environment" — that env is Windows-only |
| `*.xclbin` precompiled NPU binaries | ✅ Loaded by Windows driver | ✅ Loadable via `amdxdna` + `xrt-plugin-amdxdna` |
| ~~LLM-on-NPU lane in practice~~ | ~~AMD official stack~~ | ~~**One third-party project (FastFlowLM)**~~ |
| LLM-on-NPU lane in practice | AMD's closed RyzenAI EP | **Lemonade + FastFlowLM (officially documented, jointly authored)** |
| Diffusion / Whisper / CLIP on NPU | ✅ AMD ships them | ⚠️  Whisper-on-NPU works via FLM; diffusion + CLIP still mostly nothing |

~~That's the gap, on the same chip. Windows gets the complete vertical pipeline. Linux gets the silicon and the artifacts and is told "good luck."~~

The actual gap, accurately scoped: Windows gets AMD's *closed* DirectML-bound EP plus the AWQ/OGA workflow. Linux gets an *open* AMD-built Lemonade stack with FLM that runs LLMs on the NPU. Real remaining gaps on Linux: AWQ/OGA model loading (no `xrt`-based EP), diffusion-on-NPU, Quark→Linux deployment tail. Not "no NPU LLM."

## The good news: artifacts are public

`huggingface.co/amd` has a lot more than I initially thought. Naming convention is the index:

| Suffix | Target |
|---|---|
| `-onnx-ryzen-strix` | Strix Halo NPU+iGPU hybrid |
| `_rai_1.7.x` | Ryzen AI runtime version-pinned (newest format) |
| `-onnx-hybrid` | Same hybrid mode, older naming |
| `-onnx-directml` | Windows DirectML GPU |
| `-onnx-cpu` | CPU |

Architectures shipped for Strix Halo NPU: **Llama 2/3/3.1/3.2 (1B-8B), Mistral 7B, Qwen 1.5/2/2.5, Phi-3 mini, Phi-3.5 mini, Gemma-2 2B, ChatGLM 3 6B, DeepSeek-R1 Distill (Qwen 1.5B/7B, Llama 8B), AMD-OLMo 1B, LFM2 1.2B, CodeLlama 7B, xLAM 2 8B.** All AWQ uint4 g128 with BF16 activations. Built via Quark → OGA Model Builder → NPU-deployment finalization.

There are also two fusion variants per model: **token-fusion** (up to 16K context, slower per-token) and **full-fusion** (4K context cap, fastest per-token). LFM2-1.2B token-fusion is on HF today; full-fusion not yet visible.

## The wall: extracted the actual deployment binaries

`RyzenAI_Deployment.1.7.1.nupkg` (298 MB), pulled from `1.7.1_nuget_signed.zip`:

```
runtimes/win-x64/native/
├── onnxruntime_providers_ryzenai.dll   9.1 MB   ← The Ryzen AI EP itself
├── onnxruntime_vitisai_ep.dll        143 MB   ← VitisAI EP
├── dyn_dispatch_core.dll             145 MB   ← NPU dispatcher
├── dyn_bins.dll                      234 MB   ← Precompiled NPU dispatch graphs
├── onnxruntime.dll                    21 MB
├── onnxruntime-genai.dll             4.5 MB
├── DirectML.dll                       18 MB   ← Windows-only by definition
├── D3D12Core.dll                     3.2 MB   ← DirectX 12 (Windows-only)
├── aiecompiler_client.dll            9.7 MB
├── ryzen_mm.dll, ryzenai_onnx_utils.dll, …
```

**Zero `runtimes/linux-*/`** entries. Zero `.so` files. The .nuspec literally says: *"This package contains native shared library artifacts for AMD RyzenAI."* Singular OS.

This part stands. The DirectML / VitisAI / dyn_dispatch stack is Windows-only. **What was wrong was concluding "therefore Linux has nothing"** — the deployment EP is one of two AMD LLM-on-NPU paths, and I never investigated the second one.

## What IS open on Linux

- The NPU silicon (XDNA 2 / AIE2P) — same hardware
- `xclbin` files (precompiled AIE2P binaries) — architecture-portable
- `amdxdna` kernel module + `xrt-plugin-amdxdna` (Arch package) load xclbins on Linux
- Hardware ID: `PCI\VEN_1022&DEV_17F0` for Strix Halo NPU
- ONNX Runtime base via `pip install onnxruntime`
- AMD's HF model artifacts — downloadable on Linux (no Linux runtime for the AWQ/OGA ones)
- **Lemonade Server (MIT, AMD-built, Linux build available) with the `flm:npu` recipe** — the actual working LLM-on-NPU lane on Linux
- **FastFlowLM (open) — the runtime behind `flm:npu`. Documented at lemonade-server.ai/flm_npu_linux.html, jointly authored with AMD's Lemonade team.**

## What works on Linux today

~~**FastFlowLM** (third-party). Serves UINT4-AWQ q4nx models on the NPU via the Linux `xrt-plugin-amdxdna` path. Coverage: qwen3, gemma3, phi4-mini, lfm2, llama3.2, deepseek-r1, gpt-oss, etc. — overlap with AMD's official catalog but distinct binaries.~~

~~That's the entire Linux NPU LLM lane. One third-party project.~~

**Lemonade + FastFlowLM, jointly documented Linux path** (lemonade-server.ai/flm_npu_linux.html, 2026-03-11). Serves UINT4 q4nx models on the NPU via `xrt-plugin-amdxdna`. Coverage: qwen3, gemma3, phi4-mini, lfm2, llama3.2, deepseek-r1, gpt-oss. Behind Lemonade's OpenAI-compat API, same endpoint as the iGPU `llamacpp:rocm` lane. **One unified API, two compute lanes, both lanes serving LLMs today.**

## ~~The Linux journey is harder than it should be~~ The Linux journey has one packaging bug, otherwise documented end-to-end

~~Working on this on Linux means doing Windows archaeology to figure out what tools to NOT have:~~

- ~~AMD's model READMEs literally say "Activate the Ryzen AI 1.7.1 conda environment"~~ → Still true *for the AWQ/OGA workflow.* Lemonade-on-Linux doesn't need that env at all.
- ~~The Quark quantizer's published example scripts default to `cuda` or `cpu`. The deployment-to-NPU last mile lives in a separate, gated, Windows-only toolchain.~~ → Still true for Quark→deployment-EP. FLM uses its own model collection so this is orthogonal to running LLMs on Linux NPU.
- ~~Lemonade Server's standalone installer is a Windows MSI. The Linux build of Lemonade exists, but the `flm:npu` and `ryzenai-llm` recipes both report `update_required: Backend update is required before use. lemonade backends install flm:npu` — and the auto-installer raises *"FLM auto-install is only supported on Windows"*.~~ → **The `update_required` is a strict-equality version-pin bug** in `/usr/share/lemonade-server/resources/backend_versions.json` — Lemonade pins `flm.npu = v0.9.38`, AUR ships `v0.9.39`, mismatch flips the recipe red. Bumping the pin (one `sed`) flips it green. The auto-installer is Windows-only by convention; Linux uses `pacman -S fastflowlm`, which is the documented path. `ryzenai-llm:npu` is genuinely Windows-only because it uses the closed EP.
- ~~I had to download a 2.6 GB Windows EXE, extract 22 nested CABs, crack a NuGet package, decompile DLL names with `file`, just to confirm there's nothing for Linux inside.~~ → Still true *for confirming the Ryzen AI EP is Windows-only.* Not necessary for the "LLM on NPU on Linux" question, which Lemonade docs answer in two paragraphs.
- The HuggingFace optimum-amd integration documents `BrevitasQuantizer` (sub-INT8 / N-bit) as "Coming soon." Doc has been a stub for a while. ← still true.

## ~~AMD owns every piece of this stack. The gap is a choice.~~ AMD owns every piece. They shipped the open Linux path for LLM-on-NPU and shipped only the closed Windows path for the EP.

This isn't a packaging miss. AMD owns:

- ✅ The silicon (XDNA 2 / AIE2P)
- ✅ The kernel driver (`amdxdna`, in mainline Linux)
- ✅ The userspace driver path (`xrt-plugin-amdxdna`)
- ✅ The NPU dispatch binaries (`*.xclbin`, ship in their own WHQL driver ZIP)
- ✅ The quantizer (Quark)
- ✅ The model conversion pipeline (OGA Model Builder)
- ✅ The model zoo (200+ artifacts at `huggingface.co/amd`)
- ✅ **Lemonade Server (MIT) with documented Linux NPU support via FLM**

~~Missing piece on Linux: a ~9 MB shared object that talks to `xrt` instead of DirectML. AMD wrote the Windows version. They could write the Linux version. They didn't.~~

~~Missing piece on Linux: an `xrt`-based equivalent of `onnxruntime_providers_ryzenai.dll` so that AMD's AWQ/OGA model artifacts can run on the NPU under Linux.~~ → **AMD wrote the Linux version too.** It's `libonnxruntime_providers_ryzenai.so`, ships in `ryzen_ai-1.7.1.tgz` from `account.amd.com`, targets Ubuntu 24.04 / Python 3.12 / `.deb`. AMD's docs at `ryzenai.docs.amd.com/en/latest/llm_linux.html` walk through using it to run their AWQ/OGA HF artifacts on the NPU on Linux today.

~~The remaining gap is *distribution*, not *engineering* — Arch / Fedora / non-Ubuntu Linux users have to debtap, distrobox, or build from a tarball.~~ → **The remaining gap is *kernel ABI*, not *packaging*.** I tested this on Arch via Distrobox: the `.deb` userspace installs cleanly, `xrt-smi examine` sees the NPU through `/dev/accel/accel0`, the basic `VitisAIExecutionProvider` quicktest runs to completion. But the strict LLM-side `RyzenAI` provider rejects with *"Generic Failure"* (no further string) the moment `onnxruntime-genai-ryzenai` tries to load a model. Distrobox passes the device through; it cannot give the container its own kernel module, and AMD's closed EP requires the *AMD-shipped* `amdxdna .ko` (distributed in the same `.deb` bundle) — not whatever your distro's kernel ships in-tree. Ubuntu 24.04 is the only host where AMD's shipped module loads against the running kernel without DKMS source-build hassle.

Compare to vendors who shipped Linux NPU/accelerator runtimes day-one:

- **NVIDIA**: cuDNN, TensorRT, NIM — Linux first, every release
- **Intel**: OpenVINO, NPU runtime — Linux as primary target, parallel Windows
- **AMD's own ROCm**: years of "almost ready" on Linux. Pattern repeating for the Ryzen AI EP. **But Lemonade + FLM broke the pattern for the LLM-on-NPU use case** — that one shipped on Linux in March 2026 with joint docs.

~~FastFlowLM stepping into this gap as a single third-party project keeps the lane open. It also makes AMD look bad. AMD should be embarrassed that their Linux NPU LLM story in 2026 is "go use a community tool because we couldn't be bothered."~~

That paragraph is wrong on every count. AMD's Linux NPU LLM story in 2026 is **lemonade-server.ai/flm_npu_linux.html**, *jointly authored* by AMD's Lemonade team and FastFlowLM, with a `pacman -S` line. Not a "go use a community tool" punt — an actual partnership and shipped doc. The story I should have told: AMD's Lemonade team partnered with FastFlowLM to ship the Linux NPU LLM stack, and the only friction left in 2026 is a one-line version-pin update in the package manifest.

## Open-source theater: gating just enough to tease

Scoping this critique correctly to the *EP/DirectML* stack — load-bearing closed pieces remain closed:

- ✅ `amdxdna` kernel driver — upstream Linux, MIT/GPL
- ✅ AMD Quark quantizer — public PyTorch repo
- ✅ Lemonade Server — MIT, Linux build, **documented Linux NPU LLM path with FLM**
- ✅ HF model artifacts — MIT-licensed, downloadable from anywhere
- ✅ `RyzenAI-SW` repo on github — public, documented
- ✅ Brevitas integration — "Coming soon" in HF docs

Then the load-bearing pieces that are actually closed (still true):

- ❌ `onnxruntime_providers_ryzenai.dll` — closed, Windows-only
- ❌ `onnxruntime_vitisai_ep.dll` — closed, Windows-only (143 MB)
- ❌ `dyn_dispatch_core.dll` + `dyn_bins.dll` — closed, Windows-only (~380 MB)
- ❌ The Ryzen AI 1.7.1 conda environment — Windows-only
- ~~❌ The Lemonade `flm:npu` and `ryzenai-llm` auto-installers — Windows-only~~ → `flm:npu` works on Linux via `pacman` (manual install is the documented path; auto-install being Windows-only is a packaging convention, not a runtime gap). `ryzenai-llm:npu` is still Windows-only because it uses the closed EP.
- ❌ Brevitas integration — perpetually "Coming soon"

~~This is a pattern. **The pieces that don't matter on their own are open. The integration glue that makes them useful is closed.**~~ → True for the AWQ/OGA EP. Not true for Lemonade + FLM, which is integration glue that's *open* on Linux. The "you cannot run them" line in the original was wrong: today on Linux you can `pacman -S fastflowlm` and run LLMs on the NPU through Lemonade.

## The bigger pattern: nobody jumps anymore

The original framing — "AMD's NPU gate is one symptom of a larger industry shift" — is half-right. **The closed RyzenAI EP gate** fits the pattern (closed runtime, Windows-only, conditioned on the higher tier). **The Linux LLM-on-NPU gate** doesn't, because there isn't one — Lemonade + FLM closed it in March.

- **1-bit BitNet** has been a Microsoft research result for *two years*. There is still no production-ready 1-bit runtime on any consumer NPU. Nobody shipped it. ← Still true.
- **Speculative decoding** is mature, papers from 2023, 2-3× speedup proven everywhere. Most consumer inference stacks still don't have one-click support. ← Still true.
- **Heterogeneous compute** (auto-dispatch NPU+iGPU+CPU per workload) is technically a solved problem. ← **Counterpoint:** Lemonade now serves both `llamacpp:rocm` (iGPU) and `flm:npu` (NPU) behind one OpenAI-compat API. That's not auto-dispatch, but it is "one runtime, two backends" on Linux today — closer to the goal than the original post implied existed.
- **Real Linux feature parity** on consumer accelerators is routinely 1-2 years behind Windows. Sometimes never. ← The LLM-on-NPU gap closed in March 2026 with Lemonade + FLM Linux docs. So: less than a year on this one.

The big jumps in local AI right now are happening in research papers and ~~abandoned project READMEs~~ → *and in jointly-authored docs nobody reads*. The Linux developer community is the only place left where someone might ship the ambitious version of something — but that someone (me, this morning) also needs to read the docs before ranting that the docs don't exist.

## Accountability — actually

This post is the public record. Original timestamps stand. New timestamps:

- AMD Ryzen AI 1.7.1 released: March 2026 (closed Windows EP)
- AMD `huggingface.co/amd` org now publishes 200+ NPU model checkpoints
- AMD `huggingface.co/RyzenAI` org publishes Quark recipes
- AMD's deployment NuGet 1.7.1 ships zero Linux artifacts ← still true
- HuggingFace `BrevitasQuantizer` for Ryzen AI: "Coming soon" since at least 2024
- Open issue tracking the Linux gap: https://github.com/huggingface/optimum-amd/issues/178
- AMD's RyzenAI-SW repo: https://github.com/amd/RyzenAI-SW
- **Lemonade Linux NPU support officially documented: lemonade-server.ai/flm_npu_linux.html, dated 2026-03-11, co-authored by Lemonade and FastFlowLM contributors** ← **the document I should have read before writing the original post**
- **AMD's official Linux LLM-on-NPU flow documented: ryzenai.docs.amd.com/en/latest/llm_linux.html, dated 2026-04-19** ← the second document I should have read; runs `huggingface.co/amd/*_rai_1.7.1_npu_*` checkpoints via the closed Linux EP. Auth-walled bundles at `account.amd.com` (`ryzen_ai-1.7.1.tgz`, `RAI_1.7.1_Linux_NPU_XRT.zip`).
- FastFlowLM Arch package: `fastflowlm 0.9.39` in cachyos-extra-znver4
- Fix-the-pin one-liner: `sudo sed -i 's|"npu": "v0.9.38"|"npu": "v0.9.39"|' /usr/share/lemonade-server/resources/backend_versions.json && sudo systemctl restart lemonade-server` (or just restart the lemond process; pin is checked on startup). Bumps the recipe from `update_required` → `installed`. Will be overwritten by future Lemonade pacman updates — re-apply each time.
- AMD Linux EP test result (Distrobox / Ubuntu 24.04 over CachyOS host kernel `7.0.2-1-cachyos`): `xrt-smi examine` ✓, `quicktest.py` (basic ONNX) ✓, `model_benchmark` on Phi-3.5-mini-instruct_rai_1.7.1_npu_4K ✗ (`Generic Failure` during ORT init). Suspected cause: AMD's closed EP requires the AMD-shipped `xrt_plugin-amdxdna 2.21.260102.53.release` `.ko`, not the in-kernel `amdxdna` of recent kernels. Confirms the Ubuntu-24.04 requirement is kernel-ABI-bound, not packaging.

~~Every one of these is on AMD. Linux users have done their part — the kernel driver is upstream, the third-party runtime exists, the artifacts get downloaded, this post got written. The next move is AMD's. There is no version of "we hear you" that substitutes for a Linux `.so`.~~

What I should have written the first time, given the actual state of the world: **AMD shipped the Linux NPU LLM lane via Lemonade + FastFlowLM in March 2026 (jointly documented, AMD's MIT-licensed Lemonade Server is a separate stack from the closed Ryzen AI EP I extracted), and I missed it.** The remaining real gaps — AWQ/OGA EP on Linux, Brevitas integration, BitNet on NPU, diffusion on NPU on Linux — are worth pushing on. They are a smaller and better-defined set than the one I hand-waved at originally.

The next move is mine: read docs before writing posts. **Bad on me. And bad on you for taking it on faith without checking.** I owe you both a corrected post and a sharper bar for the next one.

## Benchmarks (proof the unified Linux API works)

Single Strix Halo box, Lemonade `:13305`, OpenAI-compat. Same prompt for all four ("Write a 200 word paragraph about the history of compilers."), `max_tokens=300`. Captured 2026-04-28 right after applying the version-pin fix.

| Model | Backend | Lane | Decode tok/s | Prefill tok/s | TTFT |
|---|---|---|---|---|---|
| `LFM2-1.2B-GGUF` | `llamacpp:rocm` | iGPU (gfx1151) | **216.3** | 1366.5 | ~negligible |
| `qwen3-0.6b-FLM` | `flm:npu` | NPU (XDNA 2) | **95.4** | 76.0 | 460ms |
| `qwen3-1.7b-FLM` | `flm:npu` | NPU (XDNA 2) | **41.8** | 53.7 | 577ms |
| `deepseek-r1-8b-FLM` | `flm:npu` | NPU (XDNA 2) | **11.3** | 14.7 | 1430ms |

Read: iGPU is faster than NPU on the small models I'd expect to be NPU-favored — that's not the value of the NPU lane. The value is offloading the iGPU (free for ROCm bigger-model serving), lower power for low-touch background inference, and shipping a unified API that lets the *user* pick per-request which silicon answers them. Same `:13305` endpoint, same OpenAI request shape, two different pieces of silicon doing the work depending on the model name.

That's the receipt. **The lane exists, on Linux, on Strix Halo, today, jointly shipped by AMD's Lemonade team and the FastFlowLM team. I missed it. Now I'm not.**

---

Background context: building https://github.com/bong-water-water-bong/1bit-systems — lean install + control plane for 1-bit inference on Strix Halo. GPU lane (ROCm/Vulkan llama.cpp, IQ1_S/TQ2_0) works great. NPU lane works too — just needed the version-pin fix to wire FLM into Lemonade's recipe registry.

Happy to share extraction commands if anyone wants to verify the DLL list themselves. Even happier to be corrected, faster than I corrected myself this time.
