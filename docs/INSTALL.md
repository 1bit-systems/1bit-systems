# Install — what AMD says vs what actually works

> Reference doc for the cross-distro / cross-stack picture. If you're on
> Arch / CachyOS, Ubuntu, or Fedora and just want this repo running,
> [`README.md`](../README.md#install) has the current entry point. This file is the "which install route do I
> actually want, and what trade-off am I taking" map.

AMD's docs say "Ubuntu 24.04 only." That's accurate for **one** of two
AMD-built stacks on Strix Halo. We tested all three install routes; they
don't give you the same thing. Pick the one that matches what you need.

## The two stacks

| Stack | Runs | License | Distribution |
|---|---|---|---|
| **Ryzen AI 1.7.1 EP** (closed) | AMD's 200+ AWQ/OGA HF checkpoints (`huggingface.co/amd/*_rai_1.7.1_npu_*`) | Closed | `.deb`, Ubuntu 24.04 only, behind `account.amd.com` login |
| **Open local stack** | GGUF models through llama.cpp, plus FLM/Lemonade where available | Open runtime mix | Native source build on Arch/CachyOS; toolbox bootstrap on Ubuntu/Fedora |

The closed EP runs models the open stack can't. The open stack runs on
distros the closed EP can't. **Neither is a superset.**

## Three install paths

### Path A — Ubuntu 24.04 native AMD EP

AMD's `.deb` bundles install cleanly. The
`xrt_plugin-amdxdna 2.21.260102.53` kernel module DKMS-builds against
Ubuntu's stock kernel. Closed EP runs the AWQ/OGA models.

**Trade-off:** you have to run Ubuntu.

For this repo's open inference path on Ubuntu, use a Strix Halo toolbox
backend and run `1bit-proxy` on the host:

```bash
distrobox create llama-vulkan-radv \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  --additional-flags "--device /dev/dri --group-add video --security-opt seccomp=unconfined"

distrobox enter llama-vulkan-radv
llama-server --host 127.0.0.1 --port 13305 \
  -m /path/to/model.gguf -c 8192 -ngl 999 -fa 1 --no-mmap

# host
node scripts/1bit-proxy.js
```

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

### Path C — Native Arch / CachyOS

Lemonade + FastFlowLM can work directly on the host. No container, no DKMS for
closed-EP modules. The in-tree `amdxdna` (kernel 7+) or `xrt-amdxdna` DKMS
(kernel 6.x) is enough when the running host stack is healthy. This is what
this repo's native installer path builds on pacman hosts
([`README.md` → Install](../README.md#install)). Treat this as the native
target lane, not the universal first bootstrap.

| Lane | Backend | Recipe |
|---|---|---|
| NPU (XDNA 2) | FastFlowLM | `flm:npu` |
| iGPU (gfx1151) — Vulkan | llama.cpp | `llamacpp:vulkan` |
| iGPU (gfx1151) — ROCm | llama.cpp | `llamacpp:rocm` |
| CPU (Zen 5) | llama.cpp | `llamacpp:cpu` |

**Trade-off:** no closed EP. AMD's 200+ AWQ/OGA checkpoints are
downloadable but unrunnable on the NPU. FLM's catalog overlaps on the
popular architectures.

### Path D — Toolbox Ubuntu / Fedora

This is the current out-of-box path for this repo outside Arch/CachyOS. Create
a Strix Halo llama.cpp toolbox, start `llama-server` on `:13305`, then start
`1bit-proxy` on the host. The proxy keeps the same app endpoint:

```text
http://127.0.0.1:13306/v1
```

**Trade-off:** first milestone is GGUF inference through llama.cpp. Lemonade,
FLM, vLLM, and remote workers still need a backend registry before they are one
clean control plane.

## Path A — Ubuntu 24.04 detailed runbook

This is AMD's officially supported flow for running their AWQ/OGA HF
checkpoints (`huggingface.co/amd/*_rai_1.7.1_npu_*`) on the NPU. Source:
[ryzenai.docs.amd.com/en/latest/llm_linux.html](https://ryzenai.docs.amd.com/en/latest/llm_linux.html)
(last updated 2026-04-19). Several earlier steps live behind
`account.amd.com` and are not fully published — those are flagged inline
as `# TODO: verify`. PRs welcome from anyone who has run this end-to-end.

Pin the version once at the top:

```bash
export VERSION=1.7.1
export DKMS_VERSION=2.21.260102.53
```

### A.1 — Prerequisites

```bash
# Ubuntu 24.04 LTS, Python 3.12, kernel >= 6.10
lsb_release -a
uname -r
python3 --version

# Strix Halo NPU PCI ID check (should print 1022:17f0)
lspci -nn | grep -i 'Signal Processing'

# Headers + DKMS toolchain for the kernel module build in A.5
sudo apt update
sudo apt install -y linux-headers-$(uname -r) dkms build-essential git-lfs
git lfs install
```

### A.2 — AMD account + bundle access

The userspace `.deb` bundle and the kernel-module source are gated behind
`account.amd.com`. There is no public mirror.

```bash
# 1. Register / log in at https://account.amd.com
# 2. Navigate to "Ryzen AI Software" downloads (the entitlement is free
#    but requires an account agreement click-through).
# 3. Download both bundles into ~/Downloads:
#      - ryzen_ai-${VERSION}.tgz                # userspace .debs + samples
#      - RAI_${VERSION}_Linux_NPU_XRT.zip       # XRT + amdxdna .deb + DKMS source
# TODO: verify exact navigation path — AMD does not publish a stable
#       deep-link. https://ryzenai.docs.amd.com/en/latest/llm_linux.html
```

### A.3 — Extract bundles

```bash
mkdir -p ~/ryzenai && cd ~/ryzenai
tar -xzf ~/Downloads/ryzen_ai-${VERSION}.tgz
unzip   ~/Downloads/RAI_${VERSION}_Linux_NPU_XRT.zip -d xrt_bundle
ls ryzen_ai-${VERSION}/ xrt_bundle/
```

### A.4 — Userspace `.deb` install (xrt → xrt-plugin-amdxdna → ryzenai)

Order matters — `xrt` provides `/opt/xilinx/xrt/`, the plugin links into
it, the runtime sits on top.

```bash
cd ~/ryzenai/xrt_bundle
sudo apt install -y ./xrt_*-amd64-xrt.deb
sudo apt install -y ./xrt_plugin*-amdxdna.deb
# TODO: verify exact ryzenai .deb name — AMD's bundle layout is not
#       published in the linked doc. Likely:
#         sudo apt install -y ~/ryzenai/ryzen_ai-${VERSION}/ryzenai_${VERSION}_amd64.deb
#       Confirm against the README inside the extracted tarball.
source /opt/xilinx/xrt/setup.sh
```

### A.5 — Kernel module via DKMS (`xrt_plugin-amdxdna` 2.21)

The strict `RyzenAI` EP requires AMD's shipped `amdxdna.ko` — not the
in-tree one. This is the step that fails on Path B.

```bash
# The xrt-plugin .deb drops sources at /usr/src/xrt_plugin-amdxdna-${DKMS_VERSION}
sudo dkms add     -m xrt_plugin-amdxdna -v ${DKMS_VERSION}
sudo dkms build   -m xrt_plugin-amdxdna -v ${DKMS_VERSION}
sudo dkms install -m xrt_plugin-amdxdna -v ${DKMS_VERSION}

# Reload — the in-tree amdxdna must be unloaded first if present
sudo modprobe -r amdxdna 2>/dev/null || true
sudo modprobe amdxdna
dmesg | tail -20 | grep -i amdxdna   # expect firmware-load lines
```

### A.6 — `xrt-smi` sanity check

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
# Expect: Device(s) Present
#         [<bdf>:00.1]  :  RyzenAI-npu5      ; bdf varies by board
#         NPU Firmware Version : <loaded, non-zero>
ls /dev/accel/accel0                                  # device node present
ldconfig -p | grep onnxruntime_providers_ryzenai      # libonnxruntime_providers_ryzenai.so resolvable
```

### A.7 — Python env + `ryzenai_llm` wheel

```bash
mkdir -p ~/ryzenai/run_llm && cd ~/ryzenai/run_llm
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --index-url https://pypi.amd.com/ryzenai_llm/${VERSION}/linux/simple/ ryzenai_llm
pip install onnxruntime-genai-ryzenai
# TODO: verify exact extra-index needs — AMD's pypi server may require
#       --extra-index-url https://pypi.org/simple to resolve transitive deps.
```

### A.8 — Pull and run an AWQ/OGA model

```bash
cd ~/ryzenai/run_llm
git clone https://huggingface.co/amd/Phi-3.5-mini-instruct_rai_${VERSION}_npu_4K

# Copy deployment helpers + benchmark binary out of the bundle
cp -r ~/ryzenai/ryzen_ai-${VERSION}/deployment/* .
cp    ~/ryzenai/ryzen_ai-${VERSION}/samples/model_benchmark .
cp    ~/ryzenai/ryzen_ai-${VERSION}/samples/amd_genai_prompt.txt .

cat > xrt.ini <<'EOF'
[Debug]
num_heap_pages = 8
EOF
export XRT_INI_PATH=$PWD/xrt.ini
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$PWD:$LD_LIBRARY_PATH
```

### A.9 — Bench (this is the call that fails on Path B)

```bash
./model_benchmark \
  -i Phi-3.5-mini-instruct_rai_${VERSION}_npu_4K/ \
  -l 128 \
  -f amd_genai_prompt.txt
# Expect (per AMD docs):
#   Prompt:     ~864 tok/s
#   Generation: ~17.6 tok/s
#   Peak mem:   ~12 GB
# Add -v for verbose ORT init; success here means
# libonnxruntime_providers_ryzenai.so bound to the AMD-shipped
# amdxdna.ko cleanly. On Path B this returns "Generic Failure" during
# ORT init.
```

### A.10 — Optional: Lemonade + FLM alongside

The open lane runs on the same host without conflict — different device
handle path, different recipe.

```bash
# Lemonade Server install on Ubuntu — see lemonade-server.ai/install_options.html
# TODO: verify exact PPA / tarball / install.sh URL for current release.
curl -fsSL https://lemonade-server.ai/install.sh | bash

# FastFlowLM on Ubuntu — see lemonade-server.ai/flm_npu_linux.html
# TODO: verify exact apt/source path; FLM ships a tarball on Linux
#       (no AUR equivalent on Ubuntu).
systemctl --user start lemonade-server   # serves :13305
```

## Path B — Distrobox detailed runbook

You're on a non-Ubuntu host (Arch / CachyOS / Fedora / etc.) with kernel
6.10+ and want to try AMD's closed Ryzen AI EP without reinstalling the
OS. Distrobox is the obvious move. **It works for everything except the
LLM workload.** Here's what passes, what fails, and why the failure
isn't yours to fix.

### B.1 — Prerequisites (host)

The host needs the in-tree `amdxdna` driver (kernel 7+) or the
`xrt-amdxdna` DKMS package (kernel 6.x). The NPU device must be exposed
at `/dev/accel/accel0` before you create the container.

```bash
ls -l /dev/accel/accel0
lspci -nn | grep -i '1022:17f0'   # Strix Halo NPU PCI ID
```

You also need `distrobox` and `podman` (or `docker`).

### B.2 — Create the Ubuntu 24.04 container with NPU passthrough

```bash
distrobox-create \
  --name rai-ubuntu \
  --image ubuntu:24.04 \
  --additional-flags "--device=/dev/accel/accel0 --device=/dev/dri" \
  --volume /dev/accel:/dev/accel
distrobox-enter rai-ubuntu
```

### B.3 — Install AMD's `.deb` userspace bundle inside the container

Same downloads as Path A. Follow Path A steps A.2 through A.4 from
inside the container shell. Python 3.12 is already the Ubuntu 24.04
default.

The `xrt_plugin-amdxdna` DKMS build *will* fail — there are no kernel
headers for your host kernel inside the container, and even if there
were, the container can't load modules. **Ignore the DKMS error.** The
userspace half is what we need; the kernel module half is the wall (see
B.5).

### B.4 — Sanity check — the part that works

```bash
xrt-smi examine          # NPU detected, firmware version reported
python quicktest.py      # AMD's basic ONNX test via VitisAIExecutionProvider
# Expected: "Test Finished"
```

This is the working ceiling of Path B. Basic ONNX inference through
`VitisAIExecutionProvider` runs end-to-end on the NPU.

### B.5 — The failure mode

Try the actual LLM flow and you get this — verbatim, no further string,
every time:

```bash
./model_benchmark \
  -i Phi-3.5-mini-instruct_rai_${VERSION}_npu_4K/ \
  -l 128 \
  -f amd_genai_prompt.txt
```

```
[E:onnxruntime:, inference_session.cc:2544 operator()] Exception during initialization: Generic Failure
```

Closed `.so`, no diagnostic. `quicktest.py` passes; `model_benchmark`
and any `onnxruntime-genai-ryzenai` LLM init fails at session
construction.

### B.6 — Why (root cause)

The strict `RyzenAI` execution provider used for LLMs probes the loaded
`amdxdna` module for symbols/ioctls that AMD's shipped module
(`xrt_plugin-amdxdna 2.21.260102.53.release`) exports, while the in-tree
`amdxdna` from kernel 7+ exposes a compatible-but-not-identical surface
— enough for `xrt-smi` and `VitisAIExecutionProvider`, not enough for
the strict EP's check. Distrobox passes the device node through, but a
container fundamentally cannot ship its own kernel module to the host.
You're stuck with whatever `.ko` the host loaded at boot, and that's
not the AMD-shipped one. Kernel ABI gate, dressed up as a packaging
restriction.

### B.7 — What to do when you hit it

This is a wall, not a misconfiguration. Two real options:

- **Path A** — boot Ubuntu 24.04 natively. AMD's `xrt_plugin-amdxdna`
  DKMS-builds against Ubuntu's stock kernel, the strict EP loads, the
  AWQ/OGA models run.
- **Path C** — pivot to Lemonade + FastFlowLM. Doesn't need the closed
  EP, runs on whatever in-tree `amdxdna` your distro ships, and is what
  this repo installs ([README → Install](../README.md#install-arch--cachyos)).

### B.8 — What still works in this container

Don't tear it down. The container is still useful for:

- `xrt-smi examine` / `xrt-smi validate` — NPU diagnostics from a
  known-good userspace
- `VitisAIExecutionProvider` — basic ONNX inference on the NPU
- ONNX model conversion / debugging against AMD's reference toolchain
- AMD's `quicktest.py` and any non-LLM Ryzen AI sample

Just not the closed-EP LLM flow. Treat the distrobox as a Ryzen AI
userspace sandbox; keep production LLM work on Path A or Path C.

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
