# Toolbox Backends

`1bit-systems` does not yet have the single control plane it wants. The current
repo has useful pieces:

- `1bit-proxy` as an OpenAI-compatible union endpoint
- a local `1bit` CLI
- docs and benchmark history
- systemd glue for an older native Lemonade/FastFlowLM layout

The brittle part is backend installation. The native `install.sh` path is still
CachyOS / Arch first and assumes `pacman`, source-built Lemonade, source-built
FastFlowLM, and host GPU/NPU device access. On Ubuntu and Fedora, use
toolbox-backed backends first, then put the 1bit control plane in front of
those backends.

## Recommended Shape

Run a known-good Strix Halo toolbox as the inference backend, then point
`1bit-proxy` at its OpenAI-compatible server:

```text
Apps / SDKs
  -> 1bit-proxy :13306/v1
       -> llama.cpp toolbox server :13305/v1
       -> optional FLM service :52625/v1
```

For the first repair pass, the toolbox server replaces Lemonade as the default
OpenAI backend on Ubuntu/Fedora. That gives the project a working inference
engine again while the real control plane is rebuilt.

## Repo Bootstrap

On Ubuntu/Fedora, run the normal installer. It detects that `pacman` is absent
and switches to the toolbox path:

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh
```

That path installs:

- host packages: `podman`, `nodejs`, `curl`, and `distrobox` or `toolbox`
- `video`/`render` group membership when those groups exist
- `1bit` CLI at `/usr/local/bin/1bit`
- `1bit-proxy.js` and companion assets under `/usr/local/share/1bit-systems`
- toolbox defaults at `~/.config/1bit-systems/toolbox.env`

Then create and start the backend:

```bash
1bit doctor
1bit toolbox bootstrap
ONEBIT_TOOLBOX_MODEL=/path/to/model.gguf 1bit toolbox up
1bit up
```

`1bit up` automatically uses the toolbox path on non-pacman hosts. Set
`ONEBIT_BACKEND=toolbox` to force that behavior on any distro.

If you want the installer to pull/create the toolbox immediately:

```bash
ONEBIT_TOOLBOX_AUTOCREATE=1 ./install.sh
```

It still does not download a model. Set `ONEBIT_TOOLBOX_MODEL` to a local GGUF
path before `1bit toolbox up`.

## Host Requirements

The host must expose GPU devices to containers:

```bash
ls -ld /dev/dri /dev/kfd
groups
```

The user should be in `video` and `render`. On Ubuntu:

```bash
sudo usermod -aG video,render "$USER"
```

Log out and back in after changing groups.

If the device permissions are still wrong on Ubuntu, community toolbox guides
use udev rules like:

```bash
printf '%s\n' \
  'SUBSYSTEM=="kfd", KERNEL=="kfd", MODE="0666"' \
  'SUBSYSTEM=="drm", KERNEL=="renderD*", MODE="0666"' |
  sudo tee /etc/udev/rules.d/70-kfd.rules

sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Llama.cpp Toolbox

Community-maintained Strix Halo llama.cpp toolboxes:

```text
https://github.com/kyuz0/amd-strix-halo-toolboxes
```

For compatibility, start with RADV/Vulkan. For performance, test ROCm after the
host devices and kernel are stable.

Fedora toolbox example, equivalent to what `1bit toolbox bootstrap` creates:

```bash
toolbox create llama-vulkan-radv \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  -- --device /dev/dri --group-add video --security-opt seccomp=unconfined

toolbox enter llama-vulkan-radv
```

Ubuntu Distrobox example, equivalent to what `1bit toolbox bootstrap` creates:

```bash
distrobox create llama-vulkan-radv \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  --additional-flags "--device /dev/dri --group-add video --security-opt seccomp=unconfined"

distrobox enter llama-vulkan-radv
```

Inside the toolbox:

```bash
llama-cli --list-devices
llama-server \
  --host 127.0.0.1 \
  --port 13305 \
  -m /path/to/model.gguf \
  -c 8192 \
  -ngl 999 \
  -fa 1 \
  --no-mmap
```

The `--host 127.0.0.1 --port 13305` choice deliberately matches
`1bit-proxy`'s default `LEMOND_URL=http://127.0.0.1:13305`. The process is not
Lemonade, but it speaks the same OpenAI `/v1` shape that the proxy needs.

Then, on the host, either use the CLI:

```bash
ONEBIT_TOOLBOX_MODEL=/path/to/model.gguf 1bit toolbox up
1bit up
curl -s http://127.0.0.1:13306/v1/models
```

Or run the proxy directly:

```bash
node scripts/1bit-proxy.js
curl -s http://127.0.0.1:13306/v1/models
```

Use this as the app base URL:

```text
http://127.0.0.1:13306/v1
```

## ROCm Toolbox

After RADV/Vulkan is stable, test the ROCm toolbox:

```bash
toolbox create llama-rocm \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.2 \
  -- --device /dev/dri --device /dev/kfd \
     --group-add video --group-add render --group-add sudo \
     --security-opt seccomp=unconfined
```

Use the same `llama-server --host 127.0.0.1 --port 13305 ...` shape.

## vLLM Toolbox

High-throughput OpenAI serving belongs in the vLLM lane, not the default
first-run lane:

```text
https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes
```

The upstream toolbox image is:

```text
docker.io/kyuz0/vllm-therock-gfx1151:stable
```

Use vLLM when batching and concurrent request throughput matter more than the
lowest-friction GGUF path. The toolbox exposes an OpenAI-compatible endpoint on
`:8000` after `start-vllm` launches a model. Until `1bit-proxy` has a backend
registry, use it as the Lemonade-compatible lane:

```bash
LEMOND_URL=http://127.0.0.1:8000 node scripts/1bit-proxy.js
curl -s http://127.0.0.1:13306/v1/models
```

CLI helper:

```bash
1bit toolbox commands vllm
```

## ComfyUI Toolbox

Image and video generation belongs in a separate lane:

```text
https://github.com/kyuz0/amd-strix-halo-comfyui-toolboxes
```

The upstream toolbox image is:

```text
docker.io/kyuz0/amd-strix-halo-comfyui:latest
```

The ComfyUI toolbox provides workflow presets and helper tooling such as
`/opt/set_extra_paths.sh`, `model_manager`, and the `start_comfy_ui` alias. Keep
this out of the default `1bit up` path for now; the first working product
surface should be text inference on the OpenAI endpoint.

CLI helper:

```bash
1bit toolbox commands comfyui
```

## Fine-Tuning Toolbox

Training belongs in a third lane:

```text
https://github.com/kyuz0/amd-strix-halo-llm-finetuning
```

The upstream toolbox image is:

```text
docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest
```

This lane is a Jupyter workspace for ROCm fine-tuning. It should produce
adapters, checkpoints, or converted GGUF artifacts that later become inference
models. It should not block the out-of-box inference bootstrap.

CLI helper:

```bash
1bit toolbox commands finetune
```

## What This Fixes

This does not magically create the final 1bit control plane. It fixes the
current broken bootstrap path:

- no host source build required before first inference
- no Arch-only package assumption for Ubuntu/Fedora users
- known-good Strix Halo containers own ROCm/Vulkan dependency churn
- `1bit-proxy` remains the stable endpoint for apps

## Control Plane Work Left

The real control plane still needs to be implemented:

- backend registry for native, toolbox, and remote backends
- service lifecycle that can start/stop toolbox-backed servers
- health checks and model discovery per backend
- explicit routing policy instead of hard-coded Lemonade/FLM assumptions
- one status surface for GAIA, Open WebUI, llama.cpp, FLM, and future vLLM

Until that exists, treat toolboxes as the backend runtime and `1bit-proxy` as
the thin compatibility surface.
