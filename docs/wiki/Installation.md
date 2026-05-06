# Installation

This page describes the current repair path. Older install notes that mention `1bit-server :8180`, `1bit-lemonade :8200`, `halo doctor`, or `install-strixhalo.sh` are obsolete. The finished single control plane is not shipping yet.

## Recommended First Path

On Ubuntu/Fedora, start with a Strix Halo toolbox backend and put
`1bit-proxy` in front of it. Use `vulkan-radv` first for compatibility, then
test `rocm-7.2.2` after GPU device access is verified.

```bash
toolbox create llama-vulkan-radv \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  -- --device /dev/dri --group-add video --security-opt seccomp=unconfined

toolbox enter llama-vulkan-radv
llama-cli --list-devices
llama-server --host 127.0.0.1 --port 13305 \
  -m /path/to/model.gguf -c 8192 -ngl 999 -fa 1 --no-mmap
```

Ubuntu users can use `distrobox` with the same image and device flags. For ROCm,
pass both `/dev/dri` and `/dev/kfd` and use
`docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-7.2.2`.

On the host:

```bash
node scripts/1bit-proxy.js
curl -s http://127.0.0.1:13306/v1/models
```

Use `http://127.0.0.1:13306/v1` as the app base URL.

## Native Reference Host

- AMD Ryzen AI MAX+ 395 / Strix Halo
- CachyOS on the running kernel
- XDNA NPU visible as `/dev/accel/accel0`
- XRT reports `RyzenAI-npu5`
- Lemonade on `:13305`
- FastFlowLM on `:52625`
- `1bit-proxy` on `:13306`
- Open WebUI on `:3000`
- GAIA as the primary UI/control surface

The native CachyOS host uses the in-tree `amdxdna` driver. DKMS packages may
exist for other kernels, but they are not active unless the running kernel
actually loads them.

## Native Install

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems.git
cd 1bit-systems
./install.sh
```

The native installer is Arch/CachyOS-first. It installs the local `1bit` CLI,
service files, Open WebUI wiring, GAIA helpers, memlock limits, and default
configuration. Lemonade and FastFlowLM are built from the maintained fork
sources under `/opt/1bit` when needed.

On first install, log out and back in, or reboot, so memlock limits apply to the NPU lane.

## Verify

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
curl -s http://127.0.0.1:13305/v1/models
curl -s http://127.0.0.1:52625/v1/models
```

For clients that require a key, use:

```text
local-no-auth
```

## Endpoints

| Service | URL | Role |
|---|---|---|
| `1bit-proxy` | `http://127.0.0.1:13306/v1` | Generic OpenAI-compatible union endpoint |
| `1bit-proxy` | `http://127.0.0.1:13306/api/v1` | GAIA / Lemonade-style union endpoint |
| Active backend | `http://127.0.0.1:13305/v1` | Toolbox `llama-server` first; Lemonade on native path |
| Lemonade | `http://127.0.0.1:13305/api/v1` or `/v1` | Native multimodal and OmniRouter lane when installed |
| FastFlowLM | `http://127.0.0.1:52625/v1` | Optional XDNA NPU chat, embeddings, opt-in ASR |
| Open WebUI | `http://127.0.0.1:3000` | Secondary browser UI |

## Start And Stop

```bash
1bit up
1bit status
1bit gaia status
1bit webui status
1bit down
```

`1bit up` starts Lemonade, FastFlowLM, `1bit-proxy`, Open WebUI, and GAIA
helpers on the native path. It does not yet manage toolbox-backed
`llama-server` lifecycle; that belongs in the backend registry/control-plane
milestone.

## Client Setup

Use `1bit-proxy` unless you intentionally need a direct backend:

```text
GAIA / Lemonade-style clients:  http://127.0.0.1:13306/api/v1
Most OpenAI clients:            http://127.0.0.1:13306/v1
Active backend direct:          http://127.0.0.1:13305/v1
Lemonade direct, native path:   http://127.0.0.1:13305/api/v1
FastFlowLM direct, optional:    http://127.0.0.1:52625/v1
Open WebUI:                     http://127.0.0.1:3000
```

The proxy keeps the app base URL stable while the default backend changes from
toolbox `llama-server` to native Lemonade. Targeted FLM routing is optional and
requires a healthy FastFlowLM service.

## Rule A Reminder

Core serving stays Python-free. Training, notebooks, build-time tools, caller-side clients, and compatibility UIs are allowed only outside the native serving path. Open WebUI is an isolated secondary UI/client behind the proxy; it is not the core engine.

## NPU Path

FastFlowLM is the intended NPU serving lane on XDNA/XRT when the host stack is
healthy. Custom NPU kernel authoring is IRON Python DSL at author time, then
MLIR-AIE, Peano, `xclbin`, and native `libxrt` runtime dispatch.

## See Also

- [Development](./Development.md)
- [Clients](./Clients.md)
- [Lemonade compatibility](./Lemonade-Compat.md)
- [NPU status](./Why-No-NPU-Yet.md)
- [Toolbox backends](../toolbox-backends.md)
