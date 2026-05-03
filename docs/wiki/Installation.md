# Installation

This page describes the current Lemonade + FastFlowLM + `1bit-proxy` stack. Older install notes that mention `1bit-server :8180`, `1bit-lemonade :8200`, `halo doctor`, or `install-strixhalo.sh` are obsolete.

## Reference Host

- AMD Ryzen AI MAX+ 395 / Strix Halo
- CachyOS on the running kernel
- XDNA NPU visible as `/dev/accel/accel0`
- XRT reports `RyzenAI-npu5`
- Lemonade on `:13305`
- FastFlowLM on `:52625`
- `1bit-proxy` on `:13306`
- Open WebUI on `:3000`
- GAIA as the primary UI/control surface

The live CachyOS host uses the in-tree `amdxdna` driver. DKMS packages may exist for other kernels, but they are not the live path unless the running kernel actually loads them.

## Install

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems.git
cd 1bit-systems
./install.sh
```

The installer is idempotent. It installs the local `1bit` CLI, service files, Open WebUI wiring, GAIA helpers, memlock limits, and default configuration. Lemonade and FastFlowLM are built from the maintained fork sources under `/opt/1bit` when needed.

On first install, log out and back in, or reboot, so memlock limits apply to the NPU lane.

## Verify

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
curl -s http://127.0.0.1:13305/api/v1/models
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
| Lemonade | `http://127.0.0.1:13305/api/v1` or `/v1` | Canonical multimodal and OmniRouter server |
| FastFlowLM | `http://127.0.0.1:52625/v1` | XDNA NPU chat, embeddings, opt-in ASR |
| Open WebUI | `http://127.0.0.1:3000` | Secondary browser UI |

## Start And Stop

```bash
1bit up
1bit status
1bit gaia status
1bit webui status
1bit down
```

`1bit up` starts Lemonade, FastFlowLM, `1bit-proxy`, Open WebUI, and GAIA helpers. `1bit down` stops the local stack in reverse order.

## Client Setup

Use `1bit-proxy` unless you intentionally need a direct backend:

```text
GAIA / Lemonade-style clients:  http://127.0.0.1:13306/api/v1
Most OpenAI clients:            http://127.0.0.1:13306/v1
Lemonade direct:                http://127.0.0.1:13305/api/v1
FastFlowLM direct:              http://127.0.0.1:52625/v1
Open WebUI:                     http://127.0.0.1:3000
```

The proxy keeps Lemonade as the default route and sends targeted FLM model families to FastFlowLM.

## Rule A Reminder

Core serving stays Python-free. Training, notebooks, build-time tools, caller-side clients, and compatibility UIs are allowed only outside the native serving path. Open WebUI is an isolated secondary UI/client behind the proxy; it is not the core engine.

## NPU Path

The live NPU serving lane is FastFlowLM on XDNA/XRT. Custom NPU kernel authoring is IRON Python DSL at author time, then MLIR-AIE, Peano, `xclbin`, and native `libxrt` runtime dispatch.

## See Also

- [Development](./Development.md)
- [Clients](./Clients.md)
- [Lemonade compatibility](./Lemonade-Compat.md)
- [NPU status](./Why-No-NPU-Yet.md)
