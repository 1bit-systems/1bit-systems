# FAQ

## What is 1bit systems?

1bit systems is a local inference workbench for the Strix Halo box. Apps speak
OpenAI-compatible HTTP to `1bit-proxy` on `:13306`; during the repair phase the
default backend should be a toolbox-backed `llama-server` on `:13305`. Native
Lemonade and FastFlowLM remain the product-direction lanes, but they are not a
finished universal control plane.

GAIA is the intended primary UI/control surface. Open WebUI is a secondary
compatibility UI on `:3000`.

## What does it run on?

The native reference host is AMD Ryzen AI MAX+ 395 / Strix Halo on CachyOS. The
toolbox repair path is meant to make Ubuntu/Fedora useful first by passing GPU
devices into known-good Strix Halo containers.

## What are the current endpoints?

| Use | Base URL |
|---|---|
| GAIA / Lemonade-style clients | `http://127.0.0.1:13306/api/v1` |
| Generic OpenAI-compatible clients | `http://127.0.0.1:13306/v1` |
| Active backend direct | `http://127.0.0.1:13305/v1` |
| Lemonade direct, native path | `http://127.0.0.1:13305/api/v1` or `/v1` |
| FastFlowLM direct NPU lane, optional | `http://127.0.0.1:52625/v1` |
| Open WebUI | `http://127.0.0.1:3000` |

Use `local-no-auth` as the placeholder API key for local clients that require one.

## How do I install it?

On Ubuntu/Fedora, start with the toolbox-backed path:

```bash
toolbox create llama-vulkan-radv \
  --image docker.io/kyuz0/amd-strix-halo-toolboxes:vulkan-radv \
  -- --device /dev/dri --group-add video --security-opt seccomp=unconfined

toolbox enter llama-vulkan-radv
llama-cli --list-devices
llama-server --host 127.0.0.1 --port 13305 \
  -m /path/to/model.gguf -c 8192 -ngl 999 -fa 1 --no-mmap
```

Use `rocm-7.2.2` after `vulkan-radv` works and `/dev/kfd` is available. See
[Toolbox backends](../toolbox-backends.md).

On Arch/CachyOS, the native installer path is:

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems.git
cd 1bit-systems
./install.sh
```

Then verify:

```bash
1bit status
curl -s http://127.0.0.1:13306/v1/models
```

The installer wires Lemonade, FastFlowLM, `1bit-proxy`, Open WebUI, GAIA
helpers, memlock limits, and systemd units for the native path. It does not yet
start and stop toolbox backends as one control plane.

## What clients work with it?

Anything that accepts an OpenAI-compatible `base_url`:

- **GAIA** — primary local agent UI and control surface
- **Open WebUI** — secondary browser UI pointed at `:13306/v1`
- **AnythingLLM, Continue, Dify, n8n, custom SDK clients** — use `:13306/v1`
- **Lemonade SDK / Lemonade-style clients** — use `:13306/api/v1` or Lemonade direct on `:13305/api/v1`
- **Hermes Agent** — external Python client on a laptop or peer, never a strixhalo runtime service

## What are the five rules?

See [Development](./Development.md). Short form: no Python in the core serving
path we own, C++20 for HIP kernels, no hipBLAS in runtime, Rust 1.88+ edition
2024, toolbox-backed llama.cpp first for repair, and FastFlowLM as the intended
NPU lane plus IRON author-time custom kernels.

## Is Python allowed anywhere?

Yes, outside the core runtime. Training, notebooks, build-time DSLs, and caller-side clients are fine. Open WebUI is an isolated secondary UI/client behind the proxy. The serving path itself stays Python-free.

## What is the NPU status?

FastFlowLM can serve supported FLM chat and embedding models on XDNA/XRT when
the native host stack is healthy. It is not the universal bootstrap path.
Custom kernel work uses IRON at author time, lowers through MLIR-AIE and Peano,
emits `xclbin`, and loads from native runtime code through `libxrt`.

## Is stablecoin payment routing part of the current stack?

No. Stablecoin machine-to-machine payment flow is a side project for future API/inference access. It does not replace the current batch-token purchase model in the live inference stack.

## Is it open source?

The public-facing repo and docs live under `bong-water-water-bong/1bit-systems`; kernel work belongs in `rocm-cpp/`. Some launch and operator material may still be private until release.
