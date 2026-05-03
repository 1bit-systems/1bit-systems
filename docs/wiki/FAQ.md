# FAQ

## What is 1bit systems?

1bit systems is a local inference and app-control stack for the Strix Halo box. Apps speak OpenAI-compatible HTTP to `1bit-proxy` on `:13306`; the proxy keeps Lemonade as the default multimodal/OmniRouter route on `:13305` and routes targeted FLM model families to FastFlowLM on `:52625` for XDNA NPU chat and embeddings.

GAIA is the primary UI/control surface. Open WebUI is a secondary compatibility UI on `:3000`.

## What does it run on?

The reference host is AMD Ryzen AI MAX+ 395 / Strix Halo on CachyOS. The current NPU device is `/dev/accel/accel0`, and XRT reports `RyzenAI-npu5` through the in-tree `amdxdna` kernel driver.

## What are the current endpoints?

| Use | Base URL |
|---|---|
| GAIA / Lemonade-style clients | `http://127.0.0.1:13306/api/v1` |
| Generic OpenAI-compatible clients | `http://127.0.0.1:13306/v1` |
| Lemonade direct | `http://127.0.0.1:13305/api/v1` or `/v1` |
| FastFlowLM direct NPU lane | `http://127.0.0.1:52625/v1` |
| Open WebUI | `http://127.0.0.1:3000` |

Use `local-no-auth` as the placeholder API key for local clients that require one.

## How do I install it?

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

The installer wires Lemonade, FastFlowLM, `1bit-proxy`, Open WebUI, GAIA helpers, memlock limits, and systemd units. On first install, log out and back in, or reboot, so memlock limits apply to the NPU lane.

## What clients work with it?

Anything that accepts an OpenAI-compatible `base_url`:

- **GAIA** — primary local agent UI and control surface
- **Open WebUI** — secondary browser UI pointed at `:13306/v1`
- **AnythingLLM, Continue, Dify, n8n, custom SDK clients** — use `:13306/v1`
- **Lemonade SDK / Lemonade-style clients** — use `:13306/api/v1` or Lemonade direct on `:13305/api/v1`
- **Hermes Agent** — external Python client on a laptop or peer, never a strixhalo runtime service

## What are the five rules?

See [Development](./Development.md). Short form: no Python in the core serving path, C++20 for HIP kernels, no hipBLAS in runtime, Rust 1.88+ edition 2024, and NPU is FastFlowLM live serving plus IRON author-time custom kernels.

## Is Python allowed anywhere?

Yes, outside the core runtime. Training, notebooks, build-time DSLs, and caller-side clients are fine. Open WebUI is an isolated secondary UI/client behind the proxy. The serving path itself stays Python-free.

## What is the NPU status?

FastFlowLM is live on XDNA/XRT for supported FLM chat and embedding models. Custom kernel work uses IRON at author time, lowers through MLIR-AIE and Peano, emits `xclbin`, and loads from native runtime code through `libxrt`.

## Is stablecoin payment routing part of the current stack?

No. Stablecoin machine-to-machine payment flow is a side project for future API/inference access. It does not replace the current batch-token purchase model in the live inference stack.

## Is it open source?

The public-facing repo and docs live under `bong-water-water-bong/1bit-systems`; kernel work belongs in `rocm-cpp/`. Some launch and operator material may still be private until release.
