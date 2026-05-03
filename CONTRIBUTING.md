# Contributing To 1bit-systems

`1bit-systems` is the install, control, docs, packaging, and glue layer for a local AMD AI stack:

```text
GAIA -> 1bit-proxy :13306/v1 -> Lemonade :13305/v1
                              -> FastFlowLM :52625/v1
Open WebUI :3000 -> 1bit-proxy :13306/v1
systemd -> 1bit-stack.target
```

## How To Help

- File issues with reproducible commands and `1bit status` output.
- Test GAIA, Open WebUI, and other OpenAI-compatible clients against `http://127.0.0.1:13306/v1`.
- Run `1bit bench` or the benchmark scripts under `benchmarks/` on real AMD hardware and attach the raw output.
- Audit docs and website copy for stale endpoint, package, or architecture claims.
- Improve installer, systemd, proxy routing, packaging, and GAIA integration.

## Code Style

- Keep the repo focused on orchestration. Do not reimplement Lemonade, FastFlowLM, or GAIA internals here.
- Prefer existing shell/Node/Python helper style over adding a new framework.
- Preserve the OpenAI-compatible API shape.
- Treat Lemonade `:13305` as canonical for multimodal and OmniRouter behavior.
- Treat `1bit-proxy :13306` as a convenience union endpoint, not a replacement for Lemonade.
- Keep FLM pinned to `:52625` for stack parity unless there is a documented reason to change it.
- Keep GAIA as the primary UI/control surface and Open WebUI as secondary.

Use Conventional Commits (`feat:`, `fix:`, `docs:`, etc.) when practical.

## Tests And Checks

Run the narrow checks for the files you changed. Common checks:

```sh
bash -n install.sh scripts/1bit benchmarks/bench-npu-ioctl-budget.sh
node --check scripts/1bit-proxy.js
python3 -m py_compile scripts/1bit-omni.py
1bit status
```

For public website changes, preview locally:

```sh
cd 1bit-site
python3 -m http.server 8765
```

Cloudflare Pages deployment is documented in `1bit-site/README.md`.

## Upstream Projects

This stack stands on:

- AMD GAIA: https://amd-gaia.ai/docs/quickstart
- Lemonade Server: https://lemonade-server.ai/docs/
- Lemonade OmniRouter: https://lemonade-server.ai/docs/omni-router/
- FastFlowLM: https://fastflowlm.com/docs/
- llama.cpp / GGUF ecosystem for iGPU GGUF benchmarking and model work.
- ROCm, XRT, and `amdxdna` for AMD GPU/NPU runtime support.
