<div align="center">

<img src="1bit-site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# Local inference, wired for Strix Halo.

### One OpenAI-compatible endpoint while the control plane is rebuilt.

`1bit.systems` is a toolbox-first local inference workbench for AMD Strix Halo. Apps connect over standard OpenAI-compatible base URLs. The stable surface is the union endpoint; the finished single control plane comes second.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Union endpoint](https://img.shields.io/badge/endpoint-:13306%2Fv1-00ff00.svg)](#connect-apps)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![Control plane](https://img.shields.io/badge/control%20plane-rebuilding-f00fd2.svg)](docs/control-plane-roadmap.md)
[![GitHub last commit](https://img.shields.io/github/last-commit/bong-water-water-bong/1bit-systems)](https://github.com/bong-water-water-bong/1bit-systems/commits/main)

</div>

---

## Routing Surface

| Lane | Local URL | Role |
|---|---:|---|
| Union endpoint | `http://127.0.0.1:13306/v1` | OpenAI-compatible app surface for clients that want one base URL |
| Backend | `http://127.0.0.1:13305/v1` | toolbox llama.cpp or Lemonade-compatible inference lane |
| NPU lane | `http://127.0.0.1:52625/v1` | optional FastFlowLM XDNA path |
| Web UI | `http://127.0.0.1:3000` | secondary browser client |
| Control | `1bit` | CLI, lifecycle, repair checks, and service wiring |

```sh
curl http://127.0.0.1:13306/v1/models
```

Apps should connect to `1bit.systems` the same way they connect to Lemonade, llama.cpp, vLLM, or any OpenAI-compatible local server: set a base URL, set any placeholder API key if the client requires one, then send normal chat, embeddings, image, audio, or tool-calling requests.

The intended single control plane is not finished yet. Today, `1bit-proxy` is the useful stable surface, the native installer is Arch/CachyOS-first, and toolbox-backed Strix Halo runtimes are the pragmatic way to get Ubuntu/Fedora systems serving again. See [`docs/control-plane-roadmap.md`](docs/control-plane-roadmap.md) and [`docs/toolbox-backends.md`](docs/toolbox-backends.md).

The control plane is the second layer: `1bit` should start and check the services, GAIA provides the primary agent/UI surface, Open WebUI is secondary, and systemd or toolbox lifecycle keeps the stack alive. Lemonade can be the canonical multimodal and OmniRouter inference server, FastFlowLM is the XDNA NPU runtime, and `1bit-proxy` is the union endpoint for clients that want multiple lanes behind one base URL.

## Current Shape

| Layer | Role | Default |
|---|---|---|
| GAIA Agent UI | Primary agent UI and control surface | AppImage + `~/.gaia/venv/bin/gaia` |
| Lemonade Server | Canonical OpenAI-compatible multimodal API and OmniRouter surface | `http://127.0.0.1:13305/api/v1` or `/v1` |
| FastFlowLM | XDNA NPU runtime for FLM models, embeddings, and optional ASR | `http://127.0.0.1:52625/v1` |
| 1bit proxy | Convenience union endpoint for OpenAI clients | `http://127.0.0.1:13306/api/v1` or `/v1` |
| Open WebUI | Secondary browser UI | `http://127.0.0.1:3000` |
| systemd | Local stack lifecycle | `1bit-stack.target` |

Lemonade remains the source of truth for multimodal OpenAI compatibility and OmniRouter-style tool routing. The proxy does not replace Lemonade; it keeps Lemonade defaults and adds targeted side-lanes for FastFlowLM models such as NPU chat, embeddings, and opt-in ASR.

## Connect Apps

Use `1bit-systems` as an inference engine by pointing apps at an OpenAI-compatible base URL.

```text
Recommended union endpoint:
  OpenAI-style clients: http://127.0.0.1:13306/v1
  GAIA/Lemonade-style:  http://127.0.0.1:13306/api/v1

Lemonade direct:
  OpenAI-style clients: http://127.0.0.1:13305/v1
  Lemonade-style:       http://127.0.0.1:13305/api/v1

FastFlowLM direct:
  http://127.0.0.1:52625/v1
```

Client examples:

| App type | Base URL |
|---|---|
| GAIA Agent UI / GAIA CLI | `http://127.0.0.1:13306/api/v1` |
| Open WebUI | `http://127.0.0.1:13306/v1` |
| AnythingLLM, Continue, Dify, n8n, custom OpenAI SDK clients | `http://127.0.0.1:13306/v1` |
| Pure Lemonade multimodal / OmniRouter work | `http://127.0.0.1:13305/api/v1` |

If a client asks for an API key, use a local placeholder such as `local-no-auth` unless you explicitly configured Lemonade authentication.

Minimal SDK test:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:13306/v1",
    api_key="local-no-auth",
)

print(client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Say stack OK in five words."}],
    max_tokens=20,
).choices[0].message.content)
```

## Install

On Arch/CachyOS, run the native installer from the repo root:

```sh
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh
```

The installer is idempotent. It installs the local control CLI, systemd units, Open WebUI wiring, memlock limits, and default service configuration. Lemonade and FastFlowLM are built from the maintained fork sources under `/opt/1bit`; legacy upstream application packages are removed so the local stack uses the forked code paths. On first install, log out and back in, or reboot, so memlock limits apply to the NPU lane.

Install behavior is distro-aware:

| Host | Installer behavior |
|---|---|
| CachyOS / Arch | Native source build for Lemonade + FastFlowLM, systemd stack, Open WebUI wiring |
| Ubuntu / Fedora | CLI/proxy install plus toolbox-backed llama.cpp bootstrap path |

Ubuntu/Fedora quick path, using the toolbox-backed repair lane:

```sh
./install.sh
1bit doctor
1bit toolbox bootstrap
ONEBIT_TOOLBOX_MODEL=/path/to/model.gguf 1bit toolbox up
1bit up
```

`ONEBIT_TOOLBOX_AUTOCREATE=1 ./install.sh` also creates the toolbox during install. It still cannot guess which GGUF you want, so set `ONEBIT_TOOLBOX_MODEL` before starting inference. See [`docs/toolbox-backends.md`](docs/toolbox-backends.md).

Useful docs:

- Q2_0 Strix Halo plan: [`docs/q2_0-strix-halo.md`](docs/q2_0-strix-halo.md)
- GAIA quickstart: https://amd-gaia.ai/docs/quickstart
- Lemonade docs: https://lemonade-server.ai/docs/
- Lemonade API overview: https://lemonade-server.ai/docs/api/
- Lemonade app integration guides: https://lemonade-server.ai/docs/server/apps/
- Lemonade OmniRouter: https://lemonade-server.ai/docs/omni-router/
- FastFlowLM docs: https://fastflowlm.com/docs/
- FastFlowLM server mode: https://fastflowlm.com/docs/instructions/server/

## Run

```sh
1bit up                       # start Lemonade, FLM, proxy, browser, GAIA
1bit status                   # show Lemonade, FLM, proxy, GAIA, memlock
1bit doctor                   # inspect host GPU/runtime/toolbox readiness
1bit toolbox list             # show Strix Halo backend lanes
1bit toolbox commands vllm    # print vLLM toolbox bootstrap commands
1bit toolbox status           # inspect toolbox-backed llama.cpp backend
1bit toolbox bootstrap        # create the Strix Halo llama.cpp toolbox
ONEBIT_TOOLBOX_MODEL=/path/model.gguf 1bit toolbox up
1bit gaia status              # exact GAIA AppImage/CLI/UI status
1bit gaia cli                 # GAIA CLI chat against the configured base URL
1bit gaia api status          # GAIA API server status
1bit gaia mcp status          # GAIA MCP bridge status
1bit webui status             # Open WebUI status
1bit bench                    # local 1-bit / ternary GGUF benchmark sweep
1bit down                     # stop GAIA, proxy, FLM, Lemonade
```

Endpoint choices:

```text
GAIA / Lemonade-style clients:  http://127.0.0.1:13306/api/v1
Most OpenAI clients:            http://127.0.0.1:13306/v1
Lemonade direct:                http://127.0.0.1:13305/api/v1
FastFlowLM direct:    http://127.0.0.1:52625/v1
Open WebUI:           http://127.0.0.1:3000
```

Use Lemonade direct when you specifically want Lemonade's canonical multimodal/OmniRouter behavior with no union routing. Use the proxy when a client should see Lemonade and FLM through one OpenAI-compatible base URL.

## Verified Live Behavior

On the reference Strix Halo box:

- `lemond.service` serves Lemonade on `:13305`.
- `flm.service` serves `qwen3:1.7b --embed 1 --socket 20 --q-len 20` on `:52625`.
- `1bit-proxy.service` serves the union endpoint on `:13306`.
- `open-webui.service` points to `http://127.0.0.1:13306/v1` on `:3000`.
- GAIA runs from `/home/bcloud/Applications/gaia-agent-ui.AppImage` and `~/.gaia/venv/bin/gaia`.

FastFlowLM ASR is intentionally opt-in because `whisper-v3:turbo` is not installed by default. Pull it first, then add `--asr 1` to `ONEBIT_FLM_FLAGS` if you want ASR routed to FLM.

## Benchmarks

The benchmark scripts live in `benchmarks/`.

Recent local checks:

- NPU ioctl budget with `qwen3:0.6b`: 19 decoded tokens, 3879 total ioctls, 204 ioctls/token, 96.3 decode tok/s. Passed the 250 ioctl/token threshold, with a warning above 200.
- GGUF pile through Lemonade llama-bench:
  - Bonsai 1.7B IQ1_S: ~4828 prompt tok/s, ~284.7 gen tok/s
  - Bonsai 4B IQ1_S: ~1904 prompt tok/s, ~142.5 gen tok/s
  - Bonsai 8B IQ1_S: ~1058 prompt tok/s, ~90.8 gen tok/s
  - Gianni BitNet 3B TQ2_0: ~1796 prompt tok/s, ~76.1 gen tok/s

Historical benchmark files remain in `benchmarks/RESULTS-*.md`; treat dated architecture notes inside them as historical context.

## Layout

```text
install.sh                         installer and systemd unit writer
scripts/1bit                       local control CLI
scripts/1bit-proxy.js              OpenAI-compatible union proxy
scripts/1bit-omni.py               Lemonade OmniRouter helper loop
scripts/omni-plugins/              Caller-side Lemonade OmniRouter plugins
benchmarks/                        reproducible benchmark scripts/results
docs/                              repo docs and architecture notes
docs/app-integration.md            how apps connect to the inference engine
docs/optc-troubleshooting.md       Strix Halo OPTC hang mitigation and soak test
1bit-site/                         Cloudflare Pages site for 1bit.systems
packaging/                         AUR and binary packaging work
```

## Security Posture

The default local endpoints are unauthenticated developer services. Bind them to localhost unless you explicitly put authentication and TLS in front of them. Cloudflare Pages only publishes the static website; it should not expose local inference ports directly.

Relevant attack surfaces are the installer, systemd units, `scripts/1bit`, `scripts/1bit-proxy.js`, Lemonade model/backend configuration, FastFlowLM service flags, GAIA/Open WebUI client settings, and any tunnel or reverse proxy you add.

## License

MIT. See `LICENSE`.
