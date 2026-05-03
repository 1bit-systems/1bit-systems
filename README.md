# 1bit-systems

Local AI control plane for AMD Strix Halo: GAIA as the agent/UI layer, Lemonade as the canonical OpenAI-compatible multimodal server, FastFlowLM on the XDNA NPU, and a small union proxy for clients that want one endpoint for both lanes.

## Current Shape

| Layer | Role | Default |
|---|---|---|
| GAIA Agent UI | Primary agent UI and control surface | AppImage + `~/.gaia/venv/bin/gaia` |
| Lemonade Server | Canonical OpenAI-compatible multimodal API and OmniRouter surface | `http://127.0.0.1:13305/v1` |
| FastFlowLM | XDNA NPU runtime for FLM models, embeddings, and optional ASR | `http://127.0.0.1:52625/v1` |
| 1bit proxy | Convenience union endpoint for OpenAI clients | `http://127.0.0.1:13306/v1` |
| Open WebUI | Secondary browser UI | `http://127.0.0.1:3000` |
| systemd | Local stack lifecycle | `1bit-stack.target` |

Lemonade remains the source of truth for multimodal OpenAI compatibility and OmniRouter-style tool routing. The proxy does not replace Lemonade; it keeps Lemonade defaults and adds targeted side-lanes for FastFlowLM models such as NPU chat, embeddings, and opt-in ASR.

## Install

```sh
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
./install.sh
```

The installer is idempotent. It installs the local control CLI, systemd units, Open WebUI wiring, memlock limits, and default service configuration. Lemonade and FastFlowLM are built from the maintained fork sources under `/opt/1bit`; legacy upstream application packages are removed so the local stack uses the forked code paths. On first install, log out and back in, or reboot, so memlock limits apply to the NPU lane.

Useful docs:

- GAIA quickstart: https://amd-gaia.ai/docs/quickstart
- Lemonade docs: https://lemonade-server.ai/docs/
- Lemonade OmniRouter: https://lemonade-server.ai/docs/omni-router/
- FastFlowLM docs: https://fastflowlm.com/docs/
- FastFlowLM server mode: https://fastflowlm.com/docs/instructions/server/

## Run

```sh
1bit up                       # start Lemonade, FLM, proxy, browser, GAIA
1bit status                   # show Lemonade, FLM, proxy, GAIA, memlock
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
GAIA / most clients:  http://127.0.0.1:13306/v1
Lemonade direct:      http://127.0.0.1:13305/v1
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
benchmarks/                        reproducible benchmark scripts/results
docs/                              repo docs and architecture notes
1bit-site/                         Cloudflare Pages site for 1bit.systems
packaging/                         AUR and binary packaging work
```

## Security Posture

The default local endpoints are unauthenticated developer services. Bind them to localhost unless you explicitly put authentication and TLS in front of them. Cloudflare Pages only publishes the static website; it should not expose local inference ports directly.

Relevant attack surfaces are the installer, systemd units, `scripts/1bit`, `scripts/1bit-proxy.js`, Lemonade model/backend configuration, FastFlowLM service flags, GAIA/Open WebUI client settings, and any tunnel or reverse proxy you add.

## License

MIT. See `LICENSE`.
