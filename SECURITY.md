# Security Policy

## Supported Versions

`1bit-systems` is pre-1.0. Only the tip of `main` is supported. Tagged releases are snapshots, not long-term support branches.

| Branch / tag | Supported |
|---|---|
| `main` | yes |
| tagged releases | best effort, no backports |

## Reporting A Vulnerability

Do not open a public GitHub issue for anything with security impact. Use GitHub private vulnerability reporting:

https://github.com/bong-water-water-bong/1bit-systems/security/advisories/new

If that path is unavailable, contact the maintainer directly with the affected version, proof of concept, and impact summary.

## Scope

Current in-scope surfaces:

- `install.sh`, especially privileged writes under `/etc`, `/usr/local`, `/opt/1bit`, systemd, and memlock configuration.
- `scripts/1bit`, including service control, GAIA launch/control, benchmark wrappers, and path discovery.
- `scripts/1bit-proxy.js`, especially request routing, body limits, WebSocket upgrade handling, and LAN binding.
- Lemonade configuration and backend/version wiring used by this stack.
- FastFlowLM service flags and NPU runtime exposure on `:52625`.
- GAIA and Open WebUI client configuration when pointed at `:13305`, `:13306`, or `:52625`.
- `1bit-stack.target` and the service units for Lemonade, FLM, proxy, and Open WebUI.
- The static Cloudflare Pages site, especially `install.sh` delivery and redirects.

Out of scope unless caused by this repo:

- Upstream vulnerabilities in Lemonade, FastFlowLM, llama.cpp, stable-diffusion.cpp, Whisper, Kokoro, Open WebUI, GAIA, ROCm, XRT, `amdxdna`, or firmware.
- Third-party OpenAI-compatible clients.
- Model behavior, hallucinations, prompt injection in generated text, or unsafe completions.

## Known-Sensitive Areas

- Local inference endpoints are unauthenticated by default. Do not expose `:13305`, `:13306`, `:52625`, `:3000`, GAIA API, or GAIA MCP to the internet without explicit authentication and TLS.
- `1bit-proxy` accepts OpenAI-compatible bodies and forwards them to local services. Oversized body handling and model-based routing are security-relevant.
- `install.sh` is a privileged bootstrap path. Any way to execute unintended code from the Cloudflare-hosted install script is critical.
- GAIA mobile/local tunnel features can expose the local UI to the LAN. Treat the token and LAN route as sensitive.

## What To Expect

- Acknowledgement within 72 hours.
- Triage within 7 days.
- Fix or advisory within 30 days for confirmed high severity issues, faster for critical issues when practical.
- Credit in the advisory if requested.
