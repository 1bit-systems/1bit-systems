# 1bit-systems wiki

Plain-English explanations of the architectural calls we made, kept aligned
with the toolbox-first repair path and the intended Lemonade + FastFlowLM +
`1bit-proxy` product direction.

## Current Baseline

- [Development](./Development.md) — repair baseline and the five review rules
- [Architecture deep](./Architecture-Deep.md) — current runtime topology and request flow
- [Complete pack](./Complete-Pack.md) — current lanes and install entry point
- [Installation](./Installation.md) — toolbox-first install and verification flow
- [Clients](./Clients.md) — GAIA, Open WebUI, raw HTTP, and SDK setup
- [FAQ](./FAQ.md) — short answers to common questions

## Decisions

- [Why Rust above + C++ below?](./Why-Rust.md) — the language split
- [Why Strix Halo?](./Why-Strix-Halo.md) — the hardware target
- [Why Caddy + systemd?](./Why-Caddy-Systemd.md) — the ops layer
- [Why no Python at runtime?](./Why-No-Python.md) — Rule A
- [NPU status](./Why-No-NPU-Yet.md) — FastFlowLM target lane + IRON/MLIR-AIE authoring path
- [Why this way + how?](./Why-This-Way-How.md) — repair path walkthrough

## Integrations

- [Lemonade compatibility](./Lemonade-Compat.md) — Lemonade as the native `:13305` lane; toolbox `llama-server` can occupy the same backend slot during repair
- [AMD GAIA integration](./AMD-GAIA-Integration.md) — GAIA as the primary UI/control surface
- [Hermes Agent integration](./Hermes-Integration.md) — Hermes as an external client on the proxy endpoint
- [Add your own app](./Add-Your-Own-App.md) — caller-side app pattern under Rule A
- [Cloudflare Tunnel setup](./Cloudflare-Tunnel-Setup.md) — optional `api.1bit.systems` tunnel plan
- [VPN-only API](./VPN-Only-API.md) — mesh and bearer posture

## Historical Or Project Notes

- [Repo layout](./Repo-Layout.md)
- [Observability](./Observability.md)
- [Fork everything](./Fork-Everything.md)
- [Ternary on AIE pack plan](./Ternary-on-AIE-Pack-Plan.md)
- [Agent fleet](./halo-agent-fleet.md)
- [Stablecoin side-project note](./tier-jwt-flow.md)
- [Historical: 1bit-lemonade](./Crate-halo-lemonade.md)
- [Historical: 1bit-helm](./Crate-halo-helm.md)
- [Historical: 1bit-landing](./Crate-halo-landing.md)

## Repo Pointers

- [Contributing](../../CONTRIBUTING.md)
- [Agent rules](../../CLAUDE.md)
