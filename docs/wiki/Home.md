# 1bit-systems wiki

Plain-English explanations of the architectural calls we made — one page per decision, citations where they exist.

## Internals

- [Architecture deep](./Architecture-Deep.md) — end-to-end request walkthrough, per-kernel provenance, FFI signatures, agent registry, training pipeline, mesh, failure surface

## Decisions

- [Why 1.58-bit ternary?](./Why-Ternary.md) — the weight format
- [Why Rust above + C++ below?](./Why-Rust.md) — the language split
- [Why Strix Halo (gfx1151)?](./Why-Strix-Halo.md) — the hardware target
- [Why shadow-burnin?](./Why-Shadow-Burnin.md) — the cutover discipline
- [Why our own `.h1b` format?](./Why-H1b-Format.md) — vs GGUF
- [Why Caddy + systemd?](./Why-Caddy-Systemd.md) — the ops layer
- [Why 1bit-agents?](./Why-Halo-Agents.md) — the self-maintaining mesh
- [Why no Python at runtime?](./Why-No-Python.md) — Rule A
- [Why shadow-traffic parity gates?](./Why-Parity-Gates.md) — cutover criteria
- [Why no NPU yet?](./Why-No-NPU-Yet.md) — XDNA 2 status + ORT/VitisAI EP evaluation
- [Why this way + how?](./Why-This-Way-How.md) — long-form walkthrough of all the decisions + end-to-end request path

## Running it

- [Installation](./Installation.md) — full build-from-source guide (requirements, ROCm, rocm-cpp, 1bit-halo-core, server, MCP, weights, gfx1201 second target, first run, systemd units, ports)
- [Clients](./Clients.md) — connecting Open WebUI, raw HTTP / curl, MCP clients (Claude Desktop + Claude Code), and SDK samples (Python, TypeScript, Rust)
- [Add your own app](./Add-Your-Own-App.md) — caller-side pattern under Rule A, session header usage, minimal agent harness example, where apps live
- [Troubleshooting](./Troubleshooting.md) — OPTC CRTC hang, SMU/VCN/PSP failures, long-context PPL, mlock, ROCm arch, latency under load, memory-sync credentials
- [Observability](./Observability.md) — journalctl, rocprof bandwidth check, PPL harness on wikitext-103, live tok/s benchmark

## Integrations

- [Hermes Agent integration](./Hermes-Integration.md) — Nous Research's agent as external client on 1bit-server; feature-port list for 1bit-agents

## Benchmarks + proof

- [Live tok/s + PPL](./Benchmarks.md) — what we measure, what it means (now includes gfx1151 vs gfx1201 cross-arch numbers + multi-arch default build)
- [Peak performance projection](./Peak-Performance-Projection.md) — end-state throughput math, WMMA peak measured 2026-04-22
- [FAQ](./FAQ.md) — short answers to common questions

## Training + funding

- [Training runs](./Training-Runs.md) — Run 4 (3:4) live status, Run 5 (2:4 canonical) plan, H200 NVL pod default
- [Funding goals](./Funding-Goals.md) — 13B / 30B / 70B Sparse-BitNet Patreon ladder

## Ops + topology

- [Network topology](./Network-Topology.md) — four-node private Headscale mesh, coordinator layout, preauth-key flow, reachability matrix, DNS gotchas

## Session logs

- [2026-04-22 — identity rebuild + Run 4 + site scaffold](./2026-04-22-Rebuild.md) — full narrative of the halo→1bit-halo rename sweep, Sparse-BitNet Run 3 autopsy + Run 4 relaunch, ryzen gfx1201 stand-up, site scaffold, brand voice lock, repo public flip

## Pointers

- Architectural data-flow: [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md)
- Cutover runbook: [`../../CUTOVER.md`](../../CUTOVER.md)
- Demo script: [`../../DEMO.md`](../../DEMO.md)
- Contributing: [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)
- Repo conventions for AI agents: [`../../CLAUDE.md`](../../CLAUDE.md)
