# 1bit systems architecture

How the runtime pillars fit together and how the C++23 monorepo at
`cpp/` is laid out.

> **Status — gen-3 cutover (2026-04-26).** Phase 2 of the
> stack-architecture plan triggered today: the Rust `crates/` workspace
> was retired; the tower above the kernels is C++23 at `cpp/`.
> Rationale: engine-language monoculture — lemond and rocm-cpp were
> already C++, so pulling everything to one toolchain removes FFI on
> hot paths, collapses debug to one ABI, and deletes a build system.
> Shadow-burnin compares gen-3 vs gen-2 during the cutover — see
> `CUTOVER.md`.

## The three pillars

The stack has three pillars at inference time — kernels, caller tower,
agents/services. Upstream projects we borrow from (Lemonade, llama.cpp,
composable_kernel, mlir-aie) are **references**, not pillars. "Pillar"
means "something the serving process actually executes."

| # | pillar | location | language | role |
|---|---|---|---|---|
| 1 | AMD HIP kernels | `rocm-cpp/` subtree | C++20 / HIP | ternary GEMV, split-KV FD attention, RMSNorm, RoPE, KV cache |
| 2 | Caller tower | `cpp/` | C++23 | CLI, services, MCP, voice, helm, NPU dispatch |
| 3 | Agents + sidecars | `cpp/mcp*`, `retrieval/`, `voice/`, `echo/` | C++23 | MCP, retrieval, voice loop |

Rule A: bare-metal C++ from kernels up. Pillar 1 is `librocm_cpp.so`
linked directly into pillar 2 — no FFI shim, no `extern "C"` shuffle.
`lemond` on `:8180` is the canonical model gateway, replacing the
retired Rust `1bit-server`/`router`/`lemonade`/`whisper` crates from
the 2026-04-25 cull.

## Data flow

```
HTTP client → Caddy :443 → lemond :8180 → Engine (in-proc)
                                              → rocm-cpp HIP kernels
```

`cpp/core`, `cpp/mcp*`, and `cpp/retrieval` sit alongside, not in the
hot path. `cpp/cli` drives systemd. `cpp/aie` + `cpp/onnx` are the NPU
lane.

## Components

- **`cli/`** — `1bit` operator binary. `status / logs / restart /
  doctor / update / install / version`. Reads `packages.toml`, drives
  systemd units.
- **`core/`** — GGUF + H1B loader, BPE tokenizer, sampler, chat
  template; backend-agnostic; `std::expected` everywhere.
- **`mcp/`** — stdio JSON-RPC bridge for external Claude/MCP clients.
- **`landing/`** — `/metrics` probe + landing on `:8190`.
- **`helm/`** + **`helm-tui/`** — Qt6 + FTXUI clients. Plasma SNI tray
  icon. Start / stop / status of the `1bit-*` units.
- **`voice/`** + **`echo/`** — sentence-boundary streaming voice loop
  (LLM SSE → TTS chunks) + browser WebSocket gateway.
- **`power/`** — RyzenAdj wrapper for `1bit power`.
- **`onnx/`** + **`aie/`** — NPU lane (ORT C++ + VitisAI EP primary;
  libxrt + aie-rt + Peano custom).
- **`kokoro/`** + **`retrieval/`** + **`ingest/`** + **`tier-mint/`** —
  TTS + memory-palace pipeline.

## Build presets

`release-strix` (gfx1151, NPU on, `-march=native`) ·
`release-ryzen` (gfx1201, NPU off, `-march=znver5`) · `debug`. Build
with `cmake --preset <name> && cmake --build --preset <name> && ctest
--preset <name>`. The fat-binary build covers eight Wave32-WMMA AMD
arches (gfx1151 plus the rest of RDNA3 / RDNA3.5 / RDNA4).

## Systemd layout

User-scope under `~/.config/systemd/user/`, installed by `1bit install
<component>`. `1bit status` rolls them up; `1bit logs <unit>` wraps
`journalctl --user -u`.

| unit | port | notes |
|---|---|---|
| `1bit-halo-lemonade.service` | 8180 | lemond — canonical model gateway |
| `1bit-landing.service` | 8190 | live `/metrics` + landing page |
| `1bit-mcp.service` | stdio | socket-activated JSON-RPC bridge |
| `1bit-halo-whisper.service` | 8181 | STT (sliger B580 Vulkan in current topology) |
| `1bit-halo-kokoro.service` | 8182 | TTS (sliger B580 Vulkan in current topology) |
| `1bit-halo-sd.service` | 8081 | SDXL image sidecar |
| `caddy.service` | 443 | TLS front door, bearer check |

## Chat template

Single source of truth in `onebit::core::chat_template`. One turn
renders as:

```
User: <msg><|eot_id|>Assistant:
```

Multi-turn concatenates rendered turns with no extra separators. A
`core/` parity test byte-compares against a gen-2 fixture; wire format
is stable across the cutover.

## Cutover history (retained for reference)

The gen-1 → gen-2 cutover happened at v0.1.0 (2026-04-24). Both gates
were satisfied:

- **PPL parity on wikitext-103 1024 tok.** gen-2 hit **9.1805** vs
  gen-1 baseline **9.1607** — delta +0.0198, within ±0.05 tolerance.
  PASS.
- **Shadow-traffic burn-in.** Sustained `/v1` (gen-1) vs `/v2` (gen-2)
  argmax comparison ran 2026-04-21 → 04-23. Byte-exact agreement
  reached **96.66%** across 1500+ rounds after the special-token fix;
  divergences attributable to sampler nondeterminism, not semantic
  drift. PASS.

The historical Caddyfile split, burn-in tap, and `/v2/*` shadow
surface have been removed from production config. The
`benchmarks/shadow-burnin.sh` harness is retained for the gen-2 →
gen-3 cutover and any future backend swap (e.g. an NPU-backed path).
The current gen-3 cutover targets gen-2 as its parity baseline; see
`CUTOVER.md`.
