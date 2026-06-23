> **Start here:** Read `docs/wiki/README.md` before any work on this project.

# CLAUDE.md — conventions for 1bit-systems

1-bit inference engine for AMD Strix Halo. One runtime layer: Lemonade SDK.
The 1bit path is Lemonade + rocm-cpp HIP kernels. No proxy, no shell script
orchestration — Lemonade manages lifecycle, routing, and the OpenAI-compatible
API surface.

## Hard rules

- **Rule A: core serving stays Python-free.** The inference hot path is
  C++/HIP (rocm-cpp kernels) wrapped by Lemonade Server (C++17). No Python
  at inference time. Build-time tooling and notebooks are allowed.
- **Rule B: C++20 for kernels.** HIP kernels live in the `bong-water-water-bong/rocm-cpp`
  repo. 1-bit/ternary GEMV, GEMM, RMSNorm, RoPE, attention — all native HIP.
- **Rule C: hipBLAS is banned in the runtime path.** All compute goes through
  our custom ternary-aware kernels in rocm-cpp.
- **Rule D: Lemonade SDK is the only runtime layer.** No 1bit-proxy, no
  toolbox containers, no shell-script service wiring. Lemonade's `lemond`
  manages backend subprocess lifecycles through `WrappedServer`.
- **Rule E: NPU is an optional side lane.** FastFlowLM via Lemonade's FLM
  backend for NPU inference. Custom NPU kernels via IRON/MLIR-AIE for
  author-time.
- **Compatibility surface is OpenAI.** `lemond` serves on `:13305`. Any
  OpenAI-compatible client connects to `http://127.0.0.1:13305/v1`.

## Architecture

```
lemonade-sdk/lemonade  →  bong-water-water-bong/1bit-lemonade (fork)
│
├── lemond (:13305)  — HTTP server, C++17
│    ├── BitNetServer  — wraps bitnet_decode --server (rocm-cpp)
│    │    └── librocm_cpp.so  (ternary HIP GEMV/GEMV kernels)
│    ├── LlamaCppServer  — fallback for non-1bit GGUF models
│    └── FastFlowLMServer  — NPU lane
│
└── bong-water-water-bong/rocm-cpp
     └── bitnet_decode  — native C++/HIP 1-bit inference binary
          └── speaks OpenAI /v1/chat/completions on its port
```

## Repos

| Repo | Role |
|---|---|
| `bong-water-water-bong/1bit-lemonade` | Fork of Lemonade SDK with BitNetServer backend |
| `bong-water-water-bong/rocm-cpp` | Native ROCm C++ kernels for 1-bit/ternary inference |
| `bong-water-water-bong/1bit-systems` | This repo — website, docs, benchmarks, install notes |

## Layout

```
.
├── 1bit-site/             # CF Pages site for 1bit.systems
├── benchmarks/            # benchmark scripts + results
├── docs/                  # architecture notes, wiki
└── install.sh             # convenience installer (installs Lemonade + rocm-cpp)
```

## What lives outside this repo

- `lemond` — built from `bong-water-water-bong/1bit-lemonade` fork, runs on `:13305`
- `bitnet_decode` — built from `bong-water-water-bong/rocm-cpp`, speaks OpenAI protocol
- ROCm 7.x — TheRock or system ROCm for gfx1151
- BitNet / Bonsai .h1b models — produced by rocm-cpp exporter from HuggingFace safetensors

## Test / verify

```sh
# Build and run Lemonade with BitNet backend
cd 1bit-lemonade && ./setup.sh && cmake --build --preset default
./build/default/bin/lemond
# In another terminal:
./build/default/bin/lemonade run BitNet-b1.58-2B-4T
```

## Deploy

`1bit-site/` deploys to Cloudflare Pages via `wrangler pages deploy`. The
CF project name is `1bit-systems`.

## Commits

Conventional Commits: `feat / fix / perf / docs / refactor / build / ci /
chore / test`. One logical change per commit. Push to `origin`.

## What NOT to do

- **Don't add a proxy layer.** Lemonade IS the proxy. No 1bit-proxy.js,
  no Node.js. The `lemond` HTTP server routes to backends directly.
- **Don't add shell-script orchestration.** No toolbox containers, no
  systemd unit generation scripts. Lemonade handles backend lifecycle.
- **Don't commit secrets.** CF tokens and gh tokens live in libsecret.
- **Don't expand scope.** Match the literal ask.
