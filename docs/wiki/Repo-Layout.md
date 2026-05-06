---
phase: implementation
owner: cartograph
---

# Repo layout — repair stack

Canonical URL: `https://github.com/bong-water-water-bong/1bit-systems`.

## Top level

```
1bit-systems/
├── README.md               public stack overview and install entry
├── CONTRIBUTING.md         contribution rules and service map
├── CLAUDE.md               hard engineering rules for agents
├── install.sh              Arch/CachyOS native bootstrap and systemd wiring
├── scripts/
│   ├── 1bit                operator CLI: up/down/status/gaia/webui helpers
│   └── 1bit-proxy.js       union OpenAI-compatible endpoint on :13306
├── docs/                   repo-local docs for app integration, AUR, OmniRouter, OPTC
├── docs/wiki/              long-form project wiki and decision records
├── 1bit-site/              static website for 1bit.systems
├── packaging/              distro/package scaffolding
├── benchmarks/             local benchmark and verification helpers
└── vendor/                 pinned/forked source trees when installed locally
```

## Runtime shape

```
GAIA / Open WebUI / SDK clients
        |
        v
1bit-proxy :13306
   |-- toolbox llama-server :13305  first repaired GGUF backend
   |-- Lemonade :13305              native multimodal / OmniRouter lane
   `-- FastFlowLM :52625            optional XDNA NPU lane
```

GAIA points at `http://127.0.0.1:13306/api/v1`. Generic OpenAI-compatible
clients use `http://127.0.0.1:13306/v1`. The active repaired backend is
toolbox `llama-server` on `:13305`; Lemonade and FastFlowLM are native lanes
when installed and healthy.

## Kernel ownership

HIP kernels belong in `rocm-cpp/` and are C++20. Do not port kernel work to Rust, and do not add hipBLAS to the runtime path. Custom NPU kernel authoring uses IRON at author time, lowers through MLIR-AIE and Peano, emits `xclbin`, and loads from native runtime code through `libxrt`.

## Brand

- **Brand:** `1bit systems`
- **Domain:** `1bit.systems`
- **Repo:** `1bit-systems`
- **CLI:** `1bit`

Do not reintroduce old `halo-*` branding in current docs unless the name is an actual historical artifact or existing systemd path.

## See also

- [Development](./Development.md) — repair path and Rules A-E
- [Clients](./Clients.md) — OpenAI-compatible client setup
- [Lemonade compatibility](./Lemonade-Compat.md) — Lemonade's native lane role
- [NPU status](./Why-No-NPU-Yet.md) — FastFlowLM target lane and IRON authoring lane
- [Toolbox backends](../toolbox-backends.md) — toolbox-first repair path
