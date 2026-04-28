# 1bit-systems control plane — single static binary

Bun-compiled port of `scripts/1bit` (bash) + `scripts/1bit-proxy.js` (Node)
into a single self-contained ELF executable. Drop-in replacement for
both `/usr/bin/1bit` and `/usr/share/1bit-systems/1bit-proxy.js`.

## Build

```sh
# 1. Install bun (Arch / CachyOS):
sudo pacman -S bun

# 2. From this directory:
bun install            # pulls @types/bun for IDE only; build does not need it
bun build --compile --target=bun-linux-x64 --outfile=1bit ./1bit.ts

# 3. Smoke-test:
./1bit --help
./1bit status
```

The compiled binary is statically linked and bundles:
- the Bun runtime,
- the `1bit` CLI (all subcommands),
- the `1bit-proxy` reverse proxy (run with the hidden `__proxy__` subcommand
  — `1bit up` spawns this internally), and
- `home.html` (embedded as a string literal via `import ... with { type: "text" }`).

## Layout

```
packaging/binary/
├── package.json     # name=@1bit-systems/cli, type=module
├── tsconfig.json
├── 1bit.ts          # main entrypoint, argv + all subcommands
├── proxy.ts         # OpenAI-compat reverse proxy (Bun.serve)
├── home.html        # embedded at build via `import "..." with { type: "text" }`
└── README.md
```

## Subcommand parity

All bash subcommands ported:

- `1bit up` — start lemond + flm + proxy, open browser, launch GAIA
- `1bit down` — stop all
- `1bit status` — health check
- `1bit pull <model>` — auto-route NPU (flm) vs GPU (lemonade)
- `1bit bench` — run benchmarks/bench-1bit-pile.sh
- `1bit npu [model]` — start FLM NPU server
- `1bit webui [up|down|status]` — Open WebUI on :3000
- `1bit __proxy__` — *internal*, runs the embedded reverse proxy

## Packaging suggestion

Ship as `1bit-systems-bin` AUR package alongside `1bit-systems-git`:
- `1bit-systems-git` (current): bash + node, depends on `nodejs`.
- `1bit-systems-bin` (new): single ELF, no `nodejs` dep, drops to
  `/usr/bin/1bit`. ~30 MB on disk.
