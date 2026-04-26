# Installation

Two install paths:

- **AppImage (recommended, non-dev)** — single-file download, no build. Bundles the Rust binaries; ROCm itself is still built from source for now.
- **Source build (dev path)** — full control, required for kernel work and non-Arch distros. Everything below.

Flatpak is on the near-term roadmap.

## AppImage — recommended non-dev path


```bash
# download → verify → symlink to ~/.local/bin/1bit
curl -fsSLO https://1bit.systems/dl/1bit-systems-0.1.0-x86_64.AppImage
echo "cb24880525750d73bb9bc75c5db993e8aa7e83837165c80a442526c67226f6eb  1bit-systems-0.1.0-x86_64.AppImage" | sha256sum -c -
chmod +x 1bit-systems-0.1.0-x86_64.AppImage
ln -sfn "$PWD/1bit-systems-0.1.0-x86_64.AppImage" ~/.local/bin/1bit
1bit --version
```

Works on any x86_64 Linux with glibc ≥ 2.31 (Ubuntu 20.04, Debian 11, CachyOS). The scripted path `INSTALL_MODE=appimage ./install.sh` does the same three steps and verifies against [`1bit-site/releases.json`](../../1bit-site/releases.json).

Symlink the AppImage to any of the 20 bundled binary names and it dispatches via `$ARGV0`. Run `<AppImage> --list` to enumerate them.

## Source build — dev path

## Requirements

The stack targets Strix Halo specifically. Other gfx1100-family hardware *may* work with minor tweaks; only Strix Halo is tested.

### Hardware — minimum

- AMD Ryzen AI Max+ 395 (or equivalent Strix Halo SKU)
- Radeon 8060S iGPU · gfx1151 · wave32 WMMA
- 64 GB unified LPDDR5X minimum · 128 GB recommended for 13B+ ternary
- 100 GB free disk for models plus build artifacts

### Software — minimum

- Linux kernel **6.18.22-lts** (newer kernels carry the amdgpu OPTC hang — see [Troubleshooting](./Troubleshooting.md))
- ROCm 7.x — built from source against gfx1151 (not on ROCm's Tier-1 list)
- LLVM / clang 18+
- CMake 3.27+
- Rust 1.82+ (stable channel)
- Node.js or Bun *only* for caller-side clients. Nothing on the serving path — Rule A.

### Recommended host

**CachyOS** with Btrfs + snapper + limine is the reference setup. Rollback-via-snapper has saved the project more than once. Fish shell is assumed in examples but not required.

### Distro policy

**Supported:** Arch Linux (and Arch-family: CachyOS, Manjaro, EndeavourOS). Every binary, systemd unit, and install script is tested here. The reference production box is CachyOS.

**Best-effort:** Ubuntu 24.04+, Fedora 40+, Debian 12+. The code compiles; regression tests do not run on these distros. ROCm install diverges from the Arch path — you are building from source.

**Unsupported:** Windows (no HIP runtime story for this stack), macOS (ARM Apple Silicon is a separate backend via the `mlx` feature-gate, not covered here).

If you are on Ubuntu, expect the following extra steps:

1. Install ROCm 7.x per AMD's official Ubuntu instructions (not the stock Ubuntu repo — that ROCm is too old). See [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html).
2. gfx1151 is not on ROCm's Tier-1 list. Expect to build from source. The `llamacpp-rocm` fork's install script is the paved road.
3. systemd user units in this repo assume paths like `~/.local/bin/`. These are distro-agnostic; copy the unit files to `~/.config/systemd/user/` unchanged.
4. Caddy's Arch package ships with permissions that let it bind to `:80` and `:443` out of the box. On Ubuntu, verify the package manager sets the `cap_net_bind_service` capability on the Caddy binary — or run it under a reverse-proxy that already has the right caps.

When asking for help on non-Arch, lead the question with distro + kernel version + ROCm version. Short example: "Ubuntu 24.04, 6.11, ROCm 7.0 from AMD repo — `1bit-halo-server` fails to link." That beats "it's broken" every time.

## Build ROCm against gfx1151

System-package ROCm drops gfx1151 from Tier-1 in most distros. Build from source, or use the `llamacpp-rocm` fork's install script as a bootstrap.

```bash
git clone https://github.com/bong-water-water-bong/llamacpp-rocm ~/repos/llamacpp-rocm
cd ~/repos/llamacpp-rocm
./scripts/install-rocm.sh --target gfx1151
```

## Build rocm-cpp kernel library

Default build is a multi-arch fat binary covering every mainstream AMD Wave32-WMMA part — RDNA3 (`gfx1100/1101/1102/1103`), RDNA3.5 (`gfx1150/1151`), RDNA4 (`gfx1200/1201`). One build runs on every consumer Radeon + APU in that list. Override `CMAKE_HIP_ARCHITECTURES` explicitly if you want a narrower build for disk / build-time reasons.

```bash
git clone https://github.com/bong-water-water-bong/rocm-cpp ~/repos/rocm-cpp
cd ~/repos/rocm-cpp

# default: multi-arch fat binary (RDNA3 + 3.5 + 4)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
sudo cmake --install build --prefix /usr/local

# or: narrow single-arch build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
```

## Build 1bit-halo-core (bitnet_decode)

```bash
# private repo today; public release gated on NPU ship-gate
cd ~/1bit-halo-core

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/bitnet_decode --help
```

## Build 1bit-halo-server and 1bit-halo-mcp

```bash
cd ~/1bit-halo-workspace

cmake --build cpp/build/strix --bin 1bit-halo-server
cmake --build cpp/build/strix --bin 1bit-halo-mcp

ls -la cpp/build/strix/server/1bit-server cpp/build/strix/mcp/1bit-mcp
```

## Fetch model weights

```bash
# 1bit-halo-pkg is not shipped yet. Manual download for now:
mkdir -p ~/1bit-halo/models
cd ~/1bit-halo/models

# 1bit-halo-v2 · BitNet 1.58 · 2B
curl -LO https://.../1bit-halo-v2.h1b   # actual URL TBD

# TriLM 3.9B Unpacked (experimental)
curl -LO https://.../trilm-3.9b.h1b
```

> **Rule A reminder** — Python may appear in caller-side tooling and dev-time scripts only. `bitnet_decode`, `1bit-halo-server`, `1bit-halo-mcp`, and kernel binaries ship zero Python. Carve-outs (Open WebUI, `lemonade-server`) are caller-side and sunset on 1bit-helm v0.3 parity.

## Second target — RX 9070 XT (gfx1201) — shipped

Radeon RX 9070 XT (Navi 48, RDNA 4) lives in the `ryzen` mesh host and is the secondary kernel target. **Port status: shipped 2026-04-22.** All 14 WMMA prefill kernels are bit-exact vs CK on both `gfx1151` and `gfx1201`. `bitnet_decode --ppl` on wikitext-103 matches to six decimals across arches (**11.9758** on both). Peak throughput is not yet re-tuned for GDDR6 bandwidth (~640 GB/s on 9070 XT vs ~270 GB/s LPDDR5X on Strix Halo); block sizes and LDS budgets remain Strix-tuned, and a K-outer tile sweep is the natural next pass. See [Benchmarks.md](./Benchmarks.md) for the cross-arch numbers.

The hot intrinsics — `__builtin_amdgcn_wmma_*_w32`, `__builtin_amdgcn_sudot4`, `__builtin_amdgcn_sdot4` — are retained on RDNA 4 (half8 + `_w32_gfx12` shape variant, dispatched by `src/wmma_compat.hpp`). The `standalone_gemm_4c_pipelined` SGPR spill on `gfx1201` was eliminated by hoisting the capturing lambda to a free `__device__ __forceinline__` function plus a flat A-cache layout (commit `dc0b1b1`).

### Build for gfx1201

```bash
# single-arch build, 9070 XT only
GFX=gfx1201 ./install.sh

# fat-binary build, runs on both strixhalo and ryzen
GFX="gfx1151;gfx1201" ./install.sh

# auto-detect via rocminfo (use on each host natively)
GFX=auto ./install.sh
```

Prereq: ROCm must be present on `ryzen` first. Easiest path is the same TheRock source build used on Strix Halo, re-targeted to Navi 48. System-package ROCm may also work on RDNA 4 in distros that ship it; verify with `rocminfo`.

```bash
ssh ryzen
ls /opt/rocm* ~/therock 2>/dev/null     # confirm a ROCm dist exists
rocminfo | grep -E 'Name:|gfx'            # expect gfx1201
```

## First run

Start `bitnet_decode` on the dev port, then `1bit-halo-server` as the OpenAI-compatible front. Verify with `curl`.

### Start the inference core

```bash
cd ~/1bit-halo-core
./build/bitnet_decode \
  --model ~/1bit-halo/models/1bit-halo-v2.h1b \
  --port 8080 \
  --context 4096 \
  --attn split-kv-fd \
  --rope-mode hf-split-half
```

### Start the HTTP surface

```bash
cd ~/1bit-halo-workspace
./cpp/build/strix/server/1bit-server \
  --upstream http://127.0.0.1:8080 \
  --bind 0.0.0.0:8180
```

### Verify

```bash
curl -s http://127.0.0.1:8180/v1/models | jq
# expect: {"data": [{"id": "1bit-halo-v2", ...}]}

curl -s http://127.0.0.1:8180/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "1bit-halo-v2",
    "messages": [{"role":"user","content":"say hello"}]
  }' | jq
```

## Services and systemd

Production is systemd. One unit per binary. Units live in `/etc/systemd/system/` (system scope) or `~/.config/systemd/user/` (user scope). LTS kernel **needs** `LimitMEMLOCK=infinity` for the inference core or pinning fails.

### 1bit-halo-bitnet.service

```ini
# /etc/systemd/system/1bit-halo-bitnet.service
[Unit]
Description=1bit bitnet_decode (HIP inference core)
After=network.target 1bit-halo-gpu-perf.service
Requires=1bit-halo-gpu-perf.service

[Service]
Type=simple
User=1bit-halo
Group=1bit-halo
ExecStart=/usr/local/bin/bitnet_decode \
  --model /var/lib/1bit-halo/models/1bit-halo-v2.h1b \
  --port 8080 \
  --context 4096 \
  --attn split-kv-fd \
  --rope-mode hf-split-half
Restart=on-failure
RestartSec=5
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### 1bit-halo-server.service

```ini
# /etc/systemd/system/1bit-halo-server.service
[Unit]
Description=1bit OpenAI-compatible HTTP surface
After=network.target 1bit-halo-bitnet.service
Requires=1bit-halo-bitnet.service

[Service]
Type=simple
User=1bit-halo
ExecStart=/usr/local/bin/1bit-halo-server \
  --upstream http://127.0.0.1:8080 \
  --bind 0.0.0.0:8180
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 1bit-halo-gpu-perf.service

Pins SCLK high to avoid latency spikes under sustained load. Required on LTS 6.18.22.

```ini
# /etc/systemd/system/1bit-halo-gpu-perf.service
[Unit]
Description=1bit GPU perf pinning (SCLK high)
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'echo high > /sys/class/drm/card0/device/power_dpm_force_performance_level'

[Install]
WantedBy=multi-user.target
```

### Enable and check

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now 1bit-halo-gpu-perf 1bit-halo-bitnet 1bit-halo-server

systemctl status 1bit-halo-bitnet 1bit-halo-server
journalctl -u 1bit-halo-bitnet -f
```

### Default ports

| Port | Service              | Purpose                                           |
|------|----------------------|---------------------------------------------------|
| 8080 | `bitnet_decode`      | Internal dev · upstream for 1bit-halo-server      |
| 8180 | `1bit-halo-server`   | OpenAI-compatible HTTP · public-facing            |
| 8181 | `1bit-halo-mcp`      | MCP server · tool & introspection surface         |
| 8190 | `1bit-halo-whisper`  | Streaming STT (planned)                           |
| 8191 | `1bit-halo-kokoro`   | TTS (planned)                                     |
