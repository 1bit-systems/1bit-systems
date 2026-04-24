# 1bit-systems v0.1.0 — Discord announcement copy

Drafted 2026-04-24 for the v0.1.0 cut (commit `e97fe4f`). Three variants,
one per channel. Post by hand via the Discord UI — see the "how to post"
section at the bottom.

---

## announcements

**1bit-systems v0.1.0 is out.**

First tagged release of the native-ternary LLM stack for AMD Strix Halo.
Everything runs bare-metal under Rule A: C++20 HIP kernels, a Rust tower
above them, no Python in the serving path.

**What's in the box**

- AppImage, ~32 MB, distro-agnostic. SHA-256 `8a964f89bdef68ed914c04fcf23092ac642c424bb70f74dc10e8558e93b94036`.
- Baseline weights `halo-1bit-2b.h1b` at 1.8 GB.
- Native HIP ternary GEMV running at **92% of LPDDR5 peak** on gfx1151.
- Split-KV Flash-Decoding attention: **10.25x** at L=2048, **11.98x** at L=8192 vs the single-block baseline.
- WMMA fp16 path: **50.17 TFLOPS**.
- Decode **80.8 tok/s** at 64-tok context, **68.23 tok/s** at 1024-tok context.
- PPL on wikitext-103: **9.16** chunk-1024, **11.98** pass-4095.
- Cross-arch efficiency: **0.904 tok/J on Strix** vs **0.439 on the 9070 XT** — 2.06x perf/W win for the iGPU.

**NPU toolchain validated**

IRON + Peano + libxrt + aie-rt now compile and load on Strix Halo npu5.
160/160 axpy tests pass, AIE2P int8 matmul is bit-exact vs reference.
Toolchain-only for now — the NPU is NOT in the serve path yet.

**Stability**

OPTC display-engine hang on concurrent workloads is mitigated by
`amdgpu.dcdebugmask=0x410`. Two-hour soak with concurrent bitnet + SDXL
load: zero `REG_WAIT` events, edge max 65C.

**Training runs live**

Run 4 (3:4 Sparse-BitNet) and Run 5 (2:4 Sparse-BitNet DDP) are both
cooking on H200s. Results will land as follow-up model drops.

**Install**

- AppImage: grab it from the GH release.
- Arch / CachyOS: `yay -S 1bit-systems-bin` (or paru).
- Source: clone the repo, `cargo build --release --workspace`.

**Links**

- Repo: https://github.com/bong-water-water-bong/1bit-systems
- Wiki (deep-dives): https://github.com/bong-water-water-bong/1bit-systems/wiki
- Landing: https://1bit.systems

**Known limitations**

> NPU is toolchain-validated, not serve-path. BitNet on XDNA2 still
> needs kernel authoring on top of the validated lower stack.
>
> Reddit relaunch is parked until both training runs finish. No public
> push to r/LocalLLaMA etc. until the numbers from Run 4 + Run 5 are in.
>
> The HIP kernel is tuned for gfx1151. gfx1201 (RX 9070 XT) builds but
> is second-target — perf numbers above are Strix Halo only except where
> called out.

---

## releases

**v0.1.0 — 2026-04-24**

Commit `e97fe4f`.

- feat: first tagged release. AppImage 32 MB, distro-agnostic, SHA-256 `8a964f89bdef68ed914c04fcf23092ac642c424bb70f74dc10e8558e93b94036`.
- feat: baseline model `halo-1bit-2b.h1b` (1.8 GB) shipped.
- perf: decode 80.8 tok/s @ 64-tok, 68.23 tok/s @ 1024-tok on gfx1151.
- perf: ternary GEMV at 92% of LPDDR5 peak.
- perf: Split-KV Flash-Decoding attention — 10.25x @ L=2048, 11.98x @ L=8192.
- perf: WMMA fp16 path 50.17 TFLOPS.
- quality: wikitext-103 PPL 9.16 (chunk-1024), 11.98 (pass-4095).
- build: AUR package `1bit-systems-bin` published.
- platform: NPU toolchain (IRON + Peano + libxrt + aie-rt) validated on npu5. Axpy 160/160, AIE2P int8 matmul bit-exact. Not in serve path yet.
- fix: `amdgpu.dcdebugmask=0x410` soaked green, 2h concurrent-workload run clean.

Repo: https://github.com/bong-water-water-bong/1bit-systems

---

## random

> 1bit-systems v0.1.0 shipped. 32 MB AppImage, 80.8 tok/s on Strix Halo, ternary GEMV at 92% of LPDDR5 peak. I know kung fu. https://github.com/bong-water-water-bong/1bit-systems/releases/tag/v0.1.0

---

## how to post

The three existing Discord binaries in `crates/1bit-agents/bin/` do NOT
cover freeform channel posting:

- `1bit-discord-admin.rs` — creates scheduled events, uploads stickers,
  starts stage instances. No plain-message post. Subcommands: `event`,
  `sticker`, `stage`. Env: `DISCORD_BOT_TOKEN`, `GUILD_ID`.
- `1bit-watch-discord.rs` — gateway listener. Lurker-only on halo's
  token; echo's token (`ECHO_BOT_TOKEN`) only replies in forum posts
  and on `@halo` mention for `/status`. No "post to channel X" command.
- `1bit-helpdesk-setup.rs` — configures a forum channel's tags + topic.
  Doesn't post messages. Env: `DISCORD_BOT_TOKEN`, `CHANNEL_ID`.

**Post manually via the Discord UI.** Paste each block into its channel
as-is; Discord markdown renders `**bold**` and `>` quotes correctly.

Optional: the v0.1.0 release is a natural fit for `1bit-discord-admin
event` if you want a scheduled-event card on the guild (e.g. "v0.1.0
release party" as an EXTERNAL event with the GH release URL as
`--location`). That's cosmetic — not required for the three channel
posts above.
