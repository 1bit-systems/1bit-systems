# Why Strix Halo (gfx1151)?

**One-line answer**: AMD Ryzen AI MAX+ 395 ships 128 GB of LPDDR5 shared between CPU and iGPU at $2–3k total, with 256 GB/s bandwidth — enough to run a ternary 2B model at 80+ tok/s without touching discrete VRAM or cloud. No other consumer-class box hits that price × memory × bandwidth point.

## The hardware

- **CPU**: AMD Ryzen AI MAX+ 395 (Strix Halo) — 16 Zen 5 cores, 32 threads, 5.1 GHz boost.
- **iGPU**: Radeon 8060S (RDNA 3.5, gfx1151) — 40 CUs, WMMA matrix accelerators.
- **Memory**: 128 GB LPDDR5-8000, 256 bit-wide, 256 GB/s.
- **NPU**: XDNA 2 at 50 TOPS (int8). Not used today by 1bit systems; ROCm-7 + amdxdna kernel driver expose it.
- **Power**: ~45 W idle, ~150 W sustained inference.
- **Form**: mini-PC, 0.5L chassis, passive-or-quiet cooling possible.

## Why it's special for ternary inference

Unified memory. The 128 GB LPDDR5 is addressable by both CPU and iGPU with zero PCIe copy. On a discrete-GPU box, a ternary 2B model's 400 MB lives in VRAM and the CPU needs PCIe to touch it. On Strix Halo, the GPU reads straight from the same DDR bank the CPU just wrote to. At ternary bitrates we don't need more bandwidth than LPDDR5 provides — we have bandwidth to spare.

## The numbers we care about

| resource | available | ternary 2B usage | headroom |
|---|---:|---:|---:|
| memory | 128 GB | 4 GB model + 500 MB KV @ N=4096 | 123 GB |
| bandwidth | 256 GB/s | ~240 GB/s at decode (92% peak) | bandwidth-bound |
| compute | ~60 TFLOPs FP16 | ~6 TFLOPs used | 10× headroom |
| power | 150 W | ~100 W under decode | low-noise cooling |

The bottleneck is memory bandwidth, not compute. This validates the ternary story: **make weights smaller, everything gets faster.**

## Why not a discrete RTX / Radeon GPU

- **VRAM ceiling** — consumer GPUs max out at 24 GB (4090, 3090, 7900 XTX). Larger models + longer contexts fit in Strix Halo's 128 GB.
- **Price** — a box capable of running Llama-3-70B at FP16 (2× A6000 = 96 GB VRAM) is ~$10 000. A Strix Halo that runs BitNet-2B + 70 GB of context + SD + whisper + kokoro + all the agents is $2–3 000.
- **Silence** — datacenter GPUs are loud. Consumer cards thermal-throttle in a closet. Strix Halo runs whisper-quiet under sustained load.
- **Idle power** — a Strix Halo at idle draws ~45 W. A workstation with an RTX 4090 idles at ~180 W. Over a year, that's $150 of electricity per box.

## Why not an Apple M4 Max / Ultra

Apple's memory bandwidth is higher (M4 Max is 546 GB/s), memory caps similar (128 GB). Real reasons we picked AMD:

- **ROCm is open** — HIP kernels compile + debug with open source. MLX is open too, but Metal's kernel-dev ergonomics lag ROCm's, and MLX's ternary path lagged BitNet's release by months. When we hit a bug in `v_dot4_i32_i8` on gfx1151 we can patch it; with Metal we'd file feedback-assistant reports.
- **Linux-first** — 1bit systems is CachyOS/Arch. Native systemd, native `rocm-smi`, native `perf`, native Caddy. macOS has launchd and a different permissions model.
- **Price/perf** — Strix Halo $2-3k vs M4 Max Mac Studio $3–5k at equivalent RAM.
- **Upgrade path** — Strix Halo boxes are x86. Add another Ryzen node to the Headscale mesh and you have 256 GB of unified-memory compute. Apple Silicon doesn't federate.

We feature-gate a `mlx-apple` path in `1bit-mlx` so the workspace still compiles and runs on M-series for developers who work cross-platform. But AMD is the performance target.

## Why gfx1151 specifically

RDNA 3.5 iGPU. The `gfx1151` ISA level gives us:

- **WMMA** (Wave Matrix Multiply Accumulate) for int8 + fp16 — used by our ternary GEMV.
- **Wave32** default — 32-lane SIMD matches our kernel tile layouts cleanly. RDNA 2 was wave64-legacy.
- **`v_dot4_i32_i8`** — 4-wide dot product in a single instruction. Our ternary GEMV is built around this.
- **`__builtin_amdgcn_ballot_w32`** — 32-wide predicate ballot for activation-sparsity gating.
- **CDNA features we don't need** — Strix Halo is not CDNA (datacenter Instinct). That's fine; CDNA would be $5k+ per card.

## Software stack on the box

- **OS**: CachyOS with kernel 7.x (for the XDNA 2 `amdxdna` driver) or any distro with equivalent kernel.
- **Compiler**: ROCm 7.x `hipcc` → `clang` 22 → gfx1151 target.
- **Runtime**: `libamdhip64.so.7`, `librocm_cpp.so` (our kernels), `libhsa-runtime64.so.1`.
- **Orchestration**: Rust 1.88+, edition 2024, plus systemd for the live Lemonade / FastFlowLM / proxy stack.
- **Web layer**: Caddy 2.x with internal CA for LAN HTTPS.

## Community

Early — Strix Halo launched mid-2025. We're one of a handful of projects publicly running native HIP ternary kernels on gfx1151. The niche is intentional; the private memory archive carries the "silent-closet BYOA inference" thesis.

## Citations

- AMD Ryzen AI MAX+ 395 — [`https://www.amd.com/en/products/processors/laptop/ryzen/ai-max.html`](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max.html)
- ROCm 7 on gfx1151 — [`https://rocm.docs.amd.com/`](https://rocm.docs.amd.com/)
- BitNet-b1.58 2B on AMD — our benchmark suite, see [`../../benchmarks/`](../../benchmarks/).
