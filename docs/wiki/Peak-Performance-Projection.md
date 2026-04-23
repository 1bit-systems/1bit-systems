# Peak Performance Projection

Quantitative projection of 1bit systems end-state throughput on Strix Halo
(Ryzen AI MAX+ 395) once all seven planned lanes ship. Every number has
a derivation or a citation. All projections carry a ±20% band unless
stated otherwise.

## 0. Fixed constants

| constant | value | source |
|---|---:|---|
| LPDDR5-8000 peak bandwidth | 256 GB/s | vendor spec, Zen5 APU |
| iGPU FP16 peak (theoretical, gfx1151) | 118.8 TFLOPS | 40 CU × 1024 FLOPs/cycle × 2.9 GHz |
| iGPU FP16 peak (measured, gfx1151) | **50.17 TFLOPS** (42% of theoretical) | `peak_probe_bench` 2026-04-22, [`wmma-peak-20260422.json`](../../benchmarks/data/wmma-peak-20260422.json) |
| dGPU FP16 peak (measured, gfx1201 RX 9070 XT) | **138.04 TFLOPS** (81% of theoretical) | same run |
| iGPU effective (measured on ternary GEMV) | 24 TFLOPS (40% util) | bench.sh + rocprof |
| NPU INT8 peak | 50 TOPS | `xrt-smi validate`, [IRON #55](https://github.com/amd/IRON/issues/55) |
| NPU effective (42% of peak, BF16 GEMM) | 21 TOPS | IRON #55, andrej comment 3706297989 |
| NPU effective, int2-via-int8 (2× pack factor) | ~42 TOPS | two ternary weights per INT8 MAC lane |
| BitNet-b1.58-2B-4T: hidden / layers / ffn | 2560 / 30 / 6912 | Microsoft HF config |
| Ternary weight footprint (core MM only) | ~630 MB | 2.4B params × 1.58 bits + packing overhead |
| .h1b on disk (full model incl. embed/tok) | 1.8 GB | [Benchmarks.md](./Benchmarks.md) |

## 1. Current ceiling per surface (measured 2026-04-20)

| surface | decode tok/s today | prefill tok/s today | bound by |
|---|---:|---:|---|
| iGPU (gfx1151) | 83 @ L=64, 33 @ L=1024 | ~220 @ L=512 (measured) | LPDDR5 bw (92% of peak) |
| CPU (32× Zen5) | 0 (idle) | 0 | not wired |
| NPU (XDNA2, 8×4 AIE2P) | 0 | 0 | IRON #93 int8 kernel not yet ported to Peano |

Bandwidth-bound ceiling on the ternary GEMV today:
**256 GB/s ÷ 630 MB/tok ≈ 406 tok/s theoretical**. Measured 83 tok/s
@ L=64 means the reachable roofline after amortizing KV + attn + sampler
is ~85–90 tok/s — consistent with [Benchmarks.md](./Benchmarks.md).
At L=1024 the KV-cache dominates: 33 tok/s × 92% × 256 GB/s ≈
7.1 GB read per token, essentially all KV.

## 2. Decode ceiling math

Formula: `tok/s ≤ (BW × util) / (W_bytes × actsparse_factor)` where
`util = 0.92` from the rocprof roofline.

| config | W_bytes | actsparse | ceiling | realistic (±20%) | source |
|---|---:|---:|---:|---:|---|
| FP16 2.4B baseline | 4.80 GB | 1.0 | 49 tok/s | 35–45 | llama.cpp fp16 on same box |
| Ternary 1.58-bit (today) | 0.63 GB | 1.0 | 374 tok/s | **83 @ L=64** | bench.sh, measured |
| +Sherry 1.25-bit 3:4 | 0.50 GB (×0.79) | 1.0 | 471 tok/s | 105–125 | 1.25/1.58 bytes ratio, spike committed 2026-04-18 |
| +Sherry +activation sparsity 30% eff | 0.35 GB | 0.70 | 673 tok/s | 140–170 | 79.91% measured sparsity, 30% usable after DRAM granularity penalty |
| +Sherry +actsparse +BitNet v2 W1.58A4 | same W | same | same @ short ctx | 140–170 @ L≤256 | [2504.18415](https://arxiv.org/abs/2504.18415) |
| +Sherry +actsparse +A4 **@ L=2048** | KV /4 | — | ~132 tok/s (vs 22 today) | 110–160 | a4 KV is 0.25× fp16 KV; fd-attn 6.78× already landed |
| +all +Medusa 1.7× accept | — | — | — | **240–290 @ L=64**; 190–270 @ L=2048 | [MedusaBitNet](./Medusa-Integration-Plan.md), 1.5–1.8× accepted-token range |

The 83 → ~280 tok/s path is **compounding on a bandwidth-bound stack**:
every lane that shrinks bytes-per-token lifts the same wall. Medusa
multiplies on top because it issues ≥2 tokens per weight-read pass.

## 3. Prefill ceiling (compute-bound, NPU wins)

Prefill per-token FLOPs (fwd, attn + FFN, BitNet-2B):

```
F/tok ≈ 2 × (4·h² + 3·h·ffn) × layers
      = 2 × (4·2560² + 3·2560·6912) × 30
      = 2 × (26.2M + 53.1M) × 30
      ≈ 4.76 GFLOPs/tok
```

| surface | effective TOPS | prefill tok/s | dispatch overhead |
|---|---:|---:|---:|
| iGPU (FP16) | 24 TFLOPS | 24e12 / 4.76e9 ≈ **5 000 tok/s** | ~0.2 ms |
| NPU (INT8 straight) | 21 TOPS | 21e12 / 4.76e9 ≈ **4 400 tok/s** | 2–5 ms |
| NPU (int2 via INT8 MAC) | ~42 TOPS | 42e12 / 4.76e9 ≈ **8 800 tok/s** | 2–5 ms |

Crossover L\* where NPU total time beats iGPU total time, solving
`t_npu_oh + L/npu_tps = t_igpu_oh + L/igpu_tps`:

```
L* = (3ms − 0.2ms) / (1/5000 − 1/8800)
   = 2.8e-3 / (2.00e-4 − 1.14e-4)
   = 2.8e-3 / 8.6e-5
   ≈ 33 tokens
```

**NPU beats iGPU at prefill beyond ~33 tokens.** For realistic chat
prompts (128–2048 in) the NPU is the right surface unconditionally.
Uncertainty band: 25–60 tokens depending on actual BD-list setup cost
([IRON #93](https://github.com/amd/IRON/issues/93) has no live timing yet).

## 4. All-surfaces-parallel latency (512-in + 256-out)

| stage | surface | time | note |
|---|---|---:|---|
| Tokenize 512 in | CPU (1 core) | 4 ms | 1bit-core BPE, already measured |
| Prefill 512 tok | NPU | 512 / 8 800 + 3 ms ≈ 61 ms | int2-via-int8 projection |
| RoPE tables + dispatch | CPU | <1 ms | precomputed, cached |
| Decode 256 tok @ L≈768 | iGPU | 256 / 220 ≈ **1.16 s** | Sherry + actsparse + a4 + Medusa 1.7× |
| Detokenize + stream out | CPU | <5 ms | overlaps decode |

**TTFT ≈ 70 ms, total wall ≈ 1.23 s for 512+256.** Today the same
workload is ~4.1 s (prefill on iGPU, decode bottleneck). Projected
wall-clock improvement: **~3.3×** end-to-end.

## 5. End-to-end decode tok/s ceiling (2B model, L=64)

| tier | shipping lanes | projection | assumptions |
|---|---|---:|---|
| **Conservative** | 2 of 7 (Sherry + fd-attn) | **110–130 tok/s** | no NPU, no Medusa, a4/actsparse deferred |
| **Realistic** | 5 of 7 (Sherry + fd-attn + actsparse + a4 + Medusa) | **190–240 tok/s** | NPU still dark; decode is iGPU-only |
| **Aspirational** | 7 of 7 + clean ROCm 7.2 driver | **260–300 tok/s** | NPU prefill frees iGPU for decode-only duty cycle; driver jitter <3% |

The 280 tok/s headline number assumes (a) Sherry packs to 1.25 bits with
no PPL regression >0.1, (b) BitNet v2 a4 KV is bit-exact vs fp16 KV on
our re-pack path, (c) Medusa 1.7× accept rate on our workload (lower
than MedusaBitNet's reported 2.3× because our baseline is already faster).

## 6. The bandwidth wall

Even at 280 tok/s decode with every lane shipping, the final ceiling is
the LPDDR5-8000 controller. Derivation of the wall:

```
theoretical max = 256 GB/s × 0.92 util / (0.35 GB/tok best-case)
               ≈ 673 tok/s
```

Medusa multiplies tokens-per-weight-fetch by 1.7×, which is the only
way past ~400 tok/s on this bus. Beyond that you need:

- **DDR6** (consumer, ~2027): 512–768 GB/s → linear 2–3× lift.
- **HBM3** (not coming to Ryzen AI APUs — Strix Halo's LPDDR5 is soldered).
- **On-die SRAM residency** (Cerebras territory — not applicable to 2B
  at batch=1).

For the 2026-04 Strix Halo box, **256 GB/s is the terminal ceiling**.
Everything we ship between now and DDR6 is bytes-per-token reduction.

## Sources

- Live measurements: [Benchmarks.md](./Benchmarks.md), `benchmarks/bench.sh`,
  `benchmarks/ppl-gen2.sh`, `benchmarks/attn_fd.sh`.
- NPU hardware: [IRON #55](https://github.com/amd/IRON/issues/55) (column count,
  TOPS), [IRON #93](https://github.com/amd/IRON/issues/93) (INT8 kernel),
  [PR #94](https://github.com/amd/IRON/pull/94).
- Model: Microsoft [BitNet b1.58](https://arxiv.org/abs/2402.17764),
  [BitNet v2](https://arxiv.org/abs/2504.18415).
- Memory notes: `project_bitnet_live_bench.md`, `project_sherry_spike.md`,
  `project_activation_sparsity_phase1.md`, `project_attention_fd.md`,
  `project_npu_path_analysis.md`.

## One-line homepage summary

> Projected ceiling: ~280 tok/s decode + NPU prefill crossover at ~33
> tokens once all seven lanes ship, at the 256 GB/s LPDDR5 wall.
