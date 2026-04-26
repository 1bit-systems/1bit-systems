# APU Aggregator Plan

Strategic pivot, 2026-04-20. 1bit systems stops framing itself as the
iGPU-only ternary runtime and becomes **the APU aggregator for Strix
Halo** — the only runtime that orchestrates CPU + iGPU + NPU across the
128 GB unified LPDDR5 pool for sub-2-bit inference.

## 1. Thesis

Strix Halo's moat is not compute density; it is **unified APU memory**.
LPDDR5-8000 at 256 GB/s is shared cache-coherently by 16 Zen5 cores,
Radeon 8060S (gfx1151, 40% util measured), and XDNA 2 (50 TOPS int8). No
other consumer platform has this shape at this price. 1bit systems' job
is to route each phase of inference to the surface that dominates it:
**CPU for prefill** (16× Zen5 AVX2 is compute-dense and already idle
during iGPU decode), **iGPU for decode** (bandwidth-bound ternary GEMV
already at 92% of LPDDR5 peak), **NPU for bulk matmul at L ≥ 2048**
(future, compute-bound territory). Trillim + oxibonsai have proven the
sub-2-bit CPU market exists; our moat is no longer "the only runtime" —
it is cross-surface orchestration on a single SoC.

## 2. Crossover thresholds

Measured 2026-04-20 on strixhalo (Ryzen AI MAX+ 395, 128 GB LPDDR5-8000):

| surface | phase | tok/s | notes |
|---|---|---:|---|
| iGPU decode (ternary HIP GEMV) | L=64 | 50–60 | HaloV2 64/55, TQ1 50/56, Bonsai-TQ2 48/60 |
| iGPU decode | L=1024 | 33 | KV-dominant, fd-attn landed |
| iGPU prefill (same kernel) | pp@64 | ~47 | M=1 GEMV, not a real prefill path |
| iGPU kernel roofline | — | 126 GB/s | 49% of LPDDR5 peak on the mixed workload |
| CPU (16× Zen5 AVX2, ternary) | pp | **projected 200–400** | extrapolated, see below |
| NPU (XDNA 2) | any | 0 | deferred, see `project_npu_path_onnx.md` |

**Trillim reference** (https://trillim.com/blog/trillim-supports-ternary-bonsai/):
Intel i7 Alder Lake ternary prefill on Bonsai-1.7B hits **69 tok/s pp**
on their 4P+8E-core AVX2 path. Scaling factors to Zen5 × 16P-core:

- Core count: 16P vs Trillim's ~4P-effective → **≈3×**
- AVX2 density per core: Zen5 has 2× 256-bit FMAC pipes fully-pipelined
  (vs Alder Lake P-core's 2× 256-bit with shared port 5) → **1.1–1.4×**
- Clock: 5.1 GHz boost vs ~5.0 → **≈1.0×**

Product: 3.3–4.2× → **227–290 tok/s pp projected** on Bonsai-class,
plausibly higher on BitNet-2B because the larger hidden dimension
amortizes per-core dispatch overhead. We carry a ±30% band because we
have no measurement yet; 200 tok/s is the floor we plan against.

**Crossover logic**:

- **Decode**: iGPU, always. CPU ternary decode would compete with iGPU
  for the same LPDDR5 bus and lose on peak bandwidth (iGPU gets ~220
  GB/s effective vs CPU ~160 GB/s streams).
- **Prefill**: CPU when prompt length > 33 tokens. The 33-token
  threshold is derived in `Peak-Performance-Projection.md` §3 from
  `t_dispatch_oh + L/tps_A = t_dispatch_oh + L/tps_B`. Below 33
  tokens the dispatch-overhead noise dominates and iGPU's already-hot
  stream wins.
- **Prefill @ L ≥ 2048**: NPU, when available. XDNA 2 at 21 TOPS
  effective = ~4 400 tok/s prefill ceiling, ~10× CPU projection. Only
  matters once AMD ships STX-H Linux NPU or we write Peano AIE kernel
  ourselves.

## 3. Architecture

```
          ┌──────────────────────────────────────────────────┐
          │                 lemond                       │
          │                                                   │
          │   route_forward(prompt_len, ctx_len, phase):      │
          │     prefill && L < 33   →  Backend::IGpu          │
          │     prefill && L ≥ 33   →  Backend::Cpu           │
          │     prefill && L ≥ 2048 →  Backend::Npu (future)  │
          │     decode              →  Backend::IGpu          │
          └────────┬──────────────────┬─────────────────┬─────┘
                   │                  │                 │
         hidden-state handoff (LPDDR5, zero-copy where lane permits)
                   │                  │                 │
          ┌────────▼───────┐  ┌───────▼────────┐  ┌─────▼─────────┐
          │  CPU lane      │  │  iGPU lane     │  │  NPU lane     │
          │                │  │                │  │  (future)     │
          │ crates/        │  │ crates/        │  │ crates/       │
          │   lemond/ │  │   1bit-hip +   │  │   1bit-npu    │
          │   cpu_lane.rs  │  │   rocm-cpp/    │  │   (ORT +      │
          │                │  │                │  │    VitisAI)   │
          │ ternary AVX2   │  │ ternary HIP    │  │ xclbin overlay│
          │ GEMV           │  │ GEMV + fd-attn │  │ AIE kernels   │
          │                │  │                │  │               │
          │ sampler +      │  │ decode loop,   │  │ INT8 / int2-  │
          │ detokenizer    │  │ KV cache       │  │ via-INT8 GEMM │
          └────────┬───────┘  └───────┬────────┘  └─────┬─────────┘
                   │                  │                 │
          ┌────────▼──────────────────▼─────────────────▼─────────┐
          │                    .h1b weights                       │
          │                                                       │
          │   CPU reads:  packed-blocks-native (AVX2 stride)      │
          │   iGPU reads: HIP-layout (wave-coalesced)             │
          │   NPU reads:  xclbin overlay + DMA descriptors        │
          │                                                       │
          │   Single file on disk; lane-specific views materialized│
          │   at load time. See docs/wiki/Why-H1b-Format.md.      │
          └───────────────────────────────────────────────────────┘
```

Hidden-state handoff between lanes is zero-copy on Strix Halo because
LPDDR5 is cache-coherent across CPU/iGPU/NPU. The handoff cost is a
cache-line flush + a 64-byte descriptor write to the receiving lane's
command queue, not a DMA copy. Measured handoff overhead is in
`Peak-Performance-Projection.md` §3 (0.2 ms iGPU dispatch, 2–5 ms NPU
BD-list setup, <1 ms CPU rayon wake).

## 4. Phases

### Phase 1 — CPU AVX2 ternary GEMV kernel *(sibling lane, kicks off today)*

- New subtree `rocm-cpp/cpu-avx2/` (sibling to the HIP kernel subtree).
  C++20, pure AVX2, no AVX-512 dependency (Zen5 has AVX-512 but we
  target the AVX2 baseline so the same binary runs on Zen4 APU boxes if
  we ship a lower-tier SKU later).
- Packing format: `packed-blocks-native` — same 5-trit → 8-bit packing
  as the HIP kernel, but strided so 32 trits land in one 256-bit YMM
  register after a `vpshufb` unpack.
- FFI surface: `crates/1bit-cpu-avx2` with the same `ForwardToken`
  trait as `crates/1bit-hip`. Router picks one at dispatch time.
- Parity gate: bit-exact vs iGPU GEMV on the Bonsai-1.7B + BitNet-2B
  reference tensors. Same parity harness as `benchmarks/ppl-gen2.sh`
  but on CPU.
- Target measurement: 200+ tok/s pp on BitNet-2B @ L=512. If we miss
  by more than 30% (i.e. below 140 tok/s), escalate before Phase 2.

### Phase 2 — `lemond` prefill/decode split

- `Backend::Cpu` is already scaffolded (see `CPU-Lane-Plan.md`). Extend
  `RouterConfig` with `PrefillBackend::Auto | IGpu | Cpu | Npu`.
- `route_forward()` consults the Phase 1 benchmark numbers at startup
  to pick the crossover L (default 33, overridable via
  `HALO_PREFILL_CROSSOVER_L`).
- Server changes: `generate_blocking()` in `crates/1bit-server` calls
  prefill through a new `prefill()` method that can target a different
  backend than decode. KV cache handoff lives on the iGPU — CPU
  prefill writes K/V tensors directly into the iGPU's KV buffer using
  the LPDDR5 cache-coherent path.
- Tests: three new parity tests in `crates/lemond/tests/` —
  `cpu_prefill_igpu_decode_matches_igpu_full`,
  `crossover_threshold_respected`, `env_override_works`.

### Phase 3 — NPU integration

Gated behind one of:

1. AMD ships STX-H Linux NPU support in Ryzen AI SDK (today's 1.7.1
   lists STX + KRK only).
2. Microsoft/BitNet merges an XDNA backend (issue #408, dormant since
   Feb 2026).
3. We complete the Peano AIE kernel ourselves — 6–12 month bet per
   `project_npu_path_analysis.md`.

Primary lane when we light this up: ONNX Runtime C++ API + VitisAI EP
(`project_npu_path_onnx.md`), with Peano AIE kernels for ops VitisAI
doesn't accelerate. Rule A clean because ORT has a C++ API and Peano is
C++ codegen.

## 5. Differentiation vs competitors

| runtime | CPU | iGPU | NPU | on Strix Halo? |
|---|:-:|:-:|:-:|:-:|
| Trillim | yes | — | — | yes (CPU only) |
| oxibonsai (cool-japan) | yes | Metal + CUDA | — | partial (CPU only) |
| microsoft/BitNet | yes | partial CUDA | — | no |
| llama.cpp + BitNet fork | yes | HIP (non-ternary) | — | partial |
| **1bit systems** | **yes (Phase 1)** | **yes (shipping)** | **yes (Phase 3)** | **native** |

No other runtime aggregates CPU + iGPU + NPU on Strix Halo. Our
position is "only", not "best" — the comparison is binary today.

## 6. Brand implication

Revise the hero line. Today's landing page (`cpp/landing`)
leads with "the only sub-2-bit runtime on Strix Halo". After the pivot
it reads:

> **The APU aggregator for Strix Halo.** One runtime, three surfaces,
> unified 128 GB memory, sub-2-bit weights.

`Why-Strix-Halo.md` already makes the unified-memory argument; the new
hero borrows that language. The "only sub-2-bit" claim moves into a
secondary badge with the Trillim/oxibonsai footnote so we stay honest.
See `project_brand_lockin.md` for the canonical wording discipline.

## 7. Risks and gates

- **CPU AVX2 kernel might underperform projection.** We have no
  measurement yet. The 200–400 tok/s pp range is extrapolation from
  Trillim's 69 tok/s × Zen5-scaling. If Phase 1 lands below 140 tok/s
  the crossover story narrows and we may need to target only very long
  prompts (L > 256) for CPU prefill. *Gate*: Phase 1 parity harness
  must hit ≥140 tok/s pp on BitNet-2B @ L=512 before Phase 2 ships.
- **Cross-lane handoff latency could eat the crossover win.** The
  zero-copy LPDDR5 path is theoretically sub-millisecond but rocprof
  traces show real HIP dispatch jitter in the 100 µs–2 ms range. If
  handoff adds more overhead than the threshold calculation assumes,
  the crossover L shifts right (maybe to 100+ tokens). *Gate*: Phase 2
  measures actual handoff cost end-to-end; if > 3 ms, recompute L*.
- **Trillim or oxibonsai could add a HIP backend.** If either does,
  our iGPU moat evaporates and we are left with only the NPU lane as
  differentiation — which is 6–12 months out. *Mitigation*: ship
  Phase 1 + Phase 2 fast so we are the first runtime with a working
  cross-surface router; being "the aggregator" is defensible even if
  each individual surface gets commoditized.
- **NPU lane is a 6–12 month bet.** Phase 3 depends on external
  conditions (AMD SDK Linux SKU coverage) we do not control. The plan
  does not assume NPU lands; the CPU + iGPU aggregator is a complete
  story on its own.
- **Trillim may extend to >16 cores fast.** Their CPU kernel is already
  well-tuned; our 3× Zen5 advantage holds for the specific 16P-core
  Strix Halo SKU but shrinks on Threadripper. That is fine — we are a
  Strix-Halo-specific runtime by design.

## 8. Cross-references

**Memory files** (`~/.claude/projects/-home-bcloud/memory/`):

- `project_halo_vision.md` — origin story, why Strix Halo, efficiency thesis.
- `project_bitnet_live_bench.md` — live iGPU numbers feeding §2.
- `project_npu_path_onnx.md` — ORT + VitisAI EP pivot, Phase 3 primary lane.
- `project_account_canonical.md` — `bong-water-water-bong` handle, commit discipline.
- `project_brand_lockin.md` — hero-line wording, claim discipline.

**Wiki docs**:

- `Peak-Performance-Projection.md` — derives the 33-token crossover
  (§3), the 7-surface roadmap, the 256 GB/s bandwidth ceiling.
- `CPU-Lane-Plan.md` — sampler-side CPU lane (surface #7); this plan
  extends the CPU lane from sampler-only into prefill GEMV.
- `Why-No-NPU-Yet.md` — full NPU evaluation, defer rationale, Phase 3 gates.
- `Why-Strix-Halo.md` — unified-memory pitch referenced in §6.
- `Why-Ternary.md` — bandwidth-bound argument underpinning the CPU/iGPU
  surface split.
- `Why-H1b-Format.md` — weight-file format consumed by all three lanes.
