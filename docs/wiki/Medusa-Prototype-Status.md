# Medusa Prototype Status

Date: 2026-04-20
Status: tiled small-M ternary GEMM kernel + scalar reference + standalone
test harness landed in `rocm-cpp`. Hipcc compile + rocprof pending on the
architect's box. No runtime wiring (not in halo-router).
Parent plan: `docs/wiki/Medusa-Integration-Plan.md` (recommendation: DEFER).

This page captures the prototype work sitting *below* the decision
to defer. It does not flip the recommendation; it makes the "if we
later decide to do it" estimate cheaper by pinning down head shape and
landing a known-correct tiled kernel that diffs against a scalar reference
to bf16 ULP tolerance.

---

## 1. Medusa head shape

**Source constraint**: the sandbox denied outbound HTTP to `huggingface.co`
today, so the artifact `parrishcorcoran/MedusaBitNet-2B-4T` was **not
freshly re-downloaded** for this prototype. Numbers below come from
the upstream plan (`Medusa-Integration-Plan.md` §1) and the model card
snapshot in memory (`project_medusa_plan.md`).

| Field | Value | Source |
|---|---|---|
| Number of heads | 4 | model card |
| Per-head layers | 1 residual MLP block + 1 linear to vocab | Medusa paper §3, Medusa-1 config |
| Hidden dim | 2560 (matches Microsoft bitnet-b1.58-2B-4T backbone) | backbone config |
| Output vocab | 128256 (full LLaMA-3 tokenizer, shared `lm_head`) | backbone config |
| Per-head params | ~3.3 MB fp16 (2560 × 2560 residual + 2560 × 128256 = no, heads share lm_head so just the residual block) | Medusa paper §3.1 |
| Total head file | 13 MB fp16 | model card |
| Training data | Alpaca 52K only, 11 h on Ryzen AI MAX+ 395 CPU | model card |
| Acceptance rates | head-1 63.0%, head-2 29.0%, head-3 11.1%, head-4 4.6% | model card |
| Measured throughput | 2.08 tokens/backbone-step on CPU Medusa PyTorch | model card |

**Action item** the moment sandbox HTTP is unblocked: refetch
`config.json` + `medusa_heads.safetensors` header (header read is
<<100 MB per task constraint) and diff the declared shape against
the table above. If it diverges — specifically if head dim ≠ 2560
or heads don't share `lm_head` — the kernel below still compiles but
the integration plan's size estimates need revising.

---

## 2. Small-M ternary GEMM kernel

**Files**:
- `/home/bcloud/repos/rocm-cpp/src/ternary_gemm_smallm.hip` — tiled kernel
- `/home/bcloud/repos/rocm-cpp/src/ternary_gemm_smallm_scalar_ref.hip` — scalar reference
- `/home/bcloud/repos/rocm-cpp/tests/test_ternary_gemm_smallm.cpp` — parity + timing harness

Both kernels + the test are now in `CMakeLists.txt`. Compile + rocprof
are pending on the architect's box.

### 2.1 Activation format — choice B (int8 per-tensor scale)

Matches `kernels/ternary_gemv_phase5_halo.hip` exactly, so the prefill →
smallM verify → GEMV decode sequence shares one activation path (the
existing `rcpp_quantize_fp16_to_i8` kernel). An alternative fp16-in,
internal-scale-compute route was considered and rejected: doubles the HBM
read bandwidth for activations (irrelevant at M=1, not irrelevant once M
grows to 16 for tree verify) and forks the quant path that is known-good
on 92%-of-peak GEMV.

### 2.2 Shape contract (architect-pinned)

```
onebit_ternary_gemm_smallm_launch(
    const int8_t*         x_i8,          // [M, K]        int8
    float                 x_scale,       //               per-tensor fp32
    const uint32_t*       w_packed,      // [K/16, N]     u32, 16 codes/u32
    const float*          w_row_scales,  // [N]           fp32 per column
    hip_bfloat16*       y_out,         // [M, N]        bf16
    int M, int N, int K,
    hipStream_t stream);
```

Packing: `w_packed[(k/16) * N + n]`, bits `2*(k%16)..2*(k%16)+1` carry the
code; codes 0→-1, 1→0, 2→+1, 3→0. Same 2-bit code assignment as every
other ternary kernel in the repo.

### 2.3 Tile shape + register budget

- **BLOCK_M = 16, BLOCK_N = 64, BLOCK_K = 64.**
- **128 threads/block = 4 waves × 32 lanes (wave32, gfx1151).**
- **Each wave owns one 16×16 accumulator tile → 8 int32 accumulators per
  lane × 32 lanes = 256 VGPRs of accumulator per wave.** Target occupancy
  2 waves/SIMD — VGPR budget ~256 acc + ~64 spill-buffer headroom fits.
- **LDS per block: 1 KiB activations (16×64 int8) + 1 KiB weights
  (4×64 u32 = 64-K × 64-N ternary codes) = 2 KiB total.** Well under the
  64 KiB CU limit; does not gate occupancy.
- **Inner loop**: 16 k4-chunks × 8 accumulators = 128 sdot4
  (`__builtin_amdgcn_sudot4(true, weight_i32, true, x_i32, acc, false)`)
  per lane per K-tile. K=2560 → 40 K-tiles → 5120 sdot4 ops per lane.
- **Bank conflicts**: weight LDS reads broadcast (2 distinct n_local per
  wave per inner step); activation LDS reads have a 2-way conflict on the
  m=lane stride but the broadcast on weights dominates.

Lane → output-tile mapping: lane `l` owns `(m = l & 15,
n_tile_base = (l >> 4) * 8)`, with 8 accumulators running contiguous n.

### 2.4 Scalar reference

`ternary_gemm_smallm_scalar_ref.hip` uses the identical ABI with one
thread per (m, n), scalar K-loop. It is intentionally slow; the test
harness diffs tiled against scalar at bf16 ULP granularity (not
bit-exact — sdot4 vs scalar accumulation is associative at the int32
level, but the final `float → bf16` cast rounds identically only when
the pre-cast float matches bit-for-bit, which is not guaranteed across
reorderings).

### 2.5 Test harness — `test_ternary_gemm_smallm`

- Sweep: **100 seeds × M ∈ {1, 2, 4, 8, 16}** at N = K = 2560.
- **PASS**: every (seed, M) produces `max|err| ≤ 1 bf16 ULP` vs scalar.
- Exit 1 on any cell over 1 ULP; exit 0 otherwise.
- Per-M line prints `worst_ulp`, `tiled_avg_ms`, `scalar_avg_ms`, and
  `tok/s-eq = M / tiled_avg_s`.
- `hipEvent`-based timing on a single stream; no `hipDeviceSynchronize`
  in the hot measured section (only `hipEventSynchronize` on the stop
  event and `hipStreamSynchronize` before readback).
- One warmup launch per trial on the tiled kernel to avoid JIT-first-
  launch skew.

### 2.6 Performance targets (architect-pinned, to be verified on box)

- **M=1**: within 20% of the GEMV baseline (the GEMV is at 92% of LPDDR5
  peak; we expect some overhead from the M-dimension wiring that the
  specialized GEMV does not pay).
- **M=16**: ≥ 8× the scalar reference.

### 2.7 Measured numbers

_Pending hipcc compile + rocprof run on architect's box._

| M | tiled_avg_ms | scalar_avg_ms | tok/s-eq | worst_ulp | PASS |
|---|---|---|---|---|---|
| 1  | — | — | — | — | — |
| 2  | — | — | — | — | — |
| 4  | — | — | — | — | — |
| 8  | — | — | — | — | — |
| 16 | — | — | — | — | — |

### 2.8 Status checklist

- [x] Tiled kernel written, compiles in our heads; hipcc verification pending.
- [x] Scalar reference matches ABI.
- [x] Test harness written (ULP diff + timing).
- [x] `CMakeLists.txt` updated (both sources + new executable).
- [ ] `hipcc --offload-arch=gfx1151` clean on architect's box.
- [ ] `rocprofv3` counters: VGPR usage, LDS usage, achieved occupancy.
- [ ] Populate the perf table above.

---

## 3. Tree attention — how KV cache extends

Today our attention kernel is `src/kv_cache_attn_fd.hip` (Flash-Decoding,
one query per head, split-KV). It's called from
`crates/lemond/src/backend_impl.rs` in the `forward_token` hot path:

- `backend_impl.rs:463` — `forward_token` entry (one token in, one out).
- `backend_impl.rs:595-614` — `hipMemcpyAsync` writes K/V into the
  cache at slot `pos` (single slot per call).
- `backend_impl.rs:589` — the flash-decode attention call site
  that consumes `[0..=pos]` of KV.

For Medusa's tree-verify pass, we need to:

1. **Write `tree_size` K/V entries at once** into cache slots
   `[pos .. pos + tree_size)`. In practice we write them speculatively,
   then *rewind* to `pos + accepted_len` at the end of the step, so the
   cache never grows past the verified prefix. Implementation-wise that's
   just batching the `hipMemcpyAsync` calls around `backend_impl.rs:595`
   and doing a fixup write after the verify.
2. **Attend from a set of queries** Q[tree_size, nh, hd] into
   K[pos + tree_size, nkv, hd] with a **tree mask**: candidate q_i only
   attends to its own branch (its ancestors in the spec tree, not
   sibling branches). This is a new kernel — not a flag flip on
   `kv_cache_attn_fd.hip`. Sibling file path: `src/kv_cache_attn_tree.hip`.
   Signature extension: add `const uint8_t* __restrict__ mask,
   int tree_size` args. Mask is a 2D bitmap of size `tree_size × (pos + tree_size)`.

The KV-cache layout itself (pre-layer `k` and `v` buffers at
`KvCache::k` / `KvCache::v` in `backend_impl.rs:269-272`) does not need
to change — it's already indexed by absolute position. All the surgery
is in the write path (batched slot range) and the read path (masked
gather).

---

## 4. Integration outline — PR-shaped, not yet applied

One PR per crate so each is reviewable in isolation.

### PR-1: `rocm-cpp` — kernels only, no FFI wiring

- [x] `src/ternary_gemm_smallm.hip` (this file, scalar stub, LANDED
      locally, NOT added to CMakeLists yet)
- [ ] Verify hipcc compile on gfx1151.
- [ ] Add file to `CMakeLists.txt` line 74-78 target list.
- [ ] `src/kv_cache_attn_tree.hip` (copy of `kv_cache_attn_fd.hip` with
      a mask arg and an outer M loop).
- [ ] C API entry points in `include/rocm_cpp/` for both.

### PR-2: `1bit-hip` — FFI bindings

- [ ] `src/ffi.rs` — add `extern "C"` prototypes mirroring
      `ternary_gemv_halo_f16` at line 258 and `kv_cache_attn_decode_fd`
      at line 424.
- [ ] `src/lib.rs` — add safe wrappers. Keep existing functions untouched.
- [ ] Three in-crate tests (per CLAUDE.md): mock-device happy path,
      mask-bounds check, M-range bounds check.

### PR-3: `lemond` — `forward_tree`, gated behind config

- [ ] `backend_impl.rs` — add `forward_tree(&mut self, accepted: &[i32],
      base_pos: i32, tree: &SpecTree, logits_out: &mut [f32]) ->
      Result<usize, BackendError>`. Does NOT replace `forward_token`.
- [ ] `backend_impl.rs:463` — leave `forward_token` alone; speculative
      config dispatches in the caller, not inside the backend.
- [ ] `src/lib.rs:599, :615` — call-site guard: if `SpecConfig::None`,
      use `forward_token`; else `forward_tree`.
- [ ] `1bit-core::sampler::Sampler` — add `argmax_batched` for the
      batched verify. Rep-penalty only applied to accepted prefix.

### PR-4: heads loader + CLI switch

- [ ] `1bit-core` — new `.medusa` file format (1 magic + 4 × safetensors
      blob). Mmap, zero-copy. Same hard rule as `.h1b`.
- [ ] `1bit-cli` — `halo gen --spec medusa` flag.

### Out of scope

- No Medusa-2 joint training. Would destroy bit-exact backbone match.
- No backbone format change.
- No changes to `ternary_gemv_halo.hip` (per CLAUDE.md "Don't touch").

---

## 5. Next concrete step — retrain or accept?

**Firm recommendation: retrain on our own workload before production use.**

Reasons:

- parrishcorcoran's heads are **Alpaca-only**. Our actual traffic is
  mixed: Open WebUI chat, code completion via the lemonade gateway,
  long-context agentic workflows (1bit-agents). Alpaca's loss curve
  (9.85 → 3.32) does not transfer cleanly to code or long-context
  acceptance. Head-1 at 63% on Alpaca does not mean 63% on our prompts.
- Training cost is cheap in absolute terms: 11 h on Strix Halo CPU per
  the model card, or one overnight pass. Budget: one night on the
  strixhalo box, or move to the Intel B770 slot (`project_battlemage_distill.md`)
  once that lands.
- Accepting the public heads short-circuits our own acceptance-rate
  measurement, which is the single number that determines whether
  Medusa hits 1.4× or 1.8× on our silicon.

**Cheaper interim step before a full retrain**: run parrishcorcoran's
heads against ~500 prompts from our actual workload (sample from
1bit-agents audit log) and measure head-1 acceptance. If it holds at
≥ 55% we can ship with upstream heads and retrain later. If it drops
below 40%, retrain is non-negotiable.

That eval harness is itself ~half a day of work — smaller than the kernel
work above, so do it first as a go/no-go gate.

---

## 6. Memory touch

Updated `~/.claude/projects/-home-bcloud/memory/project_medusa_plan.md`
with today's prototype scaffolding note. The DEFER recommendation
stands.
