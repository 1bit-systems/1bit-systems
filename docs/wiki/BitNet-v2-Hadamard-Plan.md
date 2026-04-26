# BitNet v2 — Hadamard rotation plan (1bit-systems / gen-2)

Tracker for the **H-BitLinear** activation path (paper: arXiv 2504.18415,
Wang/Ma/Huang/Wei, Microsoft, April 2025). Replaces the BitNet-a4.8 staged
A8→A4 recipe we had queued after Sherry lands.

**Landed**: 2026-04-20 — skeleton kernel + plan only. No forward-pass
wiring, no weight pipeline, no PPL runs yet.

**Updated 2026-04-20 (PM)**: CMakeLists.txt linked, FFI exported, 1bit-hip
safe wrapper landed, lemond `bitnet-v2` feature flag + hook landed.
See §10 for the wiring map. Still missing: **rotated checkpoint** — no
shipping `.h1b` carries `H1B_FLAG_HADAMARD_ROTATED` yet, so the hook is
dormant on every production box today. PPL delta measurement waits on
the offline requantizer (plan §4, item 5).

---

## 1. Algorithm summary

H-BitLinear applies an online block-diagonal **Walsh-Hadamard** rotation
`H_B` to every activation tensor **immediately before** the quantizer in a
ternary linear (Q/K/V/O/gate/up/down). Because `H` is orthogonal
(`H^T H = B · I`; we use `H / sqrt(B)` to make it unitary), it preserves
L2 norm and therefore does not disturb the upstream RMSNorm invariant.
What it *does* do is **smear outlier channels across the block** so that
the post-rotation distribution is near-Gaussian — at which point symmetric
per-token int4 quantization (`scale = max(|x|) / 7`, values in `[-8, +7]`)
becomes viable with a <0.1 PPL regression vs fp16 activations on the 2B4T
base. The rotation is **absorbed into the ternary weights offline**
(`W' = W @ H^T`), so online inference pays for the rotate but *not* for an
inverse rotate on the output side. Native W1.58A4 training, no staged
hand-off, no a4.8-style gradual-bit schedule.

## 2. Hadamard block size choice: **B = 128**

Rationale:

- **Shape math**: BitNet-2B-4T `hidden_size = 2560 = 20*128`,
  `intermediate_size = 6912 = 54*128`, `head_dim = 128`. Every GEMV input
  size in the decode forward pass is cleanly divisible. B=256 would not
  divide 2560; B=64 would leave outlier mixing incomplete on the 6912-wide
  FFN intermediate (the most outlier-heavy tensor per the paper's Table 3).
- **Register/LDS budget**: `128 * 4 B (fp32 accumulator) = 512 B` per block
  in LDS. At 4 workgroups per CU we use 2 KiB LDS — well below the 64 KiB
  gfx1151 per-CU limit; leaves headroom to co-resident with the ternary GEMV
  or the RMSNorm kernel when we fuse.
- **Butterfly depth**: `log2(128) = 7` stages. Each stage is a single
  `__shfl_xor` on wave32 (per Sylvester's `H_{2n} = [[H_n, H_n], [H_n, -H_n]]`,
  stage `s` pairs lanes `i` and `i^(1<<s)`). 7 wave-wide shuffles vs. the
  B² scalar reference (`16384` adds per block) is a ~1000× speedup before
  any further WMMA packing.
- **Paper ablation**: Table 5 of 2504.18415 shows B=128 at the knee of the
  PPL-vs-cost curve; B=256 returns ~0.03 PPL more improvement at 2× the
  kernel cost, B=64 gives up ~0.1 PPL. B=128 is what Microsoft ships.

Implementation note: the kernel is **parameterized on B** as a template /
constexpr so we can run B=256 and B=64 in our PPL sweep before committing.

## 3. Insertion points in our forward pass

All sites are in `crates/lemond/src/backend_impl.rs` (gen-2 Rust
forward driver). Line numbers are as of commit HEAD on 2026-04-20:

| # | site | line | pre-state | post-rotate step |
|---|------|------|-----------|------------------|
| 1 | input_norm → Q/K/V | 530 | `quantize_fp16_to_i8(normed, x_i8, x_scale)` | **insert** `hadamard_rotate_fp16(normed, normed_rot, hs)` before, then replace i8 path with i4 path |
| 2 | attn_sub_norm → O | 640 | same pattern | same |
| 3 | post_attn_norm → gate/up | 674 | same pattern | same |
| 4 | ffn_sub_norm → down | 714 | `quantize_fp16_to_i8(silu_out, ...)` | same, on `silu_out` |

Four rotate sites per layer × 28 layers = **112 Hadamard launches per
decode token**. At B=128 butterfly-optimized, each rotate is a single
wave32 kernel with 7 LDS-free shfl stages plus a load/store pair. Budget:
~1-2 µs per rotate on gfx1151 = ~150-200 µs per token added → ≤2% decode
time if current token budget is ~12 ms (83 tok/s). Acceptable.

We explicitly do **not** rotate the residual stream itself, and we do
**not** rotate between RoPE and attention — only the four pre-GEMV sites
above. RoPE's Givens pairs and Hadamard's ±1 signs don't commute; mixing
them is a correctness bug.

## 4. Kernel list

### New

1. **`src/hadamard_rotate.hip`** — prototype scalar reference landed
   2026-04-20 (`rcpp_hadamard_rotate_fp16_ref_launch`). B=128 hardcoded,
   O(B²) inner loop. Purpose: diff-testable oracle for the butterfly variant.
2. **`kernels/hadamard_rotate_butterfly.hip`** — wave32 butterfly landed
   2026-04-20 (`rcpp_hadamard_rotate_fp16_butterfly_launch`). 7 stages, one
   workgroup per B-sized chunk, fused `1/sqrt(B)` into the final store.
   Stages 0..4 intra-wave via `__shfl_xor` (no LDS); stages 5..6 cross-wave
   via 512 B LDS + `__syncthreads`. 135 LOC, compile-only (not in
   CMakeLists.txt yet). Standalone test at
   `tests/test_hadamard_butterfly.cpp` (136 LOC) exercises bit-exact vs
   scalar ref on 16 blocks and benches a 2048×2560 tile. **Projected**
   per-block cost ~0.1-0.3 µs (memory-bound at fp16 load+store, ~7 cycle
   compute per lane per stage × 7 stages ≈ 50 cycles/block of pure ALU
   + 2 LDS round-trips); measured numbers pending manual hipcc run
   (agent is sandboxed).
3. **`kernels/hadamard_rotate_fused_quant.hip`** *(follow-up)* — fuses
   the butterfly with per-token i4 symmetric quantization (max-reduce →
   scale → quantize → pack) in a single launch. Replaces two separate
   kernel launches per site. Target: single kernel with both the rotate
   and the `quantize_fp16_to_i4` emitted in one grid.

### Modified

4. **`kernels/ternary_gemv_phase5_i4a.hip`** — already consumes `i4`
   activations with a **per-tensor** scale (1bit systems Lane A predecessor).
   Needs: per-token scale path parameterized in (we quantize fresh every
   forward pass, so the scale is per-site-per-token — which is what the
   kernel already handles via the `x_scale` scalar; just re-plumb from
   `x_scale_dev` on the Rust side). No kernel restructure required.
   Add `ternary_gemv_halo_i4a_f16` fp16-output twin mirroring the
   `halo_f16` suffix pattern.
5. **Offline requantizer** — `1bit-core` / tools path. `W' = W @ H^T`
   applied once at `.h1b` export time. Since `H ∈ {+1, -1}` the matmul
   reduces to sign-flipped adds; no multiplies. Output is still ternary
   because `H^T` is ±1 and the ternary weight space is closed under sign
   flip. **Important**: this means model weights re-ship as a new `.h1b`
   variant flag (`H1B_FLAG_HADAMARD_ROTATED`) — existing rocm-cpp
   consumers must reject a non-matching flag cleanly (loader bounds check
   as always).

### Untouched

- `ternary_gemv_halo.hip` (i8 A path) — stays as the fallback lane; don't
  touch per CLAUDE.md Rule.
- `prim_kernels.hip` RMSNorm variants — unchanged. Hadamard lives
  between RMSNorm and quant, not inside either.
- `kv_rotorquant.cpp` + `rotorquant_pack.hip` — orthogonal lane. KV-side
  Givens rotation is unrelated to the activation-side Hadamard (different
  matrices, different tensors).

## 5. PPL budget + validation

Gate: **`benchmarks/ppl-gen2.sh` wikitext-103 ≤ 9.30** (gen-1 baseline
9.1607, gen-2 current 9.1805, i4a acceptance tolerance +0.12 PPL).
Per paper Table 2, 2B4T W1.58A4 vs W1.58A8: +0.05–0.08 PPL on wikitext. We
budget 2× paper margin (measurement variance on our PPL harness is ~0.04
per run) → gate at +0.12.

Validation ladder:

1. **Unit test** (`tests/test_hadamard.cpp`): round-trip
   `x → H x → H^T (H x) / B = x` to ±1 ULP at B=128 over 10k random fp16
   vectors. Catches butterfly-implementation bugs.
2. **Numerical diff** against scalar reference (`hadamard_rotate_fp16_ref`)
   — bit-exact fp16 at B=128. If butterfly diverges, the reference is the
   oracle.
3. **PPL regression run** on wikitext-103 via `bitnet_decode --ppl`
   (landed 2026-04-19 per `project_ppl_harness.md`). Single-layer A/B
   (only layer 0 rotated) first to confirm the sign convention and offline
   requant is correct before re-exporting all 28 layers.
4. **Shadow-burnin** parity vs the existing i8a path for 10 h, checking
   byte-exact argmax rate stays ≥90% (current baseline post special-token
   fix). Mismatch rate above ~15% is a correctness bug, not precision.

## 6. Rank vs Sherry + Medusa

Current decode headline: **83 tok/s** on gfx1151 decode-1-token, **92%
of LPDDR5 peak** per `project_bitnet_rocprof_plan.md` (memory-bound,
not compute-bound).

| candidate | bytes-read reduction | compute cost | expected tok/s | PPL cost | rank |
|-----------|---------------------|--------------|----------------|----------|------|
| **Sherry 1.25-bit** | ternary 1.6 bpw → 1.25 bpw = **22%** fewer weight bytes | +small (sparse unpack) | ~100 tok/s (1.22× memory-bound ceiling) | +0.08 PPL @ QAT | **#1** |
| **BitNet v2 (this)** | no weight reduction; activation bytes = hs*1B vs hs*2B = 50% of act stream | +Hadamard (small, reg-bound) | ~85-87 tok/s (activation bytes are <3% of total DRAM read per decode) | +0.05-0.08 PPL | **#3** |
| **Medusa-2 heads** | N/A — latency hides under speculative acceptance | +verify pass | ~150-180 tok/s (2.0-2.2× on BitNet-2B-4T per existing memory entry) | 0 (exact verify) | **#2** |

**Promotion ranking: Sherry → Medusa → BitNet v2.**

Why BitNet v2 is rank #3 despite being "the obvious next kernel":

- **Activations are not the bottleneck.** Decode at hidden_size=2560 streams
  `28 layers × 4 sites × 2560 × 2 B = ~560 KiB` of activations per token.
  Weights stream `~1.3 GiB / token`. Going fp16→i4 on activations saves
  ~280 KiB — **~0.02%** of total DRAM read per token. The paper's motivation
  is training cost + large-batch inference; for our batch=1 memory-bound
  regime, the payoff is narrower.
- **Sherry hits the actual bottleneck** (weight bytes, 22% reduction on
  the 99.96% of DRAM that is weights). Projected 1.2× decode.
- **Medusa hides latency entirely** by doing ~2 tokens per backbone pass;
  it's a different axis. Stacks cleanly on top of either of the above.
- BitNet v2 is **still worth doing** because: (a) ~2-3% activation-path
  speedup composes, (b) the *same* Hadamard rotation is a prerequisite
  for native W1.58A4 **QAT on Battlemage** (per
  `project_battlemage_distill.md` — training-cost win even if inference
  win is modest), and (c) offline requantizer gives us a cheap
  "W1.58A4 halo-2B" variant to publish alongside the main W1.58A8 weights
  as a size-matched checkpoint for tools that can't afford the i8 scratch.

Land order: finish Sherry → integrate Medusa → then BitNet v2. If
Battlemage lands before Sherry ships, promote BitNet v2 to #1 because the
training-side win dwarfs the 2-3% inference delta.

## 7. Open questions / paper-reading TODO

- Per-token vs per-channel int4 scale — paper Sec 3.2 suggests per-token
  is sufficient post-Hadamard (whole point of the rotation is to
  homogenize). Confirm before committing to per-token in the packer.
- Do we need the Hadamard on the **KV-cache** writes too? If K/V writes
  stay at fp16 we bypass the rotation for KV — but when we eventually
  ship the int8 KV cache as default we'd want to extend. Out of scope for
  this plan; revisit when int8-KV graduates from optional to default.
- The paper's "H-BitLinear" module wraps *both* the quant and the dequant;
  confirm that "absorbed into weights offline" reading matches
  Microsoft's open-source impl (not public at paper release, check for
  updates).

## 8. Memory / references

- `project_bitnet_frontier_2026_04.md` (this item's home row)
- `project_sherry_spike.md` (rank #1 lane — do first)
- `project_battlemage_distill.md` (promotes this if B770 lands early)
- `project_bitnet_rocprof_plan.md` (memory-bound baseline justifying the rank)
- `docs/wiki/Medusa-Integration-Plan.md` (rank #2 lane)

## 9. Prototype compile verification (manual)

Both prototypes now live in the tree; agent is sandboxed off `clang++` /
`hipcc`, so verify by hand before flipping the CMakeLists.txt switch.

Scalar reference:

```bash
/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1151 -O3 \
    -std=c++20 -c \
    /home/bcloud/repos/rocm-cpp/src/hadamard_rotate.hip \
    -o /tmp/hadamard_rotate.o
```

Butterfly kernel (standalone):

```bash
/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx1151 -O3 \
    -std=c++20 -c \
    /home/bcloud/repos/rocm-cpp/kernels/hadamard_rotate_butterfly.hip \
    -o /tmp/hadamard_rotate_butterfly.o
```

End-to-end correctness + bench (scalar ref vs butterfly, bit-exact check
and 2048×2560-tile timing):

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++20 \
    /home/bcloud/repos/rocm-cpp/src/hadamard_rotate.hip \
    /home/bcloud/repos/rocm-cpp/kernels/hadamard_rotate_butterfly.hip \
    /home/bcloud/repos/rocm-cpp/tests/test_hadamard_butterfly.cpp \
    -o /tmp/test_hadamard_butterfly && /tmp/test_hadamard_butterfly
```

Once both verify and the bench shows <1 µs/block, add BOTH files to
`rocm_cpp` sources and the HIP `set_source_files_properties` list in
`/home/bcloud/repos/rocm-cpp/CMakeLists.txt` (line 78-ish block), and add
the `rcpp_hadamard_rotate_fp16_butterfly_launch` prototype to
`include/rocm_cpp/ck_gemm.h`. The `_ref_` launcher stays internal (oracle
only, not production-facing).

### Measured numbers

*(Pending — agent sandbox blocks hipcc. Fill in after manual run.)*

| tile | launches | us/launch | us/block | GB/s | bit-exact vs ref |
|------|----------|-----------|----------|------|-------------------|
| 2048 × 2560 | 40960 blocks | __ | __ | __ | __ |

## 10. Wiring map (2026-04-20)

CMakeLists + FFI + router hook all landed. Dispatch is dormant by default;
flipping it on is a per-build feature flag PLUS a per-model header bit.

### Build / link

- **`rocm-cpp/CMakeLists.txt`** — `kernels/hadamard_rotate_butterfly.hip`
  is in `rocm_cpp SHARED` sources and the `set_source_files_properties …
  PROPERTIES LANGUAGE HIP` block. Build `librocm_cpp.so` via the usual
  `cmake --build build`.
- **`rocm-cpp/include/rocm_cpp/ck_gemm.h`** — exports
  `void rcpp_hadamard_rotate_fp16_butterfly_launch(const void* x_in,
  void* y_out, int K, void* stream)`. Signature takes `K` (total fp16
  element count = n_blocks × 128); device kernel has block size hard-
  coded via constexpr `HADAMARD_BLOCK = 128`.

### FFI (1bit-hip)

- **Hand-rolled**, not bindgen (matches every other `rcpp_*` symbol in
  `crates/1bit-hip/src/ffi.rs`). New raw decl + stub-branch shim for
  CI hosts without ROCm.
- Safe wrappers in `crates/1bit-hip/src/lib.rs`:
  - `hadamard_rotate_fp16_device(x, y, k, stream) -> RcppStatus` —
    device-pointer entry point matching the rest of the crate API.
  - `hadamard_rotate_fp16(in_bits, out_bits, block_size) -> Result<()>` —
    host-slice convenience wrapper (allocates scratch, H2D+D2H). Only
    for tests / offline tooling; not the hot path.
  - `HADAMARD_BLOCK: usize = 128` public constant.
- Divisibility + length preconditions enforced Rust-side before dispatch.

### Feature gate (lemond)

- New Cargo feature `bitnet-v2` in `crates/lemond/Cargo.toml`. Off
  by default. Does NOT depend on the `hip` feature (the code path is
  compiled even for stub-mode builds so tests cover both branches).
- **Model metadata**: `H1B_FLAG_HADAMARD_ROTATED = 0x1` lives in bit 0
  of `H1bConfig::reserved` (= `cfg[8]` on disk). No format version bump;
  existing loaders still parse the file.
  - Accessor: `H1bConfig::is_hadamard_rotated()` in `1bit-core`.
- **Decision matrix** (`HipBackend::resolve_hadamard_gate`):

  | feature | model flag | behaviour                |
  |---------|------------|--------------------------|
  | OFF     | *any*      | never rotate (status quo)|
  | ON      | OFF        | pass-through, warn-once  |
  | ON      | ON         | dispatch at every site   |

- **Hook sites** (4 per layer — 112 launches / token at 28 layers):
  1. input_norm → Q/K/V pre-quant
  2. attn_sub_norm → O pre-quant
  3. post_attn_norm → gate/up pre-quant
  4. ffn_sub_norm → down pre-quant (on `silu_out`, not `normed`)

  Rotation is **in place** on the respective scratch buffer; the device
  kernel completes each lane's load before the store so `x == y`
  aliasing is safe.

- **Test-only counter**: `HipBackend::hadamard_dispatch_count()` is
  compiled under `cfg(any(test, feature = "bitnet-v2"))` so shipping
  builds without the feature pay nothing for it. Integration test
  `tests/smoke_hip.rs::hadamard_hook_fires_four_times_per_layer`
  asserts it divides evenly by 4 per `forward_token`.

### Tests added

- `1bit-core::h1b::tests::is_hadamard_rotated_reads_reserved_bit0` —
  pure flag accessor + foreign-bit isolation.
- `1bit-hip::tests::hadamard_rotate_fp16_preconditions` — GPU-free
  precondition checks on the safe wrapper.
- `1bit-hip::tests::hadamard_rotate_inverse_identity` (`#[ignore]`) —
  `H · H · x / B = x` round-trip within fp16 ε on 1 KiB of random input.
- `lemond::bitnet_v2_hadamard_tests::gate_*` — three tests, one
  per decision-matrix row, `#[cfg]`-gated so we cover both feature-on
  and feature-off builds.
- `lemond::tests::hadamard_hook_fires_four_times_per_layer`
  (`#[ignore]`, needs rotated `.h1b`) — live counter check.

### Still TODO

1. **Offline requantizer** (`1bit-core` tooling path). `W' = W @ H^T`,
   write new `.h1b` with bit 0 of `reserved` set. Paper §3.2 says
   per-token int4 scale post-Hadamard — confirm before committing to
   per-token in the packer.
2. **Fused int4-quant kernel** — `kernels/hadamard_rotate_fused_quant.hip`
   (plan §4 item 3). Replaces two launches per site with one.
3. **PPL delta measurement** — blocked on #1. Gate is
   `benchmarks/ppl-gen2.sh` ≤ 9.30 on wikitext-103 (paper: +0.05-0.08
   PPL, our budget +0.12). **Actual PPL delta TBD until we have a
   rotated checkpoint to load.**
4. **Kernel perf numbers** — fill in §9's "Measured numbers" table
   after a manual `rocprof` run on a live decode.
