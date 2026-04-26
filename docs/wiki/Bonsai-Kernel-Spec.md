# Bonsai Kernel Spec — Q1_0_g128 + TQ2_0_g128 on gfx1151

_Status: 2026-04-20. File-format + scaffold only; HIP kernels deferred to a
subsequent pass._

## Why we're here

Our post-hoc Sherry requantization of MS BitNet-b1.58 collapsed — 12.98%
sign flips produced coherent-structure-but-one-token-repeats output (see
`project_sherry_spike.md`). PrismML's Bonsai line was trained at sub-2-bit
**from scratch**, so there is no post-hoc penalty: we get ternary accuracy
gratis if we ship the kernels.

OxiBonsai (`cool-japan/oxibonsai`, Apache 2.0) is the reference inference
engine. We are NOT forking it — we're teaching halo-ai's existing
`.h1b` loader + HIP kernels to consume Bonsai's on-disk block layouts.

## Byte-exact layouts

Both formats are **block-interleaved** on disk (AoS), row-major. No SoA
transform at the file level — the Metal kernel in oxibonsai repacks at
upload time for coalescing, but that's a kernel-side concern.

### Q1\_0\_g128 (1.0 bpw information, 1.125 bpw on disk)

Each block = **18 bytes**, 128 weights per block.

```
 offset   bytes   field
 0x00     2       d : FP16       group scale (shared by all 128 weights)
 0x02     16      qs[16]         128 sign bits packed LSB-first
 ─── total 18 B ─────────────────────────────────────────────────────
```

Weight reconstruction:

```
  bit_i     = (qs[i / 8] >> (i % 8)) & 1       // 0..127
  w[i]      = bit_i ? +d : -d                  // no zero code
```

Row bytes (K weights per row): `(K / 128) * 18`.

**Source:** oxibonsai `BlockQ1_0G128`
(`crates/oxibonsai-core/src/tensor.rs:25-30`). Verified layout against the
`prism-ml/Bonsai-1.7B-gguf/Bonsai-1.7B-Q1_0.gguf` header dtype tag 41.

### TQ2\_0\_g128 (~1.585 bpw information, 2.125 bpw on disk)

Each block = **34 bytes**, 128 weights per block.

```
 offset   bytes   field
 0x00     2       d : FP16       group scale (shared by all 128 weights)
 0x02     32      qs[32]         128 × 2-bit codes, 4 per byte LSB-first
 ─── total 34 B ─────────────────────────────────────────────────────
```

Code → ternary value map:

```
  0b00 → -1
  0b01 →  0
  0b10 → +1
  0b11 →  0    (reserved; treat as zero)
```

Extraction of the `j`-th weight in the block:

```
  byte_idx = 2 + j / 4
  lane     = j % 4                 // 0..3
  code     = (blk[byte_idx] >> (lane * 2)) & 0b11
  value    = {-1, 0, +1, 0}[code] * d
```

Row bytes (K weights per row): `(K / 128) * 34`.

**Note on scale position (corrected 2026-04-20):** PrismML's g128 variant
puts `d` FIRST at offset 0x00, matching the Q1_0_g128 ordering and the
GGML canonical TQ2_0. Earlier revisions of this doc (and the first cut of
`bonsai_{tq2,q1}_gemv.hip`) claimed the opposite (`d` last); empirical
dumps of the PrismML `Ternary-Bonsai-1.7B-Q2_0.gguf` file show d@0 with
zero reserved 0b11 codes across every sampled block, versus d@32 showing
1–2 reserved codes per block and scales varying across 10 orders of
magnitude — d@32 is garbage. Fixed in the 2026-04-20 pass.

**Source:** oxibonsai `BlockTQ2_0_g128`
(`crates/oxibonsai-core/src/quant_ternary.rs:62-69`). Verified layout
against `prism-ml/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf`
dtype tag 42.

### Dtype tags (PrismML's Bonsai flavour)

PrismML's GGUF converter assigns **its own** dtype tags, distinct from
ggml's canonical `TQ1_0` (34) / `TQ2_0` (35) since the block layouts
differ (g128 vs g256):

| Tag | Name        | Block bytes | Weights/block |
|-----|-------------|-------------|---------------|
| 41  | Q1\_0\_g128  | 18          | 128           |
| 42  | TQ2\_0\_g128 | 34          | 128           |

## HIP kernel FFI (proposed)

Two entry points, matching the style of our existing
`sherry_ternary_gemv_launch` and `rcpp_hadamard_rotate_fp16_butterfly_launch`:

```c
// rocm-cpp/include/rocm_cpp/bonsai.h (NEW — to be authored by HIP agent)

extern "C" {

/// Q1_0_g128 1-bit GEMV. Accumulates fp32 internally, writes fp16 output.
/// Weights: row-major, (K_in / 128) blocks per row, each 18 B.
/// Preconditions: K_in % 128 == 0; N_out > 0; stream != nullptr or NULL.
/// Shape violations are early-return no-ops; check hipGetLastError() at the
/// next sync boundary for launch failures.
void bonsai_q1_gemv_launch(
    const uint8_t* packed_weights,    // device: N_out × K_in / 128 × 18 bytes
    const uint16_t* act_fp16,         // device: K_in fp16 elements
    uint16_t*      out_fp16,          // device: N_out fp16 elements
    int            N_out,
    int            K_in,
    hipStream_t    stream);

/// TQ2_0_g128 ternary GEMV. Same preconditions as bonsai_q1_gemv_launch.
/// Weights: row-major, (K_in / 128) blocks per row, each 34 B.
void bonsai_tq2_gemv_launch(
    const uint8_t* packed_weights,    // device: N_out × K_in / 128 × 34 bytes
    const uint16_t* act_fp16,         // device: K_in fp16 elements
    uint16_t*      out_fp16,          // device: N_out fp16 elements
    int            N_out,
    int            K_in,
    hipStream_t    stream);

} // extern "C"
```

Rust-side FFI already declared in
`crates/1bit-hip/src/ffi.rs:bonsai_q1_gemv_launch` /
`bonsai_tq2_gemv_launch`. Safe wrappers at
`crates/1bit-hip/src/lib.rs:bonsai_q1_gemv_fp16` /
`bonsai_tq2_gemv_fp16` — both short-circuit to `RcppError::Unsupported`
today and should flip to dispatching the launcher once the kernel lands.

## gfx1151 implementation notes (for the HIP port agent)

### Target profile

- **Architecture:** RDNA 3.5, wavefront size **32** (wave32).
- **Target ISA:** `gfx1151`.
- **Memory ceiling:** LPDDR5-8000 ≈ 256 GB/s practical (measured 92% in
  our current halo kernel on a 92% memory-bound GEMV — see
  `project_bitnet_rocprof_plan.md`).
- **LDS budget:** 64 KiB per workgroup (same as Sherry). For a 128-weight
  macro-group we need at most one `[128 × fp16]` activation tile (256 B)
  plus scratch — tiny.
- **Register pressure target:** ≤ 64 VGPR, ≤ 16 SGPR to keep 10+ waves/SIMD
  resident. Sherry's gfx1151 kernel hits this; TQ2 should match.

### Per-block tile shape

- `BLOCK_SIZE_THREADS = 256` (8 waves × 32 lanes), one macro-wave of 64
  rows per workgroup — same pattern as `sherry_ternary_gemv_launch` so
  the dispatch harness matches.
- One wave computes 8 output rows. Within a wave, each lane handles 4
  blocks (128 weights) per iteration, striding by `waves_per_workgroup *
  128`. This matches oxibonsai's Metal per-simdgroup shape
  (`simd_sum` at the end) so we can cross-check correctness against their
  CPU + Metal paths.

### Unpack

TQ2 per-byte unpack (4 codes → 4 signed floats):

```c
// Inline. VGPR-local; no LDS LUT.
inline float4 decode_byte_tq2(uint b) {
    // select() is available on HIP via OpenCL extension; otherwise use
    // ternary. Both lower to the same v_cndmask pair per lane.
    float v0 = b & 0x3;  float v1 = (b >> 2) & 0x3;
    float v2 = (b >> 4) & 0x3;  float v3 = (b >> 6) & 0x3;
    return float4(
        v0 == 0.0f ? -1.0f : (v0 == 2.0f ? 1.0f : 0.0f),
        v1 == 0.0f ? -1.0f : (v1 == 2.0f ? 1.0f : 0.0f),
        v2 == 0.0f ? -1.0f : (v2 == 2.0f ? 1.0f : 0.0f),
        v3 == 0.0f ? -1.0f : (v3 == 2.0f ? 1.0f : 0.0f));
}
```

For AMD, the arithmetic decode `(1 - code) * step(code, 1.5f)` is also
viable and saves the branch; fall-back if the ternary emits `v_cndmask`
pairs. Benchmark both before committing.

Q1 unpack is simpler — one shift + mask per lane:

```c
inline float decode_bit_q1(uint qs_byte, uint lane) {
    return ((qs_byte >> lane) & 1u) ? 1.0f : -1.0f;
}
```

### Bandwidth ceiling

Per-layer ternary GEMV for a 2B-shape Bonsai:

- Q shape `(2048, 2048)`: K=2048 → 16 blocks × 18 B = 288 B/row →
  2048 × 288 = 576 KiB.
- TQ2 same shape: 16 × 34 = 544 B/row → 2048 × 544 = 1.09 MiB.
- Total per-layer TQ2 weight bandwidth (q+k+v+o+gate+up+down): ≈ 12 MiB.
- 28 layers × 12 MiB ≈ **336 MiB per token**.
- At LPDDR5 ceiling 256 GB/s × 92% utilization ≈ 235 GB/s → 700 tok/s
  upper bound on weight-bandwidth alone.

Our current halo kernel (2B BitNet) pulls ≈ 55 tok/s. The Bonsai-1.7B
ternary GEMV should be **faster** proportionally (smaller layers) modulo
the 2-bpw-on-disk vs halo's 2-bpw-on-disk tie — meaning Bonsai won't win
or lose on bytes-per-weight vs halo. The win vs halo is architectural:
Bonsai is Qwen3 (smaller layer count, smaller hidden), so the total
per-token ternary bandwidth drops ~3×.

### Expected perf relative to halo-1bit-2b (67 tok/s @ 64, 33 tok/s @ 1024)

- **Bonsai-1.7B TQ2**: 2B → 1.7B params ≈ 15% fewer, Qwen3 arch has 28
  layers vs BitNet's 30 → ~7% fewer ternary GEMVs per token, KV head
  count 8 vs 5 → slightly larger KV-cache pressure offsetting the weight
  win. Net: ≈ **75-80 tok/s @ 64** is the naive projection.
- **Bonsai-8B TQ2**: 8B → ~5× more ternary weight → ≈ 15-18 tok/s @ 64
  projected. Only viable once we have the kernel.

These are projections from per-parameter bandwidth arithmetic, not
measurements. Bench after the kernel lands.

## `.h1b` relationship — Option A (flag bits)

We picked **option A** from the scope spec: extend `.h1b` with two new
flag bits in `H1bConfig::reserved`, not a new file format.

| Constant               | Bit | Variant                                 |
|------------------------|-----|-----------------------------------------|
| `H1B_FLAG_HADAMARD_ROTATED` | 0x1 | (BitNet v2 activation rotation) |
| `H1B_FLAG_SHERRY_FP16` | 0x2 | (Sherry fp16 dispatch on v3)           |
| `H1B_FLAG_BONSAI_Q1`   | 0x4 | `H1bWeightFormat::BonsaiQ1 { group_size: 128 }` |
| `H1B_FLAG_BONSAI_TQ2`  | 0x8 | `H1bWeightFormat::BonsaiTQ2 { group_size: 128 }` |

Bonsai flags **take precedence** across all `.h1b` file versions (unlike
Sherry which is v3-gated) because the Bonsai row-byte math differs
fundamentally from every halo-family format — block-interleaved with
inline FP16 scales, no separate `[rows] f32` scales tensor. The loader
keys off `fmt.has_inline_block_scales()` to decide whether to reserve
scale bytes after each packed tensor; Bonsai formats reserve 0.

Mutual-exclusion rule: `H1B_FLAG_BONSAI_Q1 & H1B_FLAG_BONSAI_TQ2` is a
configuration error and `H1bWeightFormat::from_version_and_flags` returns
`HaloError::InvalidConfig`. Accessors on `H1bConfig` (`is_bonsai_q1` /
`is_bonsai_tq2`) return `false` when both are set, so a downstream
dispatcher never sees ambiguity.

## Runtime integration (NOT in this pass)

The following work is deferred to a separate agent session:

1. **HIP kernel authorship** — `rocm-cpp/kernels/bonsai_q1_gemv.hip` +
   `rocm-cpp/kernels/bonsai_tq2_gemv.hip`, exported as
   `bonsai_q1_gemv_launch` / `bonsai_tq2_gemv_launch`.
2. **`bitnet_decode` dispatch** — `rocm-cpp/tools/bitnet_decode.cpp` grows
   a branch on `cfg.reserved & (H1B_FLAG_BONSAI_Q1 | H1B_FLAG_BONSAI_TQ2)`
   to route the ternary GEMV through the new launcher. Needs matching
   Qwen3 attention layout (separate `attn_q_norm` / `attn_k_norm` — see
   "Surprise" below).
3. **Router / server model selection** — `lemond` grows a
   `BonsaiModel` variant that reads `is_bonsai_tq2()` off the header, and
   `1bit-server` / `1bit-lemonade` expose it as a switchable backend.
4. **Tokenizer** — Bonsai uses Qwen3's GPT-2-style BPE tokenizer with
   151669 tokens. The GGUF carries the full `tokenizer.ggml.tokens` +
   `tokenizer.ggml.merges` arrays; a separate pass will teach `htok` (or a
   sibling) to consume them.

## Surprises from the oxibonsai scan

Three things worth flagging to the HIP port agent:

1. **Bonsai arch is Qwen3, NOT BitNet-shaped.** This is the biggest gotcha.
   Qwen3 has per-head `attn_q_norm` and `attn_k_norm` RMSNorm tensors
   (shape `[head_dim]`) absent from our BitNet schema, and no
   `attn_sub_norm` / split `ffn_sub_norm`. The `gguf-to-h1b` converter
   zero-fills the BitNet-shaped norm slots and expects the loader to look
   up the real norms by name from the GGUF. This means the HIP port
   agent's task is NOT "swap the ternary GEMV kernel" — it's "implement
   Qwen3 forward pass with ternary weights", which is a larger scope
   than the BitNet path.

2. **SoA-at-upload, not on-disk.** oxibonsai's Metal kernel
   (`crates/oxibonsai-kernels/src/gpu_backend/kernel_sources/decode_ternary.rs`)
   expects weights in SoA order `[all d × N × 2 B][all qs × N × 32 B]`
   and repacks at upload time. The on-disk layout is AoS (interleaved).
   Our HIP kernel can either (a) replicate that and pay the upload cost
   once per model load, or (b) consume the AoS layout directly and let
   the memory subsystem stream blocks. Recommend (b) for gfx1151 — AoS
   is what the halo ternary GEMV already does and LPDDR5 isn't caching
   the scale bytes enough to hurt us.

3. **`0b11` code is reserved-not-used.** oxibonsai documents
   `0b11 → 0 (reserved)` and treats it as zero at decode time. The
   PrismML quantizer never emits `0b11` — the `quantize()` function in
   `crates/oxibonsai-core/src/quant_ternary.rs:138-149` only emits 0b00,
   0b01, 0b10. We can assume `0b11` never appears in input files; if it
   does, decode it as zero and move on. Do NOT add a check that fails
   loudly — it would false-trigger on padding.

Other surprises, minor:

- PrismML's GGUF converter uses dtype tags **41 / 42**, NOT llama.cpp's
  canonical **34 / 35**. The `tools/gguf-to-h1b` converter knows this;
  anyone writing a new decoder needs to too.
- oxibonsai's `simd_sum` at the end of each Metal wave reduces 32 partial
  dot products. On gfx1151 with wave32, this is `__builtin_amdgcn_mov_dpp`
  + cross-lane shuffle (1 hop per power-of-2 lane distance). Already used
  in our halo kernel.
- `.gguf` tensor shapes are `[cols, rows]` for matmul weights — the
  outer dim is input K, inner is output N. This conflicts with the
  `.h1b` convention `ternary(rows, cols)`. The converter flips them at
  framing time; every downstream consumer should expect the `.h1b`
  convention.
