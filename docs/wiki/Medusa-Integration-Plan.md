# Medusa Integration Plan (1bit-systems / gen-2)

Status: research brief, no code written.
Date: 2026-04-20.
Recommendation: **DEFER** (details in §5).

---

## 1. What MedusaBitNet actually is

Medusa is the 2024 speculative-decoding framework from Cai et al. (arXiv
[2401.10774](https://arxiv.org/abs/2401.10774), published at ICML 2024, repo
[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa)). Instead of
running a separate small draft model, Medusa bolts N extra MLP "heads" onto the
frozen backbone. Each head predicts the token at position t+1, t+2, … t+N from
the same backbone hidden state. At decode time, the heads produce a set of
candidate continuations that are laid out as a tree; the backbone verifies the
whole tree in a single forward pass using a **tree-attention mask** (causal
within each branch, with predecessors-only attention). Accepted prefix advances
the KV cache; rejected branches are discarded. Medusa-1 (frozen backbone) reports
~2.2× wall-clock speedup on Vicuna at batch=1; Medusa-2 (joint-train) reports
2.3–3.6× ([Together AI blog](https://www.together.ai/blog/medusa), paper §4).

**MedusaBitNet** is not a combined framework — a web search for
"MedusaBitNet" returns nothing indexed outside our own memory file. What
exists is a specific HuggingFace artifact:
[parrishcorcoran/MedusaBitNet-2B-4T](https://huggingface.co/parrishcorcoran/MedusaBitNet-2B-4T)
— 4 Medusa heads (13 MB total, fp16) trained against the frozen Microsoft
`bitnet-b1.58-2B-4T` backbone (751 MB, I2_S). Measured acceptance: head-1 63.0%,
head-2 29.0%, head-3 11.1%, head-4 4.6%; end-to-end **2.08 tokens per backbone
step** on CPU. Trained ~11 h on a Ryzen AI MAX+ 395 (same silicon class as
Strix Halo). Inference paths: a verified PyTorch loop and an incomplete
llama.cpp/bitnet.cpp C++ path; the author's notes explicitly flag "kernel
compatibility issues with I2_S" — the C++ throughput is not validated. There
is no HIP/ROCm path and the artifact card does not mention tree attention;
the Python reference appears to be naive per-head verify.

## 2. Where it would bolt into our tree

Our decode loop is a tight line through three crates. Speculative decoding
breaks the `1 token in → 1 token out` assumption at every level, so changes
touch all three.

### 2.1 `rocm-cpp` (HIP kernels)

- `src/kv_cache_attn_fd.hip` — current Flash-Decoding attention. Splits
  KV along the sequence axis, one query per head. Medusa needs **tree
  attention**: a *set* of query tokens (the speculated candidates) all
  attending to the same KV but with a sparse mask that says
  "candidate q_i attends to its own branch only." Either extend this kernel
  with a mask-table argument or add a sibling file
  `src/kv_cache_attn_tree.hip`. The FD split still applies along KV; the
  query axis grows from 1 to `tree_size` (Medusa paper uses trees of 42–64
  nodes).
- `kernels/ternary_gemv_*.hip` — all current ternary kernels assume
  **GEMV** (batch-1). A Medusa verify pass needs a small **GEMM** over the
  speculated query set against the same weights. Two options:
    1. Call the existing GEMV `tree_size` times in a loop — simple, but
       you re-stream the weights from LPDDR5 each call. Our Phase-1
       memory report already says we're at 92% of LPDDR5 peak on the
       backbone weights (`project_bitnet_rocprof_plan.md`), so naïve
       iteration **eats the entire speculative win**.
    2. Promote one kernel (candidate: `ternary_gemv_phase5_halo.hip`) to
       tile over a small M axis. New file
       `kernels/ternary_gemm_small_m.hip`, M ∈ {1…8}. This is new HIP
       work, not a flag flip.
- New kernel: the Medusa **heads themselves** are dense fp16 MLPs (~3 MB
  each). An fp16 GEMV path already exists (`src/prim_kernels.hip`
  `fp16_gemv`). Call it 4× per decode step, one per head. No new kernel
  needed for the heads.
- New kernel: **tree-mask builder / argmax-topk per head**. Small, fits
  in `src/prim_kernels.hip`. Needs a device-side top-k (current
  `argmax_fp32` is top-1 only).
- `tools/bitnet_decode.cpp` — gen-1 CLI. Only touch if we want a gen-1
  reference loop for parity-checking; otherwise leave.

### 2.2 `halo-workspace/crates/1bit-hip` (Rust FFI)

- `src/lib.rs` — add FFI bindings for the new kernels:
  - `kv_cache_attn_tree_fd(...)` mirroring `kv_cache_attn_decode_fd` at
    line 424, plus a `mask: *const u8` arg.
  - `ternary_gemm_small_m(...)` next to the existing
    `ternary_gemv_halo_f16` at line 258.
  - `fp16_gemv_batched(...)` if we choose path (1) for the heads.
- `src/ffi.rs` — matching `extern "C"` prototypes.

### 2.3 `halo-workspace/crates/lemond` (decode driver)

- `src/backend_impl.rs:410` — `HipBackend::forward_token` is the hot loop.
  It takes one token id and returns one. Medusa requires a new sibling
  method:
  ```rust
  fn forward_tree(
      &mut self,
      accepted: &[i32],       // tokens confirmed last step
      base_pos: i32,
      tree: &SpecTree,        // candidate token ids + parent indices
      logits_out: &mut [f32], // one row per candidate
  ) -> Result<usize /*accepted_len*/, BackendError>
  ```
  The existing `forward_token` stays for prefill and as the non-spec
  fallback. Medusa heads need a handle to the post-`lm_head` hidden
  state, so `forward_token`'s internal buffer layout has to expose the
  pre-softmax residual — right now it's scratch-local.
- `src/lib.rs:319` and `:429` — the two `forward_token` call sites in the
  decode loop. Wrap behind a `SpecConfig` struct on `GenRequest`; if
  `None`, call `forward_token` unchanged; if `Some`, call `forward_tree`
  and advance `cur` / `pos` by the accepted length.
- `1bit-core::sampler::Sampler` at
  `crates/1bit-core/src/sampler.rs` — currently samples one token from
  one logits row. Needs a batched variant that takes a 2D logits buffer
  and returns the argmax *per row* so the verify step can pick the
  accepted prefix. Rep-penalty bookkeeping has to be applied to the
  accepted tokens only (not the rejected branches) — easy to get wrong.

### 2.4 Things that **don't** change

- Tokenizer (`crates/lemond/src/tokenizer.rs`,
  `crates/1bit-core/src/htok.rs`) — vocabulary is the same backbone; heads
  share `lm_head`.
- GGUF/.h1b loaders — heads are a separate 13 MB fp16 file; add one new
  loader path, don't touch the backbone loader.
- RoPE / RMSNorm kernels — reused as-is for the tree verify pass.

## 3. Realistic wall-clock speedup on our 2B, batch=1

Upstream numbers:

- Medusa-1 paper / Together blog: **~2.2×** on Vicuna 7B/13B batch=1,
  on A100-class GPUs ([paper §4](https://arxiv.org/abs/2401.10774),
  [Together](https://www.together.ai/blog/medusa)).
- Medusa-2 (joint-train): **2.3–3.6×**.
- MedusaBitNet-2B-4T card: **2.08 tokens/backbone-step** measured on
  Ryzen AI MAX+ 395 **CPU**, PyTorch. CPU absolute throughput there is
  72.7 tok/s for the Medusa path.

Naïve extrapolation to our GPU path: 83 tok/s × 2.08 ≈ **172 tok/s**.
That is the upper bound and is almost certainly not reachable. Three
reasons we'd lose headroom vs. the Vicuna 2.2× figure:

1. Our backbone decode is already **bandwidth-bound** at 92% of LPDDR5
   peak on the ternary GEMV (`project_bitnet_rocprof_plan.md`). Medusa's
   win comes from amortizing one weight-stream across multiple accepted
   tokens. If the small-M GEMM doesn't keep *the same* LPDDR5 efficiency,
   we re-pay the bandwidth bill per verified token and the speedup
   collapses toward 1×.
2. Tree-attention kernel overhead is non-trivial at small tree sizes;
   the paper uses trees of 42–64 nodes to hide it, but 64 nodes × 2B ≈
   prefill pressure on a 4 GB/s inference budget.
3. The MedusaBitNet-2B-4T heads were trained **only on Alpaca 52K**
   (loss 9.85 → 3.32). Acceptance rate on our realistic prompts (code,
   long-context chat) is **unknown — would need offline eval on our own
   workload to verify**.

Honest estimate, with sources: **1.4–1.8× wall-clock on our batch=1 path,
if the small-M GEMM holds its bandwidth**. Higher is possible if
Medusa-2-style joint-training is done; lower if we hit GEMM inefficiency.
Anything ≥ 2× on our silicon is unverified claim territory.

## 4. Prerequisites not yet met

Ranked by how much of a blocker each one is.

1. **New HIP kernels — two of them.**
   - Tree-attention variant of `kv_cache_attn_fd.hip`. Not a flag flip.
     Estimate: 1–2 weeks of kernel work + bit-exact vs. reference PyTorch.
   - Small-M ternary GEMM. Required to preserve the LPDDR5 efficiency
     we already have. Can reuse layout / dequant logic from
     `ternary_gemv_phase5_halo.hip`, but the tiling is new.
2. **Draft heads retrained on our data.** The public MedusaBitNet heads
   are Alpaca-only. For any workload that isn't instruction-follow chat,
   acceptance rate will drop. Medusa-1 training is cheap (~11 h CPU per
   the model card, or one overnight on the Intel B770 slot in
   `project_battlemage_distill.md`) but it *is* a training step we have
   to run. Medusa-2 (joint finetune of backbone) is out — it would
   destroy our bit-exact match with the Microsoft backbone and blow up
   our ternary format guarantees.
3. **Hidden-state exposure from `forward_token`.** The current FFI
   returns an int (the sampled token). Medusa needs the pre-`lm_head`
   residual to feed the heads. This is a visibility change in
   `1bit-hip`, not a new kernel, but it does ripple through the
   FFI surface.
4. **Batched sampler + rep-penalty accounting.** Minor; Rust-side only.
5. **Per-request spec config plumbing.** Minor; `GenRequest` gets a new
   optional field.

Not required:

- No new tokenizer path (same vocab).
- No new GGUF format (backbone is unchanged; heads are their own file).
- No change to the prefill path.

## 5. Ranking vs. current backlog

| Item | Expected win | Risk | Prereqs |
|---|---|---|---|
| **Sherry 1.25-bit** | bytes-read −21% → ~15% decode speedup at 1024 tok | low — kernels already landed (`ternary_gemv_sherry.hip`), PPL pending | finish PPL eval, loader wiring |
| **BitNet v2 (Hadamard, native W1.58A4)** | supersedes a4.8 staged recipe | medium — format churn if MS re-releases 2B weights; new act quant kernel | paper drop 2025-04, code TBD |
| **Activation sparsity Phase 2** | upper bound 10–15%, we measured 79.91% sparsity | low speedup due to 600:1 weight-bandwidth domination | already deferred, see memory |
| **Medusa (this plan)** | 1.4–1.8× if everything works | **high** — two new HIP kernels, retrained heads, bandwidth fight, end-to-end measurement from zero | kernels + heads + FFI reshape |

**Firm recommendation: defer.** The expected-value per engineering-week is
worse than Sherry. Sherry is an in-flight kernel with the slow part
already written; Medusa is two fresh kernels, a training run, and an FFI
redesign, with the payoff capped by the same LPDDR5 ceiling Sherry
directly attacks. Revisit Medusa **after**:

- Sherry 1.25-bit is benched end-to-end (need the real post-Sherry
  baseline; the Medusa speedup is multiplicative on whatever the new
  baseline is).
- BitNet v2 decision lands (format churn risk — no point training
  Medusa heads against an about-to-be-replaced backbone).
- We have one confirmed workload where Alpaca-style acceptance rate
  is *not* the best-case (i.e., we've measured heads-1 acceptance on
  code or long-context and it isn't 63%).

If all three clear and the batch=1 decode is still the bottleneck users
complain about, spin up Medusa as its own 2–3-week sprint: one week
kernels, one week retrain heads on our traffic, one week wiring and
bit-exact verify.

## Sources

- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads* — arXiv
  [2401.10774](https://arxiv.org/abs/2401.10774), ICML 2024.
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa).
- [Together AI Medusa blog](https://www.together.ai/blog/medusa).
- [parrishcorcoran/MedusaBitNet-2B-4T](https://huggingface.co/parrishcorcoran/MedusaBitNet-2B-4T).
- Our memory: `project_ternary_community_2026_04.md`,
  `project_bitnet_rocprof_plan.md`, `project_attention_fd.md`.
