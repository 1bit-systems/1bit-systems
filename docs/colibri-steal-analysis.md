# What 1bit/Zaya Should Steal from colibrì

Analysis comparing `c/glm.c` (colibrì) with `src/zaya_engine.cpp` + `kernels/` (1bit/Zaya).

---

## 1. MLA Weight Absorption for Decode

**colibrì** (`glm.c:905-950`): During single-token decode (`S≤4`), colibrì avoids reconstructing
k_nope + value for every cached token. Instead it absorbs `W_kvB^T` into the query:

```c
q_absorbed[d] = Σ_{r} W_kvB[h,r] · q_nope[r]    // per head, once
score_t = q_absorbed · L_t + q_rope · k_rot_t    // cheap latent-space score
ctx = Σ a_t · L_t                                 // attend in latent space
output = W_o · W_kvB[v_head_offset] · ctx         // project back once
```

Cost: `O(T·kv_lora)` per head instead of `O(T·H·(nope+vh))`.

**Zaya** (`zaya_engine.cpp`): Uses CCA (Cross-Computer Attention) which is a different
mechanism — Q/K/V are projected separately with conv1d depthwise + grouped filters.
There's no weight absorption. Every decode step reconstructs full K/V pairs.

**Steal**: Port the principle. Zaya's CCA has `QKV=1280` values per token in the
convolution buffer. The conv → attention pipeline could be reordered so the
query absorbs the conv kernel weights, reducing per-step compute for decode.
**Priority: 4/5 | Effort: ~8h**

## 2. IDOT Integer Matmul Kernels (CPU Fallback)

**colibrì** (`glm.c:360-480`): The `dot_i8i8()` / `dot_i4i8()` kernels use a sign trick
that makes `_mm256_maddubs_epi16` (unsigned×signed) work with signed weights:

```c
// |w| as unsigned, x·sign(w) as signed
p = _mm256_maddubs_epi16(_mm256_sign_epi8(w,w), _mm256_sign_epi8(x,w));
```

This gives int8 matmuls at **119 GFLOP/s** on AVX2. The int4 packed version
(`dot_i4i8`) unpacks nibbles → int8 on the fly, then uses the same trick.

**Zaya**: All kernels are HIP/CUDA GPU — no CPU fallback matmul kernels exist.
If the GPU is busy or for hybrid dispatch, there's no efficient CPU path.

**Steal**: Port the IDOT kernels as a CPU fallback for 1bit's backend abstraction
(`src/backend_factory.cpp` dispatches to HIP/NPU/Vulkan). Add an `#ifdef __AVX2__`
fallback path. The sign trick is 20 lines of intrinsics.
**Priority: 3/5 | Effort: ~2h**

## 3. Coalesced Expert Loading (One pread Instead of Three)

**colibrì** (`glm.c:860-920`): Expert weights consist of 3 matrices (gate/up/down).
Colibrì checks if they're contiguous in the file, and if so issues **one** `pread()`
for all three (~19 MB) instead of three separate reads:

```c
if (contiguous) {
    pread(fd, slab, wtot, offset);  // one syscall
    // QT views point into slab (zero-copy)
} else {
    // fallback: 3 preads
}
```

This halves syscall overhead on spinning disks and NVMe.

**Zaya** (`zaya_engine.cpp`): Loads `gate_up_proj` and `down_proj` as two
separate `hipMalloc` + `hipMemcpy` calls. No coalescing.

**Steal**: Zaya's MoE experts are already in contiguous weight files. The
two expert weight tensors could be loaded with one `hipMemcpy` from a
coalesced host buffer. On the GPU side, the gate/up/down projections
for the same expert could be fused into one kernel launch.
**Priority: 3/5 | Effort: ~4h**

## 4. Batch-Union MoE

**colibrì** (`glm.c:1160-1260`): When processing a batch (`S>1`), colibrì deduplicates
expert IDs across all positions in the batch. Each unique expert is loaded from disk
**once** and applied to every position that routes to it:

```c
// Phase B: union of expert IDs across batch
for s in batch: for k in topk: seen[expert_id] = 1
unique = list of seen experts

// Phase C: for each unique expert:
expert_load(layer, eid)
for s in batch where route[s] == eid:
    compute(s)  // reuses cached expert weights
```

**Zaya** (`zaya_engine.cpp`): Uses a single expert per token (top-1 routing).
The `mm_k` kernel is launched per-position. There's no batch dimension in
the current decode path — it processes one token at a time.

**Steal**: If Zaya adds batched decode (for MTP verification or prompt
processing), the batch-union pattern is essential. Currently not urgent
since Zaya is single-token decode, but worth noting.
**Priority: 2/5 | Effort: ~6h (when batched decode is added)**

## 5. KV-Cache Persistence (Crash-Safe)

**colibrì** (`glm.c:1880-1980`): Appends the compressed MLA KV cache to
`.coli_kv` after every turn (~182 KB/token). Uses a header/record format:

```
[magic:8] [header:32] [record_0] [record_1] ... [record_n]
                                                      ^
                                              nrec written LAST
                                              (crash-safe: stale nrec = truncated)
```

On resume: reads the file, restores KV cache from disk. **Zero re-prefill**.
Byte-identical to an uninterrupted session.

**Zaya**: No KV persistence. Conversation state is lost on engine restart.
Every new session must re-prefill the entire prompt — costly for long
conversations or server mode.

**Steal**: Zaya's CCA attention has a convolution buffer per layer
(`d_conv[il][2*QKV]`). This plus the hidden state (`d_phs[il][H]`)
could be persisted. With 40 layers, that's ~40 × (2×1280 + 2048) × 2 bytes
= ~400 KB/token. A `.zaya_kv` file with the same crash-safe append pattern
would give warm restart.
**Priority: 4/5 | Effort: ~6h**

## 6. Speculative Decoding (MTP Pattern)

**colibrì** (`glm.c:1450-1620`): GLM-5.2 has a native MTP head at layer 78.
The draft-verify-absorb pipeline:

```c
// Draft: MTP head proposes G tokens
for g in 0..G:
    h' = MTP_layer(concat(embed(tok_g), h_g))
    draft[g] = argmax(lm_head(h'))

// Verify: batch-forward all G+1 positions
logits = step_all([tok, draft[0], ..., draft[G-1]])

// Accept: greedy match (or rejection sampling)
k = 0; while k < G && argmax(logits[k]) == draft[k]: k++

// Absorb: sync MTP head's KV cache with verified tokens
mtp_absorb(verified_tokens, verified_hiddens)
```

All on GPU (colibrì's CUDA backend) or CPU. **2.2-2.8 tok/forward** measured.

**Zaya**: No speculative decoding. Single token per forward. The EDA router
(Zaya's routing mechanism) selects exactly 1 expert per layer.

**Steal**: Zaya doesn't have an MTP head, but the **speculation pattern**
itself is portable. A simpler approach: n-gram lookup (colibrì's `DRAFT=4`,
`ngram_draft()` at line 1420) looks up the last bigram in context history
and proposes follow tokens. On GPU with batched decode, even a small
draft (2-3 tokens) could give 1.5-2x speedup with zero model changes.
**Priority: 3/5 | Effort: ~10h (MTP) or ~3h (n-gram)**

## 7. Router-Lookahead Prefetch

**colibrì** (`glm.c:1310-1400`): Predicts the next layer's routing from the
current layer's post-attention state. Measured **71.6% recall** on GLM-5.2.
A dedicated I/O thread (`pilot_worker`) issues `WILLNEED` for predicted
experts while the CPU computes the current layer.

```c
// After attention of layer L:
pilot_prefetch(m, L+1, post_attn_state)
// Predicts top-K experts for L+1, issues WILLNEED for missing ones
```

**Zaya**: The EDA router has a recurrent state (`d_prev_rs[il][256]`) that
could be exploited. If the router at layer L can predict L+1's routing
from `prev_rs[L]`, the GPU could prefetch the next expert's weights while
the current expert computes.

**Steal**: Add a GPU-side prefetch kernel that runs the EDA router for
layer L+1 while the MoE matmul for layer L is in-flight. Uses the existing
`d_prev_rs` state that's already maintained.
**Priority: 4/5 | Effort: ~6h**

## 8. DSA Sparse Attention (Long Context)

**colibrì** (`glm.c:968-1040`): The DSA lightning indexer selects top-k keys
per query beyond `index_topk` (2048) tokens. Cost: `O(T·index_hd)` instead
of `O(T·H)`.

```c
// Per query position:
k_idx = W_ik · x        // [index_hd=128]
score_t = Σ_h ReLU(W_ip · x) · (W_iq · q_latent · k_idx_t)
keep = qsort(score_t)[:index_topk]
```

**Zaya**: No sparse attention. Full causal attention for all context tokens.
With 40 layers, context is currently limited.

**Steal**: Zaya's CCA attention could incorporate a simpler sparse selection:
since CCA already projects Q/K through conv filters, the conv output
could double as a relevance score for key selection. This is architecture-
specific but worth exploring for >8K context.
**Priority: 2/5 | Effort: ~12h (requires model retrain or approximation)**

---

## Summary

| # | Technique | Priority | Effort | Impact |
|---|-----------|----------|--------|--------|
| 1 | Weight absorption for decode | ★★★★ | 8h | 2-3x decode speed |
| 2 | IDOT CPU fallback kernels | ★★★ | 2h | Backend flexibility |
| 3 | Coalesced expert loading | ★★★ | 4h | 1.5x expert load |
| 4 | Batch-union MoE | ★★ | 6h | (future) |
| 5 | KV-cache persistence | ★★★★ | 6h | Warm restart |
| 6 | Speculative decoding | ★★★ | 3-10h | 1.5-2.8x throughput |
| 7 | Router-lookahead prefetch | ★★★★ | 6h | Hide expert load latency |
| 8 | DSA sparse attention | ★★ | 12h | Longer context |

**Top 3 to implement first:**
1. **Weight absorption** — biggest decode speedup, directly in Zaya's CCA path
2. **KV-cache persistence** — warm restart, high user-facing value
3. **Router-lookahead prefetch** — uses existing EDA router state, hides expert latency
