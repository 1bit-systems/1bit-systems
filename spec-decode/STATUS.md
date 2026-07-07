# Speculative Decode — Status

## Architecture

```
Draft (CPU, Eagle3 1-layer / DSpark 5-layer)
    │ proposes tokens
    ▼
Target (NPU, 4-xclbin INT8 GEMM via NPUQwen3Target)
    │ verifies via forward_with_kv()
    ▼
Accept/Reject → next iteration
```

---

## ✅ Working

| Mode | Command | Target | Tok/s | Draft | Status |
|------|---------|--------|-------|-------|--------|
| 4-xclbin spec-decode | `--spec-decode` | NPUQwen3Target (4 xclbins/layer) | **1.3 tok/s** (FIXED_ASCALE+OMP) | DSpark 5-layer | ✅ Real tokens |
| 4-xclbin simple | `--bench-real` | NPUQwen3Target | **3.0s/4tok** | None | ✅ Real tokens |
| Simulated bench | `--bench` | Simulated | ~5000 tok/s | DSpark | ✅ |
| Daemon | `--daemon` | NPUQwen3Target | — | DSpark | ✅ |

## 🔧 Fused Target — NaN at Layers 24-27

The fused xclbin (`NpuFusedTarget`) produces non-zero output for layers 0-23
but **NaN at layers 24-27** due to BF16 numerical overflow in the AIE2 tile code.
Token output is all-zero (token 0 = `<unk>`). This is an xclbin build issue.

**Subprocess approach** (`FusedServerTarget`): spawns `npu_engine_fused_server`
via pipes. Same NaN issue — the fused xclbin itself has the numerical instability.

Fix needed: rebuild the fused xclbin with wider internal precision or fix the
weight scaling for upper layers.

| Mode | Tok/s | Tokens | Status |
|------|-------|--------|--------|
| `--fused` | 0.9-1.2 | All 0 | ⚠️ NaN at layers 24+ |
| `--fused-spec` | — | All 0 | ⚠️ Same NaN issue |
| `--daemon-fused` | — | All 0 | ⚠️ Same NaN issue |

## 🏋️ Training (in progress)

| Run | Examples | Epochs | Loss | Acceptance | Data Source |
|-----|----------|--------|------|------------|-------------|
| FP (old) | ~10K | 5 | — | 0% | HuggingFace FP16 |
| NPU 200 | 200 | 5 | 12.1→6.4 | **0%** | NPU INT8 hidden states |
| **NPU 1K** | **1,000** | **5 (running)** | **9.10→6.67 (ep1→ep2)** | **TBD ~2h** | **NPU INT8 hidden states** |

## 🆕 Token Router (separate project)

A **token-level router** design in Rust (`token-router/DESIGN.md`) routes
individual tokens between NPU/GPU/MLX backends based on confidence, content,
or budget. Key strategies:

- **Cascade**: NPU generates all tokens; low-confidence tokens rerouted to GPU
  (no draft model needed — confidence-based fallback)
- **Speculative decode**: draft on NPU, verify on GPU (uses unified memory on
  Strix Halo for zero-copy)
- **Content router**: keyword-based routing (port of `unified-router.py`)

This could bypass the draft training problem entirely — instead of training
a draft that matches the NPU's distribution, use the GPU to handle uncertain
tokens on the fly.

## Next Steps

1. **Wait for training** (~2h) → test acceptance with `--spec-decode` on 4-xclbin target
2. **If acceptance >0%**: scale up NPU data capture, iterate
3. **If acceptance still 0%**: pivot to the cascade token router approach
   (NPU primary, GPU fallback for low-confidence tokens) — no training needed
4. **Fix fused xclbin NaN** at layers 24+ (xclbin rebuild needed)
5. **Benchmark cascade router** — estimated 230 tok/s with 95% NPU + 5% GPU
