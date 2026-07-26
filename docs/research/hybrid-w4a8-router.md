# Hybrid W4A8 Precision Router

Extend the Token Router to dispatch per-layer by precision (ternary vs FP8 vs block-ternary) in addition to backend (NPU/GPU/CPU).

## Motivation

Decoder-only generative models suffer at W4A4 in early/late layers but middle FFN layers are robust at ternary. The INT4-for-transformers paper (arXiv:2301.12017) shows W4A4 accuracy drop concentrates in specific layers.

## Precision profile

| Precision | Weights | Activations | Backend | When |
|---|---|---|---|---|
| TERNARY | 1.58-bit | INT8 | GPU ternary | Robust layers (middle FFN, V/O)
| BLOCK_TERNARY | Block-scaled ternary | INT8 | GPU (new kernel) | Intermediate — block scale protects outliers
| FP8/W8A8 | MXFP8 | FP8 | NPU INT8 / GPU | Sensitive (early attn, Q/K, lm_head)

## Integration

- Add `precision_profile` field to `rcpp_bitnet_model_t` (one byte per sub-layer)
- Extend `zaya_gpu_router.hip` to select kernel variant from profile table
- Add kernel dispatcher: ternary calls `ternary_gemv_phase5`, block-ternary calls `ternary_gemv_block_scaled`, FP8 calls existing INT8 path
- Profile generated once during conversion via calibration

## Expected impact

~60-70% of layers are robust to ternary with block scaling. Compared to full FP8: ~2x throughput gain. Compared to full ternary: ~0.5 perplexity improvement.
