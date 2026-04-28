# Bonsai 1-bit paper notes

Notes on **"1-bit Bonsai 8B: End-to-end 1-bit language model deployment across
Apple, GPU, and mobile runtimes"**, PrismML, 2026-03-31.

1-bit Bonsai 8B is a true end-to-end 1-bit-weight language model built from
Qwen3-8B (architecture unchanged). 1-bit precision is applied across embeddings,
attention projections, MLP projections, and the LM head — no higher-precision
escape hatches. The deployable format is `Q1_0_g128`: bitpacked sign bits with
one shared FP16 scale per group of 128 weights, giving 1.125 bpw effective. The
runtime package shrinks the 8.19 B model from 16.38 GB FP16 to 1.15 GB GGUF
(14.2x) / 1.28 GB MLX (12.8x). Released under Apache 2.0 with weights, demos,
and integrations.

## Headline numbers

- **Footprint:** 14.2x vs FP16 (GGUF), 12.8x (MLX). 1/14 the memory of peer 8B
  models in their benchmark table.
- **Throughput (TG128, single-token decode):** RTX 4090 368 tok/s (6.2x),
  RTX L40S 327 tok/s (6.3x), M4 Pro Metal 85 tok/s (5.4x), M4 Pro MLX
  131 tok/s (8.4x), iPhone 17 Pro Max 44 tok/s (3.2x vs 4-bit), RTX 3060
  Laptop 81 tok/s (23x — FP16 baseline must partial-offload).
- **Energy/token:** 5.6x lower on Mac MLX, 5.1x Metal, 4.1x RTX 4090, ~2.1x on
  iPhone vs 4-bit.
- **Prompt processing (PP512):** only 1.0–1.1x gain — the format is bandwidth-
  bound, so decode benefits dominate.
- **No Strix Halo / gfx1151 number is reported.** Closest analog is the
  RTX 3060 Laptop CUDA result.

## Q1_0_g128 vs other low-bit formats

- Stores one sign bit per weight plus one FP16 scale per 128-weight group.
  Weights map `{0,1}` → `{-s_g, +s_g}`. Effective 1.125 bpw.
- Sign bits decoded *inline* in the matmul kernel — no offline expansion to
  FP16 weight tensors.
- vs **BitNet b1.58** (ternary, 1.58 bpw, native pretraining): Bonsai is binary
  (no zero state) and applied as deployment quantization on a Qwen3-8B base.
- vs **IQ1_S** (mainline llama.cpp, ~1.6 bpw): Q1_0_g128 is lower bpw and is
  *not* in upstream llama.cpp. PrismML maintain forks at
  `PrismML-Eng/llama.cpp`, `PrismML-Eng/mlx`, `PrismML-Eng/mlx-swift` with
  custom CUDA / Metal / OpenCL kernels.

## Benchmarks

EvalScope v1.4.2 on vLLM 0.15.1 / H100, 11 peer 8B models, greedy decoding,
thinking mode disabled. 1-bit Bonsai 8B average **70.5** at 1.15 GB vs Qwen3-8B
79.3 at 16.38 GB. Per-task: MMLU-Redux 65.7, MuSR 50, GSM8K 88, HumanEval+ 73.8,
IFEval 79.8, BFCLv3 65.7. Beats LFM2 8B, Llama 3.1 8B, GLM 4 9B, Hermes 3 8B,
Marin 8B, DeepSeek R1 Qwen 7B on average. PrismML also define an "intelligence
density" metric where the three Bonsai sizes (1.7B / 4B / 8B) lead all 20 peers
by a wide margin.

## Limitations & roadmap (§7)

- Results are on general-purpose hardware via software/kernel optimization,
  *not* on a native 1-bit silicon target.
- iPhone energy is estimated (Xcode Power Profiler + battery-drain), not
  hardware-metered.
- Methodology is architecture-agnostic; future Bonsai variants planned for
  non-transformer / hybrid / diffusion backbones and for additional
  efficient-deployment formats beyond 1-bit.

## Relation to what 1bit-systems uses

The 8B in this paper *is* the upstream model that `lilyanatia/Bonsai-{1.7B,4B,8B}-requantized`
re-packs from `Q1_0_g128` into mainline llama.cpp formats (IQ1_S, ~1.6 bpw).
`docs/model-priority.md` Tier 1 currently pins
`Bonsai-1.7B-requantized-Bonsai-1.7B-IQ1_S.gguf` (385 MB, ~280 tok/s on
gfx1151). Running the *native* `Q1_0_g128` would require building PrismML's
llama.cpp fork — mainline does not implement the format.

## See also

- <https://github.com/PrismML-Eng/Bonsai-demo>
- <https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized> (the GGUF
  descendant we use)
- [`docs/model-priority.md`](./model-priority.md)
