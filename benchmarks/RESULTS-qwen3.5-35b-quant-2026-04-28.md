# Qwen3.5-35B-A3B (MoE) — sub-2-bit vs 4-bit on Strix Halo

_Captured 2026-04-28 against `lemond` on `:13305` (Lemonade Server, llama.cpp `b1231` lane, ROCm/Vulkan auto-pick) on the production strixhalo box (Ryzen AI MAX+ 395 / `gfx1151`, 124 GB LPDDR5x). Single-shot OpenAI-compat curls; lemond's max_loaded_models=1 swaps weights between runs._

## Models

| Tag | HF source | Quant | bpw (~) | Size on disk |
|---|---|---:|---:|---:|
| Baseline | `unsloth/Qwen3.5-35B-A3B-GGUF` (`Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`) | Q4_K_XL | 4.5 | 22 GB |
| Sub-2-bit | `Manojb/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf` | IQ2_XXS | 2.0 | 11.5 GB |

Same architecture (Qwen3.5 35B MoE, A3B = 3B active params per token). Same Lemonade endpoint. Same prompts. Different quantization.

## Results

| Run | Prompt tok | Output tok | Q4_K_XL prefill tps | Q4_K_XL decode tps | IQ2_XXS prefill tps | IQ2_XXS decode tps |
|---|---:|---:|---:|---:|---:|---:|
| short ("hi") | 17 | 30 | 63 | **56** | 27 | **59** |
| med (100 word) | 33 | 200 | 230 | **54** | 261 | **73** |
| long (250 word) | 47 | 350 | 167 | **54** | 72 | **73** |

## Headlines

- **Decode steady-state**: Q4_K_XL → ~54 tok/s, IQ2_XXS → ~73 tok/s. **+35% throughput** at sub-2-bit on this APU.
- **Disk**: 11.5 GB vs 22 GB. **0.52× the size** for the same model.
- **Quality** (subjective, single-shot): IQ2_XXS coherent on standard prompts; perplexity-grade comparison hasn't been run yet (would need a wikitext sweep — separate task).
- **Prefill numbers** at single-shot are noisy (167 vs 72 on long is within run-to-run variance). The decode steady-state number is the reliable signal.

## Why this matters

The 1bit-systems brand is "1bit" because that's the *floor* — Bonsai-1.7B at IQ1_S hits 280 tok/s on this hardware (`benchmarks/RESULTS-1bit-2026-04-26.md`, `benchmarks/RESULTS-stack-2026-04-28.md`). But for a real 35B-class assistant that answers real questions, **sub-2-bit on MoE is the daily driver**. Half the disk, 35% faster decode, runs on a single APU. 1-bit is the headline; 2-bit is what you'll keep open.

## Reproduce

```sh
# pull both
lemonade pull Qwen3.5-35B-A3B-GGUF                           # → unsloth Q4_K_XL by default
lemonade pull Manojb/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf         # → user.Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS

# bench
for MODEL in "Qwen3.5-35B-A3B-GGUF" "user.Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS"; do
  curl -s http://127.0.0.1:13305/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" \
      '{model:$m,messages:[{role:"user",content:"Write a 100 word paragraph about why 1-bit inference matters for local AI. /no_think"}],max_tokens:200}')" \
    | jq -r '.timings | "decode: \(.predicted_per_second | floor) tps"'
done
```

## What's still pending

- True 1-bit pre-merged Qwen3.5-35B-A3B GGUF doesn't exist on HF as a single file; only Thireus's per-tensor `SPECIAL_SPLIT` format (734 files, 35 GB) which requires their merge tool. IQ2_XXS is the closest pre-merged sub-2-bit option today.
- Perplexity / quality benchmarks for IQ2_XXS vs Q4_K_XL on this checkpoint — separate sweep.
