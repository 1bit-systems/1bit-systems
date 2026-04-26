# 1-bit / Sub-2-bit / Ternary Model Bench — Strix Halo gfx1151

_Captured 2026-04-26 via `benchmarks/bench-1bit-pile.sh` on AMD Strix Halo gfx1151. ROCm 7.2 / llama-bench Vulkan backend (lemonade-bundled). `-p 512 -n 128 -r 2 -ngl 99` per model._

**GPU:** `Radeon 8060S Graphics (RADV STRIX_HALO)`

| Model | Quant | Size (MB) | Prompt-eval (tok/s) | Decode (tok/s) |
|---|---|---:|---:|---:|
| `gianni-bitnet-large` | TQ2_0 | 207 | 1362.3 ± 7.2 | 73.5 ± 0.2 |
| `lily-bonsai-1.7B` | IQ1_S | 385 | 4910.5 ± 7.4 | 281.2 ± 1.6 |
| `lily-bonsai-1.7B` | Q2_K | 595 | 4658.9 ± 0.9 | 227.6 ± 0.0 |
| `lily-bonsai-4B` | IQ1_S | 872 | 1984.1 ± 1.1 | 143.7 ± 0.3 |
| `trilm-3.9B` | TQ1_0 | 948 | 146.2 ± 0.0 | 40.0 ± 0.2 |
| `superkaiii-bonsai-4B` | TQ1_0 | 1041 | 146.8 ± 0.7 | 38.9 ± 0.3 |
| `trilm-3.9B` | TQ2_0 | 1112 | 306.9 ± 9.3 | 40.9 ± 0.1 |
| `superkaiii-bonsai-4B` | TQ2_0 | 1203 | 308.4 ± 1.8 | 37.9 ± 0.3 |
| `lily-bonsai-4B` | Q2_K | 1409 | 1935.1 ± 0.4 | 107.5 ± 2.2 |
| `lily-bonsai-8B` | IQ1_S | 1803 | 1119.1 ± 0.4 | 92.1 ± 0.2 |
| `gianni-bitnet-3B` | TQ2_0 | 1834 | 1909.8 ± 9.3 | 81.3 ± 2.1 |
| `outlier-10B-V2` | TQ1_0 | 2038 | 85.2 ± 0.4 | 30.8 ± 0.8 |
| `outlier-10B-V2` | TQ2_0 | 2330 | 173.5 ± 1.8 | 31.3 ± 0.3 |
| `lily-bonsai-8B` | Q2_K | 2836 | 1024.0 ± 6.6 | 61.4 ± 0.3 |

## Headline

- Fastest decode: **`lily-bonsai-1.7B-IQ1_S`** — **281.2 tok/s** (385 MB).
- Slowest decode: **`outlier-10B-V2-TQ1_0`** — **30.8 tok/s** (2038 MB).

## Notes

- Throughput-only sweep. PPL on wikitext-103 is a separate run via `benchmarks/sherry-ppl.sh`.
- `IQ1_S` = `lilyanatia` repacks of `prism-ml/Bonsai-*` unpacked weights → mainline-llama.cpp loadable.
- `TQ1_0` / `TQ2_0` = native ternary GGUF formats (llama.cpp PR #8151 lineage).
- `Q2_K` = mainline llama.cpp 2.5-bit, closest non-ternary baseline.
- All loaded full-GPU (`-ngl 99`) on gfx1151 unified memory (LPDDR5x ~256 GB/s).
- Source weights: `project_ternary_models_test_pile.md` in memory or `benchmarks/bench-1bit-pile.sh` for HF repo IDs.

## Reproduce

```sh
# 1. Pull pile (~36 GB):
hf download superkaiii/Ternary-Bonsai-4B-GGUF      --local-dir ./bonsai-4b-mainline/
hf download Green-Sky/TriLM_3.9B-GGUF              --local-dir ./trilm-3.9b/
hf download mradermacher/Outlier-10B-V2-GGUF       --local-dir ./outlier-v2-10b/
hf download lilyanatia/Bonsai-1.7B-requantized     --local-dir ./lily-bonsai-1.7b-rq/
hf download lilyanatia/Bonsai-4B-requantized       --local-dir ./lily-bonsai-4b-rq/
hf download lilyanatia/Bonsai-8B-requantized       --local-dir ./lily-bonsai-8b-rq/
hf download gianni-cor/bitnet_b1_58-3B-TQ2_0       --local-dir ./gianni-3b-tq2/
hf download gianni-cor/bitnet_b1_58-large-TQ2_0    --local-dir ./gianni-large-tq2/

# 2. Bench:
bash benchmarks/bench-1bit-pile.sh
# Output: /home/bcloud/claude output/bench-1bit-<TS>.{json,log}
```

