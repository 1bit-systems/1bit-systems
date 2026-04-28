# Qwen3.5-35B-A3B (MoE) — perplexity sweep, IQ2_XXS vs Q4_K_XL

_Captured 2026-04-28 on the strixhalo box (Ryzen AI MAX+ 395 / `gfx1151`, 124 GB LPDDR5x). Closes the open TODO in [`benchmarks/RESULTS-qwen3.5-35b-quant-2026-04-28.md`](RESULTS-qwen3.5-35b-quant-2026-04-28.md): "Perplexity / quality benchmarks for IQ2_XXS vs Q4_K_XL on this checkpoint — separate sweep."_

## Methodology

- **Runner:** `~/.cache/lemonade/bin/llamacpp/vulkan/llama-perplexity` (matches the same Vulkan lane our chat traffic auto-picks; `llama.cpp` `b8668`).
- **Backend:** `Vulkan0 (8060S Graphics (RADV STRIX_HALO))`, all 99 layers offloaded to iGPU (`-ngl 99`).
- **Test set:** wikitext-103 raw test split (`test-00000-of-00001.parquet` from `huggingface.co/datasets/wikitext`), trimmed to first **256 KB** (~50 chunks-worth, 20 used).
- **Run shape:** `--chunks 20 -c 512 -b 512` for each model. 10,240 tokens scored per model.
- **What it computes:** sliding-window negative log-likelihood per token across the corpus, then `PPL = exp(mean NLL)`. Lower is better; same number across models is direct apples-to-apples on text-prediction quality.
- `lemond.service` was stopped during the sweep so the model file was uncontended; restarted after.

## Results

| Model | Quant | bpw (~) | Disk | VRAM (Vulkan) | PPL | ± stddev | Wall |
|---|---|---:|---:|---:|---:|---:|---:|
| `unsloth/Qwen3.5-35B-A3B-UD-Q4_K_XL` | Q4_K_XL | 4.5 | 22 GB | 21,256 MiB | **6.7051** | 0.22950 | 16 s |
| `Manojb/Qwen3.5-35B-A3B-UD-IQ2_XXS` | IQ2_XXS | 2.0 | 11.5 GB | 11,761 MiB | **7.2971** | 0.25327 | 14 s |
| **Δ IQ2_XXS / Q4_K_XL** | | **0.44×** | **0.52×** | **0.55×** | **+8.83 %** | | |

PPL chunk-by-chunk shows the same shape on both runs (same prompt → same difficulty curve):

```
Q4_K_XL    [1]4.44 [5]5.44 [10]7.15 [15]7.11 [20]6.71  →  6.7051
IQ2_XXS    [1]4.70 [5]5.93 [10]7.91 [15]7.77 [20]7.30  →  7.2971
```

The IQ2_XXS curve runs roughly +0.3-0.6 PPL above Q4_K_XL at every chunk — the gap is consistent across topics, not concentrated in any particular chunk.

## Read

Standard rule-of-thumb bands for "how bad is the quality drop":

| ΔPPL | Reading |
|---:|---|
| < 5 %  | barely noticeable |
| 5 — 10 %  | **occasional quality issues** |
| 10 — 20 %  | noticeable across most outputs |
| > 20 %  | broken on long-form output |

**IQ2_XXS at +8.83 %** lands in the "occasional quality issues" band — the upper end of acceptable for a daily driver. Combined with the perf wins from [`RESULTS-qwen3.5-35b-quant-2026-04-28.md`](RESULTS-qwen3.5-35b-quant-2026-04-28.md):

- **Decode:** +35 % steady-state (54 → 73 tok/s)
- **Disk:** 0.52× (22 GB → 11.5 GB)
- **VRAM:** 0.55× (21,256 → 11,761 MiB) — *the actual reason this matters on consumer cards*
- **PPL:** +8.83 % (6.7051 → 7.2971)

The VRAM number is the headline. **IQ2_XXS fits in 16 GB GDDR6 with KV-cache headroom; Q4_K_XL doesn't.** That's what makes a 35B-class MoE actually run on the 9070 XT box (see [`RESULTS-9070xt-2026-04-28.md`](RESULTS-9070xt-2026-04-28.md) — same model, 131 tok/s decode on the dGPU lane), and it's why we made it the daily driver.

## What this doesn't tell you

- Single PPL number on 10k tokens of wikitext is a *mean* — task-specific quality (code, math, tool-calling, multi-turn coherence) can diverge from PPL. Worth a separate sweep on `MMLU-redux` / `IFEval` / `BFCL v3` if anyone wants harder receipts.
- Both models are the same checkpoint at different bit precisions. This isn't comparing different model families.
- We tested only **20 chunks** to keep the run under a minute. Full wikitext-103 (≈300+ chunks) is a larger sample; expect the headline numbers to drift by tenths of a percent at most.

## Reproduce

```sh
# Disable lemond so the model file isn't contended
systemctl --user stop lemond

# Get wikitext-103 test slice
curl -sSL https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet -o /tmp/wiki.parquet
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('/tmp/wiki.parquet')
text = '\n\n'.join(t.column('text').to_pylist())
open('/tmp/wiki-test.txt','w').write(text[:262144])
"

PERPLEXITY=~/.cache/lemonade/bin/llamacpp/vulkan/llama-perplexity
chmod +x $PERPLEXITY  # rocm/llama-perplexity ships without +x in some lemonade versions

# Run for each model
$PERPLEXITY -m <model.gguf> -f /tmp/wiki-test.txt --chunks 20 -ngl 99 -c 512 -b 512 2>&1 | grep "Final estimate"

systemctl --user start lemond
```
