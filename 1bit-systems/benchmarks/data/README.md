# Benchmark data

Real outputs from the running stack. Not synthetic, not cherry-picked.

| file | what it is |
|---|---|
| `shadow-burnin-state-20260422.txt` | counters from `benchmarks/shadow-burnin.sh` — rounds, exact matches, p95 latencies. ~6850 rounds over ~7h of continuous v1-vs-v2 comparison. |
| `shadow-burnin-sample-200-20260422.jsonl` | last 200 rounds of the live jsonl (full file is ~3 MB, not tracked). Each row = one prompt fired at both `bitnet_decode :8080` and `1bit-halo-server :8180`, diffed. |
| `ppl-wikitext103-2026-04-21.json` | perplexity harness run against wikitext-103. Reference numbers for the post-RoPE-fix era. |
| `anvil-bench-*.txt` | five anvil (CI runner) bench outputs across different commits. Same workload, different SHAs — shows regression detection. |
| `medusa-bench-2026-04-21.json` | Medusa speculative-decoding bench, single prompt at varying head counts. |
| `sparse-bitnet-run4-progress-20260422.log` | live training log from the H200 pod during Sparse-BitNet Run 4. Step-wise loss + tok/s. |

Updated as runs complete. Schema is whatever the bench script emits; no post-processing.
