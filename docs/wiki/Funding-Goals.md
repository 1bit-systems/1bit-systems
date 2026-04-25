# Funding goals

The tiered funding ladder for training runs, raised via GitHub Sponsors. Every tier ships Sparse-BitNet weights publicly to Hugging Face before the next tier opens.

## Ladder

| tier | model | token budget | GPU-hrs (H200) | wall-clock | cost @ $3.39/hr/GPU | inference footprint |
|---|---|---:|---:|---|---:|---:|
| Tier 1 | **13B** Sparse-BitNet | 50B | 1.5k | ~8 days on 2× NVL | **$1.5k** goal | 2 GB (any RDNA3 iGPU) |
| Tier 2 | **30B** Sparse-BitNet | 80B | 5k | ~25 days on 2× NVL | **$12k** goal | 4.8 GB (Strix Halo + any RDNA3/4 dGPU) |
| Tier 3 (headline) | **70B** Sparse-BitNet | 50B distill | 14.5k | ~75 days on 8× NVL | **$50k** goal | **11 GB** (any RX 9070 XT, any Strix Halo) |
| Tier 4 (moonshot) | 70B full pretrain + SFT | 1T | 60k | ~250 days on 8× NVL | $200k | same 11 GB ship-time |

FP16 baselines for context: 13B = 26 GB, 30B = 60 GB, 70B = 140 GB. The 1.25 bpw Sparse-BitNet ship format compresses ~11×.

## Why distillation, not from-scratch

Full from-scratch Llama-3-70B was ~6.4M H100-hours for 15T tokens — north of $2M on NVL at current rates. That is off the table for a single-donor project.

Continued-pretrain distillation from a dense Llama-3 teacher into Sparse-BitNet student weights compresses to 1.25 bpw while preserving ~95% of dense quality at 50-100B distillation tokens (AAzdi *Sparse-BitNet* paper, verified behavior on our 0.5B runs). Tier 3 70B assumes this distillation path; the $50k goal is for distillation-budget tokens, not from-scratch pretraining.

## Why the tiers go 13B → 30B → 70B

Credibility ladder. A blanket $50k ask for a 70B training run with no public Sparse-BitNet weights yet would be a hard sell, and correctly so. The shipping order before Tier 1 opens:

1. Run 4 (0.5B, 3:4) finishes ~2026-04-24 — public HF + benchmark proof
2. Run 5 (0.5B, 2:4, Microsoft canonical) — ~31h on 2× NVL, ~$210 total — proves the full recipe
3. Both public on HF + site + README — now potential donors see the stack actually trains
4. **Then** Tier 1 13B goal opens

Tier 3 70B opens only once Tier 1 + Tier 2 have shipped. Tier 4 70B-full-pretrain is a "never say never" backstop, not a serious ask.

## Training HW gate

13B and 30B train on 2× NVL DDP via NVLink bridge.

70B cannot train on 1-2 NVL — full state (weights + grads + Adam) is ~500 GB, needs 8× NVL DDP minimum. Inference ship-time is a different number: the 70B distilled weights fit in 11 GB and run on any 16 GB Radeon or on Strix Halo's 128 GB unified pool with room to spare.

## Pod default

H200 NVL at $3.39/hr on RunPod, not SXM at $3.99/hr. Identical 141 GB HBM3e + 4.8 TB/s bandwidth for single-GPU training; NVLink advantage of SXM matters only for multi-GPU tensor-parallel training, which we do not do at the 13B/30B scale. 15% cost saving at identical throughput.

Never switch pods mid-run — migrating checkpoint + tmux session + env risks corruption and invalidates shadow comparison. Always finish a run, then provision the next pod.

## Funding surface

GitHub Sponsors is the canonical (and only) public funding lane — 0% platform fee, all proceeds underwrite training compute. Sponsorship opens when the XDNA 2 NPU ship gate clears, or when the project's positioning explicitly changes. Until then this page is a forward declaration, not a live link.

## See also

- [Training-Runs.md](./Training-Runs.md) — live Run 4 status, Run 5 plan, retrospective on prior runs
- [Benchmarks.md](./Benchmarks.md) — the numbers donors will be checking
- [Peak-Performance-Projection.md](./Peak-Performance-Projection.md) — where the stack ceiling actually is
