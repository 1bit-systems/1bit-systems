# Training runs

Status of the active Sparse-BitNet retrain and the next run queued up behind it. Source of truth: `~/.claude/projects/-home-bcloud/memory/project_sparse_bitnet_run.md`, `project_run5_bitnet_2_4.md`, `reference_runpod_h200_nvl.md`.

## Run 5 in flight, Run 6 next (2026-04-23)

Run 5 (2:4 Microsoft-canonical Sparse-BitNet on Qwen-0.5B) is queued behind Run 4 and launches on **2× H200 NVL DDP** the moment Run 4 clears the pod — `accelerate launch --num_processes=2 --multi_gpu ...`, ~90 k tok/s aggregate, ~31 h wall-clock, ~$210 at $3.39/hr/GPU. DDP is the only shape that makes sense: single-GPU on NVL is identical silicon to SXM and the NVLink fabric is there to be used when we've got two cards in the pod.

**Run 6 is the pivot to ternary TTS.** Qwen3-TTS 0.6B Base, QAT'd to 1.58-bit, trained on the same H200 NVL ×2 rig. The [2026-04-23 research digest](../research-digest-20260423.md) is why: one academic paper (BitTTS, no code), one closed commercial demo (EZWhisper), and nothing else in sub-2-bit speech-synth land. Shipping a runnable ternary TTS at 0.6B would be the first public weights in that lane on any hardware. Recipe stub + GGUF tq2_0 slot are already in `packages.toml` (`model.qwen3-tts-0p6b-ternary`) — trainer work kicks off the moment Run 5 clears shadow-burnin.

## Run 4 — 3:4 Sparse-BitNet @ 0.5B (live)

| attribute | value |
|---|---|
| Status | **Live** — launched 2026-04-22, patched relaunch after Run 3 mask-monitoring bail |
| Recipe | 3:4 N:M sparsity on 1.58-bit weights → 1.25 effective bpw (Sherry-compatible pack) |
| Model | Qwen-0.5B base, Sparse-BitNet retrain |
| Pod | H200 SXM at `root@205.196.19.116:11595` — locked on current pod, can't migrate mid-run |
| Cost | SXM $3.99/hr × ~57h = **$227** |
| Token budget | 10B |
| Throughput | 49.5k tok/s measured (single-GPU H200) |
| Wall-clock ETA | ~57 h → finishes ~2026-04-24 17:00 UTC |
| Trainer | `scripts/pretrain_sparse_bitnet_qwen_0p5b.py` (TRL + HF streaming loader) |

Patch set applied after Run 3 bail:

- `model.enable_mask_monitoring()` at init (default-off was the bug)
- `mask_cache.clear()` after each verify (stale empty cache triggered false-positive integrity bail)
- `--save-every 100` (Run 3 lost all 500 steps; now every 100-step window is a recoverable ckpt)

Run 3 pre-bail numbers: loss 11.04 → 5.77 across steps 50 → 500, throughput steady at 49.5k tok/s — the training itself was healthy, the bail was a pure monitoring bug.

## Run 5 — 2:4 Sparse-BitNet @ 0.5B, Microsoft canonical (queued)

| attribute | value |
|---|---|
| Status | **Queued** — launches immediately after Run 4 completes |
| Recipe | 2:4 N:M sparsity (Microsoft `AAzdi/Sparse-BitNet` canonical), still 1.25 bpw pack |
| Model | Qwen-0.5B base |
| Pod | **H200 NVL × 2 DDP** at $3.39/hr/GPU |
| Cost | 2 × $3.39/hr × 31h = **~$210** |
| Token budget | 10B |
| Throughput | ~90k tok/s (1.85× single-GPU, gradient-sync overhead) |
| Wall-clock ETA | ~31 h |
| Trainer launch | `accelerate launch --num_processes=2 --multi_gpu pretrain_sparse_bitnet_qwen_0p5b.py` |

**Why 2:4 over 3:4.** Microsoft canonical (published PPL within ~0.3 of dense BitNet-1.58), better-studied ablations, and on H200/NVL tensor cores 2:4 fires at 2× dense throughput at the training-kernel tier. AMD bandwidth win is the same either way — both 2:4 and 3:4 pack to 1.25 bpw on our ship format.

**What changes.** One line in the trainer (`keep=3` → `keep=2` in the Sparse-BitNet mask), header docstring, new `.h1b` target (`halo-1bit-2b-sparse24.h1b`), new rocm-cpp kernel variant `kernels/ternary_gemv_sparse24.hip` (or a `--mode` flag on the existing Sherry kernel). Unpack LUT swaps from 4-bit zero_pos to 6-valued slot-select.

**Post-Run-5 bench targets.**
- PPL on wikitext-103 within 0.5 of Run 4's post-retrain PPL
- Decode @ 64 tok — target ~90 tok/s on gfx1151 (1.2× dense ternary baseline of 66), ~105 tok/s on gfx1201
- Sparse-kernel vs scalar reference — bit-exact per our usual harness

## Pod default: H200 NVL, not SXM

Default for all future training runs: **H200 NVL at $3.39/hr**, not SXM at $3.99/hr. Identical 141 GB HBM3e + 4.8 TB/s bandwidth for single-GPU training; the NVLink advantage of SXM only matters for multi-GPU tensor-parallel training, which we do not do.

**Do not switch mid-run.** Always finish the current run on its current pod, then provision NVL for the next one. Migrating checkpoint + tmux session + env across pods risks corruption and invalidates shadow comparison.

Dollar impact on a few forward runs:

| run | SXM | NVL | delta |
|---|---:|---:|---:|
| Run 4 (~57 h) | $227 | $193 | **+$34** (locked on SXM, can't switch) |
| 13B retrain (~300 h) | $1197 | $1017 | +$180 |
| Sherry-v2 continuation (~50 h) | $200 | $170 | +$30 |

## Rule A note

Trainers are Python on RunPod (TRL + HF). Python is allowed there — RunPod pods are caller-side, not the serving path. The inference side stays zero-Python (Rust + HIP). Weights flow: pod → pi archive (rsync) → requantizer (Python, one-shot) → `.h1b` → `strixhalo` → shadow-burnin → cutover.

## See also

- [Funding-Goals.md](./Funding-Goals.md) — where these runs plug into the Patreon ladder
- [Benchmarks.md](./Benchmarks.md) — post-retrain parity gates (wikitext-103 PPL, shadow-burnin)
- [Sherry-Default-Decision.md](./Sherry-Default-Decision.md) — prior decision on 1.25 bpw ship format
