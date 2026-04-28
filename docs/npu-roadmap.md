# NPU lane — known optimization roadmap

The `flm:npu` lane on Strix Halo (XDNA 2 / AIE2P, served by FastFlowLM
through `xrt-plugin-amdxdna`) is *running* — we measured 87-95 tok/s
decode on `qwen3-0.6b-FLM` and 42 tok/s on `qwen3-1.7b-FLM` (see
`benchmarks/RESULTS-stack-2026-04-28.md`).

Those numbers are far below what the silicon can do. The NPU has 8
columns × 4 rows of AIE2P tiles, each tile with vector + scalar + DMA
units, theoretical aggregate well into the hundreds of GB/s of on-tile
bandwidth at low precision. We're seeing ~200 GB/s effective at best,
and on token-by-token decode it's much lower.

The known bottleneck — recorded here as project context, not a fix
shipped in this repo — is **per-call weight DMA**.

## The two compounding optimizations

There are two known per-token-decode wins that are upstream-side, both
about eliminating overhead between the actual compute work the AIE2P
tiles do:

1. **Persistent device-resident weights** — weights live on the NPU
   across calls; only activations + accumulators DMA on each `infer()`.
2. **Fused per-token dispatch loop** — the host issues *one* AIE graph
   per token instead of N graphs (one per `tiled_gemv`).

These compound. Together they should move the lane from "host driving
the schedule, NPU mostly waiting" to "NPU dominates wall-clock, host
is just feeding tokens."

## 1. Persistent device-resident weights

| Mode | Per-layer GEMV wall | Per-token decode est. (1.7B model) |
|---|---:|---:|
| **Current** (weights re-uploaded per call) | ~13 s | the ceiling we're seeing today |
| **Persistent** (weights uploaded once at model load, BOs kept resident) | ~10 ms | ~1300× per-layer speedup; the per-tile DMA streaming activations + accumulators is the only remaining wall-clock |

The fix at the architecture level: allocate BOs (XRT Buffer Objects) for
weight tiles **once** at model-load time, copy weights into them
**once**, then on every `infer()` call only DMA the activations and
accumulators across the tiles. Never re-upload weights between calls.

### Where this fix would land

Depending on which layer of the stack is responsible for the
re-upload — needs investigation, not yet measured:

1. **FastFlowLM** — if `flm` is calling `xrt::bo::write()` on the weight
   tensors per inference, the fix is FLM-side. Worth grepping FLM's
   source for whether a "persistent weights" or `--keep-weights-resident`
   flag exists, and if not, opening an upstream issue/PR.
2. **`xrt-plugin-amdxdna`** — if the kernel-side plugin is invalidating /
   re-binding BOs between calls, that's a deeper fix. Would need an
   `xrt_bo` flag to mark a BO `KEEP_RESIDENT` (or an equivalent), or an
   API contract that BOs persist across submissions on the same xclbin.
3. **Lemonade** — if `max_loaded_models=1` swap is also flushing FLM
   state between requests, lemond would benefit from a "pin this model"
   mode so a hot model's BOs stay resident across `/v1/chat/completions`
   calls.

## 2. Fused per-token dispatch loop

The current pattern (best understood from the FLM/IRON dispatch code):
for each weight matrix in a layer, the host fires `tiled_gemv` once per
tile of the `n_block × k_block` tiling — typically ~25 separate
`kern.run().wait()` calls per matrix. Each call has fixed dispatch
overhead (host → kernel ioctl, command-queue insertion, AIE wakeup,
completion wait, host re-dispatch).

The fix: push the `n_block × k_block` loop *into a single AIE graph*
that internally issues all 25 mmul calls before signaling completion.
The host only does `kern.run().wait()` once per matrix.

| Mode | Dispatches per matrix | Per-token decode hot-path |
|---|---:|---|
| **Current** (host-side tile loop) | ~25 | dispatch overhead × 25 dominates |
| **Fused** (single-graph tile loop) | 1 | only the actual compute + on-tile DMA |

Saving 24 dispatches per matrix per token. With multiple matrices per
layer (Q, K, V, O, gate, up, down for transformer-style) and many
layers per token, that's *thousands* of avoided dispatches per
generated token.

Where this lands: **AIE graph compiler / FLM kernel layer**. Same
upstream-FLM territory as the persistent-weights fix — neither is
something this repo can or should reimplement. Combined with #1, the
FLM lane stops being "host orchestrates everything" and becomes "NPU
runs the actual work."

## Why this is upstream work, not 1bit-systems work

Per the project's `CLAUDE.md` lean rule:

> **Lean over scaffolding.** This repo is the install + control plane on
> top of upstream Lemonade Server and FastFlowLM. Do not reimplement
> inference. Do not port kernels in-tree. Upstream PRs are the right
> place for kernel work.

The persistent-weights fix is firmly inference-side. The right action
from this repo is:

- Probe FLM to confirm where the re-upload is happening (one line of
  `strace -e trace=ioctl flm serve ...` on a hot loop should make it
  obvious if XRT submissions include weight DMA or not).
- File the issue/PR upstream against `FastFlowLM/FastFlowLM` and/or the
  AMD xdna-driver project if it's plugin-side.
- Once landed upstream, our `flm:npu` decode numbers should jump
  significantly without any change to this repo.

## What we *can* do here

- **Track the upstream fix** — link to whatever issue/PR surfaces and
  link from this doc. If/when FLM ships persistent-weights, bump the
  version-pin patch in `install.sh` and add a "before / after" bench
  row in `benchmarks/`.
- **Write the bench harness now** — a fixed-prompt fixed-output bench
  that hits `flm:npu` enough times to expose the per-call DMA cost.
  Land in `benchmarks/bench-npu-resident.sh`. When the upstream fix
  ships, run the same harness and the delta is the receipt.
- **Set expectations honestly** — the NPU lane is currently a real but
  modest accelerator. The 1bit-systems hero claim ("two-bit killer") is
  about the iGPU lane today; the NPU lane has headroom that isn't ours
  to deliver, but is the next big number when upstream lands it.

## References

- AMD XDNA 2 architecture overview (Strix Halo NPU):
  `https://www.amd.com/en/technologies/xdna.html`
- AMD XDNA driver (kernel-side): `https://github.com/amd/xdna-driver`
- FastFlowLM: `https://github.com/FastFlowLM/FastFlowLM`
- Lemonade FLM-on-Linux article (joint AMD + FLM):
  `https://lemonade-server.ai/flm_npu_linux.html`
- Lane current measurements: `benchmarks/RESULTS-stack-2026-04-28.md`
