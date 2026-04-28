# NPU lane — measured optimization roadmap

The `flm:npu` lane on Strix Halo (XDNA 2 / AIE2P, served by FastFlowLM
through `xrt-plugin-amdxdna`) is *running* — we measured 87-95 tok/s
decode on `qwen3-0.6b-FLM` and 42 tok/s on `qwen3-1.7b-FLM` (see
`benchmarks/RESULTS-stack-2026-04-28.md`). Those numbers are below
the silicon ceiling.

**This doc was originally drafted from two hypothesis. We then probed
with `strace` (full report in
[`benchmarks/RESULTS-flm-strace-2026-04-28.md`](../benchmarks/RESULTS-flm-strace-2026-04-28.md))
and the data refuted both. Leaving the strikethroughs visible because
the *measurement* is the interesting bit.**

## ~~Hypothesis #1 — per-call weight DMA~~ → **REFUTED**

~~Allocate W BOs once at model load, copy weights once, never re-upload.
This alone should move the NPU lane from "13 s per layer GEMV" to
"~10 ms per layer GEMV" once the per-tile DMA is the only cost.~~

**Measured:** total `DRM_IOCTL_AMDXDNA_CREATE_BO` bytes during a 20-token
decode = ~23 MB. Weights for Qwen3-0.6B are 0.6-1.2 GB. Largest single
allocation observed: 2 MB (three of them). **Weights are mapped at
server start (likely into a persistent DEV_HEAP) and not re-uploaded
per `infer()`.** What FLM is doing here is already correct.

## ~~Hypothesis #2 — per-tile dispatch storm~~ → **REFUTED**

~~Push the n_block × k_block tile loop into a single AIE graph that
issues all 25 mmul calls in one kern.run().wait(). Saves 24
dispatches per matrix per token.~~

**Measured:** `DRM_IOCTL_AMDXDNA_EXEC_CMD` per decoded token = **2.85**,
not "thousands". The runtime already batches tile work into a small
number of command chains. EXEC_CMD count is *not* the bottleneck.

## What's actually slow — per-step DRM BO churn

Per decoded token at 97 tok/s on Qwen3-0.6B:

```
DRM_IOCTL_AMDXDNA_CREATE_BO     30.0 / token
DRM_IOCTL_AMDXDNA_GET_BO_INFO   30.0 / token
DRM_IOCTL_GEM_CLOSE             30.5 / token
DRM_IOCTL_AMDXDNA_EXEC_CMD       2.85 / token
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT  1.80 / token
                                ────
                                ~96 ioctls / token
```

The hot path is a `CREATE_BO → GET_BO_INFO → … → GEM_CLOSE` triplet
fired ~30 times per token, each round-trip allocating then freeing a
tiny scratch buffer (modal 34 KB DEV) or command buffer (224 B CMD).
Total churn = 36 MB allocated *and freed* per 48-token request —
36 MB of pure overhead that adds nothing the model needs.

At 97 tok/s × 30 alloc/free pairs/token = **~2,910 DRM round-trips per
second** for buffer churn alone. Each round-trip costs a syscall, a
`copy_from_user`, and (for CREATE_BO) a kernel allocation + GTT map.
That's the ceiling we're hitting before EXEC_CMD or compute even
becomes the bottleneck.

## The fix — per-stream BO pool

FLM-side change: keep a per-stream free-list of CMD and small-DEV BOs
keyed on `(type, size)`. On each step, pop a BO from the free-list (or
allocate if none); on each step end, return to the free-list instead
of `GEM_CLOSE`'ing. Once warm, the steady-state ioctl count per token
should collapse to roughly `EXEC_CMD + SYNCOBJ_WAIT` ≈ ~5 ioctls/token,
down from ~96.

Expected delta: hard to predict precisely without running the patched
binary, but the ~2.9 k alloc/free round-trips/sec of overhead are a
nontrivial slice of the ~97 tok/s ceiling on a 0.6B model. **A
plausible reading is +20-50% steady-state decode tok/s** (eliminating
GTT-map and kernel-alloc cost from the hot path), with the larger win
on smaller models where this overhead is a bigger relative fraction
of total wall-clock.

## The fix — keep it in our `1bit-flm` fork

We don't file this upstream. The plan is the same fork-and-patch path
we've used elsewhere (`1bit-gaia`, `1bit-sd.cpp`, `1bit-ggml`, etc.):
fork `FastFlowLM/FastFlowLM` → `bong-water-water-bong/1bit-flm`,
land the BO-pool patch on a branch, build, ship via a sibling AUR pkg
that replaces the upstream `fastflowlm` dep with our fork.

The patch description below is written as a commit message body for
the fork branch (not an upstream issue). Same content; different
audience.

### Patch description (commit body for `1bit-flm`)

> **Title:** FLM allocates and frees ~30 small DRM BOs per decoded
> token via amdxdna — pool them
>
> Probed `flm serve qwen3:0.6b` (FLM 0.9.39, amdxdna driver 0.6,
> FW 1.1.2.65, AIE2P / 8-col) with `strace -f -e trace=ioctl` against
> a hot server during a single `/v1/chat/completions` call
> (`max_tokens=20`, `temperature=0`).
>
> Per decoded token I see **30.0 `DRM_IOCTL_AMDXDNA_CREATE_BO` + 30.0
> `DRM_IOCTL_AMDXDNA_GET_BO_INFO` + 30.5 `DRM_IOCTL_GEM_CLOSE`**,
> paired roughly 1:1, with only **2.85
> `DRM_IOCTL_AMDXDNA_EXEC_CMD`** dispatches in the same window (and
> 1.8 `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`s). Allocation sizes are small
> (modal 34 KB and 7.9 KB DEV-type, 224 B CMD-type; only three 2 MB
> BOs across the whole 48-token request). Total churn is ~36 MB
> allocated *and freed* per request — i.e. these are scratch /
> command-list buffers, not weights. Weights stay resident across
> calls (no GB-scale CREATE_BO during decode), and the per-tile
> dispatch fanout is fine (single-digit EXEC_CMD per token). The
> ceiling at ~97 tok/s for qwen3:0.6b on this NPU is partly being
> eaten by ~2.9k alloc/free DRM round-trips per second of pure ioctl
> + GTT-map overhead.
>
> **Suggested fix (FLM side):** keep a per-stream pool of CMD and
> small-DEV BOs sized by the largest seen (≥34 KB), and reuse them
> across `infer()` steps instead of CREATE_BO/GEM_CLOSE'ing each
> time. Decoder-side scratch lifetime is one step, so a free-list
> keyed on `(type, size)` is sufficient. Expect the 30/30/30
> triplet to collapse to ~0 ioctls/token for the steady-state case,
> with EXEC_CMD/SYNCOBJ_WAIT becoming the only per-token DRM
> traffic. If buffer pooling already exists, it is being bypassed
> when prefill transitions to decode (the pattern is identical in
> both phases).
>
> Repro: `flm serve qwen3:0.6b --port 8001`, warm with one request,
> then `strace -f -p $(pgrep -f 'flm serve') -e trace=ioctl -c`
> while firing one 20-token completion. Idle traffic on the FLM PID
> is zero, so the count is delta-clean.

## Why this is upstream work, not 1bit-systems work

Per the project's `CLAUDE.md` lean rule, this repo is the install +
control plane on top of upstream Lemonade Server and FastFlowLM. We
don't reimplement inference, don't port kernels in-tree. The right
action from this repo is:

1. ~~Probe FLM with strace to confirm where the re-uploads happen~~
   **Done — see `benchmarks/RESULTS-flm-strace-2026-04-28.md`.**
2. **Fork FLM into `bong-water-water-bong/1bit-flm`** and land the
   BO-pool patch on a branch. The patch description above is the
   commit body. We don't file upstream — the integrated stack is the
   project's edge; we keep our patches in our forks.
3. Build a sibling AUR pkg (`1bit-flm-bin`?) that drops the patched
   `flm` binary in place of the upstream `fastflowlm` package. Bump
   `1bit-systems-{git,bin}`'s `depends=()` to point at our fork.
4. Use the **regression bench harness** at
   [`benchmarks/bench-npu-ioctl-budget.sh`](../benchmarks/bench-npu-ioctl-budget.sh)
   to assert `ioctls/token` stays under threshold. Currently set to
   fail at 250 (baseline observed: ~215 total / 18 decoded tokens at
   FLM 0.9.39). After the BO-pool patch lands, expect this number
   to collapse to <50 — bump `THRESHOLD` down then.
5. Re-bench `flm:npu` after the patch lands. Before/after decode tok/s
   delta becomes the project's receipt.

## What we *can't* and *won't* do here

Re-implement FastFlowLM, port AIE kernels in-tree, fork the amdxdna
driver. The cpp-tower archive at `archive/cpp-tower-2026-04-27` is
where that scope went and stayed when it didn't ship.

## References

- `benchmarks/RESULTS-flm-strace-2026-04-28.md` — full strace probe,
  raw counts, BO size histogram, idle baseline, repro commands.
- AMD XDNA driver (kernel-side): `https://github.com/amd/xdna-driver`
- FastFlowLM: `https://github.com/FastFlowLM/FastFlowLM`
- Lemonade FLM-on-Linux article (joint AMD + FLM):
  `https://lemonade-server.ai/flm_npu_linux.html`
- Lane current measurements: `benchmarks/RESULTS-stack-2026-04-28.md`
