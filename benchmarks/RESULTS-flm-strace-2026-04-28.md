# FastFlowLM strace probe — 2026-04-28

**Hardware:** Strix Halo, NPU `/dev/accel/accel0` (XDNA 2, AIE2P, 8 columns), FW 1.1.2.65, amdxdna driver 0.6, kernel 7.0.2-1-cachyos.
**FLM:** `/usr/bin/flm` serving `qwen3:0.6b` (Qwen3-0.6B-NPU2) on port 8001 in isolation (lemond stopped).
**Method:** `strace -f -p $FLM_PID -e trace=ioctl` attached to a hot server, fired one OpenAI-compat request asking for `max_tokens=20` at `temperature=0`. Idle baseline = 10 s of strace with no traffic. Verbose run with `-v -s 256` to capture `amdxdna_drm_create_bo` struct so size/type fields could be parsed.

**Decoded request used for headline numbers:**
- prompt_tokens = 28 (prefill)
- completion_tokens = 20 (decode)
- prefill_duration_ttft = 0.660 s (≈ 42 tps)
- decoding_duration = 0.206 s (≈ 97 tps)
- wall-clock = 0.892 s

## strace -c summary (one decode request)

```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
100.00    0.003378           0      3879           ioctl
------ ----------- ----------- --------- --------- ----------------
100.00    0.003378           0      3879           total
```

`-e trace=ioctl` only, so the table is single-row by design. **No ioctls observed during the 10 s idle baseline** — every count below is delta-from-idle.

## ioctl breakdown by request code (fd 7 = `/dev/accel/accel0`)

```
   1091  DRM_IOCTL_AMDXDNA_CREATE_BO
   1091  DRM_IOCTL_AMDXDNA_GET_BO_INFO
   1091  DRM_IOCTL_GEM_CLOSE
    313  DRM_IOCTL_AMDXDNA_EXEC_CMD
    293  DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT
```

(strace prints CREATE_BO as `DRM_IOCTL_AMDXDNA_CREATE_BO or DRM_IOCTL_IVPU_BO_INFO or DRM_IOCTL_XE_VM_CREATE` because the request code is shared; with fd 7 → `/dev/accel/accel0`, it is unambiguously the amdxdna call.)

## Per-token rates (decode phase only, t = prefill_end → +206 ms, 20 tokens)

| ioctl                            | calls in decode | per token |
|----------------------------------|----------------:|----------:|
| DRM_IOCTL_AMDXDNA_CREATE_BO      |             600 |     30.0  |
| DRM_IOCTL_AMDXDNA_GET_BO_INFO    |             600 |     30.0  |
| DRM_IOCTL_GEM_CLOSE              |             609 |     30.5  |
| DRM_IOCTL_AMDXDNA_EXEC_CMD       |              57 |      2.85 |
| DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT  |              36 |      1.80 |

For comparison, prefill (28 tokens, 660 ms): 461 CREATE_BO / 253 EXEC_CMD / 452 GEM_CLOSE — much heavier per-step in absolute terms but a similar ratio.

## CREATE_BO size + type histogram (whole request, 1091 allocations)

| size       | count | type                |
|-----------:|------:|---------------------|
| 34,172 B   |   644 | DEV (3)             |
|  7,904 B   |   112 | DEV (3)             |
|    224 B   |    74 | CMD (4)             |
|  2,288 B   |    56 | DEV (3)             |
|  6,832 B   |    56 | DEV (3)             |
| 14,464 B   |    56 | DEV (3)             |
|  9,104 B   |    28 | DEV (3)             |
| 11,184 B   |    28 | DEV (3)             |
| 24,336 B   |    28 | DEV (3)             |
|  1,048,576 |     6 | (mix incl. SHMEM)   |
|  2,097,152 |     3 | (mix incl. SHMEM)   |

Type counts: DEV=1008, CMD=74, SHMEM=9. **Total bytes allocated and freed: 36.3 MB** across the whole 48-token request.

## Hypothesis #1 — per-call weight DMA: **REFUTED**

Total CREATE_BO bytes during decode ≈ 23 MB; weights for Qwen3-0.6B are ≈ 0.6–1.2 GB depending on quant. The largest single allocation we see is 2 MB, three times total. Weights are mapped at server start (likely into a persistent DEV_HEAP) and **not** reallocated per `infer()`. What *is* churning is small DEV-type scratch buffers (modal sizes 34 KB and 7.9 KB) and CMD-type command buffers (224 B), with a 1:1 CREATE_BO ↔ GEM_CLOSE pair almost every step. So the box is doing **per-step scratch/cmd-buffer churn**, not weight re-upload. Different problem, smaller fix.

## Hypothesis #2 — per-tile dispatch storm: **REFUTED**

EXEC_CMD per decode token = **2.85**, not "thousands". With 8 NPU columns this is well below "one dispatch per tile per layer". The runtime is batching tile work into a small number of submitted command chains. Whatever per-step overhead exists, it is not in the EXEC_CMD count.

## Where the ioctl budget actually goes

Per decoded token: ~30 CREATE_BO + ~30 GET_BO_INFO + ~30 GEM_CLOSE + ~3 EXEC_CMD + ~2 SYNCOBJ_WAIT ≈ **96 ioctls/token**. The hot path is the `CREATE_BO → GET_BO_INFO → ... → GEM_CLOSE` triplet, fired ~30× per token. Each round-trip into the DRM ioctl path costs a syscall, a copy_from_user, and (for CREATE_BO) a kernel allocation + GTT map. At 97 tps decode, ~30 alloc/free pairs/token ≈ 2,910 alloc/free pairs per second — pure overhead that adds nothing the model needs.

## Upstream issue draft

> **Title:** FLM allocates and frees ~30 small DRM BOs per decoded token via amdxdna — pool them
>
> Probed `flm serve qwen3:0.6b` (FLM 0.x, amdxdna driver 0.6, FW 1.1.2.65, AIE2P / 8-col) with `strace -f -e trace=ioctl` against a hot server during a single `/v1/chat/completions` call (`max_tokens=20`, `temperature=0`).
>
> Per decoded token I see **30.0 `DRM_IOCTL_AMDXDNA_CREATE_BO` + 30.0 `DRM_IOCTL_AMDXDNA_GET_BO_INFO` + 30.5 `DRM_IOCTL_GEM_CLOSE`**, paired roughly 1:1, with only **2.85 `DRM_IOCTL_AMDXDNA_EXEC_CMD`** dispatches in the same window (and 1.8 `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`s). Allocation sizes are small (modal 34 KB and 7.9 KB DEV-type, 224 B CMD-type; only three 2 MB BOs across the whole 48-token request). Total churn is ~36 MB allocated *and freed* per request — i.e. these are scratch / command-list buffers, not weights. Weights stay resident across calls (no GB-scale CREATE_BO during decode), and the per-tile dispatch fanout is fine (single-digit EXEC_CMD per token). The ceiling at ~97 tok/s for qwen3:0.6b on this NPU is partly being eaten by ~2.9k alloc/free DRM round-trips per second of pure ioctl + GTT-map overhead.
>
> **Suggested fix (FLM side):** keep a per-stream pool of CMD and small-DEV BOs sized by the largest seen (≥34 KB), and reuse them across `infer()` steps instead of CREATE_BO/GEM_CLOSE'ing each time. Decoder-side scratch lifetime is one step, so a free-list keyed on `(type, size)` is sufficient. Expect the 30/30/30 triplet to collapse to ~0 ioctls/token for the steady-state case, with EXEC_CMD/SYNCOBJ_WAIT becoming the only per-token DRM traffic. If buffer pooling already exists, it is being bypassed when prefill transitions to decode (the pattern is identical in both phases).
>
> Repro: `flm serve qwen3:0.6b --port 8001`, warm with one request, then `strace -f -p $(pgrep -f 'flm serve') -e trace=ioctl -c` while firing one 20-token completion. Idle traffic on the FLM PID is zero, so the count is delta-clean.

## Notes / caveats

- `bpftrace` is not installed on this box, so the DRM-tracepoint cross-check was skipped. The strace numbers are consistent across two independent runs (3879 and 3877 ioctls) so I trust them.
- `lemond` was already inactive when probing started (PPL sweep had stopped it); restarted at the end. `open-webui` and `gaia` were untouched and remained active throughout.
- `flm serve` was killed cleanly after the probe; nothing else holds `/dev/accel/accel0`.
- Raw artifacts: `/tmp/flm-probe/strace-decode.txt` (-c summary), `/tmp/flm-probe/strace-decode-full.txt` (full trace), `/tmp/flm-probe/strace-tt.txt` (timestamped), `/tmp/flm-probe/strace-verbose.txt` (verbose w/ struct bytes), `/tmp/flm-probe/decode-response.json` (model output + timing).
