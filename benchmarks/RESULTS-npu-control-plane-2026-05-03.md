# NPU control-plane benchmark — 2026-05-03

**Hardware:** AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, XDNA NPU at `/dev/accel/accel0`.
**XRT / driver:** XRT 2.21.75, `amdxdna` 7.0.3-1-cachyos, NPU firmware 1.1.2.65.
**Device detected by `xrt-smi examine`:** `[0000:c6:00.1] RyzenAI-npu5`.
**Stack state after probes:** `lemond.service`, `flm.service`, `1bit-proxy.service`, `open-webui.service`, and `1bit-stack.target` active.

## Method

The probe uses `benchmarks/bench-npu-ioctl-budget.sh`, which:

1. Stops `lemond.service` and `flm.service` so a standalone `flm serve` owns the NPU lane.
2. Starts `flm serve <model>` on an isolated port.
3. Warms the model once.
4. Attaches `strace -f -e trace=ioctl -c` to the FLM PID.
5. Fires one OpenAI-compatible `/v1/chat/completions` request.
6. Counts total `ioctl` calls during decode and divides by decoded tokens.
7. Restores the services that were active before the probe.

Nonzero ioctls during the request, attached to the FLM process with `/dev/accel/accel0` present, confirm the FastFlowLM NPU driver path is active.

## Results

| Model | Prompt | Decoded tokens | Total ioctls | ioctls/token | Decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---|
| `qwen3:0.6b` | default short prompt | 19 | 3,878 | 204 | 97.4 | Warn, under fail threshold |
| `qwen3:0.6b` | long count prompt | 48 | 6,523 | 135 | 94.8 | Pass |
| `qwen3:1.7b` | default short prompt | 32 | 5,019 | 156 | 41.7 | Pass |
| `qwen3.5:4b` | default short prompt | 4 | 2,854 | 713 | 12.8 | Invalid/noisy: model stopped after 4 tokens |
| `qwen3.5:4b` | long count prompt | 48 | 7,455 | 155 | 12.9 | Pass |
| `deepseek-r1:8b` | long count prompt | 24 | 4,727 | 196 | 11.3 | Pass |

The `qwen3.5:4b` short-prompt run was a false alarm for regression tracking: the prompt asked for five short words and the model stopped after four decoded tokens, so fixed request overhead dominated the ratio. The long-prompt rerun reached the intended token budget and passed at 155 ioctls/token.

## Unified endpoint check

After the isolated probes restored services, the single control-plane endpoint was verified through `1bit-proxy`:

- `1bit status` reported FLM on `:52625`, proxy on `http://127.0.0.1:13306`, GAIA on `:41995`, and 57 unified models.
- `GET http://127.0.0.1:13306/v1/models` returned FLM-owned models including `qwen3:1.7b`, `qwen3.5:4b`, `deepseek-r1:8b`, and `embed-gemma:300m`.
- `POST http://127.0.0.1:13306/v1/chat/completions` with `model=qwen3:1.7b` returned FLM usage fields:
  - prompt tokens: 24
  - completion tokens: 32
  - prefill speed: 30.8 tok/s
  - decode speed: 42.2 tok/s
- `POST http://127.0.0.1:13306/v1/embeddings` with `model=embed-gemma:300m` returned a 768-dimensional embedding.

## Artifacts

- `/tmp/1bit-npu-bench-qwen06/`
- `/tmp/1bit-npu-bench-qwen06-long/`
- `/tmp/1bit-npu-bench-qwen17/`
- `/tmp/1bit-npu-bench-qwen35-4b/`
- `/tmp/1bit-npu-bench-qwen35-4b-long/`
- `/tmp/1bit-npu-bench-deepseek8b/`

Each directory contains `flm-server.log`, `warmup.json`, `decode.json`, and `strace.txt`.
