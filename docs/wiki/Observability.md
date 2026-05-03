# Observability

Three levels: service logs, kernel-level profiling, and model-quality benchmarking. Everything local; no telemetry leaves the host.

## Logs

```bash
# service logs
journalctl -u lemond -f
journalctl -u flm -f
journalctl -u 1bit-proxy -f --since "1h ago"
journalctl -u open-webui -f --since "1h ago"

# user-scope services
journalctl --user -u 1bit-halo-memory-sync.timer
```

Structured JSON logs with Loki + Grafana on the same host are planned; for now, `journalctl -o json` is the paved road.

## Kernel profiling

```bash
# bandwidth-bound sanity check
rocprof --stats --timestamp on \
  <local HIP benchmark command>

# expect: ternary GEMV at ~92% LPDDR5 peak
# if lower, the tile or packed layout regressed
```

## Model quality — PPL harness

```bash
# wikitext-103 perplexity · post-RoPE-fix reference numbers:
# 1bit-halo-v2: PPL ~12 on wikitext-103, ~1.04 on repetition
<local model-quality harness>
```

## Live benchmark

```bash
# clean-burn reference numbers (2026-04-18):
# 64-token context:   66 tok/s
# 1024-token context: 33 tok/s
<local backend benchmark>
```

## Output conventions

On the reference host, benchmark JSON lands in `/home/bcloud/claude output/`. Other hosts pick their own path; the convention matters for the project's internal tracking only.
