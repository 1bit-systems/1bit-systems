# Benchmarks

Dated results, raw data, and the recording tool for [`site/benchmarks.json`](../site/benchmarks.json)
— the repo's single authoritative benchmark data file (`docs/wiki/performance.md`,
`README.md`, and `site/index.html` all read from it; see [`docs/wiki/performance.md`](../docs/wiki/performance.md)
for the human-readable breakdown).

## Recording a result

```bash
bash benchmarks/record.sh <engine_key> <tok_s> [tflops] [status] [label] [table] [display_name] [backend]
```

Appends/replaces the entry for `<engine_key>` in `site/benchmarks.json` and updates the
`updated` and `commit` fields. Creates the file if absent. Used by
[`tests/download_and_run.sh`](../tests/download_and_run.sh) for smoke-test recording, and
runnable standalone after any manual benchmark run.

## Layout

- `RESULTS-*.md` — dated write-ups of individual benchmark passes
- `data/` — raw benchmark output
- `bench-*.sh` — individual benchmark drivers (NPU ioctl budget, 1-bit pile, PPL sweeps, etc.)
- `bonsai/` — Bonsai-specific benchmark assets
- `latest.json` — most recent raw run output
