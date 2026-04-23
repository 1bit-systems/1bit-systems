# 1bit lossless audio compressor — plan

Status: design, 2026-04-23. Pod work queued behind Run 5 + Run 6.

## What it is

A 1.58-bit ternary LLM used as the probability model inside an arithmetic coder over audio tokens. **Lossless by construction** — decoder has the same model + same context, so it inverts the arithmetic code to bit-exact.

## Target numbers

- Model: ~10 M params × 1.58 bpw = **~2 MB on disk**.
- Compression: ~1.3–1.6× better than FLAC on music content (Shannon floor).
- Real-time decode on gfx1151 iGPU via our HIP ternary kernel (no CPU bottleneck unlike NNCP).
- Single `.1bl` container: `[2 MB LLM weights] + [arithmetic-coded residual stream]`.

## Training plan (Run 8 candidate, post Run 6)

### Corpus

License-clean catalog candidates:

| Corpus | License | Size | Notes |
|---|---|---|---|
| **Jamendo Free Music Archive (FMA-medium)** | CC BY / CC BY-SA | 106 GB, 25 k tracks | standard ML benchmark corpus |
| **Kevin MacLeod (incompetech.com)** | CC BY 3.0 / 4.0 | ~12 GB, ~2000 tracks | broad genre coverage |
| **MTG-Jamendo** | CC | 70 GB, 55 k tracks | genre-tagged subset |
| **Free Music Archive full** | varies, CC/PD | 1 TB | widest genre base |
| Public-domain classical (IMSLP WAVs) | PD | ~50 GB | Bach/Mozart/Beethoven canon |
| Bring-your-own-catalog | user's own | n/a | trainer ships; weights don't |

**Default ship**: train on **Jamendo FMA-medium** (known ML benchmark, CC). Gives a generalist compressor that works well on any music.

### Tokenizer

- Byte-level over raw FLAC frames (predicts FLAC bytestream directly, no lossy Mimi layer).
- Or: lossless audio codec tokens (e.g. **BWS** / **WavPack residual coding**) — cleaner probability surface, smaller residual.

Decision: byte-level first (simplest, benchmarkable against gzip / bzip2 / NNCP directly).

### Architecture

- **10 M params** — 6 layers × 256 d_model × 8 heads (BitLinear ternary).
- Context: 4 k bytes lookback.
- 1-epoch warmup, 5-epoch overfit at low LR.
- QAT from scratch (not PTQ — ternary compressors need learned scale factors).

### Cost estimate

- 10 M-param ternary + 4 k ctx on 100 GB corpus.
- 2× H200 NVL DDP @ $6.78/hr.
- Estimated 50 B tokens / ~8 h wall-clock.
- **Total: ~$55**.

Small enough to run after Run 5 (LLM) + Run 6 (TTS) finish. Queue as Run 8.

## Runtime chain

```
encode(flac_bytes)
  → LLM probs over each byte position
  → arithmetic-code the actual bytes using those probs
  → store: [LLM weights ~2 MB] + [coded stream ~FLAC × 0.7]
decode(artifact)
  → load LLM weights
  → arithmetic-decode back to original FLAC bytes
  → byte-exact
```

Implementation lives in **`1bit-core` Rust crate** + HIP kernel dispatch via `1bit-hip`. No new .cpp fork needed — it's an arithmetic coder on top of our existing ternary kernel.

## Product positioning

- **`1bit-ac` CLI** — `1bit-ac compress album.flac → album.1bl`, `1bit-ac decompress album.1bl → album.flac` (byte-identical).
- **Benchmark** page on 1bit.systems showing `.1bl` vs FLAC vs WavPack vs bzip2 on FMA-small.
- **Audiophile pitch**: "Ship your FLAC library 30% smaller. Decode real-time on any gfx1151 box."

## License

- **Code**: MIT (same as rest of halo).
- **Weights**: CC BY 4.0 if trained on CC-BY corpus (Jamendo). Keeps attribution chain clean.
- **Users' own compressed artifacts**: user owns outputs of their own FLAC → `.1bl` encodes; no viral license on content.

## Related

- `project_music_llm_beatles.md` — the lossy 4000× variant (different product)
- `project_bitnet_frontier_2026_04.md` — ternary literature
- `project_hipgraph_opportunity.md` — Geramy's decode-perf advice applies to arithmetic-decode too
- Research digest 2026-04-23 — ternary audio is greenfield lane
