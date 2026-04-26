# Why our own `.h1b` format?

**One-line answer**: GGUF was built for LLaMA-family FP/Q4/Q8 shapes and force-fits ternary weights into metadata that the kernel has to un-fit at load time. `.h1b` ("halo 1-bit") is pre-tiled for one kernel layout, memory-mapped, zero-copy. The runtime never mallocs a weight tensor.

## What goes in the file

```text
magic      "H1B\0"
version    i32  (1..=4)
config     9 × i32   (hidden, intermediate, layers, heads, kv_heads,
                      vocab, max_seq, tie_emb, reserved)
extras     rope_theta, rms_norm_eps (v>=2)
per-layer  norms (f32) + 7 ternary tensors (q,k,v,o,gate,up,down)
           packed bytes + per-row f32 scale
tokenizer  blob (vocab + BPE merges)
rope       pre-computed sin/cos tables
```

Everything is little-endian, laid out in the order the HIP kernel reads it. The loader is:

```rust
let mmap = memmap2::Mmap::map(&file)?;
let header = parse_header(&mmap[..44])?;
for layer in 0..header.layers {
    // offsets computed from header; slices point into mmap, no copy
    layer_views.push(slice_layer(&mmap, layer, &header));
}
```

No allocator on the hot path. The OS page cache is our weight cache. Code: [`cpp/core/src/h1b.rs`](../../cpp/core/src/h1b.rs).

## Why not GGUF directly

- **GGUF's Q2_K / Q4_K blocks assume K-major tiling**; our ternary GEMV wants row-major with per-row scales. Converting at load time means the kernel waits on CPU-side reshape.
- **Metadata overhead** — GGUF stores per-tensor dtype, dims, name as strings. We have 7 tensors per layer × 30 layers. The string keys add up to hundreds of KB the kernel doesn't need.
- **No slot for Sherry packing** — GGUF's type enum has no ternary-with-3:4-sparsity entry. We'd have to piggyback on a reserved code, forever breaking every GGUF reader that sees our file.
- **Tooling lock-in** — llama.cpp owns the spec. Adding a format extension means a PR upstream, which means months.

## Size

`BitNet-b1.58-2B-4T` as `.h1b` v3: **1.8 GB** on disk. Same weights as FP16 safetensors at 4.2 GB. Sherry 1.25-bit packing (v4, WIP) targets 1.4 GB.

## The honest tradeoff

Every format invention is a tax. We pay it because:

1. The **requantizer runs once per model release** on a dev box. It's Python + PyTorch, reads safetensors, writes `.h1b`. Never shipped, never on the serving path — Rule A-safe (see [Why no Python?](./Why-No-Python.md)).
2. The **loader has no malloc** on the hot path. `1bit-server` boots in ~200 ms because weight setup is `mmap` + pointer arithmetic.
3. Kernel tile layouts ship **pre-tiled**. The first token out the door doesn't pay a one-time reshape penalty.

If GGUF ever adds a first-class ternary type with per-row scales and 3:4 sparsity, we'll import it directly. Until then, `.h1b` is the simplest thing that gives the kernel exactly what it wants.

## Pointers

- Reader struct: [`cpp/core/src/h1b.rs`](../../cpp/core/src/h1b.rs)
- Exporter (one-shot Python): `requantize-h1b.py` in the dev-tools folder
- Related: [Why ternary?](./Why-Ternary.md), [Why no Python?](./Why-No-Python.md)
