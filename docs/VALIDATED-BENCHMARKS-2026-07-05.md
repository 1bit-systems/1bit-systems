# Validated Benchmarks — July 5, 2026

All numbers below were **measured live on this machine this session**, with
**coherent output confirmed** (not synthetic/throughput-only). Reproduction
commands included. Hardware: AMD Strix Halo — Ryzen AI NPU (XDNA 2) + Radeon
8060S iGPU (RADV STRIX_HALO, 256 GB/s, UMA).

## Headline results

| Engine | Model | Quant | Decode tok/s | Coherent | Notes |
|---|---|---|---|---|---|
| **GPU (Vulkan)** | Ternary-Bonsai-1.7B | **Q2_0 (1.58-bit)** | **274–279** | ✅ | genuine ternary, native 2-bit storage |
| GPU (Vulkan) | Qwen2.5-0.5B | Q4_K | 300–312 | ✅ | |
| GPU (Vulkan) | Qwen2.5-1.5B | Q4_K | 161 | ✅ | |
| GPU (Vulkan) | Ternary-Bonsai-1.7B | F16 | 22.1 | ✅ | same model, F16 fallback |
| **NPU (FLM)** | Qwen3-0.6B | Q4NX | **93.2–93.8** | ✅ | production path, port 52632 |
| NPU (C++ `npu_engine_cb`) | Qwen3-0.6B | INT8 | 4.7 | ❌ repeats token | GEMM verified, full pipeline not yet coherent |

## The 1-bit number (headline for "1bit.systems")

**Ternary-Bonsai-1.7B in native Q2_0 (1.58-bit ternary) = 274–279 tok/s decode,
coherent, on the Radeon 8060S iGPU via Vulkan.**

- The model's weights are genuinely ternary {−1, 0, +1} (1.58 bits of information).
- Stored in native Q2_0 (2-bit) — **not** dequantized to F16/Q4.
- Same model forced through F16 runs at 22.1 tok/s → **native Q2_0 is 12.6× faster.**
- Prefill: ~3500 tok/s.

This is the honest "1-bit" throughput figure. (The earlier 300 tok/s figure is
Qwen2.5-0.5B **Q4** — a real GPU number but 4-bit, not 1-bit; don't market it as
1-bit. The "281 tok/s 1-bit" tagline in CLAUDE.md had no reproducible source —
this 274–279 figure replaces it with a measured, coherent one.)

### Why it's fast
Q2_0 stores 128 weights per 34-byte block (0.266 bytes/weight vs 2.0 for F16 =
7.5× less weight traffic), and the RADV STRIX_HALO path uses integer dot-product
(`KHR_coopmat`, `int dot: 1`) for the ternary matmul rather than being purely
bandwidth-bound.

## Reproduction

### GPU native 1.58-bit (llama.cpp Vulkan, PrismML-Eng fork)
```bash
git clone --depth 1 -b pr/q2_0-vulkan https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp && cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-bench llama-cli
# model: prism-ml/Ternary-Bonsai-1.7B-gguf → Ternary-Bonsai-1.7B-Q2_0_g64.gguf
./build/bin/llama-bench -m Ternary-Bonsai-1.7B-Q2_0_g64.gguf -ngl 99 -p 64 -n 64
#  → tg64 = 278.81 ± 2.95 t/s   pp64 = 3518.75 t/s
./build/bin/llama-cli  -m Ternary-Bonsai-1.7B-Q2_0_g64.gguf -ngl 99 -p "The capital of France is" -n 40 -st </dev/null
#  → "...is **Paris**. It is a major city in the Île-de-France region..."  273.8 t/s
```

Note the **g64** variant (QK2_0=64) matches this llama.cpp build. The plain
`Ternary-Bonsai-1.7B-Q2_0.gguf` uses QK2_0=128 (34 bytes/block) — see
`tools/q2_0_decode.py` for the bit-exact 128-block decoder (verified cos=1.0 vs
F16). ZINC support requires a Q2_0 (128/34) DMMV path distinct from its existing
`stq1_0` (256/42) kernel — the two collide on GGML type id 42.

### GPU Q4 (ZINC / Vulkan)
```bash
zinc -m qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "..." --chat -n 150   # 300–312 t/s
```

### NPU (FLM production)
```bash
# streaming decode-rate measurement against flm serve (port 52632), 180-token gen
#  → 93.2–93.8 tok/s, coherent
```

## Honest caveats (unchanged from this session's audit)
- **The `engine/fusion/` "fused engine" does not run inference** — its `main.zig`
  prints a dispatch table and exits. NPU (FLM) and GPU (ZINC) run as separate
  processes; there is no live fused serving path.
- The C++ `npu_engine_cb` INT8 engine still repeats a token (not coherent) despite
  a bit-exact GEMM kernel; kept out of the headline table's "shippable" set.
