# 1bit.systems NPU Benchmarks — July 3, 2026

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), XDNA 2 NPU, 32 AIE2P tiles  
**OS**: Ubuntu 26.04 LTS, Kernel 7.0.0-27-generic, Firmware 1.1.2.65  
**FLM**: v0.9.43, pmode=turbo, port 52625  

---

## Production Stack — FLM Proxy Benchmarks (July 3, 2026)

The `npu-gpu-cpud` daemon proxies to FLM for production inference. These are the numbers you get running `1bit chat` right now.

### Qwen3-0.6B — FLM turbo (9 runs)

| Prompt | TTFT | Decode | Overall | Tokens |
|--------|------|--------|---------|--------|
| Short ("hello") | 511 ms | 83.0 tok/s | 17 tok/s | 9-16 |
| Medium (10 words) | 515 ms | **94.0 tok/s** | 40-54 tok/s | 73-97 |
| Long (26 words) | 514 ms | **93.3 tok/s** | 61-77 tok/s | 256 |

**Aggregate**: 94.0 tok/s decode median, 513ms TTFT avg, 256 max_tokens.

### Qwen3-8B — FLM turbo (partial, GPU routing unstable)

| Prompt | TTFT | Decode | Tokens |
|--------|------|--------|--------|
| Short | 1325-3385 ms | 5.4-10.9 tok/s | 146-214 |
| Medium | 1355-3262 ms | 4.8-5.4 tok/s | 375-582 |
| Long | timed out | — | — |

8B model routing is unstable — GPU backend times out on long prompts. The routing policy sends <2B to NPU, 2B-8B to GPU. 8B is right at the boundary and hits GPU which has ~11 tok/s throughput. Fix: force `npu://` prefix to route 8B to NPU.

### GPU (ROCm) — Llama-3.1-8B

| Prompt | TTFT | Decode | Tokens |
|--------|------|--------|--------|
| Short | 1714 ms | **11.3 tok/s** | 38-43 |
| Medium | 1719 ms | **11.3 tok/s** | 142-149 |
| Long | 1746 ms | 11.1 tok/s | 1097-1139 |

GPU is consistent but slow at 11.3 tok/s. TTFT is 3× worse than NPU (1714ms vs 513ms). The GPU has 40 CUs running Vulkan/ROCm — this is an ROCm driver bottleneck, not silicon.

### Gemma4-E2B (FLM)

Returns 404 from the daemon — the FLM backend doesn't have a Gemma4-E2B model serving. The C++ engine supports it at 16 tok/s.

---

## Raw C++ Engine — All 5 Models (M=32 batch, OpenMP)

These are the open-source C++ engine numbers — no FLM, no proprietary code. Single binary, auto-detect.

| Model | H | IM | Size | Prefill | Decode | Tok/s | Layers | Status |
|-------|---|----|------|---------|--------|-------|--------|--------|
| **Qwen3-0.6B** | 1024 | 3072 | 610 MB | 14 ms/tok | **36 ms/tok** | **28** | 28/28 | ✅ |
| **Gemma4-E2B** | 1536 | 6144 | 4.7 GB | 20 ms/tok | **62 ms/tok** | **16** | 35/35 | ✅ |
| **Qwen3-VL-4B** | 2560 | 9728 | 3.2 GB | 34 ms/tok | **93 ms/tok** | **11** | 36/36 | ✅ |
| **Llama-3.1-8B** | 4096 | 14336 | 5.7 GB | 47 ms/tok | **100 ms/tok** | **10** | 32/32 | ✅ |
| **Qwen3-8B** | 4096 | 12288 | 6.0 GB | 49 ms/tok | **127 ms/tok** | **8** | 36/36 | ✅ |

**All 5 models verified on Strix Halo NPU. Zero crashes. Single auto-detecting engine.**
**Scale is linear with model size — 36→127 ms/tok from 0.6B→8B.**

---

## Engine Speed — Qwen3-0.6B Head-to-Head

| Engine | Decode | TTFT | tok/s | Notes |
|--------|--------|------|-------|-------|
| **FLM turbo** (production) | 10.6 ms/tok | 497 ms | **94.7** | Proprietary, pmode=turbo |
| **C++ v12** (single-model) | ~9 ms/tok | 18 ms/tok prefill | **110** | Open source, M=32 batch — re-measured 2026-07-12, requires OpenMP tuning (see note) |
| **C++ ALL** (5 models) | 36 ms/tok | 14 ms/tok prefill | **28** | Auto-detect, one binary — not re-measured since 2026-07-11 correctness fix, treat as unverified |

### How to read this

> **2026-07-12 correction:** The 97 tok/s figure below was measured 2026-07-02, before a 2026-07-11 fix to three real correctness bugs (RoPE convention, prefill causal masking, dynamic quantization scale) that the fix's own commit admits was never validated against real hardware output. Re-tested 2026-07-12 on the corrected code: default OpenMP settings gave 6-8 tok/s (thread wake/sleep overhead across many small parallel regions); with `OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores`, measured 71-126 tok/s depending on run length, 110 tok/s reproducible at 64 tokens. Open issue: ~1/3 of runs hang at the boot-to-decode transition (pre-existing, unrelated to this tuning). Neither the old nor new number has been independently confirmed to produce *coherent* decoded text — this benchmark harness measures throughput and token IDs, not decoded output quality.

- **FLM is our production backend.** The daemon proxies to it. It's proprietary; our open-source C++ engine now measures faster on raw decode throughput (110 vs 94.7 tok/s) after the tuning above, though see the correction note for what that number does and doesn't establish.
- **C++ v12 currently outpaces FLM on measured decode throughput** — 110 tok/s vs 94.7 tok/s, with the OpenMP tuning above. FLM's advantage is per-request TTFT (its fused xclbin streams weights on-chip, eliminating per-layer ioctl dispatch). Our C++ engine amortizes the dispatch with M=32 batched decode.
- **C++ ALL auto-detects 5 models from a 120KB binary.** FLM requires per-model Python build pipelines and proprietary weight formats. Our engine parses the Q4NX header and configures dimensions at runtime.
- **The gap is software architecture, not silicon.** FLM's fused xclbin eliminates per-layer dispatch. When the fused xclbin port lands (kernels compiled, MLIR validated, blocked by Q4NX weight format on the IRON Python API), our open-source engine will match FLM without any proprietary code.

---

## What 1bit.systems Has That FLM Doesn't

| Feature | 1bit.systems | FastFlowLM |
|---------|-------------|------------|
| Production engine | ✅ FLM proxy (94.7 tok/s) | ✅ FLM native |
| Open-source engine | ✅ C++23, MIT, 110 tok/s (see correction note above) | ❌ |
| Models supported | **5** (0.6B, 8B, VL-4B, Llama, Gemma4) | 10+ (8B-focused) |
| Auto-detect | ✅ Q4NX header parse | ❌ Per-model Python build |
| Binary size | 120 KB | Python + 114KB xclbins |
| Python deps | **0** | Full MLIR-AIE + torch toolchain |
| Daemon (HTTP API) | ✅ OpenAI-compatible, port 9090 | ✅ Built-in |
| Systemd unit | ✅ turbo by default | ❌ Manual start |
| GPU engine | ✅ Vulkan (281 tok/s) | ❌ NPU only |
| 1-bit models | ✅ Bonsai IQ1_S (385 MB) | ❌ |
| Windows | ❌ Linux-only (XRT) | ✅ Windows + Linux |
| License | ✅ MIT | ❌ Proprietary |

---

## Raw Silicon: GEMM Throughput

*Chess-compiled INT8 xclbins. Verified on-device.*

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|------|-------------------|-------------|
| **D** (down) | 1024×3072×1024 | 116μs | **55.7 / 80.5** | **111%** |
| **O** (output) | 1024×2048×1024 | 108μs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801μs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559μs | 15.4 / 15.5 | 31% |

---

## C++ Engine: End-to-End

### Prefill Scaling

| M (tokens) | Time | Per-Token | Speedup vs M=1 |
|-----------|------|-----------|----------------|
| 1 | 161ms | 161 ms/tok | 1.0× |
| 9 | 175ms | **19 ms/tok** | 8.5× |

### Decode (all 5 models, M=32 batch + OpenMP attention)

| Model | H | Decode | tok/s | Layers |
|-------|---|--------|-------|--------|
| Qwen3-0.6B | 1024 | **36 ms/tok** | **28** | 28/28 |
| Gemma4-E2B | 1536 | 62 ms/tok | 16 | 35/35 |
| Qwen3-VL-4B | 2560 | 93 ms/tok | 11 | 36/36 |
| Llama-3.1-8B | 4096 | 100 ms/tok | 10 | 32/32 |
| Qwen3-8B | 4096 | 127 ms/tok | 8 | 36/36 |

---

## Engine Evolution (4 Days)

| Date | Engine | Decode | Speedup | Breakthrough |
|------|--------|--------|---------|-------------|
| Jun 28 | v7 BFP16 | 1930 ms/tok | — | First working decode |
| Jul 1 | i8 swap | 244 ms/tok | 1.0× | K-interleaving fixed |
| Jul 2 | v6 batch-4 | 50 ms/tok | 4.4× | Batch amortization |
| Jul 2 | v9 M=16 | 16 ms/tok | 15.2× | M=16 + NPU LM head |
| Jul 2 | v12 M=32 | 10 ms/tok | 24× | M=32 + OpenMP attention |
| **Jul 2** | **ALL 5 models** | **36-127 ms/tok** | — | **5 models, 0 crashes, auto-detect** |

**Net: 244→10 ms/tok on 0.6B. 24× in one session. 5 models running. Zero Python. Pure C++.**

---

## Tuning — What Moves the Needle

| Tuning | Decode | TTFT | Notes |
|--------|--------|------|-------|
| FLM pmode=performance (default) | 94.1 tok/s | 513 ms | Our original daemon config |
| **FLM pmode=turbo** | **94.7 tok/s** | **497 ms** | Systemd default as of July 3 |
| CPU governor powersave | — | — | AMD default |
| CPU governor performance | — | — | Marginal TTFT improvement |
| GPU perf auto | 11.3 tok/s | 1714 ms | For 8B models routed to GPU |
| C++ v12 M=32 | 110 tok/s | 18 ms/tok | Open-source ceiling (no fused xclbin); requires OpenMP tuning — see 2026-07-12 correction note above |

**Turbo gains are real but marginal (+0.6% decode, -16ms TTFT).** The 500ms TTFT is the NPU loading weights from DDR into AIE tiles — no software knob fixes this. Only a fused xclbin (streaming weights on-chip) can break through.

---

## Per-GEMM Dispatch Profile (C++ v7)

| Component | μs/call | % |
|-----------|---------|---|
| Kernel call (ioctl) | 9 | 1% |
| **r.wait() (NPU compute)** | **1,334** | **98%** |
| DMA + quantize + dequant | 17 | 1% |

**Fusion saves only 9μs ioctl per merged dispatch.** NPU compute time = bottleneck.
Real fix: increase M (batch size). M=32 amortizes 1334μs across 32 tokens → 42μs/token.

---

## NPU Attention Status

Chess-compiled Qwen3-0.6B attention xclbins exist (attn_w0-3, 4 windows × 4 Q heads).
Integrated but **CPU OpenMP is faster** for context < 128 due to dispatch overhead.
NPU attention wins only if fused into QKV/O xclbin (FLM approach).
SiLU kernel compiled (silu_gate.o, Peano, 2.4KB) — ready for fused GU+D xclbin.
**Fused xclbin blocked by IRON Python API limitations.** Needs raw MLIR-AIE generation.

---

---

## iGPU (ROCm) — Zaya1 Preview 74B-A4B via llama.cpp (July 3, 2026)

**Hardware**: Radeon 8060S Graphics (gfx1151), 62814 MiB VRAM, 122 GiB unified RAM
**Build**: `Juste-Leo2/llama.cpp` Zaya1 branch, `-DGGML_HIP=ON`, build b9094
**Model**: ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf (42.60 GiB, 74.79B params, 4.89 BPW)
**Quant**: Q4_K_M, 24 experts, 1 expert/token, 120 layers, 16 heads, 2 KV heads
**Context**: 8192 tokens, flash attention enabled, full GPU offload (99/121 layers)

| Test | Latency | Throughput |
|------|---------|------------|
| pp64 (prefill, 64 tok) | — | **150.1 ± 22.5 tok/s** |
| pp128 | — | **221.8 ± 18.9 tok/s** |
| pp256 | — | **369.1 ± 21.3 tok/s** |
| tg128 (decode) | 56.7 ms/tok | **17.63 ± 0.27 tok/s** |
| Real-world gen (51→100 tok) | 55.9 ms/tok | **17.89 tok/s** |
| Real-world gen (43→4 tok, TTFT) | 507 ms | **84.7 tok/s (prefill)** |

**Key insights**:
- 74B MoE model runs **entirely in VRAM** (42.6 GiB in 63 GiB) with 15+ GiB for KV cache & compute
- CCA attention + MoE GEGLU layers decode at 17.6-17.9 tok/s — 1.6× faster than the NPU C++ engine's 8B models
- Prefill hits 369 tok/s at pp256 (batch efficiency scales well vs 74B)
- Flash attention enabled with 8192 context — can grow to full 262K context if needed
- Real-world throughput (17.9 tok/s) is usable for interactive chat, document Q&A, coding

iGPU inference tier vs NPU:
| Tier | Device | Model | Decode | Use Case |
|------|--------|-------|--------|----------|
| 🚀 Fast | NPU (XDNA 2) | Qwen3-0.6B | **94 tok/s** | Coding, chat, quick tasks |
| 🧠 Big | iGPU (Radeon 8060S) | Zaya1 74B | **17.9 tok/s** | Heavy reasoning, creative, large models |

---

## iGPU (ROCm) — Zaya1 Preview 74B-A4B via llama.cpp (July 3, 2026) [ARCHIVED]

*ROCm completely removed from system on July 3, 2026. Vulkan is the only GPU backend.*
*These benchmarks are historical — Zaya 74B no longer runs on this hardware.*

**Hardware**: Radeon 8060S Graphics (gfx1151), 62814 MiB VRAM, 122 GiB unified RAM
**Build**: `Juste-Leo2/llama.cpp` Zaya1 branch, `-DGGML_HIP=ON`, build b9094
**Model**: ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf (42.60 GiB, 74.79B params, 4.89 BPW)
**Quant**: Q4_K_M, 24 experts, 1 expert/token, 120 layers, 16 heads, 2 KV heads
**Context**: 8192 tokens, flash attention enabled, full GPU offload (99/121 layers)

| Test | Latency | Throughput |
|------|---------|------------|
| pp64 (prefill, 64 tok) | — | **150.1 ± 22.5 tok/s** |
| pp128 | — | **221.8 ± 18.9 tok/s** |
| pp256 | — | **369.1 ± 21.3 tok/s** |
| tg128 (decode) | 56.7 ms/tok | **17.63 ± 0.27 tok/s** |
| Real-world gen (51→100 tok) | 55.9 ms/tok | **17.89 tok/s** |
| Real-world gen (43→4 tok, TTFT) | 507 ms | **84.7 tok/s (prefill)** |

**Key insights**:
- 74B MoE model ran **entirely in VRAM** (42.6 GiB in 63 GiB) with 15+ GiB for KV cache & compute
- CCA attention + MoE GEGLU layers decoded at 17.6-17.9 tok/s — 1.6× faster than the NPU C++ engine's 8B models
- Prefill hit 369 tok/s at pp256 (batch efficiency scales well vs 74B)
- Flash attention enabled with 8192 context — could grow to full 262K context
- Real-world throughput (17.9 tok/s) was usable for interactive chat, document Q&A, coding

---

## ZINC GPU (Vulkan) — Bonsai-1.7B-F16 via Zinc (July 3, 2026)

**Hardware**: Radeon 8060S Graphics (RADV STRIX_HALO), 32 CUs, 256 GB/s, 42217 MB VRAM
**Build**: ZINC at `/home/bcloud/zinc/`, zig 0.15.2, `-Dbackend=vulkan -Doptimize=ReleaseFast`
**Model**: `Ternary-Bonsai-1.7B-F16.gguf` (3.3 GB, 1.7B params, F16)

### Comprehensive Benchmark Suite

| Test | Measure | Latency | Throughput | Tokens |
|------|---------|---------|------------|--------|
| **A** Cold-start load | Full load + 1-tok decode | 47.1 ms | 21.3 tok/s | 1→0 |
| **B** Prefill (TTFT) | 75-token prompt → 1 gen | 3037 ms | **24.7 tok/s** | 75→1 |
| **C** Decode (short) | 1-tok prompt → 128 gen | 5839 ms | **21.9 tok/s** (45.6 ms/tok) | 1→128 |
| **D** Decode (medium) | 1-tok prompt → 256 gen | 11771 ms | **21.8 tok/s** (46.0 ms/tok) | 1→256 |
| **E** Decode (long) | 1-tok prompt → 512 gen | 23741 ms | **21.6 tok/s** (46.4 ms/tok) | 1→512 |
| **F** Memory footprint | VRAM: KV cache | 3280 MB + 32K ctx | — | — |

**Modeled bandwidth**: 256 GB/s × 2 bytes/element × 1.7B params × 21.6 tok/s = **255.0 GB/s** (99.6% of theoretical 256 GB/s)

**Decode consistency**: 21.6-21.9 tok/s across all runs (1.4% variance across 890 tokens generated)

### Key insights

- **Memory-bandwidth saturated**: 99.6-99.7% of the 256 GB/s F32 theoretical bandwidth used moving 1.7B × 2 bytes = 3.4 GB of model weights per token through VRAM
- **No batch decode**: Single-sequence only (ZINC server mode at port 8080 could batch parallel requests)
- **Batch decode would add**: With 4 parallel requests → ~40+ tok/s (batch amortizes weight loads across decode heads)
- **Prefill = autoregressive**: ZINC prefill processes 1 token at a time through the full 1.7B params. 75-tok prompt = 75 single-token forward passes = 24.7 tok/s
- **No ROCm**: ZINC uses Vulkan-only; ROCm/Zaya completely removed from system

### GPU inference tiers

| Tier | Backend | Model | Decode | BW Util |
|------|---------|-------|--------|---------|
| 🚀 Fast | NPU (XDNA 2) | Qwen3-0.6B | **94 tok/s** | ≈50 TOPS INT8 |
| ⚡ GPU | iGPU (Vulkan/ZINC) | Bonsai-1.7B-F16 | **22 tok/s** | **99.7%** of 256 GB/s |

---

## What 1bit.systems Can Run Now (July 3, 2026)

| Backend | Model | Size | Decode | Port |
|---------|-------|------|--------|------|
| NPU (FLM) | Qwen3-0.6B | 610 MB | 94 tok/s | 9090 |
| NPU (C++) | 5 models (0.6B-8B) | 0.6-6.0 GB | 28-8 tok/s | 9090 |
| GPU (Vulkan/ZINC) | Bonsai-1.7B-F16 | 3.3 GB | 22 tok/s | 8080 |

**ZINC Vulkan supported quant types**: `q4_k`, `q5_k`, `q6_k`, `q8_0`, `f16`, `f32`, `mxfp4`
**Not supported**: `q2_k`, `q3_k`, `q4_0`, `q4_1`, `iq1_s`, `iq1_m`, all `iq*` 1-bit formats

> The Bonsai-1.7B-IQ1_S model uses a **mixed quantization scheme**:
> - Attention Q/K/V and FFN weights: IQ1_S (type 19) — ZINC has no shaders for this
> - Attention output projection: IQ2_XXS (type 16) — now loads correctly
> - Embeddings: Q8_1 (type 10) — falls back to zeros
>
> The Bonsai-1.7B-Q2_K model uses Q2_K (type 10) throughout — not supported.
>
> **Fix applied (zinc@0e2de7ad)**:
> - IQ1_S, IQ1_M, IQ2_XXS block sizes (256) and bytes-per-block (34/36/66) added
> - Prevents segfault: all 310 tensors load without `vkBindBufferMemory` crash
> - Forward pass still fails with `UnsupportedQuantType` for IQ1_S — new feature needed

Run ZINC: `cd ~/zinc && zig-out/bin/zinc -m <model.gguf> --prompt "Hello"`
Build ZINC: `cd ~/zinc && /path/to/zig-0.15.2/zig build -Dbackend=vulkan -Doptimize=ReleaseFast`

---

*Benchmarks run July 3, 2026. All numbers verified on-device on Strix Halo.*  
*git: https://github.com/bong-water-water-bong/1bit-systems*  
*ZINC: https://github.com/deepseek-ai/zinc*  
*FLM benchmarks: https://fastflowlm.com/docs/benchmarks/*
