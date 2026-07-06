# CLAUDE.md — 1bit.systems

**572 tok/s DSpark spec-decode (production) · 291 tok/s NPU fused (production) · 97 tok/s NPU v12 (fallback) · 94 tok/s NPU FLM (fallback) · 113 tok/s ROCm · 28 tok/s C++ all-5 · 22 tok/s GPU · 279 tok/s ternary · 38 KB binary · 15W. On a consumer laptop.**
Contact: admin@1bit.systems

**73+ models across 6 backends · 22 multi-modal (video, image, audio) · 55.7 TFLOPS INT8 GEMM · 72× speedup (244→3.4 ms/tok)**

**Three inference engines, one chip, ONE cache, ONE serving path.**
NPU (C++/Zig XRT) + GPU (ROCm AMD clang++ / Zig Vulkan/CUDA/Metal) + CPU (scheduler).
The H2O KV cache eviction layer and the **FusedEngine** (`engine/fusion/`) unify
all three inference paths into one shared serving infrastructure.

## ROCm Custom Kernel Backend (`ggml-rocm`)
Custom HIP kernels for 1-bit/ternary inference on AMD Strix Halo (gfx1151).
Folded into the `zaya-llama.cpp/` fork as `ggml/src/ggml-rocm/`.
- ✅ **Zaya model architecture** (hybrid CCA + MoE with EDA router)
- ✅ **100 exported C API symbols** — ternary GEMV, Bonsai Q1/TQ2, Sherry,
  KV cache attention (prefill/decode/FD/I8/PQ3), prefill GEMM (28.4 TFlops),
  Medusa tree attention, model loader, tokenizer
- ✅ **Bonsai-1.7B TQ2**: 113 tok/s decode (8.8 ms/tok)
- ✅ **TheRock 7.12** ROCm toolchain via nightly pip (ROCm 7.14.0a)
- ✅ Built standalone (`1bit/build-rocm/librocm_cpp.so`) or as ggml backend

## Fused Engine (`engine/fusion/`)
NPU+GPU hybrid inference engine. Dispatches per-layer or per-operation to NPU
(XRT xclbin INT8 GEMM) or GPU (Vulkan flash attention) through a single API.
- `engine.zig` — Unified `FusedEngine` wrapping NPU + GPU backends
- `dispatcher.zig` — Layer-level dispatch policy (8 policies: auto, npu_only,
  gpu_only, attention_on_npu, ffn_on_npu, qkv_on_npu, layer_by_layer,
  prefill_npu_decode_gpu)
- `memory.zig` — Cross-backend memory sharing (dma-buf or staging copy)
- `interop.zig` — NPU↔GPU KV cache bridge (sync NPU BO↔GPU buffers)
- `server.zig` — Unified HTTP server (OpenAI-compatible API)
- `main.zig` — CLI entry point

Build: `cd engine/fusion && zig build -Doptimize=ReleaseFast`

### Dispatch policies (`--policy <name>`):
| Policy | Attention | FFN | QKV | Use case |
|--------|-----------|-----|-----|----------|
| auto | GPU | NPU | NPU | Best throughput (GPU flash attn + NPU INT8 GEMM) |
| npu_only | NPU | NPU | NPU | NPU-only (for models without GPU flash attn kernels) |
| gpu_only | GPU | GPU | GPU | GPU-only (fallback when NPU xclbins unavailable) |
| attention_on_npu | NPU | GPU | GPU | NPU edge_attention kernel + GPU DMMV |
| ffn_on_npu | GPU | NPU | GPU | GPU flash attn + NPU INT8 GEMM (FFN-heavy models) |
| prefill_npu_decode_gpu | NPU(prefill) | GPU(decode) | NPU(prefill) | Fast prefill on NPU, batch decode on GPU|

## Agent Workflow (skills to invoke automatically)

### On every code change:
1. **`/verify`** — Run `curl -s http://127.0.0.1:9090/v1/chat/completions -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'` and confirm daemon responds (fused layer at 291 tok/s, falls back to C++ v12 at 97 tok/s, then FLM at 94 tok/s)
2. **`/code-review`** — Review diff for INT8 quantization bugs, context lifecycle issues, C++ memory safety

### On every push:
3. **`/deploy`** — Build site, verify links, push to Cloudflare Pages via `cp site/* 1bit-site/`
4. **`/release-branch`** — Open PR with conventional commits, verify coverage, merge

### On every PR:
5. **Qodo PR-Agent** auto-runs (`.pr_agent.toml` + `.github/workflows/pr-agent.yml`)
   - 3 AI reviewers (OpenCode GLM-5.2 → DeepSeek → GPT-4o-mini fallback)
   - Focus: INT8 quantization, NPU context lifecycle, BFP16 precision, C++ memory safety

## Engine: NPU (`engine/npu/`)
Fused layer engine: 291 tok/s production (Qwen3-0.6B, 3.4 ms/tok). One xclbin call per transformer layer (QKV→attention→O→GU→SiLU→D on NPU, no CPU attention). Engine binary: 38 KB. Fallback: C++ v12 at 97 tok/s — auto-converts GGUF→Q4NX — pass any .gguf file directly. Second fallback: FLM proxy (94 tok/s, 10.6 ms/tok) via AMD's proprietary runtime. C++ ALL: 28 tok/s, auto-detects 5 models.
- `engine/npu/src/npu_engine_cb.cpp` — Main loop (batched prefill + decode)
- `engine/npu/src/dequant_q4nx.c` — Q4NX dequantizer
- `engine/npu/kernel/edge_attention.cc` — NPU attention (Chess C++)
- `engine/npu/xclbins/n1_core_i8_v2.py` — INT8 MLIR generator

Build: `g++ -std=c++23 -O3 -o npu_engine engine/npu/src/npu_engine_cb.cpp engine/npu/build/dequant_q4nx.o -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl`

## Engine: GPU (`engine/gpu/`)
Zig inference on Vulkan/CUDA/Metal. GGUF native. Compute shaders.

## NPU+GPU Fusion Status
- ✅ Unified KV cache scheduler (KvPagePool, H2O eviction, RadixAttention, zero-page)
- ✅ FusedEngine interface (`engine/fusion/`) — unified NPU+GPU inference
- ✅ Dispatcher (`engine/fusion/dispatcher.zig`) — 8 dispatch policies
- ✅ Cross-backend memory (`engine/fusion/memory.zig`) — dma-buf + staging fallback
- ✅ NPU↔GPU interop (`engine/fusion/interop.zig`) — KV cache sync bridge
- ✅ Unified HTTP server (`engine/fusion/server.zig`) — OpenAI-compatible API
- ⬜ Fused prefix tree — RadixAttention shared across NPU+GPU paths
- ⬜ Dynamic policy switching — Runtime policy change via API
- ⬜ Auto-tuning — Benchmark each operation and choose fastest backend


## Unified KV Cache Layer (`engine/gpu/src/scheduler/`)
Backend-agnostic KV cache infrastructure shared across NPU, GPU, and CPU paths:
- `kv_cache.zig` — H2O eviction: cumulative attention scoring, min-heap eviction, zero-page technique
  (remaps evicted pages to a reserved zero-filled page — ~0 attention naturally, no shader changes)
  Three policies: h2o_attention_score, lru, fifo. Toggle via `ZINC_KV_EVICTION_POLICY` env var.
- `radix_tree.zig` — RadixAttention prefix tree for cross-request KV page sharing
- `offload_engine.zig` — CPU memory offloading for cold KV pages
- `quant_profile.zig` — Per-layer dynamic quantization scheme selection (8 schemes)
- `request.zig` + `scheduler.zig` — Request lifecycle and continuous batching with page allocation

## Key facts for agents
- **NPU+GPU are now fused** via `engine/fusion/`. The `FusedEngine` wraps NPU
  (XRT xclbin INT8 GEMM) and GPU (Vulkan flash attention/DMMV) behind a single
  API with per-layer dispatch. The unified H2O KV cache layer (`scheduler/`)
  is shared by all three inference paths.
- **Fused layer engine (291 tok/s) is production engine**, C++ v12 (97 tok/s) is fallback,
  FLM proxy (94 tok/s) is secondary fallback — narrative updated July 6, 2026. All badges/site reflect this.
- **Zaya model architecture supported** in `zaya-llama.cpp/` fork.
  Zaya is a hybrid CCA-attention + MoE architecture with EDA router.
  Custom ROCm kernels (ternary GEMV, Bonsai, Sherry) folded into
  `ggml/src/ggml-rocm/` as the ggml-rocm backend (100 exported symbols).
- NPU2 supports 8+ simultaneous hw_contexts (firmware 1.1.2.65)
- **Engine evolution**: 244→3.4 ms/tok in 6 days (72×). Timeline: v7 BFP16 (1930 ms) → i8 swap (244 ms) → v6 batch-4 (50 ms) → v9 M=16 (16 ms) → v12 M=32 (10 ms) → fused layer M=32 (3.4 ms, 291 tok/s). All 5 models auto-detected.
- INT8 xclbins at `/home/bcloud/npu-sandbox/npu-infer/build/int8/`
- Model at `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx`
- XRT toolchain at `/home/bcloud/torch2aie/toolchain/`
- Chess license at `/home/bcloud/torch2aie/licenses/Xilinx.lic`
- Site deploys from `1bit-site/` via Cloudflare Pages on push to main
- Benchmarks source of truth: `docs/wiki/performance.md` (July 6, 2026 refresh — complete with 1-bit model tables, all engine head-to-head, ROCm, Eagle3, H2O, Wave32)
- Engine bench deep-dive: `engine/npu/BENCHMARKS.md`
- Audit trail: `docs/journey.md` (1,236+ lines, 15+ updates)
- **Binary size**: 38 KB (fused engine) — all stale "120KB" and "74KB" references removed from codebase 
- Packaging: `packaging/` (deb + snap + tarball + docker + ollama)
- Pre-commit: always run `/verify` before committing engine changes
- **No Python in runtime paths** — CI enforces: `engine/npu/src/`, `engine/gpu/src/`, `engine/video/src/`, `spec-decode/engine/`, `spec-decode/draft/` must stay `.py`-free
- **⚠️ Never use `git add -A` or `git add .`** — `.local/share/containers/` has permission-denied dirs that block the entire add. Always add specific files: `git add <file1> <file2>`.
- PR description: use conventional commits, include ms/tok delta, tag [npu] or [gpu]
- Release: `gh release create` + upload deb/snap/tarball + tag vYYYY.MM.DD
- ROCm toolchain: TheRock 7.12 nightly pip at `/tmp/rocm-venv/` or system ROCm at `/opt/rocm

### Analytics skill
- Load `github-traffic` skill or simply ask "show analytics" to get the full GitHub traffic report
- `traffic-report.sh` fetches clones, views, top pages, referrers, trends, site health, and Cloudflare data
- Shields.io badges auto-update daily via `traffic-badge.yml` workflow
- Daily clone count displayed on site hero panel + benchmarks stats

### Web search auto-approve
- Always pass `workflow: "none"` or `workflow: "auto-summary"` to `web_search` to skip the interactive browser curator
- This avoids blocking on manual approval for routine searches (social media monitoring, fact-checking)
- Do NOT use `workflow: "summary-review"` (default) unless the user is actively reviewing results

## GPU 1-Bit Model Benchmarks (Radeon 8060S)

All models at ≤1.5625 bpw. Measured via Vulkan, 3 repetitions.

| Model | BPW | Size | Engine | Decode |
|-------|-----|------|--------|--------|
| Qwen2 0.5B | 1.06 (IQ1_S) | 296 MB | llama.cpp | **381 tok/s** |
| Qwen3.5-0.8B | 1.25 (Q1_0) | 268 MB | llama.cpp | **312 tok/s** |
| Hy-MT2 1.8B | 1.3125 (STQ1_0) | 441 MB | ZINC (Sherry) | **267 tok/s** |
| gemma-2-2b | 1.06 (IQ1_S) | 788 MB | llama.cpp | **158 tok/s** |
| gemma3 4B | 1.06 (IQ1_S) | 1.05 GB | llama.cpp | **122 tok/s** |
| Nemo 8B | 1.06 (IQ1_S) | 1.97 GB | llama.cpp | **79 tok/s** |
| Qwen3.5-9B | 1.25 (Q1_0) | 1.82 GB | llama.cpp | **70 tok/s** |

**Full benchmark source**: `docs/wiki/performance.md` (live-verified July 6, 2026)

## References
- `/home/bcloud/npu-sandbox/` — NPU experiments
- `/home/bcloud/torch2aie/` — AMD toolchain
- `/home/bcloud/zinc/` — Original GPU engine source
- `/home/bcloud/engine/fusion/` — NPU+GPU fused engine
- `/home/bcloud/npu-gpu-cpu/` — Shared memory (dma-buf/GTT) experiments + unified daemon
- `/home/bcloud/zaya-llama.cpp/ggml/src/ggml-rocm/` — ROCm kernel backend
- `/home/bcloud/1bit/` — Standalone kernel project
- `/home/bcloud/1bit/build-rocm/` — TheRock 7.12 build
