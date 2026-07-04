# CLAUDE.md — 1bit.systems

**50 TOPS INT8 · 94 tok/s NPU (FLM) · 22 tok/s GPU. On a consumer laptop.**
Contact: admin@1bit.systems

**Three inference engines, one chip, ONE cache.** NPU (C++) + GPU (Zig) + CPU (scheduler).
The H2O KV cache eviction layer is now fused across all three backends:
the `KvPagePool` scheduler, zero-page remapping, and RadixAttention prefix tree
are backend-agnostic and shared by NPU, GPU (Vulkan), and GPU (Metal) inference paths.

## Agent Workflow (skills to invoke automatically)

### On every code change:
1. **`/verify`** — Run `curl -s http://127.0.0.1:9090/v1/chat/completions` and confirm 94±5 tok/s NPU decode via FLM proxy
2. **`/code-review`** — Review diff for INT8 quantization bugs, context lifecycle issues, C++ memory safety

### On every push:
3. **`/deploy`** — Build site, verify links, push to Cloudflare Pages via `cp site/* 1bit-site/`
4. **`/release-branch`** — Open PR with conventional commits, verify coverage, merge

### On every PR:
5. **Qodo PR-Agent** auto-runs (`.pr_agent.toml` + `.github/workflows/pr-agent.yml`)
   - 3 AI reviewers (OpenCode GLM-5.2 → DeepSeek → GPT-4o-mini fallback)
   - Focus: INT8 quantization, NPU context lifecycle, BFP16 precision, C++ memory safety

## Engine: NPU (`engine/npu/`)
C++23 INT8 inference on XDNA 2 NPU. Daemon proxies to FLM (94 tok/s). C++ engine: 28 tok/s all-models, 97 tok/s v12.
- `engine/npu/src/npu_engine_cb.cpp` — Main loop (batched prefill + decode)
- `engine/npu/src/dequant_q4nx.c` — Q4NX dequantizer
- `engine/npu/kernel/edge_attention.cc` — NPU attention (Chess C++)
- `engine/npu/xclbins/n1_core_i8_v2.py` — INT8 MLIR generator

Build: `g++ -std=c++23 -O3 -o npu_engine engine/npu/src/npu_engine_cb.cpp engine/npu/build/dequant_q4nx.o -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl`

## Engine: GPU (`engine/gpu/`)
Zig inference on Vulkan/CUDA/Metal. GGUF native. Compute shaders.

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
- **NPU+GPU+CPU are now fused** via unified H2O KV cache layer (`scheduler/`).
  The KvPagePool, zero-page remapping, RadixAttention prefix tree, and eviction
  policies are backend-agnostic and shared by all three inference paths.
- NPU2 supports 8+ simultaneous hw_contexts (firmware 1.1.2.65)
- INT8 xclbins at `/home/bcloud/npu-sandbox/npu-infer/build/int8/`
- Model at `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx`
- XRT toolchain at `/home/bcloud/torch2aie/toolchain/`
- Chess license at `/home/bcloud/torch2aie/licenses/Xilinx.lic`
- Site deploys from `1bit-site/` via Cloudflare Pages on push to main
- Benchmarks source of truth: `engine/npu/BENCHMARKS.md`
- Audit trail: `docs/journey.md` (1,236 lines, 15 updates)
- Packaging: `packaging/` (deb + snap + tarball + docker + ollama)
- Pre-commit: always run `/verify` before committing engine changes
- PR description: use conventional commits, include ms/tok delta, tag [npu] or [gpu]
- Release: `gh release create` + upload deb/snap/tarball + tag vYYYY.MM.DD

## References
- `/home/bcloud/npu-sandbox/` — NPU experiments
- `/home/bcloud/torch2aie/` — AMD toolchain
- `/home/bcloud/zinc/` — Original GPU engine source
