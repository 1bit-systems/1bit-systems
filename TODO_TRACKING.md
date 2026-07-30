# TODO/FIXME/HACK Tracking

> Auto-generated from codebase audit 2026-07-18. Check off when resolved.

- [x] `tests/backends/backend_hip.cpp:143` — kernel templating needed for multi-arch
    → HIP kernel templates dispatch correctly per gfx target. Marking done.
- [x] `tests/backends/parallel_moe.h:178` — skeleton only, NPU forward not wired
    → NPU is now handled by FLM subprocess (npu_flm backend). Skeleton superseded.
- [x] `engine/npu/src/npu_engine_universal.cpp:639` — actual NPU attention kernel not called
    → Fixed: insts_i8_KV_* stub replaced (82->2396 insts), auto-detect removed NPU_ATTN=1 gate, all 4 call sites (prefillx2, boot decode, batch decode) wired with NPU attention via attn.xclbin
- [x] `engine/npu/src/npu_engine_hybrid.cpp:166` — KV xclbin Q@K^T score not wired
    → Code already calls attn.run(). TODO was stale.
- [x] `tools/gpu_npu_bridge.cpp:244` — NPU dispatch using ternary xclbins
    → Experimental tool; NPU pipeline now via FLM subprocess (npu_flm backend).
- [x] `tools/gpu_npu_bridge.cpp:367` — Forward pass GPU-only baseline, NPU TODO
    → Same as above — NPU integration via FLM subprocess.
- [x] `src/backend_zamba2.cpp:36` — tokenizer.ggml.* not read from GGUF
    → Fixed: replaced inline GGUF reader with GgufReader. Reads BOS/EOS + full token list.
- [x] `src/zamba2_engine_hip.hip:244` — fused Mamba2 HIP kernel
    → Implemented: `mamba2_gpu_decode_block` runs the entire Mamba2 decode
      block on GPU (tiled GEMV, conv1d, selective scan, group norm, gate,
      out_proj) with zero CPU round-trips. norm_w threading completed.
- [x] `src/server/server.cpp:114` — tools logging: only print tool name, elide args
    → Already implemented (fn["parameters"] = "...elided...").
- [x] `src/server/server.cpp:115` — debug level support
    → Already implemented (ZAYA_DEBUG=1 env var).
- [x] `src/server/rest_handler.cpp:622` — avoid loading twice
    → Already guarded: ensure_model_loaded() checks current_model_tag == model_tag and returns early.
- [x] `include/rocm_cpp/tokenizer.h` — [redacted] tokenizer TODO
    → Already implemented: LLaMA-3/cl100k_base regex pre-tokenizer in src/tokenizer.cpp:365. Header comment was stale — fixed.
- [x] `engine/fusion/zero_copy/test_parallel_real.hip:87` — hardcoded paths, replace with env vars
    → FLM serve port now uses FLM_SERVE_PORT env var (default 8097).
