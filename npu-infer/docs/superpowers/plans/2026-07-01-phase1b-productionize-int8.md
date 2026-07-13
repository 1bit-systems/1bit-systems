# Phase 1b: Productionize INT8 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the proof-of-concept INT8 engine (442 ms/tok, 286-line single-main) into a clean, fast, production-quality engine with multi-token decode and a reusable GEMM runner class.

**Architecture:** Extract `I8Ctx` into a reusable `NpuGemm` class in `include/`, create a proper `NpuInferenceEngine` in `src/engine.cpp` matching `include/engine.h`, add M=256 multi-token decode through a second set of xclbins, and clean up all stale engine variants and generator scripts.

**Tech Stack:** C++17, XRT (xrt_bo, xrt_kernel, xrt_hw_context), Python MLIR generator (n1_core_i8_v2.py), Chess-compiled INT8 kernel (mm_32x64x128.o)

---

## Global Constraints

- No changes to the working INT8 GEMM kernel or MLIR generator logic — only reuse/extract
- All xclbins must remain NPU-verified (100% element-wise correct)
- Engine must compile with: `g++ -std=c++17 -O2 -I../include -I/home/bcloud/torch2aie/examples -lxrt_coreutil -lm`
- Model weights stay in q4nx format (dequantized at init via `dequant_i8_to_float`)
- Qwen3-0.6B inference pipeline unchanged: M=128 xclbins for single-token decode
- Multi-token decode uses M=256 xclbins (2× batch, ~2 tok/s → ~4 tok/s effective)
- engine.cpp/engine.h interface: `CustomNpuEngine` with `init(model_path)` and `generate(input_tokens, num_input, output_tokens, max_output)`

---
