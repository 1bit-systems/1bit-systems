# Engine Status — July 5, 2026 (audited)

## Production: FLM Proxy (94 tok/s) ✅

NPU daemon proxies to FastFlowLM. Coherent output, OpenAI API.

- **Port**: 9090 | **Models**: Qwen3-0.6B (turbo) | **Status**: Production stable

## C++ Universal Engine (17 tok/s) ⚠️ — Output NOT Verified

The auto-detecting 5-model engine compiles and runs, but coherent output is
**not yet confirmed** after the 7-round V12 correctness fix pass. The fused
xclbin reference path is validated (max_abs=0.0078 vs CPU oracle), but the
standalone INT8 xclbins have not been re-tested for coherent output.

- Universal engine: 17 tok/s on Qwen3-0.6B (verified to run, output TBD)
- Fused xclbin engine: 4 tok/s (verified correct output)

## GPU ZINC Engine (22 tok/s) ✅

Vulkan compute shaders. Verified coherent output on Bonsai-1.7B-F16.

## GPU 1-bit (llama.cpp) — 70-381 tok/s ✅

7 models at IQ1_S/Q1_0/STQ1_0 formats on Radeon 8060S. Verified throughput.

---

*Detailed benchmarks: `engine/npu/BENCHMARKS.md`*
