# Hacker News Post Draft

## Title:
**One binary, model-agnostic across NPU + GPU + CPU, zero Python — on AMD Strix Halo**

## Body:

AMD shipped the Strix Halo with a 50 TOPS NPU, a real GPU, and 128 GB of unified
memory. The official stack only lets you use them one at a time, through
separate proprietary or semi-proprietary tools, each tied to its own model
format.

I built a single C++ binary that routes whatever GGUF model you hand it to
whichever backend can actually run it — GPU (ROCm HIP), NPU (via FastFlowLM,
since our own in-process NPU kernels aren't correctness-verified yet — more on
that below), or CPU fallback. No config files, no model registry. Drop in a
model, it works.

### What this is

- **Model-agnostic router**: reads a GGUF/safetensors/Q4NX file's own metadata
  (architecture, quantization, tensor shapes) and picks a backend automatically
- **Real quant format support**: Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/Q8_K/BF16,
  each dequantizer verified bit-exact against the independent `gguf` Python
  reference implementation — not "looks right," actually diffed
- **A CPU reference backend** (Qwen2/Qwen3/Llama/Mistral/Gemma/Phi family,
  including MoE routing and Qwen3's Q/K-norm) verified against an independent
  numpy forward pass, not just "doesn't crash"
- **Zero Python at runtime**: C++23, one binary, no pip/Docker/conda

### The honest part

Our own in-process NPU engine (`engine/npu/`) has a confirmed correctness bug
in its GEMM kernels on real hardware — the numbers looked good, the output
didn't. Rather than ship that as "NPU support," the default NPU path delegates
to FastFlowLM (an external, already-correct subprocess) until our kernel is
actually fixed. That's disclosed in the README, not buried in a footnote.

Real, validated (not synthetic) end-to-end numbers on a Radeon 8060S + XDNA 2
Strix Halo box:

| Path | tok/s | Status |
|------|:-----:|--------|
| GPU ROCm HIP (kernel-level) | 64 | validated |
| NPU via FastFlowLM | 57 | validated |
| GPU Vulkan (ZINC) | 22 | validated |
| zaya_server, Qwen 27B Q4_K, real prompt | 30 | end-to-end |
| zaya_server, Qwen 35B MoE Q4_K, real prompt | 20 | end-to-end |

For comparison, `llama.cpp` on the same ROCm hardware hits 229 tok/s
end-to-end — we're not claiming to beat it, we're claiming honesty about
where we actually stand while we close that gap.

### What's next

- Fix the in-process NPU GEMM kernel correctness bug so we can retire the
  FastFlowLM dependency
- Vision and RAG support (not started)
- Close the gap with llama.cpp's GPU decode throughput

### Links

GitHub: https://github.com/bong-water-water-bong/1bit-systems

MIT licensed. Open issues, including the ones we filed on ourselves when we
found our own numbers were wrong.

---

*"The silicon was never the bottleneck. Getting the numbers right was."*
