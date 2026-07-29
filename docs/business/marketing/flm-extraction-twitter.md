# FLM 37-Model Extraction — Twitter/X Thread

Thread: 6 tweets. Link blog post: https://1bit.systems/blog/flm-37-models-extracted

---

1/6
We extracted AMD's entire NPU model zoo from the ROCm/FastFlowLM v0.9.46 mirror.
37 pre-built models. 209 xclbins. Auto-detection from Q4NX headers — zero config files.

2/6
Highlights: Qwen3.5 Omni multi-modal (audio+vision+text C++ source), Qwen3.6-MoE-35B with 256 experts (3B active, 262k context), Gemma4, Phi4, DeepSeek-R1, Whisper. 18 architectures total, 46+ 1BP models.

3/6
How: drop a Q4NX file on our single C++ binary. It reads the header, picks the right xclbin, runs. No Python. No config. No license keys. Works with the mainline amdxdna driver.

4/6
Peak performance: 433 tok/s Q1 GEMV kernel, 79.4 tok/s e2e BlackMamba 1.5B — both validated on real Strix Halo hardware. NPU INT8 GEMM: 0/10000 errors across 22 shapes, 5 models.

5/6
What's broken: 22 xclbin shapes need Peano compilation. Qwen3.5 Omni source extracted but not wired into NPU dispatch yet. No e2e benchmarks on new models. We tell you what doesn't work.

6/6
Every Strix Halo APU (~48M shipped) has a 50 TOPS NPU gathering dust. Our binary makes it usable. MIT licensed.
Blog: https://1bit.systems/blog/flm-37-models-extracted
GitHub: https://github.com/bong-water-water-bong/1bit-systems
Models: https://1bit.systems/wiki/models
