# Show HN: We extracted AMD's entire NPU model zoo — 37 models, 209 xclbins, zero config

https://1bit.systems/blog/flm-37-models-extracted

AMD shipped 48M Strix Halo APUs this year — every one has a 32-tile XDNA 2 NPU sitting idle because the proprietary FastFlowLM stack requires a specific driver and license chain most users never set up.

We reverse-engineered that stack in 4 days, then pulled every pre-built model from ROCm/FastFlowLM v0.9.46: **37 models, 209 compiled NPU xclbins**, including multi-modal Qwen3.5 Omni (C++ source for audio + vision + text) and Qwen3.6-MoE-35B with 256 experts.

**How it works:** drop a Q4NX file onto our single C++ binary and it auto-detects the architecture from the header, picks the right xclbin, and runs. No config files. No Python at runtime. MIT.

**What's in the box:**
- 37 FLM models across Qwen3/3.5/3.6, Gemma3/4, Phi4, Llama3, DeepSeek-R1, GPT-OSS, Whisper, and more — now mapped with auto-detection
- 209 xclbins, 11 engine build variants
- Qwen3.6-MoE-35B: 256 experts, 3B active, 262k context — three Peano-compiled INT8 xclbins
- 18 model architectures total across GPU + NPU + CPU backends
- 433 tok/s peak kernel throughput (Q1 GEMV, ROCm HIP), 79.4 tok/s e2e (BlackMamba 1.5B)

**What's still broken (honesty):**
- 22 xclbin shapes still need compilation (scripts are checked in, need MLIR-AIE/Peano toolchain)
- Qwen3.5 Omni C++ source is extracted but not wired into our NPU dispatch yet
- No e2e benchmarks for the new models — xclbins are built, configs are in, token loop not yet run

**Why this matters:** every Strix Halo laptop has a 50 TOPS NPU. Our binary works with the mainline `amdxdna` driver. Extract the xclbins once, they run forever.

GitHub: https://github.com/bong-water-water-bong/1bit-systems
Full models page: https://1bit.systems/wiki/models
