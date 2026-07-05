# Roadmap

## ✅ Phase 1: INT8 Inference (complete)

- [x] INT8 K-interleaving fix (dataReuse on ObjectFifo)
- [x] 5 INT8 xclbins built and NPU-verified (QKV, O, GU, D, KV)
- [x] Per-tensor symmetric INT8 quantization
- [x] 8-live contexts with near-linear scaling (7.9× at 8 ctx)
- [x] Pre-loaded per-layer weight BOs
- [x] NPU attention kernel compiled and verified
- [x] **Coherence bug FIXED** — AIE micro-tiling root cause resolved
- [x] 244→10 ms/tok (24× speedup) on Qwen3-0.6B
- [x] Batched prefill, HTTP API, landing page, packaging

## ✅ Phase 2: Multi-Modal Engine (complete)

- [x] Model-agnostic AgnosticPipeline — 22 models auto-detected
- [x] 14 video models (Wan2.2, CogVideoX, HunyuanVideo, LTX, etc.)
- [x] 5 image models (Flux, Flux Schnell, Flux.2, SDXL, SD3.5)
- [x] 3 audio models (Stable Audio Open, AudioLDM2, LongCat-AudioDiT)
- [x] Unified LoRA loading across all backends
- [x] Single CLI: `--model` accepts any HF ID or alias

## 📋 Phase 3: Speculative Decode

- [ ] Draft model selection (KQV-only or 1-layer variant of Qwen3-0.6B)
- [ ] Draft loop: greedy-predict N tokens, queue for batch verify
- [ ] Batched verification: M=N draft tokens through existing INT8 xclbins
- [ ] Token acceptance: compare draft tokens against verified logits, discard mismatches
- [ ] Target: <50 ms/tok effective throughput
- [ ] Reuse DeepSpec draft training pipeline at `/home/bcloud/DeepSpec/`

### How speculative decoding works

```
Current:   token1(244ms) → token2(244ms) → token3(244ms) = 732ms for 3
Spec:      draft 4 tokens → batch verify(300ms) ≈ 75ms/tok effective
```

Reuses existing INT8 GEMMs at M=N. No new xclbins needed.

## ✅ Phase 4: 1-bit / BitNet (pipeline ready)

- [x] Q2_0 → INT8 Q4NX converter (`tools/q2_0_to_q4nx.py`)
- [x] Ternary MLIR generator (`engine/npu/kernel/n1_core_ternary.py`)
- [x] Ternary xclbin build script (`engine/npu/build/build_ternary_xclbin.sh`)
- [x] **279 tok/s GPU ternary** validated (Q2_0, Vulkan)
- [x] GPU→NPU bridge: ternary weights → INT8 → existing xclbin pipeline
- [ ] **Native ternary AIE kernel** — 2-bit packed ternary tile ops (4× density)
- [ ] Native ternary xclbin — replace `mm_32x64x128.o` with `mm_ternary_32x64x128.o`
- [ ] Target: <25 ms/tok on Strix Halo NPU (400+ tok/s est.)

## ✅ Phase 5: Productionization

- [x] HTTP API server (OpenAI-compatible, pure C++ sockets)
- [x] Ollama Modelfile
- [x] OpenAI SDK / LangChain / Open WebUI compatibility
- [x] Docker image
- [x] AUR package (PKGBUILD ready)
- [x] Snap package (built)
- [x] Debian package (built)

## 🔮 Phase 6: NPU Native Ternary Kernel

The last frontier. The Chess C++ kernel for 2-bit packed ternary weights on the AIE array.
Requires:

1. Write `mm_ternary_32x64x128.cpp` with ternary tile ops
2. Compile: `xchesscc mm_ternary_32x64x128.cpp -o mm_ternary_32x64x128.o`
3. Reference in `n1_core_ternary.py` (replace `mm_32x64x128.o`)
4. Build xclbin: `bash engine/npu/build/build_ternary_xclbin.sh`
5. Benchmark: expected 400+ tok/s on Strix Halo NPU

Toolchain ready at `/home/bcloud/torch2aie/toolchain/`. All pipeline pieces in place.
