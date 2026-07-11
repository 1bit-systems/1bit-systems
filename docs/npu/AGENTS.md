# AGENTS.md — NPU Optimization Session

## Session: 2026-06-29 — Full Optimization Sprint

### Engine: Qwen3-0.6B on Strix Halo XDNA2 NPU

**Final result: 210 ms/tok (4.8 tok/s) — 3.2× faster than baseline**

### Architecture
- **Hardware**: AMD Strix Halo (Ryzen AI MAX+ 395), XDNA2 NPU, 8 AIE columns
- **Precision**: BFP16 (v8bfp16ebs8) — hardware-native block floating point
- **Engine**: Custom C++ (XRT) with 6 fused xclbins, threaded CPU (4× LM head, 4× attention)
- **GEMMs**: 112 NPU calls/token (4/layer × 28 layers)

### What Was Built
| Artifact | Status | Location |
|----------|--------|----------|
| BFP16 xclbins (QKV, O, GU, D) | ✅ Running | config1/build/ |
| INT8 xclbins (QKV, O, GU, D) | ✅ Built, DMA hang | build/int8/ |
| Multi-token M=256 xclbins (×4) | ✅ Built | config1/build/ |
| 2-layer batch N=8320 xclbins (×4) | ✅ Built | config1/build/ |

### Key Fixes Applied
| Fix | File |
|-----|------|
| ScalarValue nanobind type mismatch | `extras/dialects/arith.py` |
| Peano ELF symbol rename | `utils/compile/utils.py` |
| transpose.hpp incomplete type | `aie_api/detail/aie2/transpose.hpp` |
| aie2p mm.cc SKIP_VECTORIZED + extern "C" | `aie_kernels/aie2p/mm.cc` |
| MLIR parser i8/i16 type support | `AIEXDialect.cpp`, `AIETargetModel.cpp` |
| linalg.py SKIP_VECTORIZED flag | `iron/kernels/linalg.py` |

### Key Files
- `src/npu_engine_fused.cpp` — Working engine (310 lines)
- `bf16_kernel_dev/` — All investigation artifacts
- `docs/npu/HANDOFF-NPU-OPTIMIZATION.md` — Full handoff
- `docs/npu/INT8-HANDOFF.md` — INT8 deep-dive
- `docs/npu/REDDIT_POST.md` — Community post draft

### Blockers
1. **INT8**: MLIR parser patched, xclbins build, DMA strides need recalibration
2. **BF16**: aiecc generates corrupt DMA descriptors for bfloat16 types
3. **Multi-token**: 2-row xclbins process 50% of K-iterations (aiecc instruction generation bug)

### Next Steps
- Fix INT8 DMA strides in n1_core_i8.py generator
- Community outreach via Reddit (r/StrixHalo, r/AMD, r/LocalLLaMA)
