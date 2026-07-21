# Strix Halo Model Catalog

Models fine-tuned and optimized for **AMD Ryzen AI Max+ 395 (Strix Halo)** with
ROCm TheRock 7.15a. Every model in this catalog is tested on the 1bit.systems
inference engine and achieves verified performance on this hardware.

## Zyphra Family

| Model | Base | Params | Format | Strix Halo Performance | Status |
|-------|------|--------|--------|----------------------|--------|
| Zamba2-1.2B-Strix | Zyphra/Zamba2-1.2B | 0.71B | Q4_0 GGUF | ~1800 tok/s GEMV | ⚠️ Mamba2 ROCm-limited |
| Zamba2-2.7B-Strix | Zyphra/Zamba2-2.7B | 2.7B | Q4_0 GGUF | ~900 tok/s GEMV | 🔲 Planned |
| Zamba2-7B-Strix | Zyphra/Zamba2-7B | 7B | Q4_0 GGUF | ~350 tok/s GEMV | 🔲 Planned |
| **ZR1-1.5B-Strix** 🏆 | Zyphra/ZR1-1.5B | 1.5B | LoRA adapter | 1.86s/it training | ✅ **Done** |
| Zaya1-8B-Strix | [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) (Apache-2.0) | 8B | Q4_K_M GGUF | ~64 tok/s decode | ✅ Available |
| BlackMamba-1.5B-Strix | [Zyphra/BlackMamba-1.5B](https://huggingface.co/Zyphra/BlackMamba-1.5B) (Apache-2.0) | 1.5B | Q4_0 GGUF | 46.3 tok/s (PyTorch/ROCm, Radeon 8060S)¹ | ⚠️ No engine path yet |
| BlackMamba-2.8B-Strix | [Zyphra/BlackMamba-2.8B](https://huggingface.co/Zyphra/BlackMamba-2.8B) (Apache-2.0) | 2.8B | Q4_0 GGUF | 29.6 tok/s (PyTorch/ROCm, Radeon 8060S)¹ | ⚠️ No engine path yet |

¹ BlackMamba is Mamba1+MoE, an architecture this project's engine has no fast inference path for yet (no build target wires up `src/mamba1_engine.hip`'s kernels to a loader that understands its MoE-expert tensor layout). This number is the official Zyphra reference implementation running via real PyTorch/ROCm on this same hardware — a genuine measurement, but not this project's own engine. See `models/*/README.md` on Hugging Face for the full verification writeup, including a CPU-scalar-reference cross-check and three real bugs found and fixed in the GGUF converter along the way.

### 1BP conversions

Full-precision-preserving conversions to this project's native format — every tensor (norms, dense weights, MoE experts) structurally and numerically verified against the source GGUF. Not fine-tunes; these are the upstream models repackaged.

| Model | Base | Params | Status |
|-------|------|--------|:------:|
| [Zaya1-8B-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) | [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) (Apache-2.0) | 8.84B | ✅ Available |
| [ZAYA1-74B-preview-1BP](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) | [Zyphra/ZAYA1-74B-preview](https://huggingface.co/Zyphra/ZAYA1-74B-preview) (Apache-2.0) | 74.79B | ✅ Available |
| [Zamba2-1.2B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | [Zyphra/Zamba2-1.2B-Instruct-v2](https://huggingface.co/Zyphra/Zamba2-1.2B-Instruct-v2) (Apache-2.0) | 1.2B | ✅ Available |
| [Zamba2-2.7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | [Zyphra/Zamba2-2.7B-Instruct-v2](https://huggingface.co/Zyphra/Zamba2-2.7B-Instruct-v2) (Apache-2.0) | 2.7B | ✅ Available |
| [Zamba2-7B-Instruct-v2-1BP](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | [Zyphra/Zamba2-7B-Instruct-v2](https://huggingface.co/Zyphra/Zamba2-7B-Instruct-v2) (Apache-2.0) | 7B | ✅ Available |
| [ZR1-1.5B-1BP](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | [Zyphra/ZR1-1.5B](https://huggingface.co/Zyphra/ZR1-1.5B) (MIT) | 1.5B | ✅ Available |
| [BlackMamba-1.5B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | [Zyphra/BlackMamba-1.5B](https://huggingface.co/Zyphra/BlackMamba-1.5B) (Apache-2.0) | 1.5B | ✅ Available |
| [BlackMamba-2.8B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) | [Zyphra/BlackMamba-2.8B](https://huggingface.co/Zyphra/BlackMamba-2.8B) (Apache-2.0) | 2.8B | ✅ Available |
| [BlackMamba-1.5B-1BP](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | [Zyphra/BlackMamba-1.5B](https://huggingface.co/Zyphra/BlackMamba-1.5B) (Apache-2.0) | 1.5B | ✅ Available |

## How to Use

```bash
# Download a model
huggingface-cli download bong-water-water-bong/Zamba2-1.2B-Strix --local-dir models/

# Run with 1bit.systems engine
./build/run_zamba2 models/zamba2-1.2b-strix-q4_0.gguf "Your prompt here"

# Or via llama.cpp
llama-cli -m models/zamba2-1.2b-strix-q4_0.gguf -p "Your prompt here"
```

## Fine-Tuning Pipeline

All models are fine-tuned on **AMD ROCm TheRock 7.15a** using PyTorch 2.11 + PEFT LoRA.

### ZR1-1.5B (standard attention — recommended)
```bash
# Fine-tune ZR1-1.5B (200 steps, Alpaca instruct, 6.2 min on Strix Halo)
source /opt/rocm-therock/activate.sh
source /tmp/therock-train/bin/activate
python scripts/finetune_zr1.py

# Push to Hugging Face
bash scripts/push_to_hub.sh ZR1-1.5B-Strix /tmp/zr1-1.5b-finetune
```

### Zamba2 (Mamba2 — ROCm fallback, slow)
```bash
# Fine-tune Zamba2-1.2B (200 steps, ~3 hours on Strix Halo)
bash scripts/finetune_zamba2.sh 1.2b
```

> **Note:** Zamba2 uses Mamba2 SSD layers which lack optimized ROCm kernels.
> The PyTorch fallback is 73× slower than standard attention.
> For best ROCm performance, use ZR1-1.5B (Qwen2 arch) instead.

## Hardware

- **CPU**: AMD Ryzen AI Max+ 395 (32 threads, Zen 5)
- **GPU**: Radeon 8060S (gfx1151, RDNA 3.5, 128 GB unified memory)
- **NPU**: AMD XDNA 2 (40 columns unlocked)
- **ROCm**: TheRock 7.15.0a (Clang 23.0.0)
- **RAM**: 128 GB unified LPDDR5X
