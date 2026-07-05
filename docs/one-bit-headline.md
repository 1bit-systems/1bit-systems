# 1-bit headline — copy for site / README

## The number

**279 tokens/sec, 1.58-bit ternary, on a laptop iGPU — coherent, and you can reproduce it.**

Ternary-Bonsai-1.7B runs at **274–279 tok/s** decode on a Radeon 8060S (Strix Halo)
through ZINC's Vulkan backend, generating real text — *"The capital of France is
**Paris**…"* — with weights kept in native 2-bit storage. The same model upcast to
F16 manages only 22 tok/s, so running the ternary weights natively is a **12.6×**
speedup. No datacenter GPU, no quantization-to-4-bit compromise: 1.58 bits, end to end.

## Reproduce it in three commands

```bash
# 1. build llama.cpp with Vulkan (PrismML-Eng Q2_0 branch)
git clone --depth 1 -b pr/q2_0-vulkan https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j --target llama-bench llama-cli

# 2. get the 1.58-bit model
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B-gguf Ternary-Bonsai-1.7B-Q2_0_g64.gguf --local-dir .

# 3. measure it
./build/bin/llama-bench -m Ternary-Bonsai-1.7B-Q2_0_g64.gguf -ngl 99 -p 64 -n 64
#   → tg64 = 278.81 ± 2.95 t/s   |   pp64 = 3518 t/s
```

## We can read the format, bit-for-bit

The Q2_0 ternary layout isn't publicly documented. We reverse-engineered it from raw
bytes and verified it **bit-exact against the F16 reference (cosine = 1.000000)** across
tensors from every layer — attention, FFN, embeddings. Decoder: `tools/q2_0_decode.py`.

## The full validated tier (all measured on-device, coherent output)

| Backend | Model | Precision | Decode |
|---|---|---|---|
| GPU (Vulkan, Radeon 8060S) | Ternary-Bonsai-1.7B | **1.58-bit** | **279 tok/s** |
| GPU (Vulkan, Radeon 8060S) | Qwen2.5-0.5B | 4-bit | 300 tok/s |
| NPU (XDNA 2, via FLM) | Qwen3-0.6B | INT4/NX | 94 tok/s |

*Numbers measured July 5, 2026 on AMD Strix Halo. See `docs/VALIDATED-BENCHMARKS-2026-07-05.md`.*
