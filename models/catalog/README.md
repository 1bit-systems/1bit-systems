# 1bit.systems Model Catalog — 46 Models (1BP)

All models available in **1BP format** — single-file, zero-config, memory-mappable.
Converted via C++ toolchain (`tools/gguf_to_onebp.cpp`), zero Python at runtime.

> **Data-correctness fix (2026-07-26, issue #1023)**: every `.1bp` file produced before
> this date had two bugs — every normalization weight was silently dropped, and a
> double-counted file offset made every dequantized weight value garbage regardless of
> model. Both are fixed in the converter now. Re-converted and re-published: Qwen3-0.6B,
> Llama-3.2-1B-Instruct, Qwen3-4B, ZR1-1.5B. The other pre-existing entries below have
> **not** been re-converted yet — treat their published `.1bp` files as still affected
> until re-uploaded. **`ZAYA1-8B`'s 149 MB entry is additionally missing its MoE expert
> weights** (a separate, unfixed gap — issue #1031: the converter never implemented
> 3D/expert-stacked tensors) — the correct file is ~6.6 GB; don't re-convert from GGUF
> until #1031 lands, the currently-published HF version is the complete one.

## Model Families

### Qwen — 4
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Qwen3-0.6B | 0.6B | 356 MB | ZINC / NPU / HIP | qwen3 |
| Qwen3-4B | 4B | 2.2 GB | ZINC / NPU / HIP | qwen3 |
| Qwen3-8B | 8B | 4.1 GB | ZINC / NPU / HIP | qwen3 |
| Qwen2.5-0.5B | 0.5B | 328 MB | ZINC / NPU | qwen2 |

### Llama Family — 4
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Llama-3.2-3B-Instruct | 3B | 1.7 GB | ZINC / NPU / HIP | llama |
| Llama-3.2-1B-Instruct | 1B | 581 MB | ZINC / NPU | llama |
| Llama-3.1-8B-Instruct | 8B | 4.1 GB | ZINC / NPU / HIP | llama |
| TinyLlama-1.1B | 1.1B | 328 MB | ZINC / NPU | qwen2 (compat) |

### Mistral — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Mistral-7B-Instruct-v0.3 | 7B | 4.3 GB | ZINC / NPU / HIP | mistral |
| Mixtral-8x7B-Instruct-v0.1 | 46.7B | 27.8 GB | ZINC / NPU / HIP | mistral (MoE) |
| Ministral-8B-Instruct-2410 | 8B | 4.7 GB | ZINC / NPU / HIP | mistral |

### Gemma — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Gemma-2-2B-it | 2B | 1.2 GB | ZINC / NPU / HIP | gemma2 |
| Gemma-3-4B-it | 4B | 1.9 GB | ZINC / NPU / HIP | gemma |
| Gemma-3-1B-it | 1B | 447 MB | ZINC / NPU | gemma |

### Phi — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Phi-3-mini-4k-instruct | 3.8B | 2.3 GB | ZINC / NPU / HIP | phi3 |
| Phi-4-mini-instruct | 3.8B | 1.9 GB | ZINC / NPU / HIP | phi3 |
| Phi-3.5-mini-instruct | 3.8B | 2.3 GB | ZINC / NPU / HIP | phi3 |

### DeepSeek — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| DeepSeek-R1-Distill-Qwen-7B | 7B | 3.8 GB | ZINC / NPU / HIP | qwen2 |
| ZR1-1.5B | 1.5B | 781 MB | ZINC ✅ (26 tok/s) / NPU | qwen2 (reasoning-tuned) |
| DeepSeek-R1-Distill-Llama-8B | 8B | 4.1 GB | ZINC / NPU / HIP | llama |

### Falcon3 (TII) — 4
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Falcon3-3B-Instruct | 3B | 1.4 GB | ZINC / NPU / HIP | llama |
| Falcon3-7B-Instruct | 7B | 4.0 GB | ZINC / NPU / HIP | llama |
| Falcon3-1B-Instruct | 1B | 675 MB | ZINC / NPU / HIP | llama |
| Falcon3-10B-Instruct | 10B | 5.7 GB | ZINC / NPU / HIP | llama |

### OLMo (AI2) — 2
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| OLMo-2-1124-7B-Instruct | 7B | 3.9 GB | ZINC / NPU / HIP | olmo |
| OLMo-2-1124-13B-Instruct | 13B | 7.6 GB | ZINC / NPU / HIP | olmo |

### Granite (IBM) — 2
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Granite-3.2-2B-Instruct | 2B | 1.5 GB | ZINC / NPU / HIP | granite (gemma) |
| Granite-3.2-8B-Instruct | 8B | 4.8 GB | ZINC / NPU / HIP | granite (gemma) |

### Laguna (MoE, poolside) — 3
| Model | Params | 1BP Size | 1BP TQ2 Size | Backend | Architecture |
|-------|:------:|:--------:|:------------:|---------|:------------:|
| Laguna-S-2.1 | 48×256ex | 73.5 GB | 36.7 GB | ZINC / NPU / HIP | laguna (MoE) |
| Laguna-XS-2.1 | 40×256ex | 20.9 GB | 10.5 GB | ZINC / NPU / HIP | laguna (MoE) |
| Laguna-S-2.1-DFlash (draft) | 6L dense | 665 MB | 665 MB | ZINC / NPU / HIP | dflash |

### Zaya — 2
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| ZAYA1-8B | 8.8B | 149 MB | ZINC | zaya ✅ |
| ZAYA1-74B-preview | 74B | 739 MB | ZINC | zaya |

### Mamba — 2
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| BlackMamba-1.5B | 1.5B | 970 MB | Mamba1 HIP (79.8 tok/s) | mamba |
| BlackMamba-2.8B | 2.8B | 1.8 GB | Mamba1 HIP (46.4 tok/s) | mamba |

### Zamba (Mamba2-Hybrid) — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Zamba2-1.2B-Instruct-v2 | 1.2B | 1.1 GB | ZINC ✅ / NPU | zamba2 (attn every 6th) |
| Zamba2-2.7B-Instruct-v2 | 2.7B | 2.4 GB | ZINC ✅ / NPU | zamba2 |
| Zamba2-7B-Instruct-v2 | 7B | 6.6 GB | ZINC ✅ / NPU | zamba2 |

### Zamba (Mamba1+Attn) — 1
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Zamba-7B-v1 | 7B | 4.3 GB | Mamba1 HIP | zamba |

### Ternary / 1-bit — 4
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Bonsai-1.7B | 1.7B | 841 MB | HIP GPU | qwen3 (ternary) |
| Bonsai-4B | 4B | 2.2 GB | HIP GPU | qwen3 (ternary) |
| Bonsai-8B | 8B | 4.1 GB | HIP GPU | qwen3 (ternary) |
| Bonsai-27B | 27B | 15 GB | HIP GPU | qwen3 (ternary) |

### Vision-Language — 3
| Model | Params | 1BP Size | Backend | Architecture |
|-------|:------:|:--------:|---------|:------------:|
| Qwen2-VL-2B | 2B | 781 MB | ZINC (vision) | qwen2vl ✅ |
| Qwen3-VL-4B | 4B | 2.2 GB | ZINC (vision) | qwen2vl |
| Qwen2-VL-7B-Instruct | 7B | 3.9 GB | ZINC (vision) | qwen2vl |

Vision-Language entries are the text decoder only — the vision tower/mmproj isn't
wired into any inference path in this repo yet (`tools/vision_server.cpp` has a
TODO for it). Text-only inference works the same as any other catalog entry.

## Total: 46 models
All converted via C++ toolchain (`tools/gguf_to_onebp`).

## Conversion Pipeline (C++ only)
```bash
# Build converter
g++ -std=c++17 -O3 -mavx2 -I include -I src \
    tools/gguf_to_onebp.cpp src/gguf_reader.cpp src/gguf_zamba2_loader.cpp \
    -o build/gguf_to_onebp -lpthread

# Convert model
./build/gguf_to_onebp input.gguf output.1bp
```

## Adding a New Model
1. Get GGUF format model file
2. Convert: `./build/gguf_to_onebp model.gguf models/ModelName.1bp`
3. If new architecture: add to `include/rocm_cpp/bitnet_model.h` in `rcpp_arch_from_string()`
4. If new architecture: add to `include/onebp_format.h` in `OnebpArch` enum
5. **Sanity-check the actual dequantized values before trusting the conversion** — a
   `.1bp` file can be structurally valid (opens, right tensor count, right shapes) while
   every value is garbage (see issue #1023). Spot-check a tensor's real values look like
   small floats (roughly -1 to 1), not astronomical numbers.
6. Rebuild: `cmake --build engine/npu/build --target npu_engine_universal`
7. Test: `./engine/npu/build/npu_engine_universal models/ModelName.1bp 5` — NOT
   `zaya_server`/`unified_server`, neither of which read `.1bp` at all (only
   `.gguf`/`.h1b`). `npu_engine_universal` is the only binary in this repo that actually
   loads the 1BP format.

## Batch Conversion

Use `tools/batch_convert.sh` to convert GGUF models to 1BP:

```bash
# Convert all local GGUF files to 1BP
bash tools/batch_convert.sh --all

# Download and convert catalog-missing models from HuggingFace
bash tools/batch_convert.sh --download

# Convert a single file
bash tools/batch_convert.sh path/to/model.gguf
```

The converter auto-detects architecture from GGUF metadata and maps to the
correct handler. Supported: qwen3, qwen2, llama, mistral, gemma, phi, falcon,
starcoder, olmo, granite, command-r, dbrx, jamba, deepseek2/3, zaya.
