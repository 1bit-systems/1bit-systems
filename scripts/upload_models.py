#!/usr/bin/env python3
"""Upload all .1bp models to HuggingFace under bong-water-water-bong."""

import os, sys, time
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("FATAL: set HF_TOKEN")
    sys.exit(1)

api = HfApi()
MODELS_DIR = "models"

# Model definitions: (local_file, hf_repo_name, display_name, params, arch, license, base_model, description)
# Sorted by size ascending so small models upload first
MODELS = [
    ("SmolLM2-135M.1bp",         "SmolLM2-135M-1BP",          "SmolLM2-135M",          "135M",  "llama",  "apache-2.0",   "HuggingFaceTB/SmolLM2-135M",             "Small efficient LM"),
    ("SmolLM2-360M.1bp",         "SmolLM2-360M-1BP",          "SmolLM2-360M",          "360M",  "llama",  "apache-2.0",   "HuggingFaceTB/SmolLM2-360M",             "Small efficient LM"),
    ("SmolLM2-1.7B.1bp",        "SmolLM2-1.7B-1BP",          "SmolLM2-1.7B",          "1.7B",  "llama",  "apache-2.0",   "HuggingFaceTB/SmolLM2-1.7B",             "Small efficient LM"),
    ("TinyLlama-1.1B.1bp",      "TinyLlama-1.1B-1BP",        "TinyLlama-1.1B",        "1.1B",  "llama",  "apache-2.0",   "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     "Compact llama-based"),
    ("Gemma-2-2B-it.1bp",       "Gemma-2-2B-it-1BP",         "Gemma-2-2B-it",         "2B",    "gemma2", "gemma",        "google/gemma-2-2b-it",                   "Google Gemma 2"),
    ("Qwen2.5-0.5B-Instruct.1bp", "Qwen2.5-0.5B-Instruct-1BP", "Qwen2.5-0.5B-Instruct", "0.5B", "qwen2",  "apache-2.0",   "Qwen/Qwen2.5-0.5B-Instruct",             "Small Qwen instruct"),
    ("Qwen3-VL-2B.1bp",         "Qwen3-VL-2B-1BP",           "Qwen3-VL-2B",           "2B",    "qwen2vl", "apache-2.0",  "Qwen/Qwen3-VL-2B",                       "Vision-language 2B"),
    ("Gemma-3-1B-it.1bp",       "Gemma-3-1B-it-1BP",         "Gemma-3-1B-it",         "1B",    "gemma",  "gemma",        "google/gemma-3-1b-it",                   "Google Gemma 3 1B"),
    ("Gemma-3-4B-it.1bp",       "Gemma-3-4B-it-1BP",         "Gemma-3-4B-it",         "4B",    "gemma",  "gemma",        "google/gemma-3-4b-it",                   "Google Gemma 3 4B"),
    ("Qwen3-VL-4B.1bp",         "Qwen3-VL-4B-1BP",           "Qwen3-VL-4B",           "4B",    "qwen2vl", "apache-2.0",  "Qwen/Qwen3-VL-4B",                       "Vision-language 4B"),
    ("Qwen2.5-3B-Instruct.1bp", "Qwen2.5-3B-Instruct-1BP",   "Qwen2.5-3B-Instruct",   "3B",    "qwen2",  "apache-2.0",   "Qwen/Qwen2.5-3B-Instruct",               "Qwen 2.5 3B instruct"),
    ("Phi-3-mini-4k-instruct.1bp", "Phi-3-mini-4k-instruct-1BP", "Phi-3-mini-4k-instruct", "3.8B", "phi3", "mit",          "microsoft/Phi-3-mini-4k-instruct",        "Phi-3 3.8B instruct"),
    ("Qwen2.5-VL-3B.1bp",       "Qwen2.5-VL-3B-1BP",         "Qwen2.5-VL-3B",         "3B",    "qwen2vl", "apache-2.0",  "Qwen/Qwen2.5-VL-3B-Instruct",            "Vision-language 3B"),
    ("Phi-4-mini-instruct.1bp",  "Phi-4-mini-instruct-1BP",   "Phi-4-mini-instruct",    "3.8B",  "phi3",   "mit",          "microsoft/Phi-4-mini-instruct",           "Phi-4 3.8B instruct"),
    ("Qwen3.5-4B.1bp",          "Qwen3.5-4B-1BP",             "Qwen3.5-4B",            "4B",    "qwen35", "apache-2.0",   "Qwen/Qwen3.5-4B",                        "Qwen 3.5 4B"),
    ("Falcon3-3B-Instruct.1bp", "Falcon3-3B-Instruct-1BP",    "Falcon3-3B-Instruct",   "3B",    "falcon", "apache-2.0",  "tiiuae/Falcon3-3B-Instruct",             "TII Falcon 3 3B"),
    ("Qwen2-VL-7B.1bp",         "Qwen2-VL-7B-1BP",            "Qwen2-VL-7B",           "7B",    "qwen2vl", "apache-2.0",  "Qwen/Qwen2-VL-7B-Instruct",              "Vision-language 7B"),
    ("CodeLlama-7B.1bp",        "CodeLlama-7B-1BP",           "CodeLlama-7B",          "7B",    "llama",  "llama2",       "codellama/CodeLlama-7b-hf",              "Code Llama 7B"),
    ("Llama-2-7B.1bp",          "Llama-2-7B-1BP",             "Llama-2-7B",            "7B",    "llama",  "llama2",       "meta-llama/Llama-2-7b-hf",               "Llama 2 7B"),
    ("Mistral-7B-Instruct-v0.2.1bp", "Mistral-7B-v0.2-1BP",   "Mistral-7B-Instruct-v0.2", "7B", "mistral", "apache-2.0", "mistralai/Mistral-7B-Instruct-v0.2",     "Mistral 7B v0.2"),
    ("Qwen3-8B.1bp",            "Qwen3-8B-1BP",               "Qwen3-8B",              "8B",    "qwen3",  "apache-2.0",   "Qwen/Qwen3-8B",                          "Qwen 3 8B"),
    ("Dolphin3.0-Llama3.1-8B.1bp", "Dolphin3.0-Llama3.1-8B-1BP", "Dolphin3.0-Llama3.1-8B", "8B", "llama", "apache-2.0", "cognitivecomputations/Dolphin3.0-Llama3.1-8B", "Dolphin 3.0 8B"),
    ("Falcon3-7B-Instruct.1bp", "Falcon3-7B-Instruct-1BP",    "Falcon3-7B-Instruct",   "7B",    "falcon", "apache-2.0",  "tiiuae/Falcon3-7B-Instruct",             "TII Falcon 3 7B"),
    ("Llama-3.1-8B-Instruct.1bp", "Llama-3.1-8B-Instruct-1BP", "Llama-3.1-8B-Instruct", "8B", "llama",  "llama3",       "meta-llama/Meta-Llama-3.1-8B-Instruct",    "Llama 3.1 8B Instruct"),
    ("Ministral-8B-Instruct.1bp", "Ministral-8B-Instruct-1BP", "Ministral-8B-Instruct", "8B",  "mistral", "apache-2.0", "mistralai/Ministral-8B-Instruct-2410",    "Ministral 8B"),
    ("Qwen2.5-VL-7B.1bp",       "Qwen2.5-VL-7B-1BP",          "Qwen2.5-VL-7B",         "7B",    "qwen2vl", "apache-2.0",  "Qwen/Qwen2.5-VL-7B-Instruct",            "Vision-language 7B"),
    ("DeepSeek-R1-0528-Qwen3-8B.1bp", "DeepSeek-R1-0528-Qwen3-8B-1BP", "DeepSeek-R1-0528-Qwen3-8B", "8B", "qwen3", "mit", "deepseek-ai/DeepSeek-R1-0528",           "DeepSeek R1 0528 Qwen3"),
    ("DeepSeek-R1-Distill-Qwen-14B.1bp", "DeepSeek-R1-Distill-Qwen-14B-1BP", "DeepSeek-R1-Distill-Qwen-14B", "14B", "qwen2", "mit", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek R1 14B"),
    ("Gemma-3-12B-it.1bp",      "Gemma-3-12B-it-1BP",         "Gemma-3-12B-it",        "12B",   "gemma",  "gemma",        "google/gemma-3-12b-it",                  "Google Gemma 3 12B"),
    ("Llama-2-13B.1bp",         "Llama-2-13B-1BP",            "Llama-2-13B",           "13B",   "llama",  "llama2",       "meta-llama/Llama-2-13b-hf",              "Llama 2 13B"),
    ("OLMo-2-1124-7B-Instruct.1bp", "OLMo-2-1124-7B-Instruct-1BP", "OLMo-2-1124-7B-Instruct", "7B", "olmo", "apache-2.0", "allenai/OLMo-2-1124-7B-Instruct",         "AI2 OLMo 2 7B"),
    ("Mistral-Small-3.1-24B-Instruct.1bp", "Mistral-Small-3.1-24B-1BP", "Mistral-Small-3.1-24B", "24B", "mistral", "apache-2.0", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral Small 24B"),
    ("Qwen3.5-9B.1bp",          "Qwen3.5-9B-1BP",             "Qwen3.5-9B",            "9B",    "qwen35", "apache-2.0",   "Qwen/Qwen3.5-9B",                        "Qwen 3.5 9B"),
    ("Gemma-4-26B-A4B-it.1bp",  "Gemma-4-26B-A4B-it-1BP",    "Gemma-4-26B-A4B-it",    "26B",   "gemma4", "gemma",        "google/gemma-4-26B-A4B-it",              "Google Gemma 4 26B MoE"),
    ("DeepSeek-R1-Distill-Qwen-32B.1bp", "DeepSeek-R1-Distill-Qwen-32B-1BP", "DeepSeek-R1-Distill-Qwen-32B", "32B", "qwen2", "mit", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek R1 32B"),
    ("OLMo-2-0325-32B.1bp",     "OLMo-2-0325-32B-1BP",        "OLMo-2-0325-32B",       "32B",   "olmo",   "apache-2.0",   "allenai/OLMo-2-0325-32B-Instruct",        "AI2 OLMo 2 32B"),
    ("Qwen3.6-35B-A3B.1bp",     "Qwen3.6-35B-A3B-1BP",        "Qwen3.6-35B-A3B",       "35B",   "qwen35moe", "apache-2.0", "Qwen/Qwen3.6-35B-A3B",                   "Qwen 3.6 35B MoE"),
    ("Qwen3-Coder-30B-A3B.1bp", "Qwen3-Coder-30B-A3B-1BP",    "Qwen3-Coder-30B-A3B",   "30B",   "qwen3moe", "apache-2.0", "Qwen/Qwen3-Coder-30B-A3B",               "Qwen Coder 30B MoE"),
    ("DeepSeek-Coder-V2-Lite-Instruct.1bp", "DeepSeek-Coder-V2-Lite-Instruct-1BP", "DeepSeek-Coder-V2-Lite-Instruct", "16B", "deepseek2", "mit", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "DeepSeek Coder V2 Lite"),
    ("Mage-VL-4B.1bp",          "Mage-VL-4B-1BP",             "Mage-VL-4B",            "4B",    "qwen2vl", "apache-2.0",  "Mage-EC/Mage-VL-4B",                     "Mage Vision-Language 4B"),
    ("Mage-ViT.1bp",            "Mage-ViT-1BP",               "Mage-ViT",              "0.3B",  "vit",    "apache-2.0",   "Mage-EC/Mage-VL-4B",                     "Mage Vision Transformer"),
    ("Falcon3-10B-Instruct.1bp", "Falcon3-10B-Instruct-1BP",  "Falcon3-10B-Instruct",  "10B",   "falcon", "apache-2.0",  "tiiuae/Falcon3-10B-Instruct",            "TII Falcon 3 10B"),
]

def make_readme(display_name, params, arch, license_, base_model, description):
    return f"""---
license: {license_}
base_model: {base_model}
tags:
  - 1bp
  - q4nx
  - {arch}
  - npu
---

# {display_name} — 1BP format

{base_model} ({license_}) converted to **1BP** — the single-file model format used by [1bit.systems](https://github.com/bong-water-water-bong/1bit-systems)'s inference engine.

1BP packs everything a loader needs into one memory-mappable file: a 256-byte header with model config, a variable-length tensor index, and Q4NX-tiled (32×256) 4-bit quantized weight data. No Python dependencies, no config files, no tokenizer files — just one file and it works.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | {params} |
| Architecture | {arch} |
| Quantization | Q4NX (4-bit, 32×256 tiles) |
| Source | {base_model} |
| Format | Single-file `.1bp` |

## Usage

```bash
# Download and run with npu_engine_universal
wget https://huggingface.co/bong-water-water-bong/{display_name.replace(" ","-").replace(".","")}-1BP/resolve/main/{display_name.replace(" ","-").replace(".","")}.1bp
./npu_engine_universal {display_name.replace(" ","-").replace(".","")}.1bp 5
```
"""

GITATTR = """*.1bp filter=lfs diff=lfs merge=lfs -text
"""

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")  # cd to repo root

total = len(MODELS)
ok = 0
fail = 0

for i, (local_file, repo_name, display, params, arch, lic, base, desc) in enumerate(MODELS, 1):
    local_path = os.path.join("models", local_file)
    if not os.path.exists(local_path):
        print(f"[{i}/{total}] SKIP {repo_name}: file not found ({local_path})")
        fail += 1
        continue

    file_size = os.path.getsize(local_path)
    print(f"[{i}/{total}] {repo_name} ({file_size/1e9:.1f} GB)... ", end="", flush=True)

    try:
        # 1. Create repo
        try:
            api.create_repo(repo_name, private=False, exist_ok=True)
        except Exception as e:
            print(f"create_repo failed: {e}")
            fail += 1
            continue

        # 2. Upload .gitattributes
        api.upload_file(
            path_or_fileobj=GITATTR.encode(),
            path_in_repo=".gitattributes",
            repo_id=f"bong-water-water-bong/{repo_name}",
            commit_message="Add LFS config",
        )

        # 3. Upload README
        readme = make_readme(display, params, arch, lic, base, desc)
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=f"bong-water-water-bong/{repo_name}",
            commit_message="Add model card",
        )

        # 4. Upload the .1bp file (big file — uses LFS)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=local_file,
            repo_id=f"bong-water-water-bong/{repo_name}",
            commit_message="Add 1BP model weights",
        )

        ok += 1
        print(f"OK ({file_size/1e9:.1f} GB)")
    except Exception as e:
        print(f"FAIL: {e}")
        fail += 1

print(f"\nDone: {ok} uploaded, {fail} failed out of {total}")
