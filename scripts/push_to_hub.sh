#!/bin/bash
# push_to_hub.sh — Push a fine-tuned model to Hugging Face
# Usage: ./scripts/push_to_hub.sh <model_name> <model_dir> [hf_token]

set -euo pipefail

MODEL_NAME="${1:-Zamba2-1.2B-Strix}"
MODEL_DIR="${2:-/tmp/zamba2-1.2b-finetune}"
HF_TOKEN="${3:-${HF_TOKEN:-}}"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: No HF_TOKEN set. Provide as arg 3 or set HF_TOKEN env var."
    exit 1
fi

echo "Pushing $MODEL_NAME from $MODEL_DIR to Hugging Face..."
echo "Repo: bong-water-water-bong/$MODEL_NAME"

# Login
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Create model card
cat > /tmp/model_card.md <<'EOF'
---
language: en
license: apache-2.0
library_name: transformers
tags:
- zamba2
- zyphra
- strix-halo
- amd
- rocm
- fine-tuned
datasets:
- yahma/alpaca-cleaned
---

# $MODEL_NAME

Fine-tuned from **Zyphra/Zamba2-1.2B** on **AMD Ryzen AI Max+ 395 (Strix Halo)**
using ROCm TheRock 7.15a and LoRA (attention projections only).

## Performance on Strix Halo

| Metric | Value |
|--------|-------|
| Hardware | AMD Ryzen AI Max+ 395 (Radeon 8060S) |
| GPU Memory | 128 GB unified |
| Format | LoRA adapter (PyTorch) |
| Base Model | Zyphra/Zamba2-1.2B |

## Training Details

- **Dataset**: yahma/alpaca-cleaned (instruction following)
- **LoRA Rank**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Mamba2 Layers**: frozen (not trained)
- **Steps**: 200
- **Hardware**: AMD Strix Halo (128 GB unified memory)
- **Framework**: ROCm TheRock 7.15a + PyTorch 2.11

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Zyphra/Zamba2-1.2B", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "bong-water-water-bong/$MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B")

prompt = "### Instruction:\\nWrite a poem about AMD ROCm.\\n\\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## About 1bit.systems

This model is part of the [1bit.systems](https://github.com/bong-water-water-bong/1bit-systems)
model catalog for AMD Strix Halo — an open-source inference engine optimized
for AMD Ryzen AI Max+ processors.
EOF

# The heredoc above is quoted (<<'EOF') so the markdown code-fence backticks
# are written literally instead of being run as command substitution. Inject
# the dynamic model name afterward.
sed -i "s|\$MODEL_NAME|$MODEL_NAME|g" /tmp/model_card.md

# Upload
huggingface-cli upload "bong-water-water-bong/$MODEL_NAME" "$MODEL_DIR" \
    --repo-type model \
    --commit-message "Initial upload: $MODEL_NAME fine-tuned on Strix Halo"

# Upload model card
huggingface-cli upload "bong-water-water-bong/$MODEL_NAME" /tmp/model_card.md \
    --repo-type model \
    --path-in-repo README.md \
    --commit-message "Add model card"

echo "✅ Uploaded to https://huggingface.co/bong-water-water-bong/$MODEL_NAME"
