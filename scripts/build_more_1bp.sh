#!/bin/bash
# build_more_1bp.sh — Build all remaining 1BP models from the catalog
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"

# Format: MODEL_NAME|GGUF_SOURCE|GGUF_FILE|HF_REPO|TQ2_FLAG
MODELS=(
  "Qwen2.5-0.5B-Instruct-1BP|Qwen/Qwen2.5-0.5B-Instruct-GGUF|qwen2.5-0.5b-instruct-q4_k_m.gguf|Qwen2.5-0.5B-Instruct-1BP|"
  "Qwen3-8B-1BP|Qwen/Qwen3-8B-GGUF|Qwen3-8B-Q4_K_M.gguf|Qwen3-8B-1BP|"
  "Qwen2-VL-2B-Instruct-1BP|second-state/Qwen2-VL-2B-Instruct-GGUF|Qwen2-VL-2B-Instruct-Q4_K_M.gguf|Qwen2-VL-2B-Instruct-1BP|"
  "TinyLlama-1.1B-Chat-1BP|TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF|tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf|TinyLlama-1.1B-Chat-1BP|"
  "Gemma-2-2B-IT-1BP|MaziyarPanahi/gemma-2b-it-GGUF|gemma-2b-it.Q4_K_M.gguf|Gemma-2-2B-IT-1BP|"
  "Phi-3-mini-4k-instruct-1BP|microsoft/Phi-3-mini-4k-instruct-gguf|Phi-3-mini-4k-instruct-q4.gguf|Phi-3-mini-4k-instruct-1BP|"
  "Falcon3-3B-Instruct-1BP|tiiuae/Falcon3-3B-Instruct-GGUF|Falcon3-3B-Instruct-q4_k_m.gguf|Falcon3-3B-Instruct-1BP|"
  "OLMo-2-1124-7B-Instruct-1BP|bartowski/OLMo-2-1124-7B-Instruct-GGUF|OLMo-2-1124-7B-Instruct-Q4_0.gguf|OLMo-2-1124-7B-Instruct-1BP|"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME GGUF_SOURCE GGUF_FILE HF_REPO TQ2_FLAG <<< "$spec"
  
  echo ""
  echo "=============================================="
  echo " Building: $MODEL_NAME"
  echo " Source:   $GGUF_SOURCE / $GGUF_FILE"
  echo " HF Repo:  bong-water-water-bong/$HF_REPO"
  echo "=============================================="
  
  MODELS_DIR="$SCRIPT_DIR/models"
  GGUF_PATH="$MODELS_DIR/$GGUF_FILE"
  OUTPUT_FILE="$MODELS_DIR/$MODEL_NAME.1bp"
  
  # Download GGUF
  echo "--- Download ---"
  if [ -f "$GGUF_PATH" ]; then
    echo "Already cached: $(du -h "$GGUF_PATH" | cut -f1)"
  else
    python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='$GGUF_SOURCE', filename='$GGUF_FILE',
    local_dir='$MODELS_DIR', local_dir_use_symlinks=False, resume_download=True)
import os; print(f'Downloaded: {os.path.getsize(p)/1e9:.1f} GB')
"
  fi
  
  # Convert to 1BP
  echo "--- Conversion ---"
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Already converted: $(du -h "$OUTPUT_FILE" | cut -f1)"
  else
    START=$(date +%s)
    python3 "$SCRIPT_DIR/tools/gguf_to_onebp.py" "$GGUF_PATH" "$OUTPUT_FILE" $TQ2_FLAG
    echo "Converted in $(( $(date +%s) - START ))s"
  fi
  
  # Upload
  echo "--- Upload ---"
  hf repos create "bong-water-water-bong/$HF_REPO" --type model 2>/dev/null || true
  python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj='$OUTPUT_FILE',
    path_in_repo='$MODEL_NAME.1bp',
    repo_id='bong-water-water-bong/$HF_REPO',
    repo_type='model',
    commit_message='Add $MODEL_NAME 1BP model')
print('Upload complete!')
"
  
  # Cleanup GGUF
  echo "--- Cleanup ---"
  rm -v "$GGUF_PATH"
  
  echo "✅ https://huggingface.co/bong-water-water-bong/$HF_REPO"
  echo "Disk free: $(df -h / | tail -1 | awk '{print $4}')"
done

echo ""
echo "=============================================="
echo " 🎉 All done!"
echo "=============================================="
