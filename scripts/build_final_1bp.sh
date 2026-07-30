#!/bin/bash
# build_final_1bp.sh — Build ALL remaining models the binary supports
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"

# MODEL_NAME|GGUF_SOURCE|GGUF_FILE|HF_REPO|TQ2_FLAG
MODELS=(
  "Falcon3-7B-Instruct-1BP|bartowski/Falcon3-7B-Instruct-GGUF|Falcon3-7B-Instruct-Q4_K_M.gguf|Falcon3-7B-Instruct-1BP|"
  "Qwen3-VL-4B-Instruct-1BP|Qwen/Qwen3-VL-4B-Instruct-GGUF|Qwen3VL-4B-Instruct-Q4_K_M.gguf|Qwen3-VL-4B-Instruct-1BP|"
  "Mixtral-8x7B-Instruct-v0.1-1BP|TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF|mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf|Mixtral-8x7B-Instruct-v0.1-1BP|"
  "Nanbeige4.2-3B-1BP|bartowski/Nanbeige_Nanbeige4.2-3B-GGUF|Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf|Nanbeige4.2-3B-1BP|"
  "Bonsai-8B-TQ2-1BP|prism-ml/Ternary-Bonsai-8B-gguf|Ternary-Bonsai-8B-F16.gguf|Bonsai-8B-TQ2-1BP|--tq2"
  "Bonsai-27B-TQ2-1BP|prism-ml/Ternary-Bonsai-27B-gguf|Ternary-Bonsai-27B-F16.gguf|Bonsai-27B-TQ2-1BP|--tq2"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME GGUF_SOURCE GGUF_FILE HF_REPO TQ2_FLAG <<< "$spec"
  
  echo ""
  echo "=============================================="
  echo " Building: $MODEL_NAME"
  echo " Source:   $GGUF_SOURCE / $GGUF_FILE"
  echo " HF Repo:  bong-water-water-bong/$HF_REPO"
  echo " TQ2:      ${TQ2_FLAG:-no}"
  echo "=============================================="
  
  MODELS_DIR="$SCRIPT_DIR/models"
  GGUF_PATH="$MODELS_DIR/$GGUF_FILE"
  OUTPUT_FILE="$MODELS_DIR/$MODEL_NAME.1bp"
  
  # Download
  echo "--- Download ---"
  if [ -f "$GGUF_PATH" ]; then
    echo "Cached: $(du -h "$GGUF_PATH" | cut -f1)"
  else
    python3 -c "
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download(repo_id='$GGUF_SOURCE', filename='$GGUF_FILE',
    local_dir='$MODELS_DIR', local_dir_use_symlinks=False, resume_download=True)
print(f'Downloaded: {os.path.getsize(p)/1e9:.1f} GB')
"
  fi
  
  # Convert
  echo "--- Convert ---"
  if [ -f "$OUTPUT_FILE" ]; then
    echo "Already converted: $(du -h "$OUTPUT_FILE" | cut -f1)"
  else
    START=$(date +%s)
    python3 "$SCRIPT_DIR/tools/gguf_to_onebp.py" "$GGUF_PATH" "$OUTPUT_FILE" $TQ2_FLAG
    echo "Done in $(( $(date +%s) - START ))s"
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
  echo "Disk: $(df -h / | tail -1 | awk '{print $3, "/", $4, "free"}')"
done

echo ""
echo "=============================================="
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
repos = api.list_models(author='bong-water-water-bong')
n = sum(1 for r in repos if '1BP' in r.modelId and 'LoRA' not in r.modelId)
print(f' 🎉 Total 1BP models: {n}')
print(f'==============================================')
"
