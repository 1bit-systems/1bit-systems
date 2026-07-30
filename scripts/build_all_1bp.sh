#!/bin/bash
# build_all_1bp.sh — Build and upload all missing 1BP models
# Processes one at a time to manage disk space

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"

# Define models: MODEL_NAME|GGUF_SOURCE|GGUF_FILE|HF_REPO
MODELS=(
  # Already have GGUF check - verify these exist
  "Phi-3.5-mini-instruct-1BP|bartowski/Phi-3.5-mini-instruct-GGUF|Phi-3.5-mini-instruct-Q4_K_M.gguf|Phi-3.5-mini-instruct-1BP"
  "Ministral-8B-Instruct-2410-1BP|bartowski/Ministral-8B-Instruct-2410-GGUF|Ministral-8B-Instruct-2410-Q4_K_M.gguf|Ministral-8B-Instruct-2410-1BP"
  "Granite-3.2-8B-Instruct-1BP|bartowski/ibm-granite_granite-3.2-8b-instruct-GGUF|ibm-granite_granite-3.2-8b-instruct-Q4_K_M.gguf|Granite-3.2-8B-Instruct-1BP"
  "DeepSeek-R1-Distill-Llama-8B-1BP|bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF|DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf|DeepSeek-R1-Distill-Llama-8B-1BP"
  "Falcon3-10B-Instruct-1BP|bartowski/Falcon3-10B-Instruct-GGUF|Falcon3-10B-Instruct-Q4_K_M.gguf|Falcon3-10B-Instruct-1BP"
  "OLMo-2-1124-13B-Instruct-1BP|bartowski/OLMo-2-1124-13B-Instruct-GGUF|OLMo-2-1124-13B-Instruct-Q4_K_M.gguf|OLMo-2-1124-13B-Instruct-1BP"
  "Qwen2-VL-7B-Instruct-1BP|bartowski/Qwen2-VL-7B-Instruct-GGUF|Qwen2-VL-7B-Instruct-Q4_K_M.gguf|Qwen2-VL-7B-Instruct-1BP"
)

for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME GGUF_SOURCE GGUF_FILE HF_REPO <<< "$model_spec"
  
  echo ""
  echo "=============================================="
  echo " Building: $MODEL_NAME"
  echo " Source:   $GGUF_SOURCE / $GGUF_FILE"
  echo " HF Repo:  bong-water-water-bong/$HF_REPO"
  echo "=============================================="
  
  MODELS_DIR="$SCRIPT_DIR/models"
  GGUF_PATH="$MODELS_DIR/$GGUF_FILE"
  OUTPUT_FILE="$MODELS_DIR/$MODEL_NAME.1bp"
  
  # Step 1: Download GGUF if not already present
  echo ""
  echo "--- Step 1: Downloading GGUF ---"
  if [ -f "$GGUF_PATH" ]; then
    echo "GGUF already exists at $GGUF_PATH ($(du -h "$GGUF_PATH" | cut -f1))"
  else
    echo "Downloading from $GGUF_SOURCE..."
    # Use python for better progress
    python3 << PYEOF
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id="$GGUF_SOURCE",
    filename="$GGUF_FILE",
    local_dir="$MODELS_DIR",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Downloaded: {path} ({os.path.getsize(path)/1e9:.1f} GB)")
PYEOF
  fi
  
  # Step 2: Convert to 1BP
  echo ""
  echo "--- Step 2: Converting to 1BP ---"
  TQ2_FLAG=""
  if echo "$MODEL_NAME" | grep -qi "TQ2\|bonsai\|ternary"; then
    TQ2_FLAG="--tq2"
  fi
  
  if [ -f "$OUTPUT_FILE" ]; then
    echo "1BP already exists at $OUTPUT_FILE ($(du -h "$OUTPUT_FILE" | cut -f1)), skipping conversion"
  else
    echo "Converting $GGUF_FILE → $MODEL_NAME.1bp ${TQ2_FLAG}..."
    
    # The converter needs a lot of memory and time - capture timing
    START_TIME=$(date +%s)
    
    python3 "$SCRIPT_DIR/tools/gguf_to_onebp.py" "$GGUF_PATH" "$OUTPUT_FILE" $TQ2_FLAG
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Conversion completed in ${DURATION}s"
  fi
  
  # Step 3: Upload to HuggingFace
  echo ""
  echo "--- Step 3: Uploading to HF ---"
  # Create repo if needed
  hf repos create "bong-water-water-bong/$HF_REPO" --type model 2>/dev/null || true
  
  # Upload the 1bp file
  FILE_SIZE=$(stat -c%s "$OUTPUT_FILE")
  echo "Uploading $MODEL_NAME.1bp ($(echo "scale=1; $FILE_SIZE / 1e9" | bc -l) GB)..."
  
  python3 << PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="$OUTPUT_FILE",
    path_in_repo="${MODEL_NAME}.1bp",
    repo_id="bong-water-water-bong/$HF_REPO",
    repo_type="model",
    commit_message="Add $MODEL_NAME 1BP model",
)
print("Upload complete!")
PYEOF
  
  # Step 4: Clean up GGUF to save disk space
  echo ""
  echo "--- Step 4: Cleaning up GGUF ---"
  if [ -f "$GGUF_PATH" ]; then
    rm -v "$GGUF_PATH"
    echo "Freed $(du -h "$GGUF_PATH" 2>/dev/null | cut -f1 || echo 'unknown')"
  fi
  
  # Verify the upload
  echo ""
  echo "--- Step 5: Verification ---"
  hf models ls "bong-water-water-bong/$HF_REPO" 2>&1 | grep "\.1bp" | head -1 || echo "⚠️  Warning: .1bp file not found in repo listing"
  
  echo ""
  echo "=============================================="
  echo " ✅ Done: $MODEL_NAME"
  echo "    https://huggingface.co/bong-water-water-bong/$HF_REPO"
  echo "=============================================="
  
  # Show disk usage
  echo ""
  echo "Disk free: $(df -h / | tail -1 | awk '{print $4}')"
done

echo ""
echo "=============================================="
echo " 🎉 All models built and uploaded!"
echo "=============================================="
