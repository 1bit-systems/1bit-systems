#!/bin/bash
# build_1bp.sh — Build and upload a 1BP model from a source GGUF Q4_K_M
# Usage: ./scripts/build_1bp.sh <model_name> <gguf_source_repo> <gguf_filename> [hf_repo_name]
#
# Example: ./scripts/build_1bp.sh OLMo-2-13B "bartowski/OLMo-2-1124-13B-Instruct-GGUF" "OLMo-2-1124-13B-Instruct-Q4_K_M.gguf" "OLMo-2-1124-13B-Instruct-1BP"

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> <gguf_source> <gguf_file> [hf_repo]}"
GGUF_SOURCE="${2:?}"
GGUF_FILE="${3:?}"
HF_REPO="${4:-$MODEL_NAME}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
CONVERTER="$SCRIPT_DIR/tools/gguf_to_onebp.py"
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}"

# Determine if this is a TQ2 ternary model
TQ2_FLAG=""
if echo "$MODEL_NAME" | grep -qi "TQ2\|bonsai\|ternary"; then
    TQ2_FLAG="--tq2"
fi

echo "=============================================="
echo " Building: $MODEL_NAME"
echo " Source:   $GGUF_SOURCE / $GGUF_FILE"
echo " HF Repo:  bong-water-water-bong/$HF_REPO"
echo " TQ2:      ${TQ2_FLAG:-no}"
echo "=============================================="

cd "$SCRIPT_DIR"

# Step 1: Download GGUF
echo ""
echo "--- Step 1: Downloading GGUF ---"
GGUF_PATH="$MODELS_DIR/$GGUF_FILE"
if [ -f "$GGUF_PATH" ]; then
    echo "GGUF already exists at $GGUF_PATH, skipping download"
else
    hf download "$GGUF_SOURCE" "$GGUF_FILE" --local-dir "$MODELS_DIR" --token "$HF_TOKEN"
    echo "Downloaded to $GGUF_PATH"
fi

# Step 2: Convert to 1BP
echo ""
echo "--- Step 2: Converting to 1BP ---"
OUTPUT_FILE="$MODELS_DIR/$MODEL_NAME.1bp"
python3 "$CONVERTER" "$GGUF_PATH" "$OUTPUT_FILE" $TQ2_FLAG
echo "Converted to $OUTPUT_FILE"

# Step 3: Upload to HuggingFace
echo ""
echo "--- Step 3: Uploading to HF ---"
# Create repo if it doesn't exist
hf repos create "bong-water-water-bong/$HF_REPO" --type model 2>/dev/null || true

# Upload the 1bp file
hf upload "bong-water-water-bong/$HF_REPO" "$OUTPUT_FILE" . \
    --token "$HF_TOKEN" \
    --commit-message "Add $MODEL_NAME 1BP model"

# Step 4: Clean up GGUF to save space
echo ""
echo "--- Step 4: Cleaning up GGUF ---"
rm -v "$GGUF_PATH"
echo "Cleaned up $GGUF_PATH"

# Keep the 1BP file in models/ for now
echo ""
echo "=============================================="
echo " ✅ Done: $MODEL_NAME"
echo "    https://huggingface.co/bong-water-water-bong/$HF_REPO"
echo "=============================================="
