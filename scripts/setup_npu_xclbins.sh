#!/usr/bin/env bash
# setup_npu_xclbins.sh — Symlink FLM xclbins into engine/npu/xclbins/
#
# The NPU engine (engine/npu/src/npu_engine_universal.cpp) expects xclbin
# and instruction files at engine/npu/xclbins/ with specific filenames.
# This script links them from an existing FLM build directory.
#
# Usage:
#   ./scripts/setup_npu_xclbins.sh [--flm-dir /path/to/fastflowlm-build]
#
# Default FLM directory: /home/bcloud/fastflowlm-build (Strix Halo dev box)
# Override with --flm-dir or NPU_XCLBIN_DIR env var (without model suffix).
#
# After running, set NPU_XCLBIN_DIR=engine/npu/xclbins when running the engine.

set -euo pipefail

FLM_DIR="${NPU_XCLBIN_DIR:-$HOME/fastflowlm-build}"
MODEL_TAG="${NPU_MODEL_TAG:-qwen3-0.6b}"
MODEL_DIR="${FLM_DIR}/src/xclbins/Qwen3-0.6B-NPU2"

# Parse --flm-dir
while [[ $# -gt 0 ]]; do
    case "$1" in
        --flm-dir) FLM_DIR="$2"; MODEL_DIR="${FLM_DIR}/src/xclbins/Qwen3-0.6B-NPU2"; shift 2 ;;
        --model-tag) MODEL_TAG="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

TARGET_DIR="$(cd "$(dirname "$0")/../engine/npu/xclbins" && pwd)"
mkdir -p "$TARGET_DIR"

echo "NPU xclbin setup"
echo "  FLM dir:    $FLM_DIR"
echo "  Model dir:  $MODEL_DIR"
echo "  Model tag:  $MODEL_TAG"
echo "  Target:     $TARGET_DIR"
echo ""

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Set --flm-dir or NPU_XCLBIN_DIR to point at your fastflowlm-build checkout."
    exit 1
fi

# Map FLM xclbin names to engine-expected names
declare -A XCLBIN_MAP=(
    ["attn.xclbin"]="final_i8_QKV_${MODEL_TAG}.xclbin"
    ["mm.xclbin"]="final_i8_GU_${MODEL_TAG}.xclbin"
    ["layer.xclbin"]="final_i8_D_${MODEL_TAG}.xclbin"
    ["dequant.xclbin"]="final_i8_O_${MODEL_TAG}.xclbin"
    ["dequant.xclbin"]="final_i8_U_${MODEL_TAG}.xclbin"
)

# Instruction files from packaging dir
INSTS_DIR="$(cd "$(dirname "$0")/../packaging/appimage/AppDir/usr/lib/1bit/xclbins" && pwd)"

echo "Creating symlinks..."
for flm_name in attn mm layer dequant; do
    flm_path="${MODEL_DIR}/${flm_name}.xclbin"
    if [ -f "$flm_path" ]; then
        # Find all engine names that map to this FLM name
        for engine_name in "${!XCLBIN_MAP[@]}"; do
            if [ "${XCLBIN_MAP[$engine_name]}" = "final_i8_${flm_name}_${MODEL_TAG}.xclbin" ] ||
               [ "$engine_name" = "$flm_name" ]; then
                target="${TARGET_DIR}/final_i8_$(echo $flm_name | tr 'a-z' 'A-Z')_${MODEL_TAG}.xclbin"
                ln -sf "$flm_path" "$target"
                echo "  $target -> $flm_path"
            fi
        done
    else
        echo "  WARNING: $flm_path not found"
    fi
done

# Handle dequant → O + U mapping
if [ -f "${MODEL_DIR}/dequant.xclbin" ]; then
    for suffix in O U; do
        target="${TARGET_DIR}/final_i8_${suffix}_${MODEL_TAG}.xclbin"
        ln -sf "${MODEL_DIR}/dequant.xclbin" "$target"
        echo "  $target -> ${MODEL_DIR}/dequant.xclbin"
    done
fi

# Instruction files
if [ -d "$INSTS_DIR" ]; then
    for inst in "$INSTS_DIR"/insts_i8_*.txt; do
        base=$(basename "$inst")
        target="${TARGET_DIR}/${base}"
        ln -sf "$inst" "$target"
        echo "  $target -> $inst"
    done
else
    echo "  WARNING: instruction files not found at $INSTS_DIR"
fi

echo ""
echo "Done. Set NPU_XCLBIN_DIR=$TARGET_DIR when running the engine."
echo "Or run: export NPU_XCLBIN_DIR=$TARGET_DIR"
