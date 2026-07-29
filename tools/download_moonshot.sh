#!/usr/bin/env bash
# download_moonshot.sh — Download and convert Moonshot AI models to 1BP format
#
# Usage:
#   bash tools/download_moonshot.sh              # download + convert all
#   bash tools/download_moonshot.sh kimi_k3       # Kimi K3 only
#   bash tools/download_moonshot.sh moonlight     # Moonlight-16B-A3B only
#   bash tools/download_moonshot.sh kimi_vl       # Kimi-VL-A3B-Thinking only
#
# Requires: huggingface-cli, python3, numpy
# Models are downloaded to models/moonshot/ and converted to models/*.1bp

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$REPO_DIR/models"
MOONSHOT_DIR="$MODELS_DIR/moonshot"
mkdir -p "$MOONSHOT_DIR"

# ─── Config ─────────────────────────────────────────────────────────────────
KIMI_K3_REPO="moonshotai/Kimi-K3"
MOONLIGHT_REPO="moonshotai/Moonlight-16B-A3B"
KIMI_VL_REPO="moonshotai/Kimi-VL-A3B-Thinking"
KIMI_VL_2506_REPO="moonshotai/Kimi-VL-A3B-Thinking-2506"
MOONLIGHT_INSTRUCT_REPO="moonshotai/Moonlight-16B-A3B-Instruct"

# ══════════════════════════════════════════════════════════════════════════
# Model Definitions
# ══════════════════════════════════════════════════════════════════════════
# Format: "repo|arch|quant|description"
#   repo: HuggingFace repo name
#   arch: architecture string for 1BP header
#   quant: quantization type for 1BP (Q4NX, TQ2, F16)
#   description: human-readable name

MODELS=(
    "$KIMI_K3_REPO|kimi_k3|Q4NX|Kimi K3 2.8T (too large for Strix Halo — reference conversion only)"
    "$MOONLIGHT_REPO|moonlight|Q4NX|Moonlight-16B-A3B — 16B total, 3B active MoE"
    "$MOONLIGHT_INSTRUCT_REPO|moonlight|Q4NX|Moonlight-16B-A3B-Instruct — instruction-tuned"
    "$KIMI_VL_REPO|kimi_vl|Q4NX|Kimi-VL-A3B-Thinking — 16B VL MoE, 128K ctx"
    "$KIMI_VL_2506_REPO|kimi_vl|Q4NX|Kimi-VL-A3B-Thinking-2506 — improved VL thinking"
)

# ══════════════════════════════════════════════════════════════════════════
# Prerequisites Check
# ══════════════════════════════════════════════════════════════════════════

check_prereqs() {
    local missing=0
    command -v huggingface-cli >/dev/null 2>&1 || { echo "Missing: huggingface-cli (pip install huggingface-hub)"; missing=1; }
    command -v python3 >/dev/null 2>&1 || { echo "Missing: python3"; missing=1; }
    python3 -c "import numpy" 2>/dev/null || { echo "Missing: numpy (pip install numpy)"; missing=1; }
    if [ $missing -ne 0 ]; then
        echo "Install missing dependencies and retry."
        exit 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# Download Functions
# ══════════════════════════════════════════════════════════════════════════

download_model() {
    local repo="$1"
    local target_dir="$2"
    local model_name="$3"

    if [ -d "$target_dir" ] && [ -f "$target_dir/model.safetensors.index.json" ]; then
        echo "  ✓ Already downloaded: $model_name"
        return 0
    fi

    echo "  ↓ Downloading $model_name ($repo)..."
    echo "    Target: $target_dir"
    echo "    This may take a while for large models."

    # Use huggingface-cli for reliable download
    huggingface-cli download "$repo" \
        --local-dir "$target_dir" \
        --local-dir-use-symlinks False \
        --resume-download \
        2>&1 | tail -5

    # Verify download
    if [ -f "$target_dir/model.safetensors.index.json" ] || ls "$target_dir"/*.safetensors 2>/dev/null | head -1 > /dev/null; then
        echo "  ✓ Download complete: $model_name"
        return 0
    else
        echo "  ✗ Download failed or incomplete: $model_name"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# Conversion
# ══════════════════════════════════════════════════════════════════════════

convert_model() {
    local repo="$1"
    local arch="$2"
    local quant="$3"
    local desc="$4"
    local model_name="${repo##*/}"
    local source_dir="$MOONSHOT_DIR/$model_name"
    local output_file="$MODELS_DIR/${model_name}.1bp"

    if [ -f "$output_file" ]; then
        echo "  ✓ Already converted: $model_name.1bp ($(du -h "$output_file" | cut -f1))"
        return 0
    fi

    echo "  ⚙ Converting $model_name ($desc)..."
    echo "    Architecture: $arch"
    echo "    Quantization: $quant"

    # Use the Python converter with safetensors support
    python3 "$SCRIPT_DIR/hf_to_onebp.py" \
        --input "$source_dir" \
        --output "$output_file" \
        --arch "$arch" \
        --quant "$quant" \
        2>&1 | sed 's/^/    /'

    if [ -f "$output_file" ]; then
        local size
        size=$(du -h "$output_file" | cut -f1)
        echo "  ✓ Converted: $model_name.1bp ($size)"
    else
        echo "  ✗ Conversion failed: $model_name"
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

main() {
    check_prereqs

    local filter="${1:-}"

    echo "═══════════════════════════════════════════════════════════"
    echo "  Moonshot AI → 1BP Model Downloader & Converter"
    echo "═══════════════════════════════════════════════════════════"
    echo ""

    for model_def in "${MODELS[@]}"; do
        IFS='|' read -r repo arch quant desc <<< "$model_def"
        local model_name="${repo##*/}"

        # Apply filter if specified
        if [ -n "$filter" ]; then
            case "$model_name" in
                *"$filter"*) ;;  # match
                *) continue ;;    # skip
            esac
        fi

        echo "─── $model_name ─────────────────────────────────"
        echo "  $desc"

        local target_dir="$MOONSHOT_DIR/$model_name"

        download_model "$repo" "$target_dir" "$model_name" || continue
        convert_model "$repo" "$arch" "$quant" "$desc"

        echo ""
    done

    echo "═══════════════════════════════════════════════════════════"
    echo "  All done!"
    echo ""
    echo "  Models available:"
    ls -lh "$MODELS_DIR"/*.1bp 2>/dev/null || echo "  (no .1bp files generated)"
    echo ""
    echo "  Add models to the catalog by updating models/catalog/README.md"
}

main "$@"
