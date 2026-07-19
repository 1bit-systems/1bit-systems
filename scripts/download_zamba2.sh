#!/bin/bash
# download_zamba2.sh — Download Zamba2 GGUF models for 1bit.systems
#
# Downloads existing GGUF quantizations from HuggingFace.
# Currently supports:
#   - Zamba2-2.7B-Instruct-v2 (Q4_0, Q8_0) from EchoLabs33
#   - ZR1-1.5B (various quants) from bartowski
#
# Usage:
#   ./scripts/download_zamba2.sh [model] [quant]
#
# Models:
#   zamba2-2.7b   — Zamba2 2.7B Instruct v2 (default)
#   zr1-1.5b      — ZR1 1.5B (Qwen2 arch, works on 1bit today)
#   zamba2-1.2b   — Zamba2 1.2B Instruct v2
#   zamba2-7b     — Zamba2 7B Instruct v2
#
# Quants: q4_0, q8_0 (zamba2) or IQ2_M..Q5_K_M (ZR1)

set -euo pipefail

MODEL="${1:-zamba2-2.7b}"
QUANT="${2:-q4_0}"
OUTDIR="${3:-models}"

mkdir -p "$OUTDIR"

case "$MODEL" in
    zamba2-2.7b)
        HF_REPO="EchoLabs33/Zamba2-2.7B-Instruct-v2-GGUF"
        HF_FILE="zamba2-2.7b-instruct-v2-${QUANT}.gguf"
        ;;
    zamba2-1.2b)
        HF_REPO="EchoLabs33/Zamba2-1.2B-Instruct-v2-GGUF"
        HF_FILE="zamba2-1.2b-instruct-v2-${QUANT}.gguf"
        ;;
    zamba2-7b)
        HF_REPO="EchoLabs33/Zamba2-7B-Instruct-v2-GGUF"
        HF_FILE="zamba2-7b-instruct-v2-${QUANT}.gguf"
        ;;
    zr1-1.5b)
        HF_REPO="bartowski/Zyphra_ZR1-1.5B-GGUF"
        HF_FILE="Zyphra_ZR1-1.5B-${QUANT}.gguf"
        if [ "$QUANT" = "q4_0" ]; then
            HF_FILE="Zyphra_ZR1-1.5B-Q4_K_M.gguf"
        fi
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: zamba2-2.7b, zamba2-1.2b, zamba2-7b, zr1-1.5b"
        exit 1
        ;;
esac

OUTPUT="${OUTDIR}/${HF_FILE}"

if [ -f "$OUTPUT" ]; then
    echo "✅ Already downloaded: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
    exit 0
fi

echo "⬇️  Downloading $HF_REPO/$HF_FILE ..."
echo "   → $OUTPUT"

# Use huggingface-cli if available, else curl
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$OUTDIR" --local-dir-use-symlinks False
else
    URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"
    echo "   URL: $URL"
    if command -v curl &>/dev/null; then
        curl --fail -L -o "$OUTPUT" "$URL"
    elif command -v wget &>/dev/null; then
        wget -O "$OUTPUT" "$URL"
    else
        echo "❌ Need curl or wget"
        exit 1
    fi
fi

if [ -f "$OUTPUT" ]; then
    echo "✅ Downloaded: $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
    echo ""
    echo "To run:"
    echo "  ./build/run_zamba2 $OUTPUT \"Your prompt here\""
    echo ""
    echo "Or via llama.cpp:"
    echo "  llama-cli -m $OUTPUT -p \"Your prompt here\""
else
    echo "❌ Download failed"
    exit 1
fi
