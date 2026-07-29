#!/bin/bash
# validate_model_file.sh — Validate a model file for integrity
# Checks file existence, zero-byte detection, magic bytes, and known formats.
# Usage: validate_model_file.sh <path-to-model>
#
# Supports: GGUF, 1BP, H1B, Safetensors, PyTorch
#
# Exit codes:
#   0 — validation passed
#   1 — file not found
#   2 — file is 0 bytes
#   3 — unknown/invalid format

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-model>"
    exit 1
fi

MODEL="$1"

# ── File existence ──
if [ ! -f "$MODEL" ]; then
    echo "ERROR: File not found: $MODEL"
    exit 1
fi

# ── Size check ──
if command -v stat &>/dev/null; then
    SIZE=$(stat -c%s "$MODEL" 2>/dev/null || stat -f%z "$MODEL" 2>/dev/null)
else
    SIZE=$(wc -c < "$MODEL" | tr -d ' ')
fi

if [ "$SIZE" -eq 0 ]; then
    echo "ERROR: File is 0 bytes: $MODEL"
    exit 2
fi

# ── Magic bytes ──
# Read first 4 bytes as hex
if command -v xxd &>/dev/null; then
    MAGIC=$(xxd -p -l 4 "$MODEL")
elif command -v od &>/dev/null; then
    MAGIC=$(od -A n -t x1 -N 4 "$MODEL" | tr -d ' \n')
else
    # Fallback: python
    MAGIC=$(python3 -c "print(open('$MODEL','rb').read(4).hex())")
fi

# ── Format detection ──
FORMAT=""
case "$MAGIC" in
    47475546) FORMAT="GGUF         ✅" ;;  # GGUF
    50423100) FORMAT="1BP          ✅" ;;  # 1BP / H1B
    48423200) FORMAT="H1B          ✅" ;;  # H1B alternative
    30486231) FORMAT="1BP (v0)     ✅" ;;  # 1BP big-endian variant
    6c727478) FORMAT="NVIDIA/TRT   ⚠️" ;;  # NVIDIA TensorRT plan
    534c3430) FORMAT="SafeTensors  ✅" ;;  # safetensors
    4d4f444c) FORMAT="MODL (CoreML) ⚠️" ;; # CoreML
    6f727465) FORMAT="ONNX         ⚠️" ;;  # ONNX (protobuf)
    89504e47) FORMAT="PNG          ❌" ;;  # Not a model
    25504446) FORMAT="PDF          ❌" ;;  # Not a model
    *)
        # Check if it could be a PyTorch pickle (starts with pickle protocol)
        FIRST_TWO=$(echo "$MAGIC" | head -c4)
        if [ "$FIRST_TWO" = "800" ]; then
            FORMAT="PyTorch      ✅"
        else
            FORMAT="Unknown      ❌"
        fi
        ;;
esac

# ── Human-readable size ──
if command -v numfmt &>/dev/null; then
    HR_SIZE=$(numfmt --to=iec "$SIZE")
elif command -v ls &>/dev/null; then
    # shellcheck disable=SC2012
    HR_SIZE=$(ls -lh "$MODEL" | awk '{print $5}')
else
    HR_SIZE="$SIZE bytes"
fi

echo "Model: $MODEL"
echo "Size:  $HR_SIZE ($SIZE bytes)"
echo "Magic: 0x$MAGIC"
echo "Format: $FORMAT"

# ── Exit with status ──
if echo "$FORMAT" | grep -q '❌'; then
    exit 3
fi

exit 0
