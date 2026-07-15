#!/usr/bin/env bash
# setup_all_xclbins.sh — Comprehensive xclbin setup for all NPU engine models
#
# Sources xclbins from:
#   1. Pre-built int8 directory (~/npu-sandbox/npu-infer/build/int8/)
#   2. FLM build directory (~/fastflowlm-build/src/xclbins/)
#   3. Packaging instruction files
#
# FLM xclbin → Engine mapping:
#   attn.xclbin     → final_i8_QKV_<tag>.xclbin
#   dequant.xclbin  → final_i8_O_<tag>.xclbin (and final_i8_U_<tag> if split)
#   layer.xclbin    → final_i8_D_<tag>.xclbin
#   mm.xclbin       → final_i8_GU_<tag>.xclbin (fused) or final_i8_G_<tag> (split)
#
# Supports all models defined in npu_dims.h

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/xclbins"
INT8_DIR="${HOME}/npu-sandbox/npu-infer/build/int8"
FLM_DIR="${HOME}/fastflowlm-build"
PACKAGE_INSTS="${SCRIPT_DIR}/../../packaging/appimage/AppDir/usr/lib/1bit/xclbins"
mkdir -p "$TARGET_DIR"

echo "============================================"
echo "NPU Xclbin Setup — All Models"
echo "============================================"
echo "Target: $TARGET_DIR"
echo ""

link_or_copy() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        return 0
    fi
    return 1
}

# FLM xclbin mapping: label → FLM basename
flm_xclbin() {
    local label="$1"
    case "$label" in
        QKV) echo "attn" ;;
        O|U) echo "dequant" ;;
        D)   echo "layer" ;;
        GU|G) echo "mm" ;;
        *)   echo "" ;;
    esac
}

# Try to set up an xclbin for a model from various sources
setup_xclbin() {
    local label="$1" tag="$2" flm_model="$3"
    local dst_name="final_i8_${label}_${tag}.xclbin"
    local dst="${TARGET_DIR}/${dst_name}"
    
    if [ -f "$dst" ]; then
        echo "  ✓ $dst_name (already exists)"
        return 0
    fi
    
    # Source 1: int8 dir with exact naming
    if [ -f "${INT8_DIR}/${dst_name}" ]; then
        link_or_copy "${INT8_DIR}/${dst_name}" "$dst"
        echo "  ✓ $dst_name (int8)"
        return 0
    fi
    
    # Source 2: int8 dir with expanded naming (e.g. llama_3_1_8b)
    if [ "$tag" = "llama" ] && [ -f "${INT8_DIR}/final_i8_${label}_llama_3_1_8b.xclbin" ]; then
        link_or_copy "${INT8_DIR}/final_i8_${label}_llama_3_1_8b.xclbin" "$dst"
        echo "  ✓ $dst_name (int8/llama_3_1_8b)"
        return 0
    fi
    
    # Source 3: int8 dir with _v suffix (legacy fallback)
    if [ -f "${INT8_DIR}/final_i8_${label}_v.xclbin" ]; then
        link_or_copy "${INT8_DIR}/final_i8_${label}_v.xclbin" "$dst"
        echo "  ✓ $dst_name (int8/_v legacy)"
        return 0
    fi
    
    # Source 4: FLM build directory
    local flm_name
    flm_name=$(flm_xclbin "$label")
    if [ -n "$flm_name" ] && [ -d "${FLM_DIR}/src/xclbins/${flm_model}" ]; then
        if [ -f "${FLM_DIR}/src/xclbins/${flm_model}/${flm_name}.xclbin" ]; then
            link_or_copy "${FLM_DIR}/src/xclbins/${flm_model}/${flm_name}.xclbin" "$dst"
            echo "  ✓ $dst_name (FLM ${flm_model})"
            return 0
        fi
    fi
    
    echo "  ✗ MISSING: $dst_name"
    return 1
}

# Set up instruction file
setup_insts() {
    local label="$1" tag="$2"
    local dst_name="insts_i8_${label}_${tag}.txt"
    local dst="${TARGET_DIR}/${dst_name}"
    
    if [ -f "$dst" ]; then
        echo "  ✓ $dst_name (already exists)"
        return 0
    fi
    
    # Source 1: packaging dir
    if [ -f "${PACKAGE_INSTS}/${dst_name}" ]; then
        cp "${PACKAGE_INSTS}/${dst_name}" "$dst"
        echo "  ✓ $dst_name (packaging)"
        return 0
    fi
    
    # Source 2: int8 dir
    if [ -f "${INT8_DIR}/${dst_name}" ]; then
        cp "${INT8_DIR}/${dst_name}" "$dst"
        echo "  ✓ $dst_name (int8)"
        return 0
    fi
    
    # Source 3: int8 dir with expanded naming
    if [ "$tag" = "llama" ] && [ -f "${INT8_DIR}/insts_i8_${label}_llama_3_1_8b.txt" ]; then
        cp "${INT8_DIR}/insts_i8_${label}_llama_3_1_8b.txt" "$dst"
        echo "  ✓ $dst_name (int8/llama_3_1_8b)"
        return 0
    fi
    
    # Source 4: int8 dir with _v suffix
    if [ -f "${INT8_DIR}/insts_i8_${label}_v.txt" ]; then
        cp "${INT8_DIR}/insts_i8_${label}_v.txt" "$dst"
        echo "  ✓ $dst_name (int8/_v legacy)"
        return 0
    fi
    
    echo "  ✗ MISSING: $dst_name"
    return 1
}

# ============================================================
# Setup xclbins + insts for each model
# ============================================================

# Qwen3-0.6B (tag: qwen3_0_6b, GU_FUSED=1 → GU fused)
echo ""
echo "=== Qwen3-0.6B (GU_FUSED=1) ==="
for label in QKV O GU D; do
    setup_xclbin "$label" "qwen3_0_6b" "Qwen3-0.6B-NPU2"
    setup_insts  "$label" "qwen3_0_6b"
done

# Qwen3-8B (tag: qwen3_8b, GU_FUSED=0 → G + U split)
echo ""
echo "=== Qwen3-8B (GU_FUSED=0) ==="
for label in QKV O D; do
    setup_xclbin "$label" "qwen3_8b" "Qwen3-8B-NPU2"
    setup_insts  "$label" "qwen3_8b"
done
for label in G U; do
    setup_xclbin "$label" "qwen3_8b" "Qwen3-8B-NPU2"
    setup_insts  "$label" "qwen3_8b"
done

# Qwen3-VL-4B (tag: qwen3_vl_4b, GU_FUSED=0 → G + U split)
echo ""
echo "=== Qwen3-VL-4B (GU_FUSED=0) ==="
for label in QKV O D; do
    setup_xclbin "$label" "qwen3_vl_4b" "Qwen3-VL-4B-Instruct-NPU2"
    setup_insts  "$label" "qwen3_vl_4b"
done
for label in G U; do
    setup_xclbin "$label" "qwen3_vl_4b" "Qwen3-VL-4B-Instruct-NPU2"
    setup_insts  "$label" "qwen3_vl_4b"
done

# Llama-3.1-8B (tag: llama, GU_FUSED=0 → G + U split)
echo ""
echo "=== Llama-3.1-8B (GU_FUSED=0) ==="
for label in QKV O D; do
    setup_xclbin "$label" "llama" "Llama-3.1-8B-NPU2"
    setup_insts  "$label" "llama"
done
for label in G U; do
    setup_xclbin "$label" "llama" "Llama-3.1-8B-NPU2"
    setup_insts  "$label" "llama"
done

# Gemma4-E2B (tag: gemma4_e2b, GU_FUSED=1 → GU fused)
echo ""
echo "=== Gemma4-E2B (GU_FUSED=1) ==="
for label in QKV O GU D; do
    setup_xclbin "$label" "gemma4_e2b" "Gemma4-E2B-IT-NPU2"
    setup_insts  "$label" "gemma4_e2b"
done

# ============================================================
# Legacy: _v variants for older engine versions
# ============================================================
echo ""
echo "=== Legacy _v variants ==="
for label in QKV O GU D KV; do
    if [ ! -f "${TARGET_DIR}/final_i8_${label}_v.xclbin" ]; then
        if [ -f "${INT8_DIR}/final_i8_${label}_v.xclbin" ]; then
            link_or_copy "${INT8_DIR}/final_i8_${label}_v.xclbin" \
                "${TARGET_DIR}/final_i8_${label}_v.xclbin" && echo "  ✓ final_i8_${label}_v.xclbin (legacy)"
        fi
    fi
done
for label in QKV O GU D KV; do
    if [ ! -f "${TARGET_DIR}/insts_i8_${label}_v.txt" ]; then
        if [ -f "${INT8_DIR}/insts_i8_${label}_v.txt" ]; then
            cp "${INT8_DIR}/insts_i8_${label}_v.txt" \
                "${TARGET_DIR}/insts_i8_${label}_v.txt" && echo "  ✓ insts_i8_${label}_v.txt (legacy)"
        fi
    fi
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "Xclbin summary by model:"
echo "============================================"
for tag in qwen3_0_6b qwen3_8b qwen3_vl_4b llama gemma4_e2b; do
    xcount=$(ls "${TARGET_DIR}"/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
    icount=$(ls "${TARGET_DIR}"/insts_i8_*_${tag}.txt 2>/dev/null | wc -l)
    echo "  $tag: $xcount xclbins, $icount insts files"
done
echo ""
echo "Total: $(ls "${TARGET_DIR}"/*.xclbin 2>/dev/null | wc -l) xclbins, $(ls "${TARGET_DIR}"/*.txt 2>/dev/null | wc -l) insts files"
echo ""
echo "Usage: export NPU_XCLBIN_DIR=${TARGET_DIR}"
