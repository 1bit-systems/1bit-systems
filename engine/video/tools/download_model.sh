#!/bin/bash
# Download Wan2.2 models + LoRA adapters for video engine
set -e

BASE="https://huggingface.co"

download() {
    local repo="$1"
    local file="$2"
    local dest="${3:-$file}"
    
    if [ -f "$dest" ]; then
        echo "  ✓ $dest"
        return
    fi
    
    echo "  ↓ $dest..."
    wget -q --show-progress "$BASE/$repo/resolve/main/$file" -O "$dest" 2>/dev/null && echo "  ✓ $dest" || echo "  ✗ $dest (skipped)"
}

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     Wan2.2 Model & LoRA Downloader          ║"
echo "║     1bit.systems Video Engine                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# --- Models ---
echo "── Models ──────────────────────────────────────"

# T2V 1.3B GGUF (QuantStack — public)
download "QuantStack/Wan2.2-T2V-A14B-GGUF" "wan2.2-t2v-a14b-q4_0.gguf"

# I2V 14B GGUF (QuantStack — public)
download "QuantStack/Wan2.2-I2V-A14B-GGUF" "wan2.2-i2v-a14b-q4_0.gguf"

# --- LoRA Adapters ---
echo ""
echo "── LoRA Adapters ───────────────────────────────"
echo "  (skip any that fail — not all repos are public)"
echo ""

# Camera Control LoRAs
echo "  · Camera Control LoRAs:"
download "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera" "camera_pan_left.safetensors"  "lora_camera_left.safetensors"
download "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera" "camera_zoom_in.safetensors"   "lora_zoom_in.safetensors"
download "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera" "camera_dolly_zoom.safetensors" "lora_dolly_zoom.safetensors"

# Reward LoRAs (aligned with human preference)
echo "  · Reward LoRAs:"
download "alibaba-pai/Wan2.2-Fun-Reward-LoRAs" "aesthetic.safetensors"     "lora_aesthetic.safetensors"
download "alibaba-pai/Wan2.2-Fun-Reward-LoRAs" "motion_smooth.safetensors" "lora_motion_smooth.safetensors"

# Distilled/Lightning LoRAs (4-step fast inference)
echo "  · Distilled LoRAs:"
download "lightx2v/Wan2.2-Distill-Loras" "lightx2v_t2v_a14b_4step_lora.safetensors" "lora_4step.safetensors"

# Community LoRAs (FP16 I2V)
echo "  · Community LoRAs:"
download "Rob1221rib/wan22-fp16-i2v-loras" "cinematic.safetensors"  "lora_cinematic.safetensors"
download "Rob1221rib/wan22-fp16-i2v-loras" "anime.safetensors"     "lora_anime.safetensors"

echo ""
echo "── Usage ───────────────────────────────────────"
echo ""
echo "  # Single LoRA"
echo "  ./video_engine -m wan2.2-t2v-a14b-q4_0.gguf \\"
echo '      -p "cinematic cat" --lora lora_cinematic.safetensors'
echo ""
echo "  # Multiple LoRAs (comma-separated)"
echo "  ./video_engine -m wan2.2-t2v-a14b-q4_0.gguf \\"
echo '      -p "cinematic cat" \\'
echo "      --lora lora_cinematic.safetensors,lora_motion_smooth.safetensors \\"
echo "      --lora-scale 0.7,0.3"
echo ""
echo "  # High-noise + low-noise (Wan2.2 MoE)"
echo "  ./video_engine -m wan2.2-t2v-a14b-q4_0.gguf \\"
echo '      -p "cinematic cat" \\'
echo "      --lora lora_cinematic.safetensors \\"
echo "      --lora-high-noise lora_dolly_zoom.safetensors \\"
echo "      --lora-hn-scale 0.6"
echo ""
