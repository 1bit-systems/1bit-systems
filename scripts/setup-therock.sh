#!/usr/bin/env bash
# 1bit.systems — ROCm TheRock C++ SDK Setup
# Installs + configures the TheRock nightly ROCm for Strix Halo (gfx1151)
# Run: sudo bash scripts/setup-therock.sh
set -euo pipefail

ROCK_ROOT="/opt/rocm-therock"
NIGHTLY_INDEX="https://rocm.nightlies.amd.com/whl-multi-arch/"
GPU_TARGET="gfx1151"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  1bit.systems — ROCm TheRock C++ SDK Setup              ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Install packages ──
if [ ! -f "$ROCK_ROOT/bin/hipcc" ]; then
    echo "++ Installing ROCm TheRock SDK to $ROCK_ROOT"
    if [ ! -d "$ROCK_ROOT" ]; then
        python3 -m venv "$ROCK_ROOT"
    fi
    "$ROCK_ROOT/bin/pip" install \
        "rocm[libraries,devel,device-${GPU_TARGET}]" \
        --index-url "$NIGHTLY_INDEX"
    "$ROCK_ROOT/bin/rocm-sdk" init
else
    echo "!! TheRock already installed, updating..."
    "$ROCK_ROOT/bin/pip" install --upgrade \
        "rocm[libraries,devel,device-${GPU_TARGET}]" \
        --index-url "$NIGHTLY_INDEX"
    "$ROCK_ROOT/bin/rocm-sdk" init
fi

# ── Ollama integration ──
if command -v ollama &>/dev/null; then
    echo "++ Configuring Ollama for TheRock..."
    mkdir -p /etc/systemd/system/ollama.service.d/
    cat > /etc/systemd/system/ollama.service.d/override.conf << 'OVERRIDE'
[Service]
# TheRock 7.15.0a has native gfx1151 — no HSA override needed
Environment=HSA_OVERRIDE_GFX_VERSION=
Environment=HSA_ENABLE_SDMA=0
Environment=HIP_VISIBLE_DEVICES=0
Environment=ROCR_VISIBLE_DEVICES=0
Environment=OLLAMA_DEBUG=1
# TheRock runtime paths
Environment=LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib:/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
Environment=ROCM_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel
Environment=HIP_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel
OVERRIDE
    systemctl daemon-reload
    systemctl restart ollama
    echo "!! Ollama restarted with TheRock"
fi

# ── systemd daily update timer ──
echo "++ Installing daily update timer..."
cat > /etc/systemd/system/rocm-therock-update.service << 'SVC'
[Unit]
Description=ROCm TheRock daily update
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/opt/rocm-therock/bin/pip install --upgrade "rocm[libraries,devel,device-gfx1151]" --index-url https://rocm.nightlies.amd.com/whl-multi-arch/
ExecStartPost=/opt/rocm-therock/bin/rocm-sdk init
StandardOutput=journal
User=root
SVC

cat > /etc/systemd/system/rocm-therock-update.timer << 'TMR'
[Unit]
Description=Daily ROCm TheRock update check
[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1h
[Install]
WantedBy=timers.target
TMR

systemctl daemon-reload
systemctl enable --now rocm-therock-update.timer 2>/dev/null || true

# ── Verify ──
echo ""
echo "═══ Verification ═══"
source "$ROCK_ROOT/activate.sh" 2>&1 | head -1
echo "  HIP:     $(hipcc --version 2>&1 | head -1)"
echo "  GPU:     $(rocminfo 2>/dev/null | grep 'Marketing Name' | head -1 | awk -F': *' '{print $2}')"
echo "  ROCm:    $(pip show rocm 2>/dev/null | grep Version)"
echo ""
echo "✅ TheRock C++ SDK ready"
echo "   Activate: source /opt/rocm-therock/activate.sh"
echo "   Update:   systemctl start rocm-therock-update.service"
