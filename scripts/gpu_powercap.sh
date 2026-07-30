#!/bin/bash
# gpu_powercap.sh — Set iGPU SCLK cap to prevent thermal throttling on Strix Halo.
# Reference: https://github.com/hogeheer499-commits/strix-halo-guide/issues/24
#
# The Radeon 8060S iGPU silently heat-soaks above 2500 MHz under sustained load.
# Cap at 2400 MHz for reliable, consistent benchmarks.
#
# Usage: sudo ./scripts/gpu_powercap.sh [cap_mhz] [--install]
#   cap_mhz:  SCLK ceiling in MHz (default: 2400)
#   --install: Install as a systemd oneshot + timer (re-applies every 60s)

set -euo pipefail

CAP=${1:-2400}
PCI_PATH="0000:c5:00.0"
SYSFS_OD="/sys/bus/pci/devices/${PCI_PATH}/pp_od_clk_voltage"
SYSFS_PERF="/sys/class/drm/card1/device/power_dpm_force_performance_level"

# Check if OD sysfs is available (needs amdgpu.ppfeaturemask=0xffffffff)
if [ ! -w "$SYSFS_OD" ]; then
    echo "❌ OD sysfs not writable at ${SYSFS_OD}"
    echo "   Add 'amdgpu.ppfeaturemask=0xffffffff' to kernel cmdline and reboot."
    echo "   Or: echo 0xffffffff | sudo tee /sys/module/amdgpu/parameters/ppfeaturemask"
    exit 1
fi

# Set performance level to manual
echo "manual" > "$SYSFS_PERF" 2>/dev/null || true

# Set clock cap: floor → ceiling → commit (order matters!)
echo "s 0 600"    > "$SYSFS_OD"  # min clock
echo "s 1 ${CAP}" > "$SYSFS_OD"  # max clock
echo "c"          > "$SYSFS_OD"  # commit

echo "✅ SCLK capped at ${CAP} MHz (PCI ${PCI_PATH})"

# Verify
CURRENT=$(cat "$SYSFS_OD" | grep "SCLK.*:" | tail -1 | awk '{print $3}')
echo "   Current ceiling: ${CURRENT} MHz"

if [ "${2:-}" = "--install" ]; then
    UNIT="gpu_powercap.service"
    TIMER="gpu_powercap.timer"
    
    # Create systemd service
    cat > /tmp/${UNIT} << UNITEOF
[Unit]
Description=GPU Power Cap — iGPU SCLK limiter for Strix Halo
After=multi-user.target

[Service]
Type=oneshot
ExecStart=${PWD}/scripts/gpu_powercap.sh ${CAP}
User=root

[Install]
WantedBy=multi-user.target
UNITEOF

    # Create systemd timer (re-apply every 60s — GPU reset reverts OD)
    cat > /tmp/${TIMER} << TIMEREOF
[Unit]
Description=GPU Power Cap timer — re-apply every 60s

[Timer]
OnBootSec=10
OnUnitActiveSec=60
Unit=${UNIT}

[Install]
WantedBy=timers.target
TIMEREOF

    sudo mv /tmp/${UNIT} /etc/systemd/system/
    sudo mv /tmp/${TIMER} /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now ${UNIT} ${TIMER}
    echo "✅ Systemd ${UNIT} + ${TIMER} installed (re-applies every 60s)"
fi
