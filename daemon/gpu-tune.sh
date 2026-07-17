#!/bin/bash
# GPU performance tuning for Radeon 8060S (gfx1151) on Strix Halo
# Applied at boot via gpu-tune.service

set -e

GPU=/sys/class/drm/card1/device

echo "1bit GPU tune: Radeon 8060S (gfx1151)"

# 1. Pin clocks to maximum
echo high > $GPU/power_dpm_force_performance_level 2>/dev/null && \
  echo "  DPM: high"

# 2. Disable runtime power management
echo on > $GPU/power/control 2>/dev/null && \
  echo "  PM:  on"

# 3. ROCm performance level (if available)
if command -v rocm-smi &>/dev/null; then
  rocm-smi --setperflevel high 2>/dev/null && echo "  ROCm: high" || true
fi

# Show final clocks
echo "  Clocks:"
for c in sclk mclk socclk fclk; do
  v=$(grep '\*' $GPU/pp_dpm_$c 2>/dev/null || echo "N/A")
  printf "    %-8s %s\n" "$c:" "$v"
done
