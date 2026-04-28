#!/usr/bin/env bash
# 1bit-systems installer — Strix Halo (gfx1151)
#
# Lean 1-bit inference engine: Lemonade Server (ROCm + Vulkan llama.cpp)
# and FastFlowLM (XDNA 2 NPU) behind one OpenAI-compat endpoint.
#
# Idempotent. Safe to re-run. Targets CachyOS / Arch (pacman).

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

say()  { printf '%b▸%b %s\n' "$CYAN" "$NC" "$*"; }
ok()   { printf '%b✓%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b!%b %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '%b✗%b %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

banner() {
    printf '\n'
    printf '%b╔═══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║%b  %b1bit-systems%b — strix-halo 1-bit inference engine        %b║%b\n' "$CYAN" "$NC" "$BOLD" "$NC" "$CYAN" "$NC"
    printf '%b║%b  %bgfx1151 ROCm + Vulkan + XDNA 2 NPU%b                       %b║%b\n' "$CYAN" "$NC" "$DIM" "$NC" "$CYAN" "$NC"
    printf '%b╚═══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
}

require_pacman() {
    command -v pacman >/dev/null 2>&1 || die "pacman not found — this installer targets CachyOS / Arch."
}

require_paru() {
    if ! command -v paru >/dev/null 2>&1; then
        warn "paru not found; installing AUR packages will need it."
        die "Install paru first: https://github.com/Morganamilo/paru"
    fi
}

install_packages() {
    say "Installing packages (pacman + AUR)"
    sudo pacman -S --needed --noconfirm \
        rocm-hip-sdk \
        xrt \
        xrt-plugin-amdxdna \
        fastflowlm \
        nodejs \
        npm \
        github-cli \
        ninja
    paru -S --needed --noconfirm lemonade-server
    ok "Packages installed"
}

write_memlock_limits() {
    local conf=/etc/security/limits.d/99-1bit-systems.conf
    if [[ -f "$conf" ]] && grep -q 'memlock     unlimited' "$conf"; then
        ok "memlock limits already set ($conf)"
        return
    fi
    say "Writing memlock limits to $conf"
    sudo tee "$conf" >/dev/null <<'EOF'
# 1bit-systems: NPU + GPU buffers need unlimited memlock.
*       soft    memlock     unlimited
*       hard    memlock     unlimited
EOF
    ok "memlock limits written — re-login or reboot for them to apply"
}

# Lemonade pins backend versions in /usr/share/lemonade-server/resources/backend_versions.json
# with strict equality. Arch's `fastflowlm` package can lead the pin (e.g. AUR ships v0.9.39
# while Lemonade pins v0.9.38), which flips the flm:npu recipe to `update_required` even when
# `flm validate` is fully green. Bump the pin to whatever flm actually reports.
patch_lemonade_flm_pin() {
    local manifest=/usr/share/lemonade-server/resources/backend_versions.json
    [[ -f "$manifest" ]] || { warn "Lemonade manifest not at $manifest — skipping pin patch"; return; }
    command -v flm >/dev/null 2>&1 || { warn "flm not on PATH — skipping pin patch"; return; }
    local installed pinned
    installed=$(flm version 2>/dev/null | head -1 | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' || true)
    pinned=$(grep -oE '"npu":\s*"v[0-9]+\.[0-9]+\.[0-9]+"' "$manifest" | head -1 | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' || true)
    if [[ -z "$installed" || -z "$pinned" ]]; then
        warn "Could not read FLM versions (installed=$installed pinned=$pinned) — skipping pin patch"
        return
    fi
    if [[ "$installed" == "$pinned" ]]; then
        ok "Lemonade flm:npu pin already matches installed FLM ($installed)"
        return
    fi
    say "Bumping Lemonade flm:npu pin: $pinned → $installed"
    sudo sed -i "s|\"npu\": \"$pinned\"|\"npu\": \"$installed\"|" "$manifest"
    ok "Lemonade flm:npu pin updated to $installed (restart lemond for it to take effect)"
    warn "Future lemonade-server pacman updates will overwrite this — re-run install.sh."
}

start_lemond() {
    say "Starting lemond on :13305"
    if pgrep -f 'lemonade-server.*serve' >/dev/null 2>&1 || pgrep -x lemond >/dev/null 2>&1; then
        ok "lemond already running"
        return
    fi
    nohup lemond >/tmp/lemond.log 2>&1 &
    sleep 2
    if curl -s --max-time 3 http://127.0.0.1:13305/api/v1/health >/dev/null 2>&1 \
       || curl -s --max-time 3 http://127.0.0.1:13305/v1/models >/dev/null 2>&1; then
        ok "lemond responding on http://127.0.0.1:13305"
    else
        warn "lemond didn't respond — check /tmp/lemond.log"
    fi
}

pull_default_model() {
    local dest=/home/$(id -un)/halo-ai/models/ternary-test/lily-bonsai-1.7b-rq
    local file=Bonsai-1.7B-IQ1_S.gguf
    if [[ -f "$dest/$file" ]]; then
        ok "Default 1-bit model already present ($file)"
        return
    fi
    say "Pulling default 1-bit model: $file (385 MB)"
    mkdir -p "$dest"
    curl -L --fail \
        -o "$dest/$file" \
        "https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized/resolve/main/$file" \
        || die "Failed to fetch $file"
    ok "Default model cached at $dest/$file"
}

install_cli() {
    say "Installing /usr/local/bin/1bit + proxy"
    sudo install -m 0755 "$(dirname "$0")/scripts/1bit" /usr/local/bin/1bit
    sudo install -d /usr/local/share/1bit-systems
    sudo install -m 0644 "$(dirname "$0")/scripts/1bit-proxy.js" /usr/local/share/1bit-systems/1bit-proxy.js
    ok "1bit CLI + proxy installed — try: 1bit up"
}

main() {
    banner
    require_pacman
    require_paru
    install_packages
    write_memlock_limits
    patch_lemonade_flm_pin
    install_cli
    start_lemond
    pull_default_model
    echo
    printf '%b✓%b Done. Run %b1bit up%b to launch lemond + flm + proxy.\n' "$GREEN" "$NC" "$BOLD" "$NC"
    printf '   %bUnified OpenAI endpoint:%b http://127.0.0.1:13306/v1/  (recommended — both lanes)\n' "$DIM" "$NC"
    printf '   %bGAIA users:%b https://amd-gaia.ai (.deb on Linux). Point it at http://127.0.0.1:13306\n' "$DIM" "$NC"
}

main "$@"
