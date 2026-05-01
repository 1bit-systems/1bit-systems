#!/usr/bin/env bash
# 1bit-systems installer — Strix Halo (gfx1151)
#
# Lean 1-bit inference engine: Lemonade Server (ROCm + Vulkan llama.cpp)
# and FastFlowLM (XDNA 2 NPU) behind one OpenAI-compat endpoint.
#
# Idempotent. Safe to re-run. Targets CachyOS / Arch (pacman).
#
# Usage:
#   ./install.sh             # actually install
#   ./install.sh --dry-run   # preview every action without mutating anything
#   ./install.sh --help      # show this help

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

DRY_RUN=0

say()  { printf '%b▸%b %s\n' "$CYAN" "$NC" "$*"; }
ok()   { printf '%b✓%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b!%b %s\n' "$YELLOW" "$NC" "$*"; }
die()  { printf '%b✗%b %s\n' "$RED" "$NC" "$*" >&2; exit 1; }
dry()  { printf '%b≈%b %bDRY%b %s\n' "$YELLOW" "$NC" "$BOLD" "$NC" "$*"; }

# Run a command, or just show what would have run when DRY_RUN=1.
# Read-only probes (command -v, [[ -f ... ]], grep, flm version) bypass this.
run() {
    if (( DRY_RUN )); then
        dry "$*"
    else
        "$@"
    fi
}

usage() {
    cat <<EOF
1bit-systems installer

Usage:
  ./install.sh             actually install (idempotent)
  ./install.sh --dry-run   preview every action; mutate nothing
  ./install.sh -h | --help show this help

What runs in dry-run mode:
  - Read-only probes: pacman / paru / flm presence, FLM version,
    Lemonade manifest presence and current pin, model-cache check,
    lemond running-state check.
  - Every mutating command (pacman -S, paru -S, sudo tee/sed/install,
    nohup lemond, curl download) is printed prefixed with "DRY"
    instead of executing.

Targets CachyOS / Arch. Run on the box you want to host the stack.
EOF
}

banner() {
    local tag=""
    if (( DRY_RUN )); then tag=" — DRY-RUN"; fi
    printf '\n'
    printf '%b╔═══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║%b  %b1bit-systems%b — strix-halo 1-bit inference engine%s%b║%b\n' "$CYAN" "$NC" "$BOLD" "$NC" "$tag" "$CYAN" "$NC"
    printf '%b║%b  %bgfx1151 ROCm + Vulkan + XDNA 2 NPU%b                       %b║%b\n' "$CYAN" "$NC" "$DIM" "$NC" "$CYAN" "$NC"
    printf '%b╚═══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
    # Don't end the function on a `(( ... )) && cmd` line — under `set -e`
    # the arithmetic returns 1 when DRY_RUN=0, the && chain returns 1, the
    # function returns 1, and `set -e` exits the whole script silently
    # right after the banner.
    if (( DRY_RUN )); then
        warn "Dry-run mode — no commands will mutate the system."
    fi
}

require_pacman() {
    command -v pacman >/dev/null 2>&1 || die "pacman not found — this installer targets CachyOS / Arch."
    # NB: don't `pacman --version | head -1` here — that SIGPIPEs pacman, and
    # with `set -o pipefail` the whole script silently exits after the banner.
    ok "pacman found"
}

require_paru() {
    if ! command -v paru >/dev/null 2>&1; then
        warn "paru not found; installing AUR packages will need it."
        die "Install paru first: https://github.com/Morganamilo/paru"
    fi
    ok "paru found"
}

# Detect if this box has the AMD XDNA NPU (Strix Halo / Strix Point / Kraken Point).
# Returns 0 if NPU is present, 1 otherwise. Used to decide whether to install the
# xrt / xrt-plugin-amdxdna / fastflowlm trio (no point on a dGPU-only machine).
has_xdna_npu() {
    # Hardware ID 1022:17f0 (XDNA 2 — Strix Halo / Strix Point) or
    #             1022:1502 (XDNA 1 — Phoenix / Hawk Point — not supported by FLM).
    if command -v lspci >/dev/null 2>&1; then
        if lspci -n 2>/dev/null | grep -qE '1022:(17f0|1502)'; then
            return 0
        fi
    fi
    [[ -e /dev/accel/accel0 ]] && return 0
    lsmod 2>/dev/null | grep -q '^amdxdna ' && return 0
    return 1
}

install_packages() {
    local base_pkgs=(rocm-hip-sdk nodejs npm github-cli ninja)
    local xdna_pkgs=(xrt xrt-plugin-amdxdna fastflowlm)

    say "Installing base packages (pacman)"
    run sudo pacman -S --needed --noconfirm "${base_pkgs[@]}"

    if has_xdna_npu; then
        say "XDNA NPU detected — installing NPU lane (xrt + fastflowlm)"
        run sudo pacman -S --needed --noconfirm "${xdna_pkgs[@]}"
    else
        warn "no XDNA NPU detected — skipping xrt / xrt-plugin-amdxdna / fastflowlm"
        warn "(this box runs the iGPU/dGPU lane only; flm:npu recipe will stay 'unsupported')"
    fi

    say "Installing Lemonade Server (AUR)"
    run paru -S --needed --noconfirm --skipreview --noprovides lemonade-server
    ok "Packages installed"
}

write_memlock_limits() {
    local conf=/etc/security/limits.d/99-1bit-systems.conf
    if [[ -f "$conf" ]] && grep -q 'memlock     unlimited' "$conf"; then
        ok "memlock limits already set ($conf)"
        return
    fi
    say "Writing memlock limits to $conf"
    if (( DRY_RUN )); then
        dry "sudo tee $conf <<EOF"
        printf '       %b# 1bit-systems: NPU + GPU buffers need unlimited memlock.%b\n' "$DIM" "$NC"
        printf '       %b*       soft    memlock     unlimited%b\n' "$DIM" "$NC"
        printf '       %b*       hard    memlock     unlimited%b\n' "$DIM" "$NC"
        return
    fi
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
    if [[ ! -f "$manifest" ]]; then
        if (( DRY_RUN )); then
            warn "Lemonade manifest $manifest does not exist yet (would after install_packages); skipping pin probe"
        else
            warn "Lemonade manifest not at $manifest — skipping pin patch"
        fi
        return
    fi
    if ! command -v flm >/dev/null 2>&1; then
        if (( DRY_RUN )); then
            warn "flm not on PATH yet (would be after install_packages); skipping pin probe"
        else
            warn "flm not on PATH — skipping pin patch"
        fi
        return
    fi
    command -v python3 >/dev/null 2>&1 || { warn "python3 not found — skipping pin patch"; return; }
    local installed pinned
    installed=$(flm version 2>/dev/null | head -1 | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' || true)
    # Read flm.npu specifically (the manifest also has whispercpp.npu and others — never sed-replace blindly).
    pinned=$(python3 -c "import json,sys; print(json.load(open('$manifest')).get('flm',{}).get('npu',''))" 2>/dev/null || true)
    if [[ -z "$installed" || -z "$pinned" ]]; then
        warn "Could not read FLM versions (installed=$installed pinned=$pinned) — skipping pin patch"
        return
    fi
    if [[ "$installed" == "$pinned" ]]; then
        ok "Lemonade flm:npu pin already matches installed FLM ($installed)"
        return
    fi
    say "Bumping Lemonade flm:npu pin: $pinned → $installed"
    # Edit the JSON via python (preserves structure; only touches flm.npu).
    run sudo python3 -c "
import json,sys
p = '$manifest'
m = json.load(open(p))
m['flm']['npu'] = '$installed'
json.dump(m, open(p,'w'), indent=2)
open(p,'a').write('\n')
"
    ok "Lemonade flm:npu pin updated to $installed (restart lemond for it to take effect)"
    warn "Future lemonade-server pacman updates will overwrite this — re-run install.sh."
}

start_lemond() {
    say "Starting lemond on :13305"
    if pgrep -f 'lemonade-server.*serve' >/dev/null 2>&1 || pgrep -x lemond >/dev/null 2>&1; then
        ok "lemond already running"
        return
    fi
    if (( DRY_RUN )); then
        dry "nohup lemond >/tmp/lemond.log 2>&1 &"
        dry "sleep 2 && health-probe http://127.0.0.1:13305"
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
    local dest file
    dest=/home/$(id -un)/halo-ai/models/ternary-test/lily-bonsai-1.7b-rq
    file=Bonsai-1.7B-IQ1_S.gguf
    if [[ -f "$dest/$file" ]]; then
        ok "Default 1-bit model already present ($file)"
        return
    fi
    say "Pulling default 1-bit model: $file (385 MB)"
    run mkdir -p "$dest"
    if (( DRY_RUN )); then
        dry "curl -L --fail -o $dest/$file https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized/resolve/main/$file"
        return
    fi
    curl -L --fail \
        -o "$dest/$file" \
        "https://huggingface.co/lilyanatia/Bonsai-1.7B-requantized/resolve/main/$file" \
        || die "Failed to fetch $file"
    ok "Default model cached at $dest/$file"
}

install_cli() {
    say "Installing /usr/local/bin/1bit + proxy + omni"
    run sudo install -m 0755 "$(dirname "$0")/scripts/1bit" /usr/local/bin/1bit
    run sudo install -d /usr/local/share/1bit-systems
    run sudo install -m 0644 "$(dirname "$0")/scripts/1bit-home.html" /usr/local/share/1bit-systems/1bit-home.html
    run sudo install -m 0644 "$(dirname "$0")/scripts/1bit-proxy.js" /usr/local/share/1bit-systems/1bit-proxy.js
    run sudo install -m 0755 "$(dirname "$0")/scripts/1bit-omni.py" /usr/local/share/1bit-systems/1bit-omni.py
    ok "1bit CLI + proxy + omni installed — try: 1bit up   or   1bit omni \"...\""
}

parse_args() {
    while (( $# )); do
        case "$1" in
            -n|--dry-run) DRY_RUN=1 ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown arg: $1 (try --help)" ;;
        esac
        shift
    done
}

main() {
    parse_args "$@"
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
    if (( DRY_RUN )); then
        printf '%b≈%b %bDry-run complete.%b No system state was modified.\n' "$YELLOW" "$NC" "$BOLD" "$NC"
        printf '   Re-run without %b--dry-run%b to actually install.\n' "$BOLD" "$NC"
    else
        printf '%b✓%b Done. Run %b1bit up%b to launch lemond + flm + proxy.\n' "$GREEN" "$NC" "$BOLD" "$NC"
        printf '   %bUnified OpenAI endpoint:%b http://127.0.0.1:13306/v1/  (recommended — both lanes)\n' "$DIM" "$NC"
        printf '   %bGAIA users:%b https://amd-gaia.ai (.deb on Linux). Point it at http://127.0.0.1:13306\n' "$DIM" "$NC"
    fi
}

main "$@"
