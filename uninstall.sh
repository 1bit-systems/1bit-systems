#!/usr/bin/env bash
# 1bit-systems uninstall. Mirrors `install.sh`: stops services, removes
# binaries, peels back the systemd units, and (with consent) the system-
# level changes (memlock conf, pacman packages, the wrapper's $HOME/src
# clone).
#
# Idempotent — safe to re-run; missing pieces are quietly skipped.
#
# Tiers:
#   1 (default, no prompt) — user-level: services, ~/.local/bin/* binaries,
#       systemd unit files. Reverses install.sh + `1bit install core`.
#   2 (prompts) — system-level: /etc/security/limits.d/99-*.conf (sudo).
#   3 (prompts) — pacman pkgs (xrt, xrt-plugin-amdxdna), the wrapper clone
#       at ~/src/1bit-systems, the NPU dev venvs (~/.venvs/ironenv,
#       ~/repos/IRON), and ~/repos/rocm-cpp if BUILD_ROCM_CPP=1 was used.
#
# Flags:
#   --yes / -y     accept all prompts (still respects --keep-data unless --purge)
#   --purge        wipe everything: tiers 1-3 + user data
#   --keep-data    preserve ~/.local/share/1bit/ (models, indexes, caches)
#   --dry-run      print what would happen, change nothing
#   -h / --help    show this help

set -euo pipefail

# ── palette + feedback (mirrors install.sh) ──────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m';  DIM='\033[2m';       BOLD='\033[1m'; NC='\033[0m'

banner() {
    printf '\n'
    printf '%b╔══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║  %b 1bit-systems · uninstall %b                                %b║%b\n' "$CYAN" "$BOLD" "$NC" "$CYAN" "$NC"
    printf '%b╚══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
}
step() { printf '\n%b▸ %s%b\n' "$BOLD" "$*" "$NC"; }
info() { printf '%b  · %s%b\n' "$DIM" "$*" "$NC"; }
ok()   { printf '%b  ✓ %s%b\n' "$GREEN" "$*" "$NC"; }
warn() { printf '%b  ⚠ %s%b\n' "$YELLOW" "$*" "$NC" >&2; }
die()  { printf '%b  ✗ %s%b\n' "$RED" "$*" "$NC" >&2; exit 1; }

# ── flags ────────────────────────────────────────────────────
ASSUME_YES=0
PURGE=0
KEEP_DATA=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)        ASSUME_YES=1 ;;
        --purge)         PURGE=1; ASSUME_YES=1 ;;
        --keep-data)     KEEP_DATA=1 ;;
        --dry-run)       DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) die "unknown flag: $1 (try --help)" ;;
    esac
    shift
done

ask() {
    # ask "prompt" -> sets REPLY=y or n. --yes always says yes.
    local prompt=$1
    if (( ASSUME_YES )); then REPLY=y; return 0; fi
    printf '%b  ? %s [y/N]: %b' "$YELLOW" "$prompt" "$NC"
    read -r REPLY || REPLY=n
    [[ "$REPLY" =~ ^[Yy] ]] && REPLY=y || REPLY=n
}

run() {
    # run a command, respecting --dry-run.
    if (( DRY_RUN )); then
        printf '%b    (dry-run) %s%b\n' "$DIM" "$*" "$NC"
    else
        "$@"
    fi
}

# ── identification ───────────────────────────────────────────
# Every binary / unit name placed by install.sh + packages.toml.
# Keep this list in sync with packages.toml when components are added.
USER_BINS=(
    1bit
    1bit-helm
    1bit-helm-tui
    1bit-landing
    1bit-voice
    1bit-echo
    1bit-mcp
    1bit-mcp-linuxgsm
    1bit-stream
    1bit-ingest
    1bit-tier-mint
    1bit-power
    1bit-watchdog
    1bit-tts-server
    1bit-sd-server
    1bit-halo-ralph
    halo-agent
    bitnet-to-tq2
    gguf-to-h1b
    h1b-sherry
)

USER_UNITS=(
    strix-landing.service
    strix-echo.service
    1bit-tts.service
    1bit-whisper.service
    1bit-sd.service
    1bit-mcp.service
    1bit-stream.service
)

SYSTEM_LIMITS=(
    /etc/security/limits.d/99-1bit-systems.conf
    /etc/security/limits.d/99-npu-memlock.conf
)

PACMAN_PKGS=(
    xrt
    xrt-plugin-amdxdna
)

USER_DATA=(
    "$HOME/.local/share/1bit"
    "$HOME/.cache/1bit-systems"
)

DEV_PATHS=(
    "$HOME/src/1bit-systems"
    "$HOME/.venvs/ironenv"
    "$HOME/repos/IRON"
    "$HOME/repos/rocm-cpp"
)

banner
[[ "$DRY_RUN" == "1" ]] && info "DRY RUN — nothing will be modified"

# ── tier 1: stop + disable + remove user units ───────────────
step "stopping + disabling systemd user units"
for unit in "${USER_UNITS[@]}"; do
    if systemctl --user list-unit-files --no-legend 2>/dev/null | awk '{print $1}' | grep -Fxq "$unit"; then
        info "disabling $unit"
        run systemctl --user disable --now "$unit" 2>/dev/null || warn "$unit didn't disable cleanly (continuing)"
    fi
done
ok "user units stopped"

step "removing systemd unit files"
for unit in "${USER_UNITS[@]}"; do
    for d in "$HOME/.config/systemd/user" "$HOME/.local/share/systemd/user"; do
        if [[ -e "$d/$unit" ]]; then
            info "rm $d/$unit"
            run rm -f "$d/$unit"
        fi
    done
done
run systemctl --user daemon-reload 2>/dev/null || true
ok "unit files removed"

# ── tier 1: remove user binaries ─────────────────────────────
step "removing 1bit binaries from ~/.local/bin/"
for bin in "${USER_BINS[@]}"; do
    if [[ -L "$HOME/.local/bin/$bin" || -f "$HOME/.local/bin/$bin" ]]; then
        info "rm $HOME/.local/bin/$bin"
        run rm -f "$HOME/.local/bin/$bin"
    fi
done
ok "binaries removed"

# ── user data: only when --purge OR explicit consent ─────────
if (( PURGE )) && (( ! KEEP_DATA )); then
    step "purging user data (models, caches)"
    for p in "${USER_DATA[@]}"; do
        if [[ -e "$p" ]]; then info "rm -rf $p"; run rm -rf "$p"; fi
    done
    ok "user data purged"
elif (( KEEP_DATA )); then
    info "--keep-data: preserving ${USER_DATA[*]}"
else
    step "user data — preserved by default (use --purge to remove)"
    for p in "${USER_DATA[@]}"; do [[ -e "$p" ]] && info "kept: $p"; done
    ok "user data preserved"
fi

# ── tier 2: system memlock confs (sudo) ──────────────────────
step "system memlock configs (/etc/security/limits.d/)"
NEED_LIMITS=0
for f in "${SYSTEM_LIMITS[@]}"; do
    [[ -e "$f" ]] && { info "found: $f"; NEED_LIMITS=1; }
done
if (( NEED_LIMITS )); then
    ask "remove the memlock conf(s)? sudo will be needed"
    if [[ "$REPLY" == "y" ]]; then
        for f in "${SYSTEM_LIMITS[@]}"; do
            if [[ -e "$f" ]]; then
                info "sudo rm $f"
                run sudo rm -f "$f"
            fi
        done
        ok "memlock confs removed (re-login for limits to revert)"
    else
        info "skipped (memlock=unlimited stays in effect)"
    fi
else
    ok "no memlock confs to remove"
fi

# ── tier 3: pacman packages ──────────────────────────────────
step "pacman packages (xrt + xrt-plugin-amdxdna)"
INSTALLED_PKGS=()
for p in "${PACMAN_PKGS[@]}"; do
    pacman -Q "$p" >/dev/null 2>&1 && INSTALLED_PKGS+=("$p")
done
if [[ ${#INSTALLED_PKGS[@]} -gt 0 ]]; then
    info "installed: ${INSTALLED_PKGS[*]}"
    ask "uninstall via pacman -Rns? (other software using XRT will break)"
    if [[ "$REPLY" == "y" ]]; then
        run sudo pacman -Rns --noconfirm "${INSTALLED_PKGS[@]}"
        ok "pacman packages removed"
    else
        info "skipped"
    fi
else
    ok "no 1bit-systems pacman packages installed"
fi

# ── tier 3: dev paths (wrapper clone, NPU venvs, repos) ──────
step "developer paths"
EXISTING_DEV=()
for p in "${DEV_PATHS[@]}"; do
    [[ -e "$p" ]] && { info "found: $p"; EXISTING_DEV+=("$p"); }
done
if [[ ${#EXISTING_DEV[@]} -gt 0 ]]; then
    ask "remove these directories? (wrapper clone, NPU venvs — does NOT touch ~/Projects/1bit-systems)"
    if [[ "$REPLY" == "y" ]]; then
        for p in "${EXISTING_DEV[@]}"; do
            info "rm -rf $p"
            run rm -rf "$p"
        done
        ok "developer paths removed"
    else
        info "skipped"
    fi
else
    ok "no developer paths to clean"
fi

# ── done ─────────────────────────────────────────────────────
printf '\n%b╔══════════════════════════════════════════════════════════╗%b\n' "$GREEN" "$NC"
printf     '%b║  ✓ 1bit-systems uninstall complete                         ║%b\n' "$GREEN" "$NC"
printf     '%b╚══════════════════════════════════════════════════════════╝%b\n\n' "$GREEN" "$NC"

cat <<EOF
Re-install: \`bash ~/Projects/1bit-systems/install.sh\`
                  or: \`curl -fsSL https://1bit.systems/install.sh | bash\`
Restored at next login (memlock back to default), or run:
  systemctl --user daemon-reload  # already done
  ulimit -l                       # confirm memlock is back to 8192
EOF
