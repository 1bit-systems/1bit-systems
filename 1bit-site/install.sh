#!/usr/bin/env bash
# 1bit.systems bootstrap — wake the 1bit monster on your Strix Halo box.
#
# Usage: curl -fsSL https://1bit.systems/install.sh | bash
#
# What it does, in order:
#   1. Refuses to run on anything that isn't "AMD Ryzen AI MAX+ 395" (Strix Halo).
#   2. Installs CachyOS AMDXDNA + XRT packages (cachyos-extra-znver4 repo).
#   3. Raises memlock so HIP can pin big buffers.
#   4. Clones bong-water-water-bong/1bit-systems to ~/src/1bit-systems.
#   5. Hands off to the repo's own install.sh for the real build + systemd wiring.
#
# Idempotent — safe to re-run to repair or upgrade.

set -Eeuo pipefail

# ── colours + helpers ────────────────────────────────────────
C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1m'; D='\033[2m'; N='\033[0m'
banner() {
  printf '\n'
  printf '%b╔══════════════════════════════════════════════════════════════╗%b\n' "$C" "$N"
  printf '%b║                                                              ║%b\n' "$C" "$N"
  printf '%b║   %b1bit.systems · the 1bit monster · strix halo bootstrap%b   %b║%b\n' "$C" "$B" "$N" "$C" "$N"
  printf '%b║   %bgfx1151 · ternary BitNet-b1.58 · bare metal · zero cloud%b   %b║%b\n' "$C" "$D" "$N" "$C" "$N"
  printf '%b║                                                              ║%b\n' "$C" "$N"
  printf '%b╚══════════════════════════════════════════════════════════════╝%b\n\n' "$C" "$N"
}
info() { printf '%b  · %s%b\n' "$D" "$*" "$N"; }
ok()   { printf '%b  ✓ %s%b\n' "$G" "$*" "$N"; }
warn() { printf '%b  ⚠ %s%b\n' "$Y" "$*" "$N" >&2; }
die()  { printf '%b  ✗ %s%b\n' "$R" "$*" "$N" >&2; exit 1; }
step() { printf '\n%b▸ %s%b\n' "$B" "$*" "$N"; }

banner

# ── hardware gate ────────────────────────────────────────────
step "checking hardware"
MODEL="$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')"
info "detected: ${MODEL:-unknown}"
# Lowercase before matching: /proc/cpuinfo capitalization varies across
# microcode/kernel versions (we've seen "Ryzen" and "RYZEN" in the wild).
MODEL_LC="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
case "$MODEL_LC" in
  *"amd ryzen ai max+ 395"*) ok "Strix Halo confirmed" ;;
  *"ryzen ai max"*)
    warn "this is a Ryzen AI MAX variant but not the 395 we've tested"
    warn "continuing at your own risk — file bugs with /proc/cpuinfo attached"
    ;;
  *)
    printf '\n%b┌──────────────────────────────────────────────────────────┐%b\n' "$R" "$N"
    printf '%b│  STOP — 1bit.systems runs on Strix Halo only, for now.   │%b\n' "$R" "$N"
    printf '%b│                                                          │%b\n' "$R" "$N"
    printf '%b│  Your CPU reports:                                       │%b\n' "$R" "$N"
    printf '%b│    %-56s│%b\n' "$R" "$MODEL" "$N"
    printf '%b│                                                          │%b\n' "$R" "$N"
    printf '%b│  The stack may work on other AMD / NVIDIA / Apple hosts  │%b\n' "$R" "$N"
    printf '%b│  but it is untested. Please do not file bugs against     │%b\n' "$R" "$N"
    printf '%b│  non-Strix-Halo hardware until we label them supported.  │%b\n' "$R" "$N"
    printf '%b└──────────────────────────────────────────────────────────┘%b\n\n' "$R" "$N"
    die "unsupported CPU — aborting"
    ;;
esac

# ── distro check (soft) ──────────────────────────────────────
step "checking distro"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  info "distro: ${PRETTY_NAME:-unknown}"
  case "$ID" in
    cachyos|arch|endeavouros|manjaro) ok "pacman-family distro detected" ;;
    *) warn "non-pacman distro ($ID) — package step may fail; you'll need to adapt it" ;;
  esac
else
  warn "/etc/os-release missing — unknown distro"
fi

# ── packages ─────────────────────────────────────────────────
step "installing NPU + XRT packages"
PKGS=(
  xrt
  xrt-plugin-amdxdna
)
if command -v pacman >/dev/null 2>&1; then
  info "enabling cachyos-extra-znver4 repo (if not already)"
  # CachyOS ships cachyos-extra-znver4 by default; on plain Arch users add it manually.
  # Non-destructive: only install if the repo exposes the packages.
  info "updating package db"
  sudo pacman -Sy --noconfirm >/dev/null || warn "pacman -Sy failed; continuing"
  info "pacman -S --needed ${PKGS[*]}"
  if sudo pacman -S --needed --noconfirm "${PKGS[@]}"; then
    ok "NPU + XRT packages installed"
  else
    warn "some packages missing from enabled repos — add cachyos-extra-znver4 or build from AUR"
  fi
else
  warn "no pacman found — skipping package step (install xrt / xrt-plugin-amdxdna manually)"
fi

# ── memlock limits ───────────────────────────────────────────
step "raising memlock limits for HIP"
LIMITS_FILE=/etc/security/limits.d/99-1bit systems.conf
if [ ! -f "$LIMITS_FILE" ] || ! grep -q 1bit systems "$LIMITS_FILE"; then
  sudo tee "$LIMITS_FILE" >/dev/null <<EOF
# 1bit.systems / 1bit systems — HIP needs to pin large contiguous buffers.
*       soft    memlock     unlimited
*       hard    memlock     unlimited
EOF
  ok "wrote $LIMITS_FILE (log out + in once to apply to new shells)"
else
  ok "memlock config already present"
fi

# ── clone + hand off ─────────────────────────────────────────
step "cloning 1bit-systems"
SRC="${HOME}/src/1bit-systems"
mkdir -p "${HOME}/src"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/bong-water-water-bong/1bit-systems.git "$SRC"
  ok "cloned into $SRC"
else
  info "repo already present — pulling latest"
  git -C "$SRC" pull --ff-only || warn "pull failed; local changes present?"
  ok "$SRC up-to-date"
fi

step "handing off to 1bit-systems install.sh"
if [ -x "$SRC/install.sh" ]; then
  cd "$SRC"
  exec bash "$SRC/install.sh" "$@"
else
  die "$SRC/install.sh not found or not executable — clone may be corrupt"
fi
