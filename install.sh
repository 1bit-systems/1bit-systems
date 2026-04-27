#!/usr/bin/env bash
# 1bit-systems bootstrap. Clones + builds the C++23 tower and its
# native-HIP kernel dependency (rocm-cpp), then hands off to the
# in-tree `1bit install` package manager for per-component wiring.
#
# Idempotent: re-runs only fetch-and-rebuild what changed.
#
# Supported hosts: CachyOS / Arch (pacman) on gfx1151. Other distros can
# crib the deps-check section manually; the build path itself is generic.
#
# Rust gut 2026-04-26: this script used to drive cargo. The whole
# orchestration tower is C++23 in `cpp/` now. We invoke
# `cmake --preset release-strix` instead. See CLAUDE.md ("I know kung fu").

set -euo pipefail

# ── palette + feedback helpers ───────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m';  DIM='\033[2m';       BOLD='\033[1m'; NC='\033[0m'

TOTAL_STEPS=7
STEP_IDX=0
STEP_START=0

banner() {
    printf '\n'
    printf '%b╔══════════════════════════════════════════════════════════╗%b\n' "$CYAN" "$NC"
    printf '%b║  %b 1bit-systems · strix-halo bootstrap %b                      %b║%b\n' "$CYAN" "$BOLD" "$NC" "$CYAN" "$NC"
    printf '%b║  %bgfx1151 · ternary BitNet · C++23 tower%b                     %b║%b\n' "$CYAN" "$DIM" "$NC" "$CYAN" "$NC"
    printf '%b╚══════════════════════════════════════════════════════════╝%b\n' "$CYAN" "$NC"
}

step() {
    STEP_IDX=$((STEP_IDX + 1))
    STEP_START=$(date +%s)
    printf '\n%b▸ [%d/%d] %s%b\n' "$BOLD" "$STEP_IDX" "$TOTAL_STEPS" "$*" "$NC"
    printf '%b  ─────────────────────────────────────────────────────%b\n' "$DIM" "$NC"
}

ok() {
    local dt=$(( $(date +%s) - STEP_START ))
    printf '%b  ✓ %s%b %b(%ds)%b\n' "$GREEN" "$*" "$NC" "$DIM" "$dt" "$NC"
}

info() { printf '%b  · %s%b\n' "$DIM" "$*" "$NC"; }
warn() { printf '%b  ⚠ %s%b\n' "$YELLOW" "$*" "$NC" >&2; }
die()  { printf '%b  ✗ %s%b\n' "$RED" "$*" "$NC" >&2; exit 1; }

# Progress dots for a long-running command. Reads stdin, tees to a per-label
# logfile under /tmp/1bit-install-logs/ for post-mortem, emits a heartbeat
# line counter, and returns 0. Pipefail (set -e) catches upstream failures;
# on failure the trap (LOG_DUMP_ON_ERR) prints the tail of every captured
# log so the operator sees *why* cmake/hipcc/git died.
# Usage: `some_long_cmd 2>&1 | progress_pipe "label"`.
LOG_DIR="${LOG_DIR:-/tmp/1bit-install-logs}"
mkdir -p "$LOG_DIR"
progress_pipe() {
    local label=$1
    local logfile="$LOG_DIR/${label// /_}.log"
    local n=0
    local line=""
    : > "$logfile"
    while IFS= read -r line; do
        printf '%s\n' "$line" >> "$logfile"
        n=$((n + 1))
        if (( n % 20 == 0 )); then
            printf '%b    %s · %d lines%b\r' "$DIM" "$label" "$n" "$NC" >&2
        fi
    done
    printf '%b    %s · %d lines total → %s%b\n' "$DIM" "$label" "$n" "$logfile" "$NC" >&2
}

# On any failure, dump tail of every captured log so the operator sees the
# real error instead of just "cmake exited 1".
LOG_DUMP_ON_ERR() {
    local rc=$?
    [[ "$rc" == "0" ]] && return 0
    printf '\n%b━━ install failed (exit %d) — log tails ━━%b\n' "$RED" "$rc" "$NC" >&2
    if compgen -G "$LOG_DIR/*.log" > /dev/null; then
        for f in "$LOG_DIR"/*.log; do
            [[ -s "$f" ]] || continue
            printf '\n%b── %s (last 40 lines) ──%b\n' "$YELLOW" "$f" "$NC" >&2
            tail -40 "$f" >&2
        done
        printf '\n%bfull logs in %s%b\n' "$DIM" "$LOG_DIR" "$NC" >&2
    fi
    return $rc
}
trap LOG_DUMP_ON_ERR ERR

# ── configuration ────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$HOME/repos}"
ROCM_CPP_URL="${ROCM_CPP_URL:-https://github.com/bong-water-water-bong/rocm-cpp.git}"
ROCM_CPP_DIR="${ROCM_CPP_DIR:-$REPO_ROOT/rocm-cpp}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
CI_MODE="${CI:-${GITHUB_ACTIONS:-0}}"
INSTALL_MODE="${INSTALL_MODE:-source}"
# BUILD_ROCM_CPP=1 enables the external rocm-cpp clone+build (steps 2-3).
# Default is OFF: the cpp tower's release-strix preset has
# ONEBIT_BUILD_ROCM_CPP=OFF, so the librocm_cpp.so artifact this step
# produces is only consumed by the out-of-tree lemond inference server.
# Most operators bringing up the CLI / landing / voice tower don't need it,
# and the build requires either TheRock or a complete rocm-cpp source tree.
# Set BUILD_ROCM_CPP=1 explicitly when deploying with lemond.
BUILD_ROCM_CPP="${BUILD_ROCM_CPP:-0}"

# ── appimage fast-path ───────────────────────────────────────
# INSTALL_MODE=appimage downloads the latest release AppImage, verifies
# its sha256 against the release feed, and drops a symlink at
# $HOME/.local/bin/1bit. No ROCm build, no cargo, no systemd. The operator
# still needs to install ROCm + rocm-cpp separately before `1bit install
# core` will work — but the CLI itself (status / doctor / update) is live.
#
# Canonical release feed URL is 1bit.systems/releases.json; overridable
# via APPIMAGE_RELEASES_URL for testing / self-hosting.
if [[ "$INSTALL_MODE" == "appimage" ]]; then
    banner
    info "mode:     appimage (single-file install)"
    [[ "$CI_MODE" != "0" ]] && info "CI mode — download-only dry run"

    RELEASES_URL="${APPIMAGE_RELEASES_URL:-https://1bit.systems/releases.json}"
    INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local/bin}"
    CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/1bit-systems"
    mkdir -p "$INSTALL_PREFIX" "$CACHE_DIR"

    step "fetching release feed"
    FEED_JSON="$CACHE_DIR/releases.json"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$FEED_JSON" "$RELEASES_URL" || die "failed to fetch $RELEASES_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$FEED_JSON" "$RELEASES_URL" || die "failed to fetch $RELEASES_URL"
    else
        die "need curl or wget for appimage mode"
    fi
    ok "feed cached at $FEED_JSON"

    step "parsing release feed"
    # Prefer jq if available; fall back to a minimal grep/awk parser that
    # handles the known flat schema (single-artifact x86_64-linux-gnu).
    if command -v jq >/dev/null 2>&1; then
        LATEST="$(jq -r '.latest' "$FEED_JSON")"
        DL_URL="$(jq -r --arg v "$LATEST" '.releases[] | select(.version==$v) | .artifacts[] | select(.platform=="x86_64-linux-gnu" and .kind=="appimage") | .url' "$FEED_JSON")"
        DL_SHA="$(jq -r --arg v "$LATEST" '.releases[] | select(.version==$v) | .artifacts[] | select(.platform=="x86_64-linux-gnu" and .kind=="appimage") | .sha256' "$FEED_JSON")"
    else
        LATEST="$(grep -E '"latest"' "$FEED_JSON" | head -1 | sed -E 's/.*"([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')"
        DL_URL="$(grep -E '"url"' "$FEED_JSON" | head -1 | sed -E 's|.*"(https?://[^"]+)".*|\1|')"
        DL_SHA="$(grep -E '"sha256"' "$FEED_JSON" | head -1 | sed -E 's/.*"([a-f0-9]{64})".*/\1/')"
    fi
    [[ -n "$LATEST" && -n "$DL_URL" && -n "$DL_SHA" ]] || die "could not parse release feed"
    info "latest:   $LATEST"
    info "url:      $DL_URL"
    info "sha256:   $DL_SHA"
    ok "feed parsed"

    step "downloading AppImage"
    AI_PATH="$CACHE_DIR/1bit-systems-$LATEST-x86_64.AppImage"
    if [[ -f "$AI_PATH" ]] && sha256sum "$AI_PATH" 2>/dev/null | grep -q "^$DL_SHA "; then
        info "cached AppImage matches sha256 — reusing"
    else
        if command -v curl >/dev/null 2>&1; then
            curl -fL --progress-bar -o "$AI_PATH" "$DL_URL" || die "download failed"
        else
            wget -O "$AI_PATH" "$DL_URL" || die "download failed"
        fi
    fi
    ok "downloaded $(du -h "$AI_PATH" | awk '{print $1}')"

    step "verifying sha256"
    ACTUAL="$(sha256sum "$AI_PATH" | awk '{print $1}')"
    if [[ "$ACTUAL" != "$DL_SHA" ]]; then
        rm -f "$AI_PATH"
        die "sha256 mismatch — expected $DL_SHA, got $ACTUAL"
    fi
    ok "sha256 verified"

    step "installing symlink → $INSTALL_PREFIX/1bit"
    chmod +x "$AI_PATH"
    if [[ "$CI_MODE" == "0" ]]; then
        ln -sfn "$AI_PATH" "$INSTALL_PREFIX/1bit"
        ok "symlinked $INSTALL_PREFIX/1bit → $AI_PATH"
        if ! printf '%s\n' "$PATH" | tr ':' '\n' | grep -Fxq "$INSTALL_PREFIX"; then
            warn "$INSTALL_PREFIX is not on \$PATH — add it to your shell rc"
        fi
    else
        ok "CI mode — symlink skipped"
    fi

    printf '\n%b╔══════════════════════════════════════════════════════════╗%b\n' "$GREEN" "$NC"
    printf     '%b║  ✓ 1bit-systems AppImage installed                         ║%b\n' "$GREEN" "$NC"
    printf     '%b╚══════════════════════════════════════════════════════════╝%b\n\n' "$GREEN" "$NC"
    cat <<EOF
Next steps:
  1bit --version
  1bit doctor          # checks ROCm / rocm-cpp separately — install via full source bootstrap
  1bit --list          # list bundled binaries

AppImage path:  $AI_PATH
To switch to source install: rerun with \`INSTALL_MODE=source ./install.sh\`
EOF
    exit 0
fi

banner
info "workspace: $WORKSPACE_DIR"
info "rocm-cpp:  $ROCM_CPP_DIR"
info "rocm root: $ROCM_ROOT"
[[ "$CI_MODE" != "0" ]] && info "mode:      CI (syntax-only)"

# ── step 1: host check ───────────────────────────────────────
step "host + dependency check"
if [[ "$CI_MODE" == "0" ]]; then
    # Multi-probe: rocminfo string, amdgpu-arch, amd-smi product name. ROCm
    # version layouts vary (/opt/rocm vs /opt/rocm-7.13.0), and 8060S/Radeon
    # 8060S/AMD RYZEN AI MAX+ 395 are all gfx1151 marketing names. We accept
    # any of these as a positive ID rather than insist on a literal "gfx1151"
    # string in rocminfo output.
    GFX_HIT=0
    if [[ -x "$ROCM_ROOT/bin/rocminfo" ]]; then
        if "$ROCM_ROOT/bin/rocminfo" 2>/dev/null | grep -q gfx1151; then
            info "gfx1151 detected (rocminfo)"; GFX_HIT=1
        fi
    elif command -v rocminfo >/dev/null 2>&1; then
        if rocminfo 2>/dev/null | grep -q gfx1151; then
            info "gfx1151 detected (rocminfo on PATH)"; GFX_HIT=1
        fi
    fi
    if [[ "$GFX_HIT" == "0" ]] && command -v amdgpu-arch >/dev/null 2>&1; then
        if amdgpu-arch 2>/dev/null | grep -q gfx1151; then
            info "gfx1151 detected (amdgpu-arch)"; GFX_HIT=1
        fi
    fi
    if [[ "$GFX_HIT" == "0" ]] && command -v amd-smi >/dev/null 2>&1; then
        if amd-smi static 2>/dev/null | grep -Eqi '8060S|RYZEN AI MAX'; then
            info "Strix Halo detected (amd-smi: Radeon 8060S / Ryzen AI MAX)"
            GFX_HIT=1
        fi
    fi
    if [[ "$GFX_HIT" == "0" ]]; then
        warn "no gfx1151 detected — gen-2 kernels target this specifically"
        warn "(checked rocminfo, amdgpu-arch, amd-smi; install rocm-hip-sdk if missing)"
    fi
else
    info "CI mode — skipping gfx1151 gate"
fi
for t in git cmake; do
    command -v "$t" >/dev/null 2>&1 || die "missing: $t (install first)"
    info "$t $(${t} --version 2>&1 | head -1)"
done
ok "host looks good"

# ── step 2: rocm-cpp sync ────────────────────────────────────
step "fetching rocm-cpp (HIP kernels)"
if [[ "$BUILD_ROCM_CPP" == "0" ]]; then
    info "BUILD_ROCM_CPP=0 (default) — skipping clone (cpp tower's release-strix preset doesn't link librocm_cpp.so)"
    info "set BUILD_ROCM_CPP=1 if you're deploying lemond and need librocm_cpp.so"
    ok "skipped"
else
    mkdir -p "$REPO_ROOT"
    if [[ -d "$ROCM_CPP_DIR/.git" ]]; then
        info "existing checkout — pulling origin/main"
        git -C "$ROCM_CPP_DIR" pull --ff-only 2>&1 | progress_pipe "git pull"
    else
        info "fresh clone — $ROCM_CPP_URL"
        git clone "$ROCM_CPP_URL" "$ROCM_CPP_DIR" 2>&1 | progress_pipe "git clone"
    fi
    ok "rocm-cpp HEAD $(git -C "$ROCM_CPP_DIR" rev-parse --short HEAD)"
fi

# ── step 3: rocm-cpp build ───────────────────────────────────
step "building rocm-cpp (bitnet_decode + librocm_cpp.so)"
if [[ "$BUILD_ROCM_CPP" == "0" ]]; then
    info "BUILD_ROCM_CPP=0 (default) — skipping build"
    ok "skipped"
elif [[ "$CI_MODE" != "0" ]]; then
    info "CI mode — skipping (no ROCm on runners)"
    ok "skipped"
else
    # Some operators have CC / CXX / HIPCXX inherited from a TheRock dev
    # tree (or other non-ROCm-aware build) that point at compiler binaries
    # which no longer exist. CMake will refuse to configure with a bogus
    # CMAKE_*_COMPILER path. Validate first and unset cleanly so we fall
    # back to the system clang/clang++ on PATH.
    for var in CC CXX HIPCXX; do
        val="${!var:-}"
        if [[ -n "$val" && ! -x "$val" ]]; then
            warn "$var=$val is not executable — unsetting so CMake uses PATH"
            unset "$var"
        fi
    done
    # Wipe any stale CMakeCache that cached a now-dead compiler path.
    if [[ -f "$ROCM_CPP_DIR/build/CMakeCache.txt" ]] && \
       grep -q '^CMAKE_\(C\|CXX\|HIP\)_COMPILER:' "$ROCM_CPP_DIR/build/CMakeCache.txt" && \
       ! grep -q "^CMAKE_CXX_COMPILER:.*=$(command -v c++ 2>/dev/null || echo /usr/bin/c++)" "$ROCM_CPP_DIR/build/CMakeCache.txt"; then
        warn "stale CMakeCache in $ROCM_CPP_DIR/build — removing"
        rm -rf "$ROCM_CPP_DIR/build"
    fi
    mkdir -p "$ROCM_CPP_DIR/build"
    pushd "$ROCM_CPP_DIR/build" >/dev/null
    info "cmake configure"
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | progress_pipe "cmake"
    info "cmake build (this takes ~2 min on first run)"
    cmake --build . -j "$(nproc)" --target bitnet_decode rocm_cpp 2>&1 | progress_pipe "hipcc"
    popd >/dev/null
    ok "bitnet_decode built"
fi

# ── step 3.5: XDNA 2 NPU userspace (optional) ────────────────
### XDNA 2 NPU userspace (optional)
#
# XRT + amdxdna driver + memlock limits. Required to drive the AIE
# tiles from the NPU lane (ONNX Runtime + VitisAI Execution Provider,
# 2026-04-21 pivot — see project_npu_path_onnx.md). Skipped silently
# on boxes without an NPU device (lspci/accel0 probe). Package install
# intent only — we don't actually run pacman here; the operator runs
# install.sh with sudo or answers the one pacman prompt.
step "XDNA 2 NPU userspace (optional)"
NPU_PRESENT=0
if [[ -e /dev/accel/accel0 ]]; then
    NPU_PRESENT=1
    info "found /dev/accel/accel0"
elif lspci 2>/dev/null | grep -qi 'npu\|signal processing controller'; then
    NPU_PRESENT=1
    info "lspci shows an NPU-class device"
fi

if [[ "$NPU_PRESENT" == "0" ]]; then
    info "no NPU detected — skipping XDNA 2 userspace"
    ok "skipped (no NPU)"
elif [[ "$CI_MODE" != "0" ]]; then
    info "CI mode — skipping pacman + limits.d"
    ok "skipped (CI)"
else
    # 1. XRT + amdxdna plugin (CachyOS extra; no AUR).
    if pacman -Q xrt >/dev/null 2>&1 && pacman -Q xrt-plugin-amdxdna >/dev/null 2>&1; then
        info "xrt + xrt-plugin-amdxdna already installed"
    else
        info "installing xrt + xrt-plugin-amdxdna via pacman"
        sudo pacman -S --noconfirm xrt xrt-plugin-amdxdna 2>&1 | progress_pipe "pacman"
    fi

    # 2. memlock limits.d from tracked template (substitute @USER@).
    MEMLOCK_TMPL="$WORKSPACE_DIR/strixhalo/security/99-npu-memlock.conf.tmpl"
    MEMLOCK_DST="/etc/security/limits.d/99-npu-memlock.conf"
    if [[ -f "$MEMLOCK_DST" ]]; then
        info "memlock limits already deployed at $MEMLOCK_DST"
    elif [[ ! -f "$MEMLOCK_TMPL" ]]; then
        warn "missing template $MEMLOCK_TMPL — skipping memlock install"
    else
        info "deploying $MEMLOCK_DST (USER=$USER)"
        sed "s|@USER@|$USER|g" "$MEMLOCK_TMPL" | sudo tee "$MEMLOCK_DST" >/dev/null
        info "re-login required for memlock=unlimited to take effect"
    fi

    # 3. kernel-side sanity.
    if [[ -e /dev/accel/accel0 ]]; then
        info "/dev/accel/accel0 present"
    else
        warn "/dev/accel/accel0 missing — amdxdna module may not be loaded"
    fi

    # 4. xrt-smi examine (don't fail on non-zero: CachyOS first-login memlock
    #    issue produces a warning until the user re-logs in).
    if command -v xrt-smi >/dev/null 2>&1; then
        info "xrt-smi examine:"
        xrt-smi examine 2>&1 | sed 's/^/    /' || true
    else
        info "xrt-smi not on PATH yet (source /etc/profile.d/xrt_setup.sh or re-login)"
    fi
    ok "NPU userspace staged"
fi

# ── step 4: cpp tower build ─────────────────────────────────
step "building 1bit-systems cpp tower (C++23, release-strix)"
if [[ "$CI_MODE" != "0" ]]; then
    info "cmake configure-only (CI mode — no link)"
    ( cd "$WORKSPACE_DIR/cpp" && cmake --preset release-strix 2>&1 | progress_pipe "cmake" )
    ok "configure clean"
else
    info "cmake configure"
    ( cd "$WORKSPACE_DIR/cpp" && cmake --preset release-strix 2>&1 | progress_pipe "cmake" )
    info "cmake --build --preset release-strix (this takes ~1 min on first run)"
    ( cd "$WORKSPACE_DIR/cpp" && cmake --build --preset release-strix 2>&1 | progress_pipe "build" )
    ok "build clean"
fi

# ── step 5: 1bit CLI install ─────────────────────────────────
step "installing 1bit CLI → \$HOME/.local/bin/1bit"
mkdir -p "$HOME/.local/bin"
install -Dm755 "$WORKSPACE_DIR/cpp/build/strix/cli/1bit" "$HOME/.local/bin/1bit"
if [[ -x "$HOME/.local/bin/1bit" ]]; then
    ok "$($HOME/.local/bin/1bit --version 2>&1 | head -1)"
else
    warn "1bit binary not at ~/.local/bin — is \$PATH set?"
fi

# ── step 6: packages.toml handoff ────────────────────────────
step "running 1bit install core (C++23 tower + HIP backend)"
if [[ "$CI_MODE" != "0" ]]; then
    info "CI mode — skipping"
    ok "skipped"
else
    # `1bit install core` writes user-level systemd units via tee, which
    # doesn't create parent dirs. mkdir -p so a fresh box works without
    # `loginctl enable-linger` having pre-created ~/.config/systemd/user.
    mkdir -p "$HOME/.config/systemd/user" "$HOME/.local/bin" "$HOME/.local/share/1bit"
    if [[ "$BUILD_ROCM_CPP" != "0" ]]; then
        export ROCM_CPP_LIB_DIR="$ROCM_CPP_DIR/build"
    fi
    if "$HOME/.local/bin/1bit" install core 2>&1 | progress_pipe "1bit install"; then
        ok "core component up"
    else
        warn "1bit install core failed; inspect with '1bit doctor'"
    fi
fi

# ── banner ────────────────────────────────────────────────────
printf '\n%b╔══════════════════════════════════════════════════════════╗%b\n' "$GREEN" "$NC"
printf     '%b║  ✓ 1bit-systems bootstrap complete                         ║%b\n' "$GREEN" "$NC"
printf     '%b╚══════════════════════════════════════════════════════════╝%b\n\n' "$GREEN" "$NC"

# Detect LAN IP so the printed URLs are reachable from a phone / laptop
# on the same network, not just the local machine.
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -z "${LAN_IP:-}" ] && LAN_IP="localhost"
LEMONADE_URL="http://${LAN_IP}:8000/app/"
GAIA_URL="http://${LAN_IP}:8000/gaia/"

cat <<EOF
Next steps:
  1bit status             # 7-service snapshot
  1bit doctor             # full health check (GPU, kernel, ports, endpoints)
  1bit install --list     # optional components (agents, voice, sd, gaia)
  1bit chat               # interactive REPL against :8180
  1bit say "hello"        # TTS via 1bit-halo-kokoro :8083
  1bit bench              # shadow-burnin parity summary
  1bit ppl                # perplexity vs gen-1 baseline 9.1607

Apps live in your browser:
  Lemonade  ${LEMONADE_URL}
  GAIA      ${GAIA_URL}

Landing page:  https://strixhalo.local/
Docs:          README.md, ARCHITECTURE.md, DEMO.md

EOF

# ── auto-open the two web UIs in default browser ──────────────
# Skip in CI, headless boxes, and when the user explicitly opted out.
# xdg-open is the freedesktop standard; fall back to sensible-browser.
if [ "${CI:-0}" = "1" ] || [ "${NO_OPEN:-0}" = "1" ] || [ -z "${DISPLAY:-${WAYLAND_DISPLAY:-}}" ]; then
  echo "  (set NO_OPEN=1 or unset DISPLAY/WAYLAND_DISPLAY -> not auto-opening browser tabs)"
else
  for OPENER in xdg-open sensible-browser gnome-open kde-open; do
    if command -v "$OPENER" >/dev/null 2>&1; then
      printf '  opening %s ... ' "$LEMONADE_URL"; "$OPENER" "$LEMONADE_URL" >/dev/null 2>&1 && echo "ok" || echo "failed"
      sleep 1
      printf '  opening %s ... ' "$GAIA_URL";     "$OPENER" "$GAIA_URL"     >/dev/null 2>&1 && echo "ok" || echo "failed"
      break
    fi
  done
fi

