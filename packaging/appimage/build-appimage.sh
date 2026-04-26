#!/usr/bin/env bash
# build-appimage.sh — build the 1bit-systems AppImage.
#
# Usage:
#   packaging/appimage/build-appimage.sh              # full build
#   SKIP_BUILD=1 packaging/appimage/build-appimage.sh # reuse cpp/build/strix
#   APPIMAGETOOL=/path/to/appimagetool ... override tool location
#   VERSION=0.1.8 packaging/appimage/build-appimage.sh # override version
#
# Steps:
#   1. `cmake --build --preset release-strix` (ROCm/MLX targets skipped
#      because they aren't part of release-strix anyway)
#   2. Copy the user-facing binaries into AppDir/usr/bin/
#   3. Copy AppRun + .desktop + .png to AppDir/ root
#   4. Download appimagetool to $HOME/.cache/appimagetool/ if missing
#   5. Invoke appimagetool → dist/1bit-systems-<version>-x86_64.AppImage
#   6. Compute sha256 → .sha256 sidecar
#   7. Print the artifact path
#
# Rule A: this is a build-time script run on dev/CI hosts. The output
# AppImage bundles only the C++23 orchestration binaries — no Python,
# no ROCm, no librocm_cpp.so. ROCm stays a host prereq installed by
# ./install.sh.
#
# Rust gut 2026-04-26: this script used to drive cargo. We use cmake
# now via `cmake --preset release-strix`.

set -euo pipefail

# ── resolve paths ────────────────────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
APPDIR="$HERE/1bit-systems.AppDir"
BIN_DST="$APPDIR/usr/bin"
DIST="$HERE/dist"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/appimagetool"
CPP_BUILD="$WORKSPACE/cpp/build/strix"

# Version: env override, else parse latest [X.Y.Z] from CHANGELOG.md.
VERSION="${VERSION:-}"
if [[ -z "$VERSION" ]]; then
    VERSION="$(grep -E '^## \[[0-9]+\.[0-9]+\.[0-9]+' "$WORKSPACE/CHANGELOG.md" | head -1 | sed -E 's/.*\[([0-9]+\.[0-9]+\.[0-9]+)([^]]*)\].*/\1\2/')"
fi
if [[ -z "${VERSION}" ]]; then
    echo "build-appimage: could not parse version from CHANGELOG.md (set VERSION=X.Y.Z env)" >&2
    exit 1
fi
ARCH="x86_64"
OUT_NAME="1bit-systems-${VERSION}-${ARCH}.AppImage"

log() { printf '\033[0;36m▸\033[0m %s\n' "$*"; }
ok()  { printf '\033[0;32m✓\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m⚠\033[0m %s\n' "$*" >&2; }
die() { printf '\033[0;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ── bins to bundle ───────────────────────────────────────────
# Matches binaries built by `cmake --preset release-strix` — the
# user-facing subset that does not link librocm_cpp.so or libmlx.
# Kept in sync by hand; if you add a new user-facing binary, append
# its (component, binname) pair below.
BUNDLED_BINS=(
    "cli/1bit"
    "helm/1bit-helm"
    "helm/1bit-halo-helm-tray"
    "landing/1bit-landing"
    "mcp/1bit-mcp"
    "mcp-linuxgsm/1bit-mcp-linuxgsm"
    "voice/1bit-voice"
    "echo/1bit-echo"
    "power/1bit-power"
    "halo-ralph/1bit-halo-ralph"
    "watchdog/1bit-watchdog"
    "ingest/1bit-ingest"
    "stream/1bit-stream"
    "tier-mint/1bit-tier-mint"
    "helm-tui/1bit-helm-tui"
)

# ── step 1: cmake build ──────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "cmake configure (release-strix preset)"
    ( cd "$WORKSPACE/cpp" && cmake --preset release-strix )
    log "cmake build (release-strix preset)"
    ( cd "$WORKSPACE/cpp" && cmake --build --preset release-strix )
    ok "cmake build complete"
else
    log "SKIP_BUILD=1 — reusing existing $CPP_BUILD"
fi

# ── step 2: populate AppDir/usr/bin ──────────────────────────
log "staging AppDir at $APPDIR"
rm -rf "$BIN_DST"
mkdir -p "$BIN_DST"

MISSING=()
for entry in "${BUNDLED_BINS[@]}"; do
    src="$CPP_BUILD/$entry"
    name="$(basename "$entry")"
    if [[ -x "$src" ]]; then
        install -Dm755 "$src" "$BIN_DST/$name"
    else
        MISSING+=("$entry")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "binaries not found in $CPP_BUILD (may be pre-build artefact gap):"
    for m in "${MISSING[@]}"; do
        warn "  - $m"
    done
fi

ok "copied $(( ${#BUNDLED_BINS[@]} - ${#MISSING[@]} )) / ${#BUNDLED_BINS[@]} binaries"

# ── step 3: AppDir root files ────────────────────────────────
log "installing AppRun + .desktop + .png into AppDir/"
# AppRun + .desktop are already in AppDir/ (version-controlled). Ensure
# they're executable / present.
chmod +x "$APPDIR/AppRun"
if [[ ! -f "$APPDIR/1bit-systems.desktop" ]]; then
    die "missing $APPDIR/1bit-systems.desktop — check repo state"
fi

# Icon: generate on demand if missing.
if [[ ! -f "$APPDIR/1bit-systems.png" ]]; then
    log "generating icon via build_icon.sh"
    bash "$HERE/build_icon.sh"
fi

# AppImage wants the icon to also exist at AppDir/.DirIcon (symlink is fine).
if [[ ! -e "$APPDIR/.DirIcon" ]]; then
    ( cd "$APPDIR" && ln -sf 1bit-systems.png .DirIcon )
fi

ok "AppDir layout ready"

# ── step 4: fetch appimagetool ───────────────────────────────
APPIMAGETOOL="${APPIMAGETOOL:-}"
if [[ -z "$APPIMAGETOOL" ]]; then
    APPIMAGETOOL="$CACHE/appimagetool-x86_64.AppImage"
    if [[ ! -x "$APPIMAGETOOL" ]]; then
        mkdir -p "$CACHE"
        log "downloading appimagetool to $APPIMAGETOOL"
        URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL -o "$APPIMAGETOOL" "$URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -q -O "$APPIMAGETOOL" "$URL"
        else
            die "need curl or wget to fetch appimagetool"
        fi
        chmod +x "$APPIMAGETOOL"
        ok "appimagetool fetched"
    else
        log "reusing cached appimagetool at $APPIMAGETOOL"
    fi
fi

# ── step 5: run appimagetool ─────────────────────────────────
mkdir -p "$DIST"
OUT="$DIST/$OUT_NAME"
rm -f "$OUT" "$OUT.sha256"

log "building $OUT_NAME"
# ARCH env tells appimagetool the target arch for the runtime blob.
# --no-appstream because we don't ship an AppStream manifest yet.
# If FUSE is unavailable (CI containers) fall back to --appimage-extract-and-run.
APPIMAGETOOL_RUN=( "$APPIMAGETOOL" )
if ! "$APPIMAGETOOL" --version >/dev/null 2>&1; then
    # Try extract-and-run — needed on hosts without FUSE (some CI).
    APPIMAGETOOL_RUN=( "$APPIMAGETOOL" --appimage-extract-and-run )
fi

ARCH="$ARCH" "${APPIMAGETOOL_RUN[@]}" --no-appstream "$APPDIR" "$OUT" \
    || die "appimagetool failed — ensure FUSE is available or set APPIMAGE_EXTRACT_AND_RUN=1"

chmod +x "$OUT"
ok "AppImage built at $OUT"

# ── step 6: sha256 sidecar ───────────────────────────────────
SHA="$(sha256sum "$OUT" | awk '{print $1}')"
printf '%s  %s\n' "$SHA" "$(basename "$OUT")" > "$OUT.sha256"
ok "sha256: $SHA"

# ── step 7: summary ──────────────────────────────────────────
SIZE="$(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT")"
SIZE_MB=$(( SIZE / 1024 / 1024 ))
printf '\n\033[0;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n'
printf '\033[1;32m  1bit-systems AppImage\033[0m\n'
printf '  path  : %s\n' "$OUT"
printf '  size  : %d bytes (%d MB)\n' "$SIZE" "$SIZE_MB"
printf '  sha256: %s\n' "$SHA"
printf '\033[0;32m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m\n\n'

# Machine-readable line for CI scraping.
printf 'APPIMAGE_ARTIFACT=%s\n' "$OUT"
printf 'APPIMAGE_SHA256=%s\n'   "$SHA"
printf 'APPIMAGE_SIZE=%s\n'     "$SIZE"
