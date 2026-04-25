#!/usr/bin/env bash
# build-appimage.sh — build the 1bit-systems AppImage.
#
# Usage:
#   packaging/appimage/build-appimage.sh              # full build
#   SKIP_CARGO=1 packaging/appimage/build-appimage.sh # reuse target/release
#   APPIMAGETOOL=/path/to/appimagetool ... override tool location
#
# Steps:
#   1. `cargo build --release --workspace --bins` (ROCm/MLX crates excluded)
#   2. Copy the user-facing binaries into AppDir/usr/bin/
#   3. Copy AppRun + .desktop + .png to AppDir/ root
#   4. Download appimagetool to $HOME/.cache/appimagetool/ if missing
#   5. Invoke appimagetool → dist/1bit-systems-<version>-x86_64.AppImage
#   6. Compute sha256 → .sha256 sidecar
#   7. Print the artifact path
#
# Rule A: this is a build-time script run on dev/CI hosts. The output
# AppImage bundles only pure-Rust binaries — no Python, no ROCm,
# no librocm_cpp.so. ROCm stays a host prereq installed by ./install.sh.

set -euo pipefail

# ── resolve paths ────────────────────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
APPDIR="$HERE/1bit-systems.AppDir"
BIN_DST="$APPDIR/usr/bin"
DIST="$HERE/dist"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/appimagetool"

# Read version from workspace Cargo.toml.
VERSION="$(grep -E '^version\s*=' "$WORKSPACE/Cargo.toml" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "${VERSION}" ]]; then
    echo "build-appimage: could not parse version from Cargo.toml" >&2
    exit 1
fi
ARCH="x86_64"
OUT_NAME="1bit-systems-${VERSION}-${ARCH}.AppImage"

log() { printf '\033[0;36m▸\033[0m %s\n' "$*"; }
ok()  { printf '\033[0;32m✓\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m⚠\033[0m %s\n' "$*" >&2; }
die() { printf '\033[0;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# ── bins to bundle ───────────────────────────────────────────
# Matches `[[bin]]` entries across the workspace for non-GPU crates.
# Kept in sync by hand; if you add a new user-facing binary, add it
# here. Anything that links librocm_cpp.so or libmlx is NOT included.
BUNDLED_BINS=(
    1bit
    1bit-helm
    1bit-halo-helm-tray
    onebit-server
    1bit-landing
    1bit-lemonade
    1bit-watch-discord
    1bit-watch-github
    1bit-mcp
    1bit-mcp-discord
    1bit-mcp-linuxgsm
    1bit-voice
    1bit-echo
    1bit-whisper
    1bit-kokoro
    1bit-halo-pkg
    1bit-halo-power
    1bit-halo-ralph
)

# Crates excluded from the release build because they need ROCm /
# Apple frameworks that aren't available on clean AppImage build hosts.
# They stay buildable via feature flags on the operator box.
EXCLUDE_CRATES=(
    --exclude onebit-hip
    --exclude onebit-mlx
)

# ── step 1: cargo build ──────────────────────────────────────
if [[ "${SKIP_CARGO:-0}" != "1" ]]; then
    log "cargo build --release --workspace --bins (excluding onebit-hip, onebit-mlx)"
    ( cd "$WORKSPACE" && cargo build --release --workspace --bins "${EXCLUDE_CRATES[@]}" )
    ok "cargo build complete"
else
    log "SKIP_CARGO=1 — reusing existing target/release"
fi

# ── step 2: populate AppDir/usr/bin ──────────────────────────
log "staging AppDir at $APPDIR"
rm -rf "$BIN_DST"
mkdir -p "$BIN_DST"

MISSING=()
for b in "${BUNDLED_BINS[@]}"; do
    src="$WORKSPACE/target/release/$b"
    if [[ -x "$src" ]]; then
        cp -f "$src" "$BIN_DST/$b"
        # Keep perms + strip no further — workspace [profile.release] already
        # strips symbols.
    else
        MISSING+=("$b")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "binaries not found in target/release (may be pre-build artefact gap):"
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
