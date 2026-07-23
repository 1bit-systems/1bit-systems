#!/usr/bin/env bash
set -euo pipefail
# sync-version.sh — propagate the canonical version (root VERSION file) into
# every packaging manifest so they can't drift apart again (issue #117).
#
# Usage:
#   scripts/sync-version.sh            # rewrite all manifests from VERSION
#   scripts/sync-version.sh --check    # fail if any manifest is out of sync (CI)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="$(tr -d '[:space:]' < VERSION)"
[ -n "$VERSION" ] || { echo "VERSION file is empty" >&2; exit 1; }

MODE="${1:-write}"
fail=0

# file : sed-expression that rewrites the version line in place
sync_file() {
  local file="$1" expr="$2"
  [ -f "$file" ] || return 0
  if [ "$MODE" = "--check" ]; then
    if ! grep -Eq "$3" "$file"; then
      echo "OUT OF SYNC: $file (expected version $VERSION)"
      fail=1
    fi
  else
    sed -i -E "$expr" "$file"
    echo "synced: $file -> $VERSION"
  fi
}

# package.json  ->  "version": "<VERSION>"
sync_file package.json \
  "s/(\"version\"[[:space:]]*:[[:space:]]*\")[^\"]*(\")/\1${VERSION}\2/" \
  "\"version\"[[:space:]]*:[[:space:]]*\"${VERSION}\""

# snap manifests  ->  version: '<VERSION>'
for f in snap/snapcraft.yaml packaging/snap/snapcraft.yaml; do
  sync_file "$f" \
    "s/^(version:[[:space:]]*').*(')/\1${VERSION}\2/" \
    "^version:[[:space:]]*'${VERSION}'"
done

# deb control  ->  Version: <VERSION>
sync_file packaging/deb/DEBIAN/control \
  "s/^(Version:[[:space:]]*).*/\1${VERSION}/" \
  "^Version:[[:space:]]*${VERSION}$"

# AUR PKGBUILD  ->  pkgver=<VERSION>
sync_file packaging/aur/PKGBUILD \
  "s/^(pkgver=).*/\1${VERSION}/" \
  "^pkgver=${VERSION}$"

# Homebrew formula  ->  version "<VERSION>"
sync_file packaging/homebrew/1bit-systems.rb \
  "s/^(  version \")[^\"]*(\")/\1${VERSION}\2/" \
  "^  version \"${VERSION}\"$"

# Homebrew url tag  ->  tags/v<VERSION>.tar.gz
sync_file packaging/homebrew/1bit-systems.rb \
  "s|(tags/v)[0-9]{4}\.[0-9]{2}\.[0-9]{2}[A-Za-z0-9.-]*|\1${VERSION}|" \
  "tags/v${VERSION}"

# deb postinst banner  ->  v<VERSION>
sync_file packaging/deb/DEBIAN/postinst \
  "s/v[0-9]{4}\.[0-9]{2}\.[0-9]{2}[A-Za-z0-9.-]*/v${VERSION}/" \
  "v${VERSION}"

if [ "$MODE" = "--check" ] && [ "$fail" -ne 0 ]; then
  echo "Run scripts/sync-version.sh to fix." >&2
  exit 1
fi
