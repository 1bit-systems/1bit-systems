#!/usr/bin/env bash
# check-no-absolute-symlinks.sh — Guard against absolute symlinks tracked in git
#
# Absolute symlinks break on any clone that isn't the exact machine that
# created them (see #1043). This script checks that no git-tracked files
# are absolute symlinks — they must be either regular files, directories,
# or relative symlinks within the repo.
#
# Exit: 0 = clean, 1 = absolute symlinks found

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

broken=0

while IFS= read -r -d '' f; do
    if [ -L "$f" ]; then
        target=$(readlink "$f")
        case "$target" in
            /*) 
                echo "ERROR: absolute symlink: $f -> $target"
                broken=$((broken + 1))
                ;;
        esac
    fi
done < <(git ls-files -z)

if [ "$broken" -gt 0 ]; then
    echo ""
    echo "Found $broken git-tracked absolute symlink(s)."
    echo "These break on any clone other than the machine that created them."
    echo "Replace absolute symlinks with real files, relative symlinks, or"
    echo "document them as external dependencies with a fetch/build step."
    exit 1
fi

echo "OK: No absolute symlinks tracked in git."
