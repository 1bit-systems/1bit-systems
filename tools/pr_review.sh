#!/usr/bin/env bash
# Local PR review — checks C/C++ diff for common issues before pushing.
# Usage:  bash pr_review.sh [base_ref]
set -euo pipefail

BASE="${1:-HEAD~1}"
PROJECT="$(git rev-parse --show-toplevel 2>/dev/null || exit 1)"
cd "$PROJECT"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }

FAILED=0
check() { local n="$1"; shift; if "$@"; then green "  ✅ $n"; else red "  ❌ $n"; FAILED=$((FAILED+1)); fi; }

echo "━━━ PR Review (C/C++ only) against ${BASE} ━━━"

# Get diff for C/C++ files only
DIFF=$(git diff "${BASE}" -- '*.cpp' '*.h' '*.hpp' '*.c' '*.cc' '*.cu' 2>/dev/null) || DIFF=""
if [ -z "$DIFF" ]; then
    yellow "  No C/C++ changes to review"
    exit 0
fi

check "No TODO/FIXME/HACK" \
    bash -c '! echo "$1" | grep -E "^\+.*\b(TODO|FIXME|HACK|XXX)\b" | grep -vE "^\+\s*(//|#|/\*| \*).*TODO"' _ "$DIFF"

check "No stray debug prints" \
    bash -c '! echo "$1" | grep -E "^\+.*\b(printf|std::cout)\b" | grep -v "//.*printf\|help/usage\|npu_gemm\|gpu_stubs"' _ "$DIFF"

check "No tab characters" \
    bash -c '! echo "$1" | grep "^+	"' _ "$DIFF"

check "No trailing whitespace" \
    bash -c '! echo "$1" | grep -E "^\+.*[[:space:]]+$"' _ "$DIFF"

check "No raw new/delete" \
    bash -c '! echo "$1" | grep -E "^\+.*\bnew\b" | grep -vE "make_unique|make_shared|unique_ptr|shared_ptr|return new|= new|auto .*= new"' _ "$DIFF"

echo
echo "━━━ Diff stat ━━━"
git diff "${BASE}" --stat -- '*.cpp' '*.h' '*.hpp' '*.c' '*.cc' '*.cu' 2>/dev/null || true

echo
if [ "$FAILED" -eq 0 ]; then green "✅ PR Review PASSED"; else red "❌ PR Review FAILED — ${FAILED} issue(s)"; exit 1; fi
