#!/usr/bin/env bash
# check.sh — Quality Gate
# Runs build + test + diff checks after agent completes.
#
# Usage:
#   check.sh                    # gate on current state
#   check.sh --strict           # block on test failures too
#   check.sh --against HEAD~1   # gate against specific commit
#   check.sh --build-only       # only run build gate

set -euo pipefail

STRICT=false
BUILD_ONLY=false
AGAINST=""
PASSED=0
FAILED=0
WARNED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict) STRICT=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        --against) AGAINST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ─── Colors ────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'

PASS="${GREEN}✓ PASS${NC}"
FAIL="${RED}✗ FAIL${NC}"
WARN="${YELLOW}⚠ WARN${NC}"
SKIP="${GRAY}— SKIP${NC}"

# ─── Header ────────────────────────────────────────────────

echo ""
echo -e "  ${BOLD}═══ Quality Gate ═══${NC}"
echo ""

# ─── Layer 1: Build Gate ──────────────────────────────────

echo -e "  ${BOLD}Layer 1: Build Gate${NC}"

# Zig build
if [[ -f /home/bcloud/build.zig || -d /home/bcloud/engine/fusion/build.zig ]]; then
    echo -n "  zig build ... "
    BUILD_DIRS=()
    for d in /home/bcloud/engine/fusion /home/bcloud/engine/npu; do
        if [[ -f "$d/build.zig" ]]; then BUILD_DIRS+=("$d"); fi
    done
    if [[ ${#BUILD_DIRS[@]} -gt 0 ]]; then
        BUILD_OUTPUT=""
        BUILD_OK=true
        for bd in "${BUILD_DIRS[@]}"; do
            out=$(cd "$bd" && zig build 2>&1) || BUILD_OK=false
            BUILD_OUTPUT+="$out"$'\n'
        done
        if $BUILD_OK; then
            echo -e "${PASS}"
            PASSED=$((PASSED + 1))
        else
            echo -e "${FAIL}"
            echo -e "  ${RED}$(echo "$BUILD_OUTPUT" | tail -5 | sed 's/^/    /')${NC}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e "${SKIP} (no Zig build files)"
    fi
fi

# C++ build (spec-decode)
if [[ -f /home/bcloud/spec-decode/CMakeLists.txt ]]; then
    echo -n "  spec-decode build ... "
    BUILD_OUT=$(cd /home/bcloud/spec-decode && cmake --build build 2>&1) || true
    if echo "$BUILD_OUT" | grep -q "error:"; then
        echo -e "${FAIL}"
        echo -e "  ${RED}$(echo "$BUILD_OUT" | grep "error:" | tail -5 | sed 's/^/    /')${NC}"
        FAILED=$((FAILED + 1))
    elif echo "$BUILD_OUT" | grep -q "Built target"; then
        echo -e "${PASS}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${SKIP} (no build output detected)"
    fi
fi

# TypeScript build
if [[ -f /home/bcloud/tsconfig.json ]]; then
    echo -n "  tsc --noEmit ... "
    TS_OUT=$(cd /home/bcloud && npx tsc --noEmit 2>&1) || true
    if echo "$TS_OUT" | grep -q "error TS"; then
        echo -e "${FAIL}"
        echo -e "  ${RED}$(echo "$TS_OUT" | grep "error TS" | tail -5 | sed 's/^/    /')${NC}"
        FAILED=$((FAILED + 1))
    else
        echo -e "${PASS}"
        PASSED=$((PASSED + 1))
    fi
fi

[[ $BUILD_ONLY == true ]] && echo "" && echo -e "  ${BOLD}Result:${NC} ${PASSED} passed, ${FAILED} failed" && echo "" && exit $FAILED

# ─── Layer 2: Test Gate ────────────────────────────────────

echo ""
echo -e "  ${BOLD}Layer 2: Test Gate${NC}"

# Zig tests
if [[ -d /home/bcloud/engine ]]; then
    for td in /home/bcloud/engine/fusion /home/bcloud/engine/npu; do
        if [[ -f "$td/build.zig" ]]; then
            echo -n "  zig test $(basename $td) ... "
            TEST_OUT=$(cd "$td" && zig build test 2>&1) || true
            if echo "$TEST_OUT" | grep -q "All [0-9]* tests passed"; then
                echo -e "${PASS}"
                PASSED=$((PASSED + 1))
            elif echo "$TEST_OUT" | grep -qi "error\|FAILED\|panic"; then
                if $STRICT; then
                    echo -e "${FAIL}"
                    FAILED=$((FAILED + 1))
                else
                    echo -e "${WARN}"
                    WARNED=$((WARNED + 1))
                fi
                echo -e "  ${GRAY}$(echo "$TEST_OUT" | grep -iE "error|FAILED|panic" | tail -3 | sed 's/^/    /')${NC}"
            else
                echo -e "${SKIP} (no tests found)"
            fi
        fi
    done
fi

# C++ tests
if [[ -f /home/bcloud/spec-decode/build/CTestTestfile.cmake ]]; then
    echo -n "  ctest ... "
    CTEST_OUT=$(cd /home/bcloud/spec-decode/build && ctest --output-on-failure 2>&1) || true
    if echo "$CTEST_OUT" | grep -q "100% tests passed"; then
        echo -e "${PASS}"
        PASSED=$((PASSED + 1))
    elif echo "$CTEST_OUT" | grep -qi "fail\|error"; then
        if $STRICT; then
            echo -e "${FAIL}"
            FAILED=$((FAILED + 1))
        else
            echo -e "${WARN}"
            WARNED=$((WARNED + 1))
        fi
    else
        echo -e "${SKIP}"
    fi
fi

# Python tests
if ls /home/bcloud/tools/test_*.py &>/dev/null || [[ -d /home/bcloud/tests ]]; then
    echo -n "  pytest ... "
    PY_OUT=$(cd /home/bcloud && python3 -m pytest tests/ tools/ -x -q 2>&1) || true
    if echo "$PY_OUT" | grep -q "passed"; then
        echo -e "${PASS}"
        PASSED=$((PASSED + 1))
    elif echo "$PY_OUT" | grep -qi "failed\|error"; then
        if $STRICT; then
            echo -e "${FAIL}"
            FAILED=$((FAILED + 1))
        else
            echo -e "${WARN}"
            WARNED=$((WARNED + 1))
        fi
    else
        echo -e "${SKIP}"
    fi
fi

# ─── Layer 3: Diff Integrity ───────────────────────────────

echo ""
echo -e "  ${BOLD}Layer 3: Diff Integrity${NC}"

DIFF_STAT=$(git -C /home/bcloud diff --stat HEAD 2>/dev/null || echo "")
DIFF_SHORT=$(git -C /home/bcloud diff --shortstat HEAD 2>/dev/null || echo "")

if [[ -n "$DIFF_SHORT" ]]; then
    # Check for deleted files
    DELETED=$(git -C /home/bcloud diff --diff-filter=D --name-only HEAD 2>/dev/null || echo "")
    if [[ -n "$DELETED" ]]; then
        echo -e "  ${WARN} Deleted files: $(echo "$DELETED" | wc -l)"
        echo -e "  ${GRAY}$(echo "$DELETED" | sed 's/^/    /')${NC}"
        WARNED=$((WARNED + 1))
    fi

    # Check for binary changes
    BINARIES=$(git -C /home/bcloud diff --numstat HEAD 2>/dev/null | awk '$1 == "-" && $2 == "-" {print $3}' || echo "")
    if [[ -n "$BINARIES" ]]; then
        echo -e "  ${WARN} Binary file changes: $(echo "$BINARIES" | wc -l)"
        WARNED=$((WARNED + 1))
    fi

    # Check for large net deletions
    INSERTIONS=$(echo "$DIFF_SHORT" | grep -oP '\d+(?= insertion)' || echo 0)
    DELETIONS=$(echo "$DIFF_SHORT" | grep -oP '\d+(?= deletion)' || echo 0)
    NET_DEL=$((DELETIONS - INSERTIONS))
    if [[ $NET_DEL -gt 500 ]]; then
        echo -e "  ${WARN} Large net deletion: +${INSERTIONS}/-${DELETIONS} (net -${NET_DEL})"
        WARNED=$((WARNED + 1))
    else
        echo -e "  ${PASS} Diff: +${INSERTIONS}/-${DELETIONS}"
        PASSED=$((PASSED + 1))
    fi

    # Check unexpected file changes
    SENSITIVE_FILES=$(git -C /home/bcloud diff --name-only HEAD 2>/dev/null | grep -E '\.gitignore$|package-lock\.json$|CMakeLists\.txt$' || echo "")
    if [[ -n "$SENSITIVE_FILES" ]]; then
        echo -e "  ${WARN} Sensitive files changed:"
        echo -e "  ${GRAY}$(echo "$SENSITIVE_FILES" | sed 's/^/    /')${NC}"
        WARNED=$((WARNED + 1))
    fi
else
    echo -e "  ${SKIP} No uncommitted changes"
fi


# ─── Layer 4: Output Validation ────────────────────────────

echo ""
echo -e "  ${BOLD}Layer 4: Output Validation${NC}"
echo -e "  ${SKIP} (run on agent output, not applicable to check.sh)"

# ─── Summary ───────────────────────────────────────────────

echo ""
echo -e "  ${BOLD}═══════════════════════════════════${NC}"
echo -e "  ${BOLD}Gate Result:${NC}"
echo -n "  "
[[ $PASSED -gt 0 ]] && echo -ne "${GREEN}${PASSED} passed${NC}  "
[[ $WARNED -gt 0 ]] && echo -ne "${YELLOW}${WARNED} warnings${NC}  "
[[ $FAILED -gt 0 ]] && echo -ne "${RED}${FAILED} failed${NC}  "
echo ""

if [[ $STRICT == true && $FAILED -gt 0 ]]; then
    echo -e "  ${RED}🚫 GATE FAILED (strict mode)${NC}"
    echo ""
    exit 1
elif [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}🚫 GATE FAILED — build errors must be fixed${NC}"
    echo ""
    exit 1
else
    echo -e "  ${GREEN}✅ GATE PASSED${NC}"
    echo ""
    exit 0
fi
