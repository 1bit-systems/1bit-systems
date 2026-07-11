#!/usr/bin/env bash
# q2_0-strix-halo-validate.sh — build and validate llama.cpp Q2_0 on gfx1151.
#
# This harness is intentionally format-neutral: pass a Q2_0 g64 model for the
# upstream path, or a fork-specific g128 model when validating Prism/1bit code.
#
# Examples:
#   Q2_MODEL=/models/Ternary-Bonsai-1.7B-Q2_0-g64.gguf \
#     benchmarks/q2_0-strix-halo-validate.sh --dry-run
#
#   Q2_MODEL=/models/Ternary-Bonsai-1.7B-Q2_0-g64.gguf \
#     LLAMA_CPP_DIR=/home/bcloud/prism-llama.cpp \
#     benchmarks/q2_0-strix-halo-validate.sh --build --bench --server-smoke

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/home/bcloud/prism-llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$LLAMA_CPP_DIR/build-q2-hip}"
Q2_MODEL="${Q2_MODEL:-}"
Q2_FORMAT="${Q2_FORMAT:-unknown}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1151}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/benchmarks/data}"
SERVER_PORT="${SERVER_PORT:-18090}"
SERVER_LOG="${SERVER_LOG:-/tmp/1bit-q2_0-llama-server.log}"
RUN_BUILD=0
RUN_BENCH=0
RUN_SERVER=0
DRY_RUN=0

if [[ -d /opt/rocm/bin ]]; then
    export PATH="/opt/rocm/bin:$PATH"
fi

say() { printf '▸ %s\n' "$*"; }
ok() { printf '✓ %s\n' "$*"; }
warn() { printf '! %s\n' "$*" >&2; }
die() { printf '✗ %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage:
  Q2_MODEL=/path/to/q2_0.gguf $0 [--build] [--bench] [--server-smoke] [--dry-run]

Environment:
  LLAMA_CPP_DIR     llama.cpp checkout (default: $LLAMA_CPP_DIR)
  BUILD_DIR         build directory (default: $BUILD_DIR)
  Q2_MODEL          Q2_0 GGUF to validate (required unless --dry-run)
  Q2_FORMAT         label for results, e.g. g64 or g128 (default: unknown)
  AMDGPU_TARGETS    HIP target (default: gfx1151)
  RESULTS_DIR       output directory (default: $RESULTS_DIR)
  SERVER_PORT       llama-server smoke port (default: $SERVER_PORT)

Recommended upstream path:
  Q2_MODEL=/path/to/q2_0-g64.gguf Q2_FORMAT=g64 $0 --build --bench --server-smoke
EOF
}

run() {
    if (( DRY_RUN )); then
        printf 'DRY '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

have() {
    command -v "$1" >/dev/null 2>&1
}

parse_args() {
    while (($#)); do
        case "$1" in
            --build) RUN_BUILD=1 ;;
            --bench) RUN_BENCH=1 ;;
            --server-smoke) RUN_SERVER=1 ;;
            --dry-run|-n) DRY_RUN=1 ;;
            --help|-h) usage; exit 0 ;;
            *) die "unknown arg: $1" ;;
        esac
        shift
    done
    if (( RUN_BUILD == 0 && RUN_BENCH == 0 && RUN_SERVER == 0 )); then
        RUN_BUILD=1
        RUN_BENCH=1
        RUN_SERVER=1
    fi
}

preflight() {
    say "preflight"
    [[ -d "$LLAMA_CPP_DIR/.git" ]] || die "LLAMA_CPP_DIR is not a git checkout: $LLAMA_CPP_DIR"
    have cmake || die "cmake not found"
    have curl || die "curl not found"

    if (( ! DRY_RUN && (RUN_BENCH || RUN_SERVER) )); then
        [[ -n "$Q2_MODEL" ]] || die "Q2_MODEL is required"
        [[ -r "$Q2_MODEL" ]] || die "Q2_MODEL not readable: $Q2_MODEL"
    fi

    if [[ -e /dev/dri ]]; then ok "/dev/dri visible"; else warn "/dev/dri missing"; fi
    if [[ -e /dev/kfd ]]; then ok "/dev/kfd visible"; else warn "/dev/kfd missing"; fi
    printf 'active groups: %s\n' "$(id -nG)"
    printf 'configured groups: %s\n' "$(id -Gn "$(id -un)" 2>/dev/null || id -nG)"

    if have rocminfo; then
        rocminfo 2>/dev/null | awk '/Name:.*gfx|Marketing Name|Uuid:/ { print }' | head -20 || true
    else
        warn "rocminfo not found"
    fi

    if (( RUN_BUILD )); then
        have hipcc || warn "hipcc not found; install hip-dev and ensure /opt/rocm/bin is on PATH"
        [[ -f /opt/rocm/lib/cmake/hip-lang/hip-lang-config.cmake ]] \
            || warn "hip-lang CMake package not found; install hip-dev"
        [[ -f /opt/rocm/lib/cmake/hipblas/hipblas-config.cmake ]] \
            || warn "hipBLAS CMake package not found; install hipblas-dev and rocblas-dev"
    fi
}

write_metadata() {
    local out="$1"
    mkdir -p "$(dirname "$out")"
    {
        printf '{\n'
        printf '  "timestamp": "%s",\n' "$(date -Is)"
        printf '  "host": "%s",\n' "$(hostname)"
        printf '  "kernel": "%s",\n' "$(uname -r)"
        printf '  "llama_cpp_dir": "%s",\n' "$LLAMA_CPP_DIR"
        printf '  "llama_cpp_head": "%s",\n' "$(git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null || true)"
        printf '  "model": "%s",\n' "$Q2_MODEL"
        printf '  "format": "%s",\n' "$Q2_FORMAT"
        printf '  "amdgpu_targets": "%s"\n' "$AMDGPU_TARGETS"
        printf '}\n'
    } >"$out"
}

build_hip() {
    say "configure/build HIP for $AMDGPU_TARGETS"
    run cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$AMDGPU_TARGETS" \
        -DCMAKE_BUILD_TYPE=Release
    run cmake --build "$BUILD_DIR" -j"$(nproc)"
}

run_correctness() {
    local test_bin="$BUILD_DIR/bin/test-quantize-fns"
    if [[ -x "$test_bin" ]]; then
        say "run quant correctness smoke"
        run "$test_bin"
    else
        warn "test-quantize-fns not found at $test_bin; skipping"
    fi
}

run_bench() {
    local bench_bin="$BUILD_DIR/bin/llama-bench"
    [[ -x "$bench_bin" || "$DRY_RUN" -eq 1 ]] || die "llama-bench not found: $bench_bin"
    local stamp out meta
    stamp="$(date +%Y%m%dT%H%M%S)"
    out="$RESULTS_DIR/q2_0-strix-halo-${Q2_FORMAT}-${stamp}.jsonl"
    meta="$RESULTS_DIR/q2_0-strix-halo-${Q2_FORMAT}-${stamp}.meta.json"
    say "write metadata: $meta"
    if (( ! DRY_RUN )); then write_metadata "$meta"; else run write_metadata "$meta"; fi
    say "llama-bench -> $out"
    if (( DRY_RUN )); then
        run "$bench_bin" -m "$Q2_MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 2 -o jsonl
    else
        "$bench_bin" -m "$Q2_MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 2 -o jsonl | tee "$out"
    fi
}

server_smoke() {
    local server_bin="$BUILD_DIR/bin/llama-server"
    [[ -x "$server_bin" || "$DRY_RUN" -eq 1 ]] || die "llama-server not found: $server_bin"
    say "start llama-server smoke on :$SERVER_PORT"
    if (( DRY_RUN )); then
        run "$server_bin" --host 127.0.0.1 --port "$SERVER_PORT" -m "$Q2_MODEL" -ngl 99 -fa 1
        return
    fi

    pkill -f "llama-server.*--port $SERVER_PORT" 2>/dev/null || true
    "$server_bin" --host 127.0.0.1 --port "$SERVER_PORT" -m "$Q2_MODEL" -ngl 99 -fa 1 >"$SERVER_LOG" 2>&1 &
    local pid=$!
    trap 'kill "$pid" 2>/dev/null || true' EXIT

    for _ in $(seq 1 30); do
        if curl -s --max-time 1 "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null 2>&1; then
            ok "llama-server up at http://127.0.0.1:$SERVER_PORT/v1"
            break
        fi
        sleep 1
    done
    curl -sS --max-time 3 "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null
    curl -sS --max-time 30 "http://127.0.0.1:$SERVER_PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"'"$(basename "$Q2_MODEL")"'","messages":[{"role":"user","content":"Say Q2_0 OK"}],"max_tokens":16,"temperature":0}'
    echo
}

main() {
    parse_args "$@"
    preflight
    if (( RUN_BUILD )); then
        build_hip
    fi
    if (( RUN_BENCH )); then
        run_correctness
        run_bench
    fi
    if (( RUN_SERVER )); then
        server_smoke
    fi
}

main "$@"
