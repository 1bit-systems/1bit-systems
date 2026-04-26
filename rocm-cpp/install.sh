#!/usr/bin/env bash
# rocm-cpp install — end-to-end setup for Strix Halo (gfx1151).
#
# Clones rocm-cpp + halo-1bit, builds librocm_cpp + bitnet_decode against an
# existing ROCm dist (TheRock or /opt/rocm), quantizes a BitNet-b1.58-2B-4T
# checkpoint to .h1b, runs a smoke test.
#
# Usage:
#   ./install.sh                # defaults — builds in $HOME, skips model DL
#   ROCM_ROOT=/opt/rocm ./install.sh
#   SKIP_MODEL=1 ./install.sh   # build only, no weights
#   PREFIX=/path/to/workspace ./install.sh

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PREFIX="${PREFIX:-$HOME}"
ROCM_ROOT="${ROCM_ROOT:-$HOME/therock/build/dist/rocm}"
SKIP_MODEL="${SKIP_MODEL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
MODEL_REPO="${MODEL_REPO:-microsoft/bitnet-b1.58-2B-4T-bf16}"
MODEL_DIR="${MODEL_DIR:-$PREFIX/halo-1bit/models/bitnet-2b-base}"
H1B_OUT="${H1B_OUT:-$PREFIX/halo-1bit/models/halo-1bit-2b-absmean.h1b}"
JOBS="${JOBS:-$(nproc)}"
# Arch: override with e.g. GFX=gfx1201 (RX 9070 XT / Navi 48) or
# GFX="gfx1151;gfx1201" for a fat binary that runs on both targets.
# If GFX=auto, detect via rocminfo (resolved after helpers are defined below).
GFX="${GFX:-gfx1151}"

ROCM_CPP_REPO="${ROCM_CPP_REPO:-https://github.com/bong-water-water-bong/rocm-cpp.git}"
HALO_1BIT_REPO="${HALO_1BIT_REPO:-https://github.com/bong-water-water-bong/halo-1bit.git}"

blue()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }
step()  { blue "── $* ─────────────────────────────────"; }

# Resolve GFX=auto now that helpers are defined.
if [ "$GFX" = "auto" ]; then
    if [ -x "$ROCM_ROOT/bin/rocminfo" ]; then
        GFX="$("$ROCM_ROOT/bin/rocminfo" | awk '/gfx1[12][0-9][0-9]/ {gsub(/.*Name:[[:space:]]*/,""); print; exit}')"
    elif command -v rocminfo >/dev/null 2>&1; then
        GFX="$(rocminfo | awk '/gfx1[12][0-9][0-9]/ {gsub(/.*Name:[[:space:]]*/,""); print; exit}')"
    fi
    [ -z "$GFX" ] && { red "GFX=auto: could not detect gfx arch via rocminfo"; exit 1; }
    green "GFX=auto detected: $GFX"
fi

# ─── Preflight ────────────────────────────────────────────────────────────────
step "Preflight"
if [ ! -d "$ROCM_ROOT" ]; then
    red "ROCM_ROOT=$ROCM_ROOT not found."
    red "Set ROCM_ROOT to a TheRock build (\$HOME/therock/build/dist/rocm)"
    red "or a system ROCm install (/opt/rocm)."
    exit 1
fi
if [ ! -x "$ROCM_ROOT/lib/llvm/bin/clang++" ] && [ ! -x "$ROCM_ROOT/bin/clang++" ]; then
    red "No clang++ under $ROCM_ROOT/{lib/llvm/bin,bin}. Is this a real ROCm dist?"
    exit 1
fi
for cmd in git cmake ninja python3; do
    command -v "$cmd" >/dev/null 2>&1 || { red "missing: $cmd"; exit 1; }
done
green "OK — ROCm at $ROCM_ROOT"

# ─── Clone rocm-cpp ───────────────────────────────────────────────────────────
step "Clone rocm-cpp"
cd "$PREFIX"
if [ -d rocm-cpp/.git ]; then
    green "rocm-cpp already present, pulling latest"
    git -C rocm-cpp pull --ff-only || true
else
    git clone --depth 1 "$ROCM_CPP_REPO" rocm-cpp
fi

# ─── Build librocm_cpp + bitnet_decode ────────────────────────────────────────
if [ "$SKIP_BUILD" != "1" ]; then
    step "Build librocm_cpp + tests + bitnet_decode"
    cd "$PREFIX/rocm-cpp"
    mkdir -p build
    cd build
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_ARCHITECTURES="$GFX" \
        -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++" \
        -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang" \
        -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++"
    cmake --build . -j "$JOBS" --target rocm_cpp bitnet_decode test_prim_and_attn
    green "Built: librocm_cpp.so, bitnet_decode, test_prim_and_attn"
fi

# ─── Clone halo-1bit (for .h1b exporter + tokenizer) ──────────────────────────
step "Clone halo-1bit"
cd "$PREFIX"
HALO_1BIT_BRANCH="${HALO_1BIT_BRANCH:-absmean-quant}"
if [ -d halo-1bit/.git ]; then
    green "halo-1bit already present, fetching $HALO_1BIT_BRANCH"
    git -C halo-1bit fetch origin "$HALO_1BIT_BRANCH" || true
    git -C halo-1bit checkout "$HALO_1BIT_BRANCH" 2>/dev/null || \
        git -C halo-1bit checkout -b "$HALO_1BIT_BRANCH" "origin/$HALO_1BIT_BRANCH"
    git -C halo-1bit pull --ff-only || true
else
    git clone --depth 1 --branch "$HALO_1BIT_BRANCH" "$HALO_1BIT_REPO" halo-1bit
fi

# ─── Model download + export ──────────────────────────────────────────────────
if [ "$SKIP_MODEL" != "1" ]; then
    step "Fetch BitNet-b1.58-2B-4T weights → $MODEL_DIR"
    if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
        mkdir -p "$MODEL_DIR"
        python3 -m pip install --user --quiet huggingface_hub safetensors numpy
        python3 - "$MODEL_REPO" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=out, local_dir_use_symlinks=False,
                  allow_patterns=["config.json","*.safetensors","tokenizer*","special_tokens_map.json","generation_config.json"])
PY
    else
        green "weights already present"
    fi

    step "Quantize to .h1b (BitNet absmean)"
    if [ ! -f "$H1B_OUT" ] || [ "$MODEL_DIR/model.safetensors" -nt "$H1B_OUT" ]; then
        mkdir -p "$(dirname "$H1B_OUT")"
        cd "$PREFIX/halo-1bit"
        python3 scripts/export_base_h1b.py "$MODEL_DIR" "$H1B_OUT"
    else
        green ".h1b already built"
    fi
fi

# ─── Smoke test ───────────────────────────────────────────────────────────────
if [ "$SKIP_MODEL" != "1" ] && [ "$SKIP_BUILD" != "1" ]; then
    step "Smoke test — bitnet_decode"
    export LD_LIBRARY_PATH="$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}:$PREFIX/rocm-cpp/build"
    export HSA_OVERRIDE_GFX_VERSION=11.5.1
    export HIP_VISIBLE_DEVICES=0
    "$PREFIX/rocm-cpp/build/bitnet_decode" "$H1B_OUT" 128000 16
    echo
    green "ALL GREEN — rocm-cpp is live on this box."
    echo
    echo "  Binary : $PREFIX/rocm-cpp/build/bitnet_decode"
    echo "  Model  : $H1B_OUT"
    echo
    echo "  Add to your shell rc:"
    echo "    export LD_LIBRARY_PATH=$ROCM_ROOT/lib:$PREFIX/rocm-cpp/build:\$LD_LIBRARY_PATH"
    echo "    export HSA_OVERRIDE_GFX_VERSION=11.5.1"
fi

green "Done."
