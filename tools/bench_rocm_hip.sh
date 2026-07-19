#!/usr/bin/env bash
# Benchmark: GPU HIP kernel decode tok/s on Radeon 8060S (gfx1151) via
# bitnet_decode (rocm_cpp ternary GEMV) on a compiled .h1b ternary model.
#
# Prints a single parseable line for tools/validate_claims.py:
#     rocm_hip_tok_s <value>
#
# Requires a pre-built .h1b model. The default points at the Bonsai-1.7B
# ternary model compiled from Ternary-Bonsai-1.7B-F16.gguf via
# tools/gguf_to_h1b.
set -euo pipefail

MODEL="${ROCM_HIP_MODEL:-$HOME/work/1bit-systems/models/bonsai.h1b}"
BINDIR="${ROCM_HIP_BINDIR:-$HOME/work/1bit-systems/build}"
N="${ROCM_HIP_N:-128}"
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

[ -x "${BINDIR}/bitnet_decode" ] || { echo "MISSING: ${BINDIR}/bitnet_decode" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }

export HSA_OVERRIDE_GFX_VERSION HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH="${BINDIR}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

out=$("${BINDIR}/bitnet_decode" --model "$MODEL" --ctx "$N" --iters "$N" 2>&1 || true)
tok_s=$(printf '%s\n' "$out" | grep -oE 'decode [0-9]+ tok in [0-9.]+ ms \(.*, [0-9.]+ tok/s\)' | grep -oE '[0-9.]+ tok/s' | head -1 | grep -oE '^[0-9.]+')
if [ -n "${tok_s:-}" ] && [ "$(echo "$tok_s > 0" | bc 2>/dev/null)" = "1" ]; then
  echo "rocm_hip_tok_s ${tok_s}"
else
  echo "rocm_hip_tok_s: could not parse decode speed" >&2
  exit 2
fi