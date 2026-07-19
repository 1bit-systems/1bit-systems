#!/usr/bin/env bash
# Benchmark: GPU ternary (Vulkan) decode tok/s via the ZINC engine on
# AMD Radeon 8060S (gfx1151).
#
# Prints a single parseable line for tools/validate_claims.py:
#     gpu_ternary_tok_s <value>
set -euo pipefail

ZINC="${ZINC_BIN:-$HOME/zinc/zig-out/bin/zinc}"
MODEL="${ZINC_MODEL:-$HOME/models/Ternary-Bonsai-1.7B-Q2_0.gguf}"
N="${ZINC_N:-128}"
PROMPT="${ZINC_PROMPT:-The capital of France is}"

[ -x "$ZINC" ]  || { echo "MISSING: $ZINC" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }

out=$("$ZINC" -m "$MODEL" --prompt "$PROMPT" --max-tokens "$N" --raw 2>&1 || true)
tok_s=$(printf '%s\n' "$out" | grep -oE 'Generated [0-9]+ tokens in [0-9.]+ ms — [0-9.]+ tok/s' | grep -oE '[0-9.]+ tok/s' | head -1 | grep -oE '^[0-9.]+')
if [ -n "${tok_s:-}" ] && [ "$(echo "$tok_s > 0" | bc 2>/dev/null)" = "1" ]; then
  echo "gpu_ternary_tok_s ${tok_s}"
else
  echo "gpu_ternary_tok_s: could not parse ZINC speed" >&2
  exit 2
fi