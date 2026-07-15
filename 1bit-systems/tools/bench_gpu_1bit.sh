#!/usr/bin/env bash
set -euo pipefail

LLAMACPP_BIN="${LLAMACPP_BIN:-/home/bcloud/Bonsai-demo/bin/rocm}"
MODEL="${GPU_1BIT_MODEL:-/home/bcloud/models/Ternary-Bonsai-1.7B-Q2_0.gguf}"
PROMPT="${GPU_1BIT_PROMPT:-The capital of France is}"
N="${GPU_1BIT_N:-128}"

export LD_LIBRARY_PATH="${LLAMACPP_BIN}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

BIN="${LLAMACPP_BIN}/llama-completion"
[ -x "$BIN" ]  || { echo "MISSING: $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }

out=$("$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" --temp 0.0 -ngl 99 -no-cnv 2>&1 || true)
# Pull the decode eval-time tokens/second (excludes prompt eval).
# "prompt eval time" also contains "eval time" -- skip it by matching lines
# with "ms / N runs" (decode) vs "ms / N tokens" (prompt eval).
tok_s=$(printf '%s\n' "$out" | grep -E 'eval time *=.*ms /.*runs' | grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+')
if [ -n "${tok_s:-}" ] && [ "${tok_s}" != "0" ]; then
  echo "gpu_1bit_tok_s ${tok_s}"
else
  echo "gpu_1bit_tok_s: could not parse decode tok/s" >&2
  exit 2
fi