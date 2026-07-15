#!/usr/bin/env bash
# Benchmark: 1-bit GPU decode tok/s on AMD Radeon 8060S (gfx1151) via
# llama.cpp/ROCm on a 1.58-bit ternary GGUF.
#
# Prints a single parseable line so tools/validate_claims.py can re-measure it:
#     gpu_1bit_tok_s <value>
#
# Reuses the on-box llama.cpp/ROCm build and the existing Ternary-Bonsai-1.7B
# Q2_0 model (1.58-bit). No model download, no new files on disk.
#
# Override the model path with GPU_1BIT_MODEL and the llama.cpp bin dir with
# LLAMACPP_BIN (defaults below point at the author's Strix Halo box layout).
set -euo pipefail

LLAMACPP_BIN="${LLAMACPP_BIN:-/home/bcloud/Bonsai-demo/bin/rocm}"
MODEL="${GPU_1BIT_MODEL:-/home/bcloud/models/Ternary-Bonsai-1.7B-Q2_0.gguf}"
PROMPT="${GPU_1BIT_PROMPT:-The capital of France is}"
N="${GPU_1BIT_N:-128}"

export LD_LIBRARY_PATH="${LLAMACPP_BIN}:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
# gfx1151 is not in stock ROCm 7.2; the project standard override works.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"

BIN="${LLAMACPP_BIN}/llama-completion"
# Missing infra -> exit 2 (validate_claims treats it as a soft skip, not a
# claim drift) and DO NOT print the key line (so 0 can never be published).
[ -x "$BIN" ] || { echo "MISSING: $BIN" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING: $MODEL" >&2; exit 2; }

# llama.cpp prints "eval time = ... ms / N runs ( X ms per token, Y tokens per second)".
# With -n N (max tokens) and temp 0, generation is deterministic.
out=$("$BIN" -m "$MODEL" -p "$PROMPT" -n "$N" --temp 0.0 -ngl 99 -no-cnv 2>&1 || true)
# Pull the eval-time tokens/second (decode rate, excludes prompt eval).
tok_s=$(printf '%s\n' "$out" | grep -oE 'eval time *=.*[0-9.]+ tokens per second' | grep -oE '[0-9.]+ tokens per second' | head -1 | grep -oE '^[0-9.]+')
# Only print the key line on a real measurement; a parse miss means we did
# not actually measure, so soft-skip (exit 2) rather than publish a junk 0.
if [ -n "${tok_s:-}" ] && [ "${tok_s}" != "0" ]; then
  echo "gpu_1bit_tok_s ${tok_s}"
else
  echo "gpu_1it_tok_s: could not parse eval tok/s" >&2
  exit 2
fi