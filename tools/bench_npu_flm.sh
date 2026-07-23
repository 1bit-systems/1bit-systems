#!/usr/bin/env bash
set -euo pipefail
# Benchmark: NPU (XDNA 2) decode tok/s on AMD Strix Halo via the FastFlowLM
# "FLM" engine and the on-box Qwen3-0.6B-NPU2 q4nx model + xclbins.
#
# Prints a single parseable line for tools/validate_claims.py:
#     npu_flm_tok_s <value>
#
# Method: start `flm serve`, wait for readiness, POST a deterministic
# chat completion (temp 0, max_tokens 128), compute completion_tokens / wall.
# Reuses the on-box FLM build + /flm model mount; no downloads, no new files.
#
# Override: FLM_BIN, FLM_CONFIG_PATH, FLM_XCLBIN_PATH, NPU_FLM_TAG, NPU_FLM_PORT.
set -euo pipefail

FLM_BIN="${FLM_BIN:-$HOME/fastflowlm-build/src/build/flm}"
FLM_CONFIG_PATH="${FLM_CONFIG_PATH:-$HOME/fastflowlm-build/src/model_list.json}"
FLM_XCLBIN_PATH="${FLM_XCLBIN_PATH:-$HOME/fastflowlm-build/src/xclbins}"
TAG="${NPU_FLM_TAG:-qwen3:0.6b}"
PORT="${NPU_FLM_PORT:-8098}"
PROMPT="${NPU_FLM_PROMPT:-List ten European capital cities, one per line, brief.}"
N="${NPU_FLM_N:-128}"

LD_LIBRARY_PATH="$(dirname "$FLM_BIN"):${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
export FLM_CONFIG_PATH FLM_XCLBIN_PATH

cleanup() { pkill -f "flm serve .*--port ${PORT}" 2>/dev/null || true; }
trap cleanup EXIT
cleanup   # in case a stale one is bound

[ -x "$FLM_BIN" ]  || { echo "MISSING: $FLM_BIN" >&2; exit 2; }
[ -e /dev/accel/accel0 ] || { echo "NO NPU (/dev/accel/accel0)" >&2; exit 2; }

nohup "$FLM_BIN" serve "$TAG" --port "$PORT" --pmode performance >/tmp/flm_bench.log 2>&1 &
# shellcheck disable=SC2034
PID=$!

ready=0
# shellcheck disable=SC2034
for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 1
done
[ "$ready" -eq 1 ] || { echo "npu_flm_tok_s 0"; echo "serve not ready" >&2; exit 2; }

tok_s=$(python3 - "$PORT" "$PROMPT" "$N" <<'PY'
import sys, time, json, urllib.request
port, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model":"qwen3:0.6b","messages":[{"role":"user","content":prompt}],
                         "max_tokens":n,"temperature":0,"stream":False}).encode(),
        headers={"Content-Type":"application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=200)
    d = json.load(r)
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    print(f"{(ct/dt):.2f}" if dt > 0 and ct else "")
except Exception as e:
    print("")
PY
)
if [ -n "${tok_s:-}" ]; then
  echo "npu_flm_tok_s ${tok_s}"
else
  echo "npu_flm_tok_s: no tokens / request failed" >&2
  exit 2
fi
