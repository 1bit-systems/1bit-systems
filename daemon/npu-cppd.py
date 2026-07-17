#!/usr/bin/env python3
"""
npu-cppd — C++ NPU Engine Daemon

OpenAI-compatible HTTP API for the NPU inference engine.
Routes chat completions to the NPU runner and handles tokenization.

Usage:
  sudo ./daemon/npu-cppd.py [--port PORT] [--model MODEL]

Environment:
  NPU_XCLBIN_DIR       Override xclbin directory (default: auto-detect)
  NPU_MODEL_PATH        Override model path (default: auto-detect)
  TORCH2AIE_ROOT        Path to torch2aie checkout (required)
  TORCH2AIE_VENV        Path to torch2aie virtualenv (default: TORCH2AIE_ROOT/.venv)
  TOKENIZER_JSON        Path to tokenizer.json (default: Qwen3-0.6B-NPU2)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Configuration ──────────────────────────────────────────────────────────

def _find_default(paths, desc):
    for p in paths:
        if Path(p).exists():
            return p
    return None

# Auto-detect tokenizer path from common locations
_TOKENIZER_CANDIDATES = [
    os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"),
    str(REPO_ROOT / "engine/npu/tokenizer/tokenizer.json"),
    os.path.join(os.environ.get("NPU_MODEL_PATH", ""), "tokenizer.json"),
]
_TOKENIZER_JSON = os.environ.get("TOKENIZER_JSON") or _find_default(_TOKENIZER_CANDIDATES, "tokenizer.json")

_TORCH2AIE_ROOT = os.environ.get("TORCH2AIE_ROOT", "")
_TORCH2AIE_VENV = os.environ.get("TORCH2AIE_VENV", os.path.join(_TORCH2AIE_ROOT, ".venv"))


class CppEngineBackend:
    """NPU inference backend."""

    def __init__(self):
        self.ready = False
        self.env = os.environ.copy()

        # Set up PYTHONPATH for torch2aie if configured
        if _TORCH2AIE_ROOT:
            parts = [
                os.path.join(_TORCH2AIE_ROOT, "toolchain/mlir_aie/python"),
                os.path.join(_TORCH2AIE_ROOT, "examples/qwen3-decode-layer"),
                os.path.join(_TORCH2AIE_ROOT, "examples/qwen3-decode-layer/cases"),
            ]
            existing = self.env.get("PYTHONPATH", "")
            self.env["PYTHONPATH"] = ":".join(parts + ([existing] if existing else []))

    def start(self):
        if not _TORCH2AIE_ROOT:
            print("  TORCH2AIE_ROOT not set — skipping NPU runner init", flush=True)
            self.ready = False
            return

        python_bin = os.path.join(_TORCH2AIE_VENV, "bin/python3")
        try:
            result = subprocess.run(
                [python_bin, "-c",
                 "import sys, os; sys.path.insert(0, os.environ.get('REPO_ROOT','')); "
                 "from tools.npu_runner import NPURunner; print('OK')"],
                capture_output=True, text=True, timeout=120,
                env={**self.env, "REPO_ROOT": str(REPO_ROOT)},
                cwd=str(REPO_ROOT))
            if result.returncode == 0:
                self.ready = True
                print("  NPU runner ready", flush=True)
            else:
                print(f"  NPU runner failed: {result.stderr[:200]}", flush=True)
        except Exception as e:
            print(f"  NPU runner startup error: {e}", flush=True)

    def stop(self):
        self.ready = False

    # Native tokenizer: engine/fusion/tokenize (pure C++17, bit-exact with HF).
    # Build with: make -C engine/fusion tokenize
    # This replaces the previously-wired paths, which were BOTH BROKEN:
    #   - fused-engine --tokenize-only : emitted no token output
    #   - engine/npu/tokenizer/detokenize : mangled byte-level spaces (Hello,\xc4\xa0world)
    _TOKENIZE_BIN = str(REPO_ROOT / "engine/fusion/tokenize")

    def tokenize(self, text: str) -> list:
        if not _TOKENIZER_JSON:
            return []
        try:
            r = subprocess.run(
                [self._TOKENIZE_BIN, "--model", _TOKENIZER_JSON, "--encode", text],
                capture_output=True, timeout=10)
            if r.returncode == 0:
                return [int(t) for t in r.stdout.decode().strip().split() if t.strip()]
        except Exception:
            pass
        return []

    def detokenize(self, tokens: list) -> str:
        if not _TOKENIZER_JSON or not tokens:
            return ""
        try:
            r = subprocess.run(
                [self._TOKENIZE_BIN, "--model", _TOKENIZER_JSON, "--decode",
                 *[str(t) for t in tokens]],
                capture_output=True, timeout=10)
            if r.returncode == 0:
                return r.stdout.decode()
        except Exception:
            pass
        return " ".join(str(t) for t in tokens)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        if not self.ready or not _TORCH2AIE_ROOT:
            return {"error": "NPU engine not ready", "hint": "Set TORCH2AIE_ROOT env var"}

        prompt_parts = []
        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "")
            prompt_parts.append(f"<|im_start|>{r}\n{c}<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "".join(prompt_parts)

        prompt_tokens = self.tokenize(full_prompt)
        if not prompt_tokens:
            return {"error": "Empty prompt"}

        max_new = min(kwargs.get("max_tokens", 64), 256)

        python_bin = os.path.join(_TORCH2AIE_VENV, "bin/python3")
        runner_path = str(REPO_ROOT / "tools/npu_runner.py")
        req = json.dumps({"tokens": prompt_tokens, "max_new_tokens": max_new})

        try:
            proc = subprocess.run(
                [python_bin, runner_path],
                input=req, capture_output=True, text=True, timeout=300,
                env=self.env, cwd=str(REPO_ROOT))
        except subprocess.TimeoutExpired:
            return {"error": "Inference timeout (>300s)"}
        except Exception as e:
            return {"error": str(e)}

        if proc.returncode != 0:
            return {"error": f"Runner failed: {proc.stderr[:200]}"}

        try:
            resp = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {"error": f"Bad JSON: {proc.stdout[:200]}"}

        out_tokens = resp.get("tokens", [])
        out_text = self.detokenize(out_tokens)
        return {
            "id": "chatcmpl-npu-cpp",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": out_text},
                         "finish_reason": "stop" if resp.get("finished") else "length"}],
            "usage": {"prompt_tokens": len(prompt_tokens), "completion_tokens": len(out_tokens),
                      "total_tokens": len(prompt_tokens) + len(out_tokens)},
        }


# ── HTTP Server ────────────────────────────────────────────────────────────

class ChatHandler(BaseHTTPRequestHandler):
    backend: CppEngineBackend = None
    model_name: str = "qwen3-0.6b-npu"

    def log_message(self, format, *args):
        print(f"  {self.client_address[0]} - {format % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "ready": self.backend.ready})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": self.model_name, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        model = body.get("model", self.model_name)
        messages = body.get("messages", [])
        kwargs = {k: v for k, v in body.items() if k not in ("model", "messages")}
        result = self.backend.chat(model, messages, **kwargs)
        if "error" in result:
            self._json(500, result)
        else:
            self._json(200, result)

    def _json(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main():
    parser = argparse.ArgumentParser(description="NPU C++ Engine Daemon")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--model", default="qwen3-0.6b-npu")
    args = parser.parse_args()

    print(f"=== NPU C++ Engine Daemon ===", flush=True)
    print(f"  Port: {args.port}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Tokenizer: {_TOKENIZER_JSON or 'not found'}", flush=True)
    print(f"  torch2aie: {_TORCH2AIE_ROOT or 'not set (NPU runner disabled)'}", flush=True)

    backend = CppEngineBackend()
    backend.start()
    ChatHandler.backend = backend
    ChatHandler.model_name = args.model

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    print(f"\n  Listening on http://0.0.0.0:{args.port}", flush=True)
    print(f"  Health: http://localhost:{args.port}/health", flush=True)
    print(f"  API:    http://localhost:{args.port}/v1/chat/completions", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...", flush=True)
        backend.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
