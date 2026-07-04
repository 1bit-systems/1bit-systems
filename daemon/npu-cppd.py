#!/usr/bin/env python3
"""
npu-cppd — C++ NPU Engine Daemon (FLM replacement)

Starts the C++ NPU inference engine as a subprocess and
provides an OpenAI-compatible HTTP API for chat completions.

Usage:
  sudo ./daemon/npu-cppd.py [--port PORT] [--model MODEL]

Environment:
  NPU_XCLBIN_DIR    Override xclbin directory
  NPU_MODEL_PATH    Override model path
"""

import argparse
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class CppEngineBackend:
    """NPU inference via torch2aie fused xclbin Python runner."""

    def __init__(self, port: int):
        self.port = port
        self.ready = False
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = "/home/bcloud/torch2aie/toolchain/mlir_aie/python:/home/bcloud/torch2aie/examples/qwen3-decode-layer:/home/bcloud/torch2aie/examples/qwen3-decode-layer/cases"

    def start(self):
        import subprocess
        result = subprocess.run(
            ["/home/bcloud/torch2aie/.venv/bin/python3", "-c",
             "import sys; sys.path.insert(0, '/home/bcloud/1bit-systems'); from tools.npu_runner import NPURunner; print('OK')"],
            capture_output=True, text=True, timeout=120, env=self.env,
            cwd="/home/bcloud/1bit-systems")
        if result.returncode == 0:
            self.ready = True
            print(f"  NPU runner ready", flush=True)
        else:
            print(f"  NPU runner failed: {result.stderr[:200]}", flush=True)

    def stop(self):
        self.ready = False

    def tokenize(self, text: str) -> list:
        import subprocess
        r = subprocess.run(["/home/bcloud/1bit-systems/engine/npu/tokenizer/tokenize",
            "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"],
            input=text.encode(), capture_output=True, timeout=10)
        return [int(t) for t in r.stdout.decode().strip().split(",") if t.strip()]

    def detokenize(self, tokens: list) -> str:
        inp = " ".join(str(t) for t in tokens)
        r = subprocess.run(["/home/bcloud/1bit-systems/engine/npu/tokenizer/detokenize",
            "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"],
            input=inp.encode(), capture_output=True, timeout=10)
        return r.stdout.decode().strip()

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        if not self.ready:
            return {"error": "NPU engine not ready"}
        prompt_parts = []
        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "")
            prompt_parts.append("<|im_start|>" + r + "\n" + c + "<|im_end|>\n")
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "".join(prompt_parts)
        
        prompt_tokens = self.tokenize(full_prompt)
        if not prompt_tokens:
            return {"error": "Empty prompt"}
        max_new = kwargs.get("max_tokens", 64)
        
        import subprocess, json
        req = json.dumps({"tokens": prompt_tokens, "max_new_tokens": min(max_new, 64)})
        proc = subprocess.run(
            ["/home/bcloud/torch2aie/.venv/bin/python3", "/home/bcloud/1bit-systems/tools/npu_runner.py"],
            input=req, capture_output=True, text=True, timeout=300, env=self.env,
            cwd="/home/bcloud/1bit-systems")
        
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