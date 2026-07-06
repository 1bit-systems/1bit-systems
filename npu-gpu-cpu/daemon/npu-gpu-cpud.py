#!/usr/bin/env python3
"""
npu-gpu-cpud — Unified Control Plane Daemon (v2 with live backends)

Routes inference requests to NPU, GPU, or CPU based on policy.
Backends:
  NPU: Custom INT8 NPU engine (npu_engine_all, subprocess)
  GPU: Lemonade/ROCm server (lemond)
  CPU: Lemonade CPU (lemond --backend cpu)

Policy (model_size → device):
  < 2B params  → NPU (lowest power)
  2B-8B params → GPU (fastest compute)
  > 8B params  → CPU (fallback)

Usage:
  sudo ./npu-gpu-cpud.py [--port PORT] [--npu-port PORT] [--gpu-port PORT]
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import urllib.request
import urllib.error

# Fused engine backend (NPU+GPU hybrid)
FUSED_ENGINE_AVAILABLE = False
try:
    from fused_backend import FusedEngine
    FUSED_ENGINE_AVAILABLE = True
except ImportError:
    FusedEngine = None

def _reject_json_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")

def _loads_json(body: bytes):
    return json.loads(body, parse_constant=_reject_json_constant)

def _parse_token(value) -> int:
    if isinstance(value, bool):
        raise ValueError("token must be an integer")
    if isinstance(value, int):
        token = value
    elif isinstance(value, str):
        if not value.strip().isdigit():
            raise ValueError(f"invalid token: {value!r}")
        token = int(value)
    else:
        raise ValueError("token must be an integer")
    if token < 0:
        raise ValueError("token must be non-negative")
    return token

def _parse_token_list(values) -> list[int]:
    if not isinstance(values, list):
        raise ValueError("tokens must be a list")
    return [_parse_token(value) for value in values]

# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------

class NPUBackend:
    """NPU inference via FLM (Fast Flow LM) — proxy to flm serve."""
    FLM_MODEL = os.environ.get("FLM_MODEL", "qwen3:0.6b")
    FLM_BIN = os.environ.get("FLM_BIN", "/usr/bin/flm")
    # FLM's internal port; we proxy from our port to FLM's API
    # NOTE: FLM v0.9.44 has a bug in /v1/chat/completions (substr error in
    # response serialization). /v1/completions works fine, so we convert chat
    # messages to a text prompt using a lightweight Qwen3 chat template.

    def __init__(self, port: int = 52625):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.flm_url = f"http://127.0.0.1:{port}/v1/completions"

    @staticmethod
    def _apply_chat_template(messages: list) -> str:
        """Lightweight Qwen3 chat template (no transformers dependency).
        Handles: system, user, assistant roles. Uses <|im_start|>/<|im_end|> format.
        """
        prompt = ""
        first_user_prompt = None
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                if first_user_prompt is None:
                    first_user_prompt = i
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def start(self):
        print(f"  Starting FLM NPU backend on port {self.port}...")
        env = os.environ.copy()
        # Raise memlock limit for FLM
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except Exception:
            pass
        self.process = subprocess.Popen(
            [self.FLM_BIN, "serve", self.FLM_MODEL, "--port", str(self.port), "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        # Wait for FLM to be ready
        for attempt in range(30):
            time.sleep(1)
            try:
                r = urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/v1/chat/completions",
                    data=b'{"model":"qwen3:0.6b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}',
                    timeout=5)
                r.read()
                break
            except Exception:
                if attempt == 29:
                    print(f"  ⚠️  FLM failed to start within 30s")
                    self.stop()
                    return
        print(f"  FLM NPU backend ready (pid={self.process.pid})")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        """Proxy chat completion to FLM via /v1/completions.

        FLM's /v1/chat/completions has a bug in response serialization
        (basic_string::substr error). We work around it by converting
        chat messages to a text prompt using a lightweight chat template
        and calling /v1/completions instead.
        """
        try:
            prompt = self._apply_chat_template(messages)
        except Exception as e:
            return {"error": f"Chat template error: {e}", "x-device": "npu"}

        req_body = {
            "model": self.FLM_MODEL,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
        }
        try:
            data = json.dumps(req_body).encode()
            r = urllib.request.urlopen(
                urllib.request.Request(self.flm_url, data=data,
                    headers={"Content-Type": "application/json"}),
                timeout=120)
            resp = json.loads(r.read())
            # Convert text_completion response to chat_completion format
            completion_text = resp.get("choices", [{}])[0].get("text", "")
            chat_resp = {
                "id": resp.get("id", "chatcmpl-npu"),
                "object": "chat.completion",
                "created": resp.get("created", int(time.time())),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text,
                    },
                    "finish_reason": resp.get("choices", [{}])[0].get("finish_reason", "stop"),
                }],
                "usage": resp.get("usage", {}),
                "x-device": "npu",
            }
            return chat_resp
        except Exception as e:
            return {"error": str(e), "x-device": "npu"}

class GPUBackend:
    """Lemonade server on GPU (port 13305 by default)"""
    def __init__(self, port: int = 13305):
        self.port = port
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        print(f"  Starting GPU backend (lemond) on port {self.port}...")
        cache_dir = f"/tmp/lemonade-gpu-{self.port}"
        self.process = subprocess.Popen(
            ["lemond", cache_dir, "--port", str(self.port), "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Wait for server to be ready
        for _ in range(30):
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/v1/health", timeout=2)
                if r.status == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        print(f"  GPU backend running (pid={self.process.pid})")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        body = {"model": model, "messages": messages, **kwargs}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=600)
        return json.loads(resp.read())

# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def estimate_model_size(model_name: str) -> float:
    """Estimate model size in billions of parameters from name."""
    model_lower = model_name.lower()
    for pattern, size in [
        ("0.5b", 0.5), ("0.6b", 0.6), ("0.8b", 0.8),
        ("1b", 1), ("1.2b", 1.2), ("1.5b", 1.5), ("1.7b", 1.7),
        ("2b", 2), ("2.6b", 2.6), ("3b", 3), ("4b", 4),
        ("7b", 7), ("8b", 8), ("9b", 9),
        ("13b", 13), ("14b", 14),
        ("20b", 20), ("34b", 34), ("70b", 70),
    ]:
        if pattern in model_lower:
            return size
    return 7  # default

def select_device(model_size_b: float) -> str:
    """Returns 'npu', 'gpu', or 'cpu' based on model size."""
    if model_size_b < 2:
        return "npu"
    elif model_size_b <= 8:
        return "gpu"
    else:
        return "cpu"

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

backends: dict = {}
start_time = time.time()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/health":
            self._json(200, {
                "status": "ok",
                "uptime_sec": int(time.time() - start_time),
                "devices": {
                    "npu": {"backend": "NPU INT8 engine", "port": backends["npu"].port,
                            "available": backends["npu"].process is not None},
                    "gpu": {"backend": "Lemonade (ROCm)", "port": backends["gpu"].port,
                            "available": backends["gpu"].process is not None},
                    "cpu": {"backend": "Lemonade (CPU)", "available": True},
                    "fused": {"backend": "NPU+GPU fused engine",
                              "available": backends.get("fused") is not None},
                },
                "policy": {
                    "< 2B params": "npu",
                    "2B-8B params": "gpu",
                    "> 8B params": "cpu",
                    "fused:// prefix": "NPU+GPU hybrid (fused engine)",
                },
            })
        elif self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [
                    {"id": "Qwen3-0.6B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Qwen3-8B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Llama-3.1-8B-NPU2", "object": "model", "owned_by": "npu"},
                    {"id": "Gemma4-E2B-IT-NPU2", "object": "model", "owned_by": "npu"},
                ]
            })
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = _loads_json(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return

            model = req.get("model", "unknown")
            messages = req.get("messages", [])
            stream = req.get("stream", False)
            extra_kwargs = {k: v for k, v in req.items() if k not in ("model", "messages", "stream")}
            print(f"  REQ model={model!r} msgs={len(messages)} stream={stream} extra_keys={list(extra_kwargs.keys())}", flush=True)

            # Streaming requested but not (yet) truly supported — just respond
            # non-streaming. Most clients handle this gracefully.
            _ = stream  # consumed, ignored

            # Route — npu:// bypasses size estimation
            # fused:// prefix routes to the fused NPU+GPU engine
            if model.startswith("fused://"):
                device = "fused"
                model_size = 0.6
            elif model.startswith("npu://"):
                device = "npu"
                model_size = 0.6  # Qwen3-0.6B
            else:
                model_size = estimate_model_size(model)
                device = select_device(model_size)
                # 2B-8B models can use fused engine if available
                if device == "gpu" and FUSED_ENGINE_AVAILABLE and "fused" in backends and backends["fused"] is not None:
                    device = "fused"

            try:
                if device == "fused":
                    if FUSED_ENGINE_AVAILABLE and backends.get("fused"):
                        resp = backends["fused"].chat(model, messages, **extra_kwargs)
                    else:
                        # Fallback to NPU if fused unavailable
                        resp = backends["npu"].chat(model, messages, **extra_kwargs)
                elif device == "npu":
                    resp = backends["npu"].chat(model, messages, **extra_kwargs)
                elif device == "gpu":
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)
                else:
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)

                # Tag response with routing info
                if isinstance(resp, dict):
                    resp["x-device"] = device
                    resp["x-model-size"] = f"{model_size}B"
                    if "error" in resp:
                        print(f"  RESP error: {resp['error']}", flush=True)
                    else:
                        txt = resp.get("choices",[{}])[0].get("message",{}).get("content","")[:50]
                        print(f"  RESP ok: {txt!r}", flush=True)

                self._json(200, resp)

            except urllib.error.HTTPError as e:
                self._json(e.code, {"error": e.reason, "x-device": device})
            except Exception as e:
                self._json(502, {"error": str(e), "x-device": device})
        elif self.path == "/v1/batch/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = _loads_json(body)
            except (json.JSONDecodeError, ValueError):
                self._json(400, {"error": "Invalid JSON"})
                return
            try:
                tokens = _parse_token_list(req.get("tokens", []))
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            if not tokens:
                self._json(400, {"error": "No tokens in request"})
                return
            if len(tokens) > 256:
                self._json(400, {"error": "Max 256 tokens"})
                return
            model_path = req.get("model",
                "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx")
            import subprocess
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = (
                "/home/bcloud/torch2aie/toolchain/mlir_aie.libs:"
                "/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64:" +
                env.get("LD_LIBRARY_PATH", ""))
            engine = "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_mt"
            try:
                proc = subprocess.run([engine, model_path] + [str(t) for t in tokens],
                    capture_output=True, text=True, env=env, timeout=600)
                stderr_lines = proc.stderr.strip().splitlines()
                tokens_out = []; total_ms = 0
                for line in stderr_lines:
                    if "===" in line:
                        parts = line.strip().split(",")
                        for p in parts:
                            if "ms" in p:
                                total_ms = float(p.split()[0].rstrip('ms'))
                if not math.isfinite(total_ms):
                    total_ms = 0
                import re
                for line in stderr_lines:
                    m = re.search(r'token\s+(\d+).*top8=\[([^\]]+)\]', line)
                    if m:
                        top8 = [int(x) for x in m.group(2).split(",")]
                        tokens_out.append({"index": int(m.group(1)), "top8": top8})
                self._json(200, {
                    "object": "batch.completion", "tokens": tokens_out,
                    "total_ms": total_ms,
                    "ms_per_token": total_ms / len(tokens) if tokens else 0,
                    "x-tokens": len(tokens), "x-device": "npu"})
            except subprocess.TimeoutExpired:
                self._json(504, {"error": "Engine timeout"})
            except Exception as e:
                self._json(502, {"error": str(e)})
        else:
            self._json(404, {"error": "Not found"})

    def _json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        method, path, code = args[0], args[1], args[2]
        device = self.headers.get("X-Device", "?")
        print(f"[{time.strftime('%H:%M:%S')}] {method} {path} → {code}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NPU+GPU+CPU Control Plane Daemon")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port")
    parser.add_argument("--npu-port", type=int, default=52625, help="NPU backend port")
    parser.add_argument("--gpu-port", type=int, default=13305, help="GPU backend port")
    parser.add_argument("--fused-port", type=int, default=18080, help="Fused NPU+GPU engine port")
    parser.add_argument("--fused-policy", default="auto",
        choices=["auto", "npu_only", "gpu_only", "attention_on_npu", "ffn_on_npu", "qkv_on_npu"],
        help="Fused engine dispatch policy")
    parser.add_argument("--no-auto", action="store_true", help="Don't auto-start backends")
    args = parser.parse_args()

    global backends
    # Parse extra args
    fused_policy = os.environ.get("FUSED_POLICY", "auto")

    backends = {
        "npu": NPUBackend(port=args.npu_port),
        "gpu": GPUBackend(port=args.gpu_port),
    }
    if FUSED_ENGINE_AVAILABLE:
        backends["fused"] = None  # lazy init

    print("=" * 60)
    print("  NPU + GPU + CPU = Unified Control Plane")
    if FUSED_ENGINE_AVAILABLE:
        print("  ★ Fused NPU+GPU engine available via fused:// prefix")
        print(f"  ★ Dispatch policy: {fused_policy}")
    print("  Gateway: http://0.0.0.0:{}".format(args.port))
    print("=" * 60)
    print()

    # Fused engine: lazy-initialize when first request arrives for fused://
    fused_engine_ref = {"eng": None}

    def get_fused_engine() -> Optional[FusedEngine]:
        if not FUSED_ENGINE_AVAILABLE:
            return None
        if fused_engine_ref["eng"] is None:
            try:
                eng = FusedEngine(policy=fused_policy)
                eng.start_server(args.fused_port)
                fused_engine_ref["eng"] = eng
                backends["fused"] = eng
                print(f"  Fused NPU+GPU engine started (policy={fused_policy})")
            except Exception as e:
                print(f"  ⚠️  Fused engine failed: {e}")
        return fused_engine_ref["eng"]

    if not args.no_auto:
        print("Starting backends...")
        backends["npu"].start()
        # GPU backend is optional — skip if lemond not found
        try:
            backends["gpu"].start()
        except FileNotFoundError:
            print("  ⚠️  lemond not found — GPU backend disabled")
        print()

    # Signal handling for clean shutdown
    def shutdown(sig, frame):
        print("\nShutting down...")
        backends["npu"].stop()
        backends["gpu"].stop()
        if fused_engine_ref["eng"]:
            fused_engine_ref["eng"].stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Gateway listening on http://0.0.0.0:{args.port}")
    print(f"  GET  /v1/health     — Device and backend status")
    print(f"  GET  /v1/models     — List available models")
    print(f"  POST /v1/chat/completions — Route to NPU/GPU/CPU")
    print(f"  POST /v1/batch/completions — Batch prefill (NPU multi-token)")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
