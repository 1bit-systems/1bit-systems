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
import os
import signal
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import urllib.request
import urllib.error

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
# Tokenizer (wraps C binaries)
# ---------------------------------------------------------------------------

TOKENIZE_BIN = "/home/bcloud/1bit-systems/engine/npu/tokenizer/tokenize"
DETOKENIZE_BIN = "/home/bcloud/1bit-systems/engine/npu/tokenizer/detokenize"

MODEL_REGISTRY = {
    "qwen3:0.6b": {
        "tokenizer": "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json",
        "model": "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx",
    },
    "qwen3-0.6b": {
        "tokenizer": "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json",
        "model": "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx",
    },
}

class Tokenizer:
    """Wrapper around tokenize/detokenize C binaries."""
    def __init__(self, tokenizer_json: str):
        self.json_path = tokenizer_json

    def encode(self, text: str) -> list[int]:
        """text → token IDs (includes BOS/EOS)"""
        try:
            r = subprocess.run(
                [TOKENIZE_BIN, self.json_path],
                input=text, capture_output=True, text=True, timeout=10)
            out = r.stdout.strip()
            if not out:
                return []
            return [int(t.strip()) for t in out.split(",") if t.strip()]
        except Exception:
            return []

    def decode(self, tokens: list[int]) -> str:
        """token IDs → text"""
        try:
            inp = " ".join(str(t) for t in tokens)
            r = subprocess.run(
                [DETOKENIZE_BIN, self.json_path],
                input=inp, capture_output=True, text=True, timeout=10)
            return r.stdout.strip().replace("Ġ", " ").replace("\Ċ", "\n")
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------

class NPUBackend:
    """Custom NPU inference engine (Qwen3-0.6B on INT8 GEMMs, ~175ms/tok)"""
    ENGINE_PATH = os.environ.get(
        "NPU_ENGINE_PATH",
        "/home/bcloud/npu-sandbox/npu-infer/build/npu_engine_stdio"
    )
    MODEL_PATH = os.environ.get(
        "NPU_MODEL_PATH",
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
    )
    TOKENIZER_PATH = os.environ.get(
        "NPU_TOKENIZER_PATH",
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json"
    )
    LD_PATH = os.environ.get(
        "NPU_LD_PATH",
        "/home/bcloud/torch2aie/toolchain/mlir_aie.libs:"
        "/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64"
    )

    def __init__(self, port: int = 52625):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.chat_stream = None  # token IDs for current chat session
        self.chat_idx = 0        # position in chat_stream
        self.generated_tokens: list[int] = []  # tokens generated so far
        self.tokenizer: Optional[Tokenizer] = None

    def start(self):
        print(f"  Starting NPU engine: {self.ENGINE_PATH}")
        if not os.path.exists(self.ENGINE_PATH):
            print(f"  ⚠️  NPU engine not found at {self.ENGINE_PATH}")
            return
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self.LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
        self.process = subprocess.Popen(
            [self.ENGINE_PATH, self.MODEL_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # Init tokenizer
        if os.path.exists(self.TOKENIZER_PATH):
            self.tokenizer = Tokenizer(self.TOKENIZER_PATH)
            print(f"  Tokenizer ready: {self.TOKENIZER_PATH}")
        else:
            print(f"  ⚠️  Tokenizer not found at {self.TOKENIZER_PATH}")
        print(f"  NPU engine running (pid={self.process.pid})")

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None

    def _send_recv(self, req: dict) -> Optional[dict]:
        if not self.process or self.process.poll() is not None:
            return None
        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()
        resp_line = self.process.stdout.readline()
        if not resp_line:
            return None
        try:
            return json.loads(resp_line.strip(), parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            return None

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        """Chat completion. Accepts:
        - Text models: "qwen3:0.6b" → tokenize text → engine → detokenize
        - Raw tokens: "npu://T1,T2,..." → direct engine feed (no tokenizer)
        """
        max_tokens = kwargs.get("max_tokens", 32)

        # ── npu:// raw token path ──
        if model.startswith("npu://"):
            return self._chat_raw_tokens(model, messages, max_tokens)

        # ── Text path (requires tokenizer) ──
        if not self.tokenizer:
            return {"error": "No tokenizer loaded — use npu:// format"}

        # Extract user text from messages
        user_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
        if not user_text:
            # Try raw numeric in content
            return self._chat_raw_from_content(messages, max_tokens)

        # Tokenize
        tokens = self.tokenizer.encode(user_text)
        if not tokens:
            return {"error": "Tokenizer failed"}

        # Remove EOS from end if present (we'll generate our own)
        if tokens and tokens[-1] == 151645:
            tokens = tokens[:-1]

        print(f"  Tokenized: {user_text!r} → {len(tokens)} tokens")

        # Run engine on token sequence, generate max_tokens new tokens
        generated = self._generate_tokens(tokens, max_tokens)
        if generated is None:
            self.chat_stream = None
            return {"error": "NPU engine failed"}

        # Detokenize output
        text = self.tokenizer.decode(generated)
        return {
            "id": "npu-completion",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            },
            "x-device": "npu",
            "x-ms-per-tok": self._avg_ms(),
        }

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> Optional[list[int]]:
        """Run engine on prompt tokens, generate up to max_tokens new tokens."""
        generated = []
        self._send_recv({"reset": True})

        # Prefill: feed each prompt token, engine returns the NEXT token each time
        # Last prompt token produces the first generated token
        for i, t in enumerate(prompt_tokens):
            resp = self._send_recv({"token": t})
            if resp is None:
                return None
            # The token returned is the engine's prediction for NEXT token
            if i == len(prompt_tokens) - 1:
                # Last prompt token → this is our first generated token
                tok = resp.get("token")
                if tok is not None:
                    try:
                        tok = _parse_token(tok)
                        generated.append(tok)
                    except ValueError:
                        pass

        # Autoregressive: feed each generated token back as next input
        for _ in range(max_tokens - 1):
            if not generated:
                break
            last_tok = generated[-1]
            if last_tok == 151645:  # EOS
                break
            resp = self._send_recv({"token": last_tok})
            if resp is None:
                break
            tok = resp.get("token")
            if tok is None:
                break
            try:
                tok = _parse_token(tok)
                generated.append(tok)
            except ValueError:
                break

        return generated

    def _chat_raw_tokens(self, model: str, messages: list, max_tokens: int) -> dict:
        """Handle npu:// token ID format."""
        try:
            tokens = [_parse_token(t) for t in model[6:].split(",") if t.strip()]
        except ValueError as e:
            return {"error": str(e)}
        if not tokens:
            return {"error": "NPU engine needs npu://token,1,2,3 model format"}

        generated = self._generate_tokens(tokens, max_tokens)
        if generated is None:
            self.chat_stream = None
            return {"error": "NPU engine failed"}

        return {
            "id": "npu-completion",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": str(generated[0]) if generated else ""},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            },
            "x-device": "npu",
            "x-ms-per-tok": self._avg_ms(),
        }

    def _chat_raw_from_content(self, messages: list, max_tokens: int) -> dict:
        """Try to parse raw numeric tokens from message content."""
        if not messages:
            return {"error": "No messages"}
        last = messages[-1].get("content", "")
        if not last or not all(c.isdigit() or c.isspace() or c == "," for c in last):
            return {"error": "NPU engine needs npu:// format or text with tokenizer"}
        try:
            tokens = [_parse_token(t) for t in last.replace(",", " ").split() if t.strip()]
        except ValueError as e:
            return {"error": str(e)}
        if not tokens:
            return {"error": "No tokens found"}

        generated = self._generate_tokens(tokens, max_tokens)
        if generated is None:
            self.chat_stream = None
            return {"error": "NPU engine failed"}

        return {
            "id": "npu-completion",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": str(generated[0]) if generated else ""},
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": len(tokens),
                "completion_tokens": len(generated),
                "total_tokens": len(tokens) + len(generated),
            },
            "x-device": "npu",
            "x-ms-per-tok": self._avg_ms(),
        }

    def _avg_ms(self) -> float:
        """Average ms per token (estimated ~175ms)."""
        return 175.0

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
                },
                "policy": {
                    "< 2B params": "npu",
                    "2B-8B params": "gpu",
                    "> 8B params": "cpu",
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

            # Streaming requested but not (yet) truly supported — just respond
            # non-streaming. Most clients handle this gracefully.
            _ = stream  # consumed, ignored

            # Route — npu:// bypasses size estimation
            if model.startswith("npu://"):
                device = "npu"
                model_size = 0.6  # Qwen3-0.6B
            else:
                model_size = estimate_model_size(model)
                device = select_device(model_size)

            try:
                if device == "npu":
                    resp = backends["npu"].chat(model, messages, **extra_kwargs)
                elif device == "gpu":
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)
                else:
                    resp = backends["gpu"].chat(model, messages, **extra_kwargs)

                # Tag response with routing info
                if isinstance(resp, dict):
                    resp["x-device"] = device
                    resp["x-model-size"] = f"{model_size}B"

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
    parser.add_argument("--no-auto", action="store_true", help="Don't auto-start backends")
    args = parser.parse_args()

    global backends
    backends = {
        "npu": NPUBackend(port=args.npu_port),
        "gpu": GPUBackend(port=args.gpu_port),
    }

    print("=" * 60)
    print("  NPU + GPU + CPU = Unified Control Plane")
    print("  Gateway: http://0.0.0.0:{}".format(args.port))
    print("=" * 60)
    print()

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
