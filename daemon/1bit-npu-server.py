#!/usr/bin/env python3
"""1bit NPU Chat Server — Zero FLM. Uses npu_engine_universal worker for GEMM.

Architecture:
  - Spawns npu_engine_universal --worker for NPU GEMM ops
  - Loads embed + norm weights directly from model.q4nx
  - CPU handles: RoPE, attention, KV cache, RMSNorm, SiLU, sampling
  - NPU handles: QKV, O, GU, D GEMM projections
  - Serves Ollama-compatible HTTP API on port 9090

Usage:
  sudo ./1bit-npu-server.py --port 9090
  curl http://localhost:9090/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"hi"}],"stream":false}'

Requires: NPU_MODEL_PATH, NPU_TOKENIZER_PATH, pip install tokenizers
"""

import argparse, json, os, socket, struct, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np

# Content-Length limit to prevent DoS via oversized POST bodies
MAX_BODY_SIZE = 16 * 1024 * 1024      # 16 MB

# ── Config ──
ENGINE = os.environ.get("NPU_ENGINE_BIN",
    str(Path.home() / "1bit-systems" / "engine" / "npu" / "build" / "npu_engine_universal"))
MODEL = os.environ["NPU_MODEL_PATH"]
TOKENIZER = os.environ["NPU_TOKENIZER_PATH"]
XCLBIN = os.environ.get("NPU_XCLBIN_DIR",
    str(Path.home() / "1bit-systems" / "engine" / "npu" / "xclbins"))

EPS = 1e-6

def bf16_to_f32(v):
    """Convert uint16 bfloat16 to float32."""
    return struct.unpack('<f', struct.pack('<I', v << 16))[0]

class Q4NXReader:
    """Read weights from Q4NX format model file."""
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()
        hsz = struct.unpack('<Q', self.data[:8])[0]
        self.header = self.data[8:8+hsz].decode('utf-8', errors='replace')
        self.data_start = 8 + hsz
        self._parse_offsets()

    def _parse_offsets(self):
        self.offsets = {}
        import re
        for m in re.finditer(r'"([^"]+)":\s*\{[^}]*"data_offsets":\s*\[(\d+),(\d+)\]', self.header):
            name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            self.offsets[name] = (start, end)

    def read_bf16(self, name, count):
        """Read bf16 values as float32 array."""
        if name not in self.offsets:
            return None
        start, end = self.offsets[name]
        raw = self.data[self.data_start+start:self.data_start+end]
        vals = np.frombuffer(raw, dtype=np.uint16)
        # Convert bf16 to f32
        out = np.zeros(len(vals), dtype=np.float32)
        for i, v in enumerate(vals):
            out[i] = bf16_to_f32(v)
        return out[:count]

class NpuWorker:
    """Subprocess worker for NPU GEMM operations."""
    def __init__(self):
        self.proc = None

    def start(self):
        os.environ['NPU_XCLBIN_DIR'] = XCLBIN
        model_tag = os.environ.get('NPU_MODEL_TAG', 'qwen3_0_6b')
        
        # Pass full environment (env=dict() replaces, not appends!)
        proc_env = os.environ.copy()
        proc_env['NPU_XCLBIN_DIR'] = os.environ.get('NPU_XCLBIN_DIR',
            str(Path.home() / '1bit-systems' / 'engine' / 'npu' / 'xclbins'))
        self.proc = subprocess.Popen(
            [ENGINE, MODEL, '--model-tag', model_tag, '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
            env=proc_env)
        
        # Wait for READY on stdout (xclbin loader may print to stdout first)
        import select
        t0 = time.time()
        buf = b''
        while time.time() - t0 < 60:
            r, _, _ = select.select([self.proc.stdout], [], [], 1)
            if r:
                buf += self.proc.stdout.read(1)
                if b'READY\n' in buf:
                    break
            if time.time() - t0 > 60:
                raise RuntimeError(f'Worker not ready after 60s')
        print(f'  Worker ready (PID {self.proc.pid})')
        print(f"  Worker ready (PID {self.proc.pid})")

    def gemm(self, op, layer, batch, in_dim, data):
        """Run a GEMM operation on NPU. data is float32 numpy array."""
        hdr = struct.pack('<4I', op, layer, batch, in_dim)
        self.proc.stdin.write(hdr)
        self.proc.stdin.write(data.tobytes())
        self.proc.stdin.flush()
        resp = self.proc.stdout.read(8)
        status, out_dim = struct.unpack('<2I', resp)
        if status != 0:
            raise RuntimeError(f"GEMM failed: op={op} status={status}")
        out = np.frombuffer(self.proc.stdout.read(out_dim * 4), dtype=np.float32)
        return out.reshape(batch, out_dim // batch)

    def stop(self):
        if self.proc:
            self.proc.stdin.write(struct.pack('<I', 0))
            self.proc.wait(2)

class Inference:
    """Full transformer inference loop."""
    def __init__(self, worker, model):
        self.wk = worker
        self.H = 1024; self.NC = 28; self.NH = 16; self.NKV = 8
        self.HD = 128; self.IM = 3072; self.NV = 151936
        self.GQA = self.NH // self.NKV
        self.QD = self.NH * self.HD
        self.KD = self.NKV * self.HD
        self.max_seq = 4096

        # Load weights
        self.embed = model.read_bf16('model.embed_tokens.weight', self.NV * self.H)
        self.final_norm = model.read_bf16('model.norm.weight', self.H)
        lo = model.read_bf16('lm_head.weight', self.NV * self.H)
        self.lm_emb = lo if lo is not None else self.embed
        self.in_n = [model.read_bf16(f'model.layers.{i}.input_layernorm.weight', self.H) for i in range(self.NC)]
        self.pa_n = [model.read_bf16(f'model.layers.{i}.post_attention_layernorm.weight', self.H) for i in range(self.NC)]

        # KV cache
        self.k_cache = [np.zeros((self.max_seq, self.NKV, self.HD), dtype=np.float32) for _ in range(self.NC)]
        self.v_cache = [np.zeros((self.max_seq, self.NKV, self.HD), dtype=np.float32) for _ in range(self.NC)]
        self.pos = 0

        # RoPE precompute
        self._build_rope()

    def _build_rope(self):
        hd2 = self.HD // 2
        self.cos_c = np.zeros((self.max_seq, self.HD), dtype=np.float32)
        self.sin_c = np.zeros((self.max_seq, self.HD), dtype=np.float32)
        for p in range(self.max_seq):
            for d in range(hd2):
                theta = 1000000.0 ** (-2.0 * d / self.HD)
                a = p * theta
                self.cos_c[p, d] = np.cos(a)
                self.sin_c[p, d] = np.sin(a)
                self.cos_c[p, d + hd2] = np.cos(a)
                self.sin_c[p, d + hd2] = np.sin(a)

    def rope(self, x, p):
        hd2 = self.HD // 2
        out = x.copy()
        out[:, :hd2] = x[:, :hd2] * self.cos_c[p:p+1, :hd2] - x[:, hd2:] * self.sin_c[p:p+1, :hd2]
        out[:, hd2:] = x[:, :hd2] * self.sin_c[p:p+1, :hd2] + x[:, hd2:] * self.cos_c[p:p+1, :hd2]
        return out

    def rmsnorm(self, x, w):
        ss = np.sum(x * x, axis=-1, keepdims=True) / x.shape[-1]
        return x / np.sqrt(ss + EPS) * w

    def attn(self, q, layer, cl):
        """CPU attention: Q@K^T → softmax → @V."""
        NH, NKV, HD, GQA = self.NH, self.NKV, self.HD, self.GQA
        k, v = self.k_cache[layer][:cl], self.v_cache[layer][:cl]
        out = np.zeros((NH, HD), dtype=np.float32)
        for h in range(NH):
            kvh = h // GQA
            sc = q[h] @ k[:, kvh].T  # (cl,)
            sc = sc / np.sqrt(HD)
            sc = np.exp(sc - np.max(sc))
            sc /= np.sum(sc) + 1e-10
            out[h] = sc @ v[:, kvh]  # (cl,) @ (cl, HD)
        return out.reshape(-1)

    def forward(self, token):
        """Process one token, return next token logits."""
        h = self.embed[token].copy()

        for l in range(self.NC):
            sb = h.copy()
            h = self.rmsnorm(h, self.in_n[l])

            # QKV on NPU
            qkv = self.wk.gemm(1, l, 1, self.H, h.reshape(1, -1))[0]
            q = qkv[:self.QD].reshape(self.NH, self.HD)
            k = qkv[self.QD:self.QD+self.KD].reshape(self.NKV, self.HD)
            v = qkv[self.QD+self.KD:].reshape(self.NKV, self.HD)

            # RoPE
            q = self.rope(q, self.pos)
            k = self.rope(k, self.pos)

            # Store KV
            self.k_cache[l][self.pos] = k
            self.v_cache[l][self.pos] = v
            cl = self.pos + 1

            # Attention
            at = self.attn(q, l, cl)

            # O projection on NPU
            oo = self.wk.gemm(2, l, 1, self.QD, at.reshape(1, -1))[0]
            h = sb + oo

            # FFN
            sb = h.copy()
            h = self.rmsnorm(h, self.pa_n[l])
            gu = self.wk.gemm(3, l, 1, self.H, h.reshape(1, -1))[0]
            gate, up = gu[:self.IM], gu[self.IM:]
            gate = gate / (1.0 + np.exp(-gate))  # SiLU
            activated = gate * up
            dw = self.wk.gemm(5, l, 1, self.IM, activated.reshape(1, -1))[0]
            h = sb + dw

        self.pos += 1

        # Final norm + lm_head
        h = h / np.sqrt(np.sum(h*h) / self.H + EPS) * self.final_norm
        logits = h @ self.lm_emb.T
        return logits

    def generate(self, tokens, max_new=256):
        """Generate text from input tokens."""
        self.pos = 0
        result = []
        last = tokens[0] if tokens else 0
        for _ in range(max_new):
            logits = self.forward(last)
            # Greedy sampling
            next_tok = int(np.argmax(logits))
            result.append(next_tok)
            if next_tok == 2:  # EOS
                break
            last = next_tok
        return result

# ── HTTP Handler ──
inf = None
tok = None

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"object":"list","data":[{"id":"default","object":"model"}]})
        elif self.path == "/health":
            self.send_json({"status":"ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/api/generate", "/api/chat"):
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY_SIZE:
            self.send_error(413, "Payload too large")
            return
        body = json.loads(self.rfile.read(length))
        prompt = self._get_prompt(body)
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens", 256)

        # Tokenize
        ids = tok.encode(prompt).ids[:512]
        out_ids = inf.generate(ids, max_tokens)
        text = tok.decode(out_ids)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for word in text.split():
                chunk = {"choices":[{"delta":{"content":word+" "},"index":0}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            resp = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
                "usage": {"prompt_tokens":len(ids),"completion_tokens":len(out_ids),"total_tokens":len(ids)+len(out_ids)}
            }
            self.send_json(resp)

    def _get_prompt(self, body):
        if "messages" in body:
            return "\n".join(m.get("content","") for m in body["messages"])
        return body.get("prompt", body.get("input",""))

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    global tok, inf
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    tok.no_truncation()

    model = Q4NXReader(MODEL)
    print(f"  Model loaded: {Path(MODEL).name}")

    wk = NpuWorker()
    wk.start()

    global inf
    inf = Inference(wk, model)
    print(f"  Inference ready: {inf.NC} layers, {inf.NH} heads, H={inf.H}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n  🚀 1bit NPU chat server: http://127.0.0.1:{args.port}")
    print(f"  Zero FLM dependency — using npu_engine_universal worker\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        wk.stop()
        server.server_close()

if __name__ == "__main__":
    main()
