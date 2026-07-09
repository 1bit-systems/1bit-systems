#!/usr/bin/env python3
"""OpenAI-compatible HTTP server wrapping engine_final with logprobs.

Usage: python3 engine_server.py
Listens on :52627, returns logprobs for cascade routing.
"""

import json, subprocess, uuid, math, time
from http.server import HTTPServer, BaseHTTPRequestHandler

ENGINE = "./engine_final"
MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
HOST, PORT = "127.0.0.1", 52627

def run_engine(prompt="Hi", max_tokens=50):
    """Run engine_final --json, yield (token_id, logit)."""
    proc = subprocess.Popen(
        [ENGINE, "--json", "-m", MODEL, "-p", prompt, "-n", str(max_tokens)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        env={"LD_LIBRARY_PATH": "/opt/rocm-7.2.4/lib"}
    )
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                yield d["token"], d.get("logit", 0.0)
            except: pass
    proc.wait()

def logit_to_logprob(logit, vocab_size=151936):
    """Convert a single logit to an approximate logprob.
    
    True softmax requires computing exp for ALL vocab items.
    We estimate by assuming all other logits are near zero:
    logprob ≈ logit - log(exp(logit) + vocab_size - 1)
    For high logits, this is accurate enough for routing decisions.
    """
    if vocab_size <= 1: return 0.0
    max_l = max(logit, 0.0)
    log_sum = logit + math.log(1 + (vocab_size - 1) * math.exp(-max_l))
    return logit - log_sum


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"object":"list","data":[{"id":"qwen3:0.6b","object":"model","created":int(time.time()),"owned_by":"engine_final"}]})
        elif self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
            prompt = " ".join(m["content"] for m in body.get("messages",[]) if "content" in m) or "Hi"
            max_tokens = body.get("max_tokens", 50)
            need_logprobs = body.get("logprobs", False)
            
            tokens = []
            for tok_id, logit in run_engine(prompt, max_tokens):
                tokens.append((tok_id, logit))
            
            text = " ".join(str(t[0]) for t in tokens)
            choice = {"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}
            
            if need_logprobs and tokens:
                last_id, last_logit = tokens[-1]
                lp = logit_to_logprob(last_logit)
                choice["logprobs"] = {"content":[{"token":str(last_id),"logprob":lp,"bytes":None}]}
            
            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:16]}","object":"chat.completion",
                "created": int(time.time()),"model":"qwen3:0.6b","choices":[choice],
                "usage":{"prompt_tokens":len(prompt)//4,"completion_tokens":len(tokens),"total_tokens":len(prompt)//4+len(tokens)}
            }
            self.send_json(resp)
        else:
            self.send_error(404)
    
    def send_json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())
    
    def send_error(self, code):
        self.send_response(code); self.end_headers()

if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Engine server on {HOST}:{PORT}")
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
