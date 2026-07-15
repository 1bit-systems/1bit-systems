#!/usr/bin/env python3
import json, os, re, subprocess, sys, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from jarvis.rag import _kb
from jarvis.routing import resolve_model, flm_chat, ollama_chat, ollama_chat_stream, MODEL_ROUTING
from jarvis.stt import transcribe_audio
from jarvis.tts import synthesize_speech
from jarvis.ui import CHAT_HTML

NPU_URL = os.environ.get("NPU_URL", "http://127.0.0.1:52625")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_PORT = int(os.environ.get("JARVIS_PORT", "8080"))


def _j(d): return json.dumps(d).encode()
def _sl(d): return b"data: " + json.dumps(d).encode() + b"\n\n"
def _sd(): return b"data: [DONE]\n\n"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        sys.stderr.write(f"[J] {a[0]} {a[1]} {a[2]}\n")

    def _s(self, s, h, b=None):
        self.send_response(s)
        for k, v in h.items(): self.send_header(k, v)
        self.end_headers()
        if b: self.wfile.write(b)

    def _j(self, s, d): self._s(s, {"Content-Type": "application/json"}, _j(d))
    def _h(self, s, c): self._s(s, {"Content-Type": "text/html; charset=utf-8"}, c.encode())

    def do_GET(self):
        if self.path in ("/", "/chat"): return self._h(200, CHAT_HTML)
        if self.path == "/health": return self._j(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._j(200, {"data": [{"id": m, "object": "model", "backend": c["backend"]} for m, c in MODEL_ROUTING.items()]})
        if self.path == "/v1/knowledge":
            entries = []
            for f in _kb.all_files():
                try:
                    t = f.read_text()
                    ti = ""
                    tags = []
                    for l in t.split("\n"):
                        if l.startswith("# "): ti = l[2:].strip()
                        if l.startswith("tags:"): tags = [x.strip().strip("'") for x in l[5:].strip().strip("[]").split(",")]
                    r = str(f.relative_to(_kb.root))
                    entries.append({"path": r, "title": ti or f.name, "category": r.split("/")[0], "tags": tags, "size": len(t)})
                except: pass
            return self._j(200, {"entries": entries})
        self._j(404, {"error": "nf"})

    def do_POST(self):
        p = self.path
        if p == "/v1/audio/transcriptions": return self._stt()
        if p == "/v1/audio/speech": return self._tts()
        if p in ("/v1/chat/completions", "/api/chat"): return self._chat()
        if p == "/v1/knowledge/search": return self._kbs()
        if p == "/v1/knowledge/upload": return self._kbu()
        self._j(404, {"error": "nf"})

    def _chat(self):
        l = int(self.headers.get("Content-Length", 0))
        b = self.rfile.read(l) if l else b"{}"
        try: d = json.loads(b)
        except: return self._j(400, {"error": "json"})
        mid = d.get("model")
        msgs = d.get("messages", [])
        stream = d.get("stream", False)
        mt = d.get("max_tokens", 256)
        temp = d.get("temperature", 0.7)
        rag = d.get("rag", True)

        if not mid or mid == "auto":
            has_img = any(isinstance(m.get("content"), list) for m in msgs)
            if has_img: mid = "qwen3vl:4b"
            else:
                ln = sum(len(m.get("content", "")) for m in msgs if isinstance(m.get("content"), str))
                mid = "qwen3:0.6b" if ln < 500 else "qwen3.5:9b"

        if rag:
            q = ""
            for m in msgs:
                if isinstance(m.get("content"), str) and m["role"] == "user": q = m["content"]
            if q:
                ctx = _kb.get_knowledge_context(q)
                if ctx:
                    for i, m in enumerate(msgs):
                        if m["role"] == "user" and isinstance(m.get("content"), str):
                            msgs[i]["content"] = ctx + "\n\nQ: " + m["content"]
                            break

        r = resolve_model(mid)
        bkd = r.get("backend", "npu")
        if bkd == "npu_vision": return self._vis(r, msgs, stream, mt, temp)
        if bkd == "gpu": return self._gpu(r["ollama_model"], msgs, stream, mt, temp)
        return self._npu(r.get("flm_model", mid), msgs, stream, mt, temp)

    def _npu(self, m, msgs, s, mt, t):
        r = flm_chat(m, msgs, mt, t)
        if "error" in r: return self._j(502, r)
        if s:
            c = r.get("choices", [{}])[0].get("message", {}).get("content", "")
            self._s(200, {"Content-Type": "text/event-stream"})
            self.wfile.write(_sl({"choices": [{"delta": {"content": c}, "index": 0}]}))
            self.wfile.write(_sd())
        else: self._j(200, r)

    def _vis(self, route, msgs, s, mt, t):
        fm = route.get("flm_model", "qwen3vl-it:4b")
        txt = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p)
                         for m in msgs for p in (m.get("content") if isinstance(m.get("content"), list) else [m.get("content", "")]))
        if not txt: txt = "Describe this image."
        r = flm_chat(fm, [{"role": m["role"], "content": txt} for m in msgs], mt, t)
        if "error" in r: return self._j(502, r)
        if s:
            c = r.get("choices", [{}])[0].get("message", {}).get("content", "")
            self._s(200, {"Content-Type": "text/event-stream"})
            self.wfile.write(_sl({"choices": [{"delta": {"content": c}, "index": 0}]}))
            self.wfile.write(_sd())
        else: self._j(200, r)

    def _gpu(self, m, msgs, s, mt, t):
        if s:
            self._s(200, {"Content-Type": "text/event-stream"})
            for c in ollama_chat_stream(m, msgs, mt, t):
                self.wfile.write(_sl(c))
                self.wfile.flush()
            self.wfile.write(_sd())
        else:
            r = ollama_chat(m, msgs, mt, t)
            if "error" in r: return self._j(502, r)
            self._j(200, {"id": f"c-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "model": m,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": r.get("response", "")}, "finish_reason": "stop"}]})

    def _stt(self):
        ct = self.headers.get("Content-Type", "")
        bd = ct.split("boundary=")[-1].strip()
        l = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(l)
        audio = None
        for p in body.split(b"--" + bd.encode()):
            if b"filename" in p:
                i = p.find(b"\r\n\r\n")
                if i > 0: audio = p[i+4:].rstrip(b"\r\n--"); break
        if not audio: return self._j(400, {"error": "no audio"})
        wav = audio
        if audio[:4] != b"RIFF":
            try:
                p = subprocess.run(["ffmpeg", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"], input=audio, capture_output=True, timeout=30)
                if p.returncode == 0: wav = p.stdout
            except: pass
        return self._j(200, {"text": transcribe_audio(wav)})

    def _tts(self):
        l = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(l) if l else b"{}"
        try: d = json.loads(body)
        except: return self._j(400, {"error": "json"})
        w = synthesize_speech(d.get("input", ""), d.get("voice", "en_US-lessac-medium"))
        if not w: return self._j(500, {"error": "tts failed"})
        self._s(200, {"Content-Type": "audio/wav"}, w)

    def _kbs(self):
        l = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(l) if l else b"{}"
        try: d = json.loads(body)
        except: return self._j(400, {"error": "json"})
        return self._j(200, {"results": _kb.search(d.get("query", ""), d.get("max_results", 5))})

    def _kbu(self):
        ct = self.headers.get("Content-Type", "")
        if "multipart" in ct:
            bd = ct.split("boundary=")[-1].strip()
            l = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(l)
            fn = "doc.txt"; content = b""
            for p in body.split(b"--" + bd.encode()):
                if b"Content-Disposition" in p:
                    m = re.search(r'filename="([^"]+)"', p.decode(errors="ignore"))
                    if m: fn = m.group(1)
                    i = p.find(b"\r\n\r\n")
                    if i > 0: content = p[i+4:].rstrip(b"\r\n--"); break
            if not content: return self._j(400, {"error": "empty"})
            return self._j(200, {"path": _kb.add_document(fn, content.decode("utf-8", errors="replace"))})
        l = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(l) if l else b"{}"
        try: d = json.loads(body)
        except: return self._j(400, {"error": "json"})
        return self._j(200, {"path": _kb.add_document(d.get("filename", "note.md"), d.get("content", ""))})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = p.parse_args()
    s = HTTPServer(("0.0.0.0", a.port), H)
    print(f"JARVIS @ http://localhost:{a.port}/chat")
    try: s.serve_forever()
    except KeyboardInterrupt: s.server_close()
