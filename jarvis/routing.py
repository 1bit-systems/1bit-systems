import json
from urllib.request import Request, urlopen
from urllib.error import URLError

NPU_URL = "http://127.0.0.1:52625"
OLLAMA_URL = "http://127.0.0.1:11434"
# tools/unified_server.cpp — the native engine's own OpenAI-compatible server
# (npu_xrt/hip_gpu/mamba1_gpu/vulkan_gpu/cpu, auto-selected per model, live
# per-request model switching via the standard "model" field). Zero
# proprietary code: no subprocess wraps a closed-source binary here.
UNIFIED_URL = "http://127.0.0.1:8088"

MODEL_ROUTING = {
    "qwen3:0.6b": {"backend": "npu", "flm_model": "qwen3:0.6b"},
    "qwen3:1.7b": {"backend": "npu", "flm_model": "qwen3:1.7b"},
    "qwen3:4b": {"backend": "npu", "flm_model": "qwen3:4b"},
    "bonsai:1.7b": {"backend": "npu", "flm_model": "bonsai:1.7b"},
    "gemma4:e2b": {"backend": "npu", "flm_model": "gemma4:e2b"},
    "phi4-mini:4b": {"backend": "npu", "flm_model": "phi4-mini:4b"},
    "qwen3.6:35b": {"backend": "npu", "flm_model": "qwen3.6:35b"},
    "qwen3vl:4b": {"backend": "npu_vision", "flm_model": "qwen3vl-it:4b"},
    "qwen3-vl:4b": {"backend": "npu_vision", "flm_model": "qwen3vl-it:4b"},
    "qwen3.5:9b": {"backend": "gpu", "ollama_model": "qwen3.5:9b"},
    "llama3.1:8b": {"backend": "gpu", "ollama_model": "llama3.1:8b"},
    "deepseek-r1:8b": {"backend": "gpu", "ollama_model": "deepseek-r1:8b"},
    "qwen2.5:7b": {"backend": "gpu", "ollama_model": "qwen2.5:7b"},
    "mistral:7b": {"backend": "gpu", "ollama_model": "mistral:7b"},
    "gpt-oss:20b": {"backend": "gpu", "ollama_model": "gpt-oss:20b"},
    "llama3.2-vision": {"backend": "gpu", "ollama_model": "llama3.2-vision"},

    # Zyphra family, served locally via tools/unified_server.cpp (models/*.gguf,
    # 1BP conversions available for all of these — see models/catalog/README.md).
    # zr1-1.5b confirmed producing coherent output live 2026-07-21. The
    # zamba2 sizes are real but slower to load/switch (7B: >120s cold), and
    # blackmamba-1.5b/2.8b are EXCLUDED here — their GGUF conversion has no
    # usable embedded tokenizer vocab right now, so unified_server falls back
    # to an ASCII passthrough tokenizer and emits raw token IDs, not text.
    # That's a real bug in the conversion, not something to route chat to
    # until it's fixed (separate from the Mamba1 HIP *inference* path itself,
    # which does work — see PR #584).
    "zr1:1.5b": {"backend": "unified", "unified_model": "ZR1 1.5B"},
    "zamba2:1.2b": {"backend": "unified", "unified_model": "zamba2-1.2b-instruct-v2-q4_0"},
    "zamba2:2.7b": {"backend": "unified", "unified_model": "zamba2-2.7b-instruct-v2-q4_0"},
    "zamba2:7b": {"backend": "unified", "unified_model": "zamba2-7b-instruct-v2-q4_0"},
    "zaya1:8b": {"backend": "unified", "unified_model": "ZAYA1 8B"},
}


def _json_bytes(data):
    return json.dumps(data).encode()


def resolve_model(model_id):
    if model_id in MODEL_ROUTING:
        return MODEL_ROUTING[model_id]
    try:
        req = Request(f"{OLLAMA_URL}/api/tags")
        resp = urlopen(req, timeout=5)
        for m in json.loads(resp.read()).get("models", []):
            if model_id in m["name"]:
                return {"backend": "gpu", "ollama_model": model_id}
    except Exception as e:
        # Ollama is unavailable; fall back to NPU
        pass
    return {"backend": "npu", "flm_model": model_id}


def flm_chat(model, messages, max_tokens=256, temp=0.7):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temp, "stream": False}
    req = Request(f"{NPU_URL}/v1/chat/completions",
                  data=_json_bytes(payload),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=120)
        return json.loads(resp.read())
    except URLError as e:
        return {"error": f"NPU: {e.reason}"}


def unified_chat(model, messages, max_tokens=256, temp=0.7):
    # Live model-switch on the server side means the first request for a
    # not-currently-active model pays a reload cost -- the larger Zyphra
    # sizes (7B) can take well over a minute. Give this a generous timeout
    # rather than fail a cold switch that would otherwise succeed.
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temp, "stream": False}
    req = Request(f"{UNIFIED_URL}/v1/chat/completions",
                  data=_json_bytes(payload),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=180)
        return json.loads(resp.read())
    except URLError as e:
        return {"error": f"unified: {e.reason}"}


def ollama_chat(model, messages, max_tokens=256, temp=0.7):
    # /api/chat (not the legacy /api/generate) — takes the full messages
    # array natively. /api/generate only takes a flat prompt string, which a
    # prior version of this function approximated with messages[-1]["content"]
    # alone, silently dropping every earlier turn (including server-side
    # multi-turn memory prepended by jarvis.server._chat).
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"num_predict": max_tokens, "temperature": temp}}
    req = Request(f"{OLLAMA_URL}/api/chat",
                  data=_json_bytes(payload),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return {"response": data.get("message", {}).get("content", "")}
    except URLError as e:
        return {"error": f"GPU: {e.reason}"}


def ollama_chat_stream(model, messages, max_tokens=256, temp=0.7):
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"num_predict": max_tokens, "temperature": temp}}
    req = Request(f"{OLLAMA_URL}/api/chat",
                  data=_json_bytes(payload),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=300)
        for line in resp:
            line = line.decode().strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)
                if content:
                    yield {"choices": [{"delta": {"content": content}, "index": 0}]}
                if done:
                    break
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Log the error but continue streaming — a single corrupt chunk
                # shouldn't kill the entire stream connection.
                import sys
                print(f"  [routing] stream chunk parse error: {e}", file=sys.stderr)
    except URLError as e:
        yield {"error": str(e)}
