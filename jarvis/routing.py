import json
from urllib.request import Request, urlopen
from urllib.error import URLError

NPU_URL = "http://127.0.0.1:52625"
OLLAMA_URL = "http://127.0.0.1:11434"

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
    except:
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


def ollama_chat(model, messages, max_tokens=256, temp=0.7):
    prompt = messages[-1]["content"] if messages else ""
    system = None
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"num_predict": max_tokens, "temperature": temp}}
    if system:
        payload["system"] = system
    req = Request(f"{OLLAMA_URL}/api/generate",
                  data=_json_bytes(payload),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urlopen(req, timeout=300)
        return json.loads(resp.read())
    except URLError as e:
        return {"error": f"GPU: {e.reason}"}


def ollama_chat_stream(model, messages, max_tokens=256, temp=0.7):
    prompt = messages[-1]["content"] if messages else ""
    system = None
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": {"num_predict": max_tokens, "temperature": temp}}
    if system:
        payload["system"] = system
    req = Request(f"{OLLAMA_URL}/api/generate",
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
                content = data.get("response", "")
                done = data.get("done", False)
                if content:
                    yield {"choices": [{"delta": {"content": content}, "index": 0}]}
                if done:
                    break
            except:
                pass
    except URLError as e:
        yield {"error": str(e)}
