#!/usr/bin/env python3
"""
1bit omni — opinionated OmniRouter loop for the 1bit-systems stack.

Wraps the OpenAI tool-calling pattern that AMD's Lemonade team
documented on 2026-04-28 (lemonade-sdk/lemonade/examples/lemonade_tools.py),
but with our tier-priority defaults baked in:

  LLM-in-the-loop  =  Qwen3.5-35B-A3B-UD-IQ2_XXS  (Tier 2 daily driver, sub-2-bit)
  Image gen        =  SD-Turbo                      (sd-cpp:rocm lane)
  TTS              =  kokoro-v1                     (kokoro:cpu lane)
  STT              =  Whisper-Tiny                  (whispercpp:vulkan lane)

Override any of these with the matching env var:
  ONEBIT_OMNI_LLM, ONEBIT_OMNI_IMAGE, ONEBIT_OMNI_TTS, ONEBIT_OMNI_STT.
Add caller-side Lemonade OmniRouter plugins with ONEBIT_OMNI_PLUGIN_DIRS.

Endpoint follows LEMONADE_BASE_URL if set, else http://127.0.0.1:13305/v1.

Usage:
  1bit-omni.py "Generate an image of a sunset"
  1bit-omni.py "Say 'hello world' out loud"
  1bit-omni.py "Transcribe /path/to/audio.wav"
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

LEMONADE_URL = os.environ.get("LEMONADE_BASE_URL", "http://127.0.0.1:13305/v1").rstrip("/")
LLM = os.environ.get("ONEBIT_OMNI_LLM", "user.Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS")
IMG = os.environ.get("ONEBIT_OMNI_IMAGE", "SD-Turbo")
TTS = os.environ.get("ONEBIT_OMNI_TTS", "kokoro-v1")
STT = os.environ.get("ONEBIT_OMNI_STT", "Whisper-Tiny")

OUT_DIR = Path(os.environ.get("ONEBIT_OMNI_OUT", "/tmp/1bit-omni"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLUGIN_DIRS = [
    Path(p).expanduser()
    for p in os.environ.get(
        "ONEBIT_OMNI_PLUGIN_DIRS",
        ":".join(
            [
                str(Path(__file__).resolve().parent / "omni-plugins"),
                "/usr/share/1bit-systems/omni-plugins",
                "/usr/local/share/1bit-systems/omni-plugins",
            ]
        ),
    ).split(":")
    if p
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text description.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "Detailed visual description"}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "Convert text into a spoken audio WAV.",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "Text to speak"}},
                "required": ["input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe a local audio file (.wav, .mp3) to text.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute path to the audio file"}},
                "required": ["path"],
            },
        },
    },
]

SYSTEM = (
    "You are a 1bit-systems assistant on local Strix Halo hardware. "
    "Use the tools when the user asks for an image, TTS, or transcription. "
    "Otherwise reply directly. Keep replies short. /no_think"
)


def http_post(path: str, body: bytes, headers: dict, timeout: int = 600) -> bytes:
    req = urllib.request.Request(f"{LEMONADE_URL}{path}", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_get(path: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(f"{LEMONADE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def http_post_multipart(path: str, fields: dict, files: dict, timeout: int = 120) -> bytes:
    """Bare-bones multipart/form-data using urllib (no `requests` dep)."""
    boundary = "----1bitomni" + os.urandom(8).hex()
    parts: list[bytes] = []
    for k, v in fields.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
            v.encode() if isinstance(v, str) else v,
            b"\r\n",
        ]
    for k, (fname, data) in files.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            data,
            b"\r\n",
        ]
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return http_post(path, body, headers, timeout=timeout)


def tool_generate_image(prompt: str) -> str:
    body = json.dumps({"model": IMG, "prompt": prompt, "n": 1, "size": "512x512"}).encode()
    raw = http_post("/images/generations", body, {"Content-Type": "application/json"})
    d = json.loads(raw)
    item = d["data"][0]
    out = OUT_DIR / "image.png"
    if "b64_json" in item and item["b64_json"]:
        out.write_bytes(base64.b64decode(item["b64_json"]))
        return f"Wrote {out} ({out.stat().st_size:,} bytes)"
    if "url" in item and item["url"]:
        return f"Image URL: {item['url']}"
    return "(image gen returned no content)"


def tool_text_to_speech(text: str) -> str:
    body = json.dumps({"model": TTS, "input": text, "response_format": "wav"}).encode()
    raw = http_post("/audio/speech", body, {"Content-Type": "application/json"})
    out = OUT_DIR / "tts.wav"
    out.write_bytes(raw)
    return f"Wrote {out} ({out.stat().st_size:,} bytes)"


def tool_transcribe_audio(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Audio file not found: {path}"
    raw = http_post_multipart(
        "/audio/transcriptions",
        fields={"model": STT},
        files={"file": (p.name, p.read_bytes())},
    )
    d = json.loads(raw)
    return d.get("text", "(no text)").strip()


DISPATCH = {
    "generate_image": lambda args: tool_generate_image(args.get("prompt", "")),
    "text_to_speech": lambda args: tool_text_to_speech(args.get("input", "")),
    "transcribe_audio": lambda args: tool_transcribe_audio(args.get("path", "")),
}


def build_plugin_body(spec: dict, args: dict) -> dict:
    out = {}
    for key, source in (spec or {}).items():
        if isinstance(source, dict) and "arg" in source:
            out[key] = args.get(source["arg"], source.get("default"))
        elif isinstance(source, dict) and "env" in source:
            out[key] = os.environ.get(source["env"], source.get("default", ""))
        else:
            out[key] = source
    return out


def execute_plugin_tool(name: str, args: dict, handler: dict) -> str:
    kind = handler.get("kind")
    endpoint = handler.get("endpoint")
    if not endpoint:
        return f"plugin tool {name} has no endpoint"

    if kind == "http_get":
        raw = http_get(endpoint)
    elif kind == "http_json":
        body = build_plugin_body(handler.get("body", {}), args)
        raw = http_post(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    else:
        return f"plugin tool {name} has unsupported kind: {kind}"

    text = raw.decode(errors="replace")
    max_chars = int(handler.get("max_chars", 4000))
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... truncated {len(text) - max_chars} chars"
    return text


def load_omni_plugins() -> None:
    seen = {tool["function"]["name"] for tool in TOOLS}
    for plugin_dir in PLUGIN_DIRS:
        if not plugin_dir.is_dir():
            continue
        for path in sorted(plugin_dir.glob("*.json")):
            try:
                plugin = json.loads(path.read_text())
            except Exception as e:
                print(f"warning: failed to load OmniRouter plugin {path}: {e}", file=sys.stderr)
                continue

            handlers = plugin.get("handlers", {})
            for tool in plugin.get("tools", []):
                fn = tool.get("function", {})
                name = fn.get("name")
                if not name or name in seen:
                    continue
                handler = handlers.get(name)
                if not isinstance(handler, dict):
                    print(f"warning: plugin {path} tool {name} has no handler; skipping", file=sys.stderr)
                    continue
                TOOLS.append(tool)
                DISPATCH[name] = lambda args, n=name, h=handler: execute_plugin_tool(n, args, h)
                seen.add(name)


def chat_completion(messages: list, tools: list | None = None) -> dict:
    body = {"model": LLM, "messages": messages, "max_tokens": 600, "temperature": 0}
    if tools:
        body["tools"] = tools
    raw = http_post("/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(raw)


def banner():
    sys.stderr.write(f"\033[2m1bit omni · LLM={LLM} · img={IMG} · tts={TTS} · stt={STT} · base={LEMONADE_URL}\033[0m\n")


def run(prompt: str) -> int:
    load_omni_plugins()
    banner()
    print(f"\033[1;36m›\033[0m {prompt}")

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    resp = chat_completion(messages, tools=TOOLS)
    msg = resp["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []

    if not tool_calls:
        content = msg.get("content") or "(empty reply)"
        print(f"\033[1;32m✓\033[0m {content}")
        return 0

    # Append the assistant message + each tool result; one more pass for the final reply
    messages.append({
        "role": "assistant",
        "content": msg.get("content") or "",
        "tool_calls": tool_calls,
    })

    for tc in tool_calls:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"] or "{}")
        print(f"\033[1;33m⚙\033[0m tool: \033[1m{name}\033[0m({json.dumps(args)})")
        handler = DISPATCH.get(name)
        if handler is None:
            result = f"unknown tool: {name}"
        else:
            try:
                result = handler(args)
            except urllib.error.HTTPError as e:
                result = f"HTTP {e.code}: {e.read().decode()[:200]}"
            except Exception as e:
                result = f"tool error: {e}"
        print(f"\033[2m   → {result}\033[0m")
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": name,
            "content": result,
        })

    final = chat_completion(messages, tools=None)
    final_content = final["choices"][0]["message"].get("content") or ""
    print(f"\033[1;32m✓\033[0m {final_content}")
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    prompt = " ".join(sys.argv[1:])
    return run(prompt)


if __name__ == "__main__":
    sys.exit(main())
