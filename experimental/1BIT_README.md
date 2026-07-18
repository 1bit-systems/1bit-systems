# bit1-mlx — Extracted Rapid-MLX Components for 1bit.systems

> Extracted pure-Python API layer, tool parsers, middleware, and model registry from [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX). Zero MLX dependency.

## 📦 What's Inside

| Module | Description | Lines |
|--------|-------------|-------|
| `bit1_mlx/model_profile.py` | Single frozen dataclass for per-model metadata | 250 |
| `bit1_mlx/model_aliases.py` | 163-model alias registry with fuzzy suggestion | 280 |
| `bit1_mlx/aliases.json` | 163 model alias entries (HF paths + profiles) | 48 KB |
| `bit1_mlx/api/models.py` | Full OpenAI-compatible Pydantic models | 137 KB |
| `bit1_mlx/api/anthropic_adapter.py` | Anthropic ↔ OpenAI message conversion | 38 KB |
| `bit1_mlx/api/tool_calling.py` | Tool call building, JSON schema extraction | 46 KB |
| `bit1_mlx/api/utils.py` | Content parsing, streaming routers, think tags | 61 KB |
| `bit1_mlx/api/guided.py` | Structured JSON generation via outlines | 26 KB |
| `bit1_mlx/api/responses_adapter.py` | OpenAI Responses API adapter (Codex CLI) | 52 KB |
| `bit1_mlx/tool_parsers/` | **17 parsers** (45 registration variants) | 290 KB |
| `bit1_mlx/middleware/` | Auth, body-size, body-depth, CORS, exception handlers | 60 KB |
| `bit1_mlx/engine/base.py` | `BaseEngine` ABC — clean interface contract | 20 KB |
| `bit1_mlx/cache/protocol.py` | KV cache export/import protocol | 33 KB |
| `bit1_mlx/output_router.py` | Token-level channel routing | 34 KB |
| `bit1_mlx/output_router_harmony.py` | Harmony protocol streaming parser | 31 KB |

## 🚀 Quick Start

```python
# Adapter — speak both protocols
from bit1_mlx.api.anthropic_adapter import anthropic_to_openai
from bit1_mlx.api.anthropic_models import AnthropicRequest

anthropic_req = AnthropicRequest(
    model="my-model",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
openai_req = anthropic_to_openai(anthropic_req)
```

```python
# Tool parsers — parse model-native tool call formats
from bit1_mlx.tool_parsers import ToolParserManager

tpm = ToolParserManager()
# Get a parser for any model family
parser_cls = tpm.get_tool_parser("hermes")  # or mistral, llama, qwen, deepseek, etc.
parser = parser_cls(tokenizer)
result = parser.extract_tool_calls(model_output)
```

```python
# Model registry — 163 aliases with fuzzy matching
from bit1_mlx.model_aliases import resolve_model, suggest_similar

hf_path = resolve_model("qwen3.5-4b-4bit")
# → "mlx-community/Qwen3.5-4B-MLX-4bit"

suggestions = suggest_similar("gemma4-27b", n=3)
# → ['gemma3-27b-4bit', 'gemma-4-e2b-4bit', 'gemma-4-12b-4bit']
```

```python
# OpenAI-compatible requests
from bit1_mlx.api.models import ChatCompletionRequest, ToolCall, FunctionCall

req = ChatCompletionRequest(
    model="my-model",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
    }]
)
```

## 🔧 Architecture

```
bit1_mlx/
├── model_profile.py      # Frozen dataclass: parser, hybrid, MoE, modality specs
├── model_aliases.py      # 163 alias → profile resolution + fuzzy suggestion
├── aliases.json          # The alias database
├── api/
│   ├── models.py         # Pydantic models (OpenAI + Anthropic wire format)
│   ├── anthropic_adapter.py  # Anthropic ↔ OpenAI conversion
│   ├── tool_calling.py   # Tool call build/parse helpers
│   ├── utils.py          # Content cleaning, think tag routing
│   └── ...
├── tool_parsers/         # 17 model-native tool-call format parsers
├── middleware/            # ASGI middleware (auth, body limits, CORS)
├── engine/
│   └── base.py           # Abstract engine interface
├── cache/
│   └── protocol.py       # KV cache export/import
└── config/               # ServerConfig singleton
```

## 📝 License

Apache 2.0 — compatible with the original Rapid-MLX license.
