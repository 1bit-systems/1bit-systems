# App Integration

`1bit-systems` is an inference engine first. The single control plane comes second.

Apps connect to the engine over OpenAI-compatible base URLs. The control plane exists to start, monitor, and route the inference services; it is not the app contract.

## Base URLs

| Use case | Base URL |
|---|---|
| Full 1bit engine for generic OpenAI-compatible apps | `http://127.0.0.1:13306/v1` |
| Full 1bit engine for GAIA / Lemonade-style clients | `http://127.0.0.1:13306/api/v1` |
| Lemonade direct, canonical multimodal and OmniRouter | `http://127.0.0.1:13305/api/v1` |
| Lemonade direct, generic OpenAI-style path | `http://127.0.0.1:13305/v1` |
| FastFlowLM direct NPU runtime | `http://127.0.0.1:52625/v1` |

If a client requires an API key, use `local-no-auth` unless you explicitly configured Lemonade authentication.

## Recommended Client Settings

| App | Provider / mode | Base URL |
|---|---|---|
| GAIA Agent UI / CLI | Lemonade / OpenAI-compatible | `http://127.0.0.1:13306/api/v1` |
| Open WebUI | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| AnythingLLM | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Continue | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Dify / n8n / custom tools | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Direct Lemonade workflows | Lemonade/OpenAI-compatible | `http://127.0.0.1:13305/api/v1` |

## Minimal SDK Test

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:13306/v1",
    api_key="local-no-auth",
)

response = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Say stack OK in five words."}],
    max_tokens=20,
)

print(response.choices[0].message.content)
```

## Routing Rules

The union endpoint keeps Lemonade as the default route because Lemonade owns canonical multimodal OpenAI compatibility and OmniRouter behavior. The proxy only diverts targeted FastFlowLM model families to the NPU lane, such as FLM chat models and `embed-*` embeddings.

```text
App
  -> 1bit-proxy :13306
      -> Lemonade :13305     default, multimodal, OmniRouter
      -> FastFlowLM :52625   NPU chat, embeddings, opt-in ASR
```

## Security

These endpoints are local developer services by default. Do not expose them directly to the internet. Put authentication, TLS, and an explicit reverse proxy policy in front of any remote access.

## References

- Lemonade docs: https://lemonade-server.ai/docs/
- Lemonade API overview: https://lemonade-server.ai/docs/api/
- Lemonade app integration guides: https://lemonade-server.ai/docs/server/apps/
- Lemonade server configuration: https://lemonade-server.ai/docs/server/configuration/
