# App Integration

`1bit-systems` is an inference engine first. The single control plane comes second.

Apps connect to the engine over OpenAI-compatible base URLs. The intended
control plane will start, monitor, and route inference services; today the
working repair surface is `1bit-proxy` plus a toolbox-backed backend.

## Base URLs

| Use case | Base URL |
|---|---|
| Full 1bit engine for generic OpenAI-compatible apps | `http://127.0.0.1:13306/v1` |
| Full 1bit engine for GAIA / Lemonade-style clients | `http://127.0.0.1:13306/api/v1` |
| Active backend direct, toolbox llama.cpp or Lemonade | `http://127.0.0.1:13305/v1` |
| Lemonade direct, native multimodal and OmniRouter | `http://127.0.0.1:13305/api/v1` |
| FastFlowLM direct NPU runtime, optional | `http://127.0.0.1:52625/v1` |

If a client requires an API key, use `local-no-auth` unless you explicitly configured Lemonade authentication.

## Recommended Client Settings

| App | Provider / mode | Base URL |
|---|---|---|
| GAIA Agent UI / CLI | Lemonade / OpenAI-compatible | `http://127.0.0.1:13306/api/v1` |
| Open WebUI | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| AnythingLLM | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Continue | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Dify / n8n / custom tools | OpenAI-compatible | `http://127.0.0.1:13306/v1` |
| Direct backend tests | toolbox llama.cpp or Lemonade/OpenAI-compatible | `http://127.0.0.1:13305/v1` or `http://127.0.0.1:13305/api/v1` |

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

The union endpoint keeps the app URL stable while the active backend changes.
During repair, the default backend can be toolbox `llama-server` on `:13305`.
On the native path, Lemonade owns multimodal OpenAI compatibility and
OmniRouter behavior. The proxy can divert targeted FastFlowLM model families to
the NPU lane, such as FLM chat models and `embed-*` embeddings, when FLM is
enabled.

```text
App
  -> 1bit-proxy :13306
      -> toolbox llama-server or Lemonade :13305
      -> optional FastFlowLM :52625
```

For pure OmniRouter tool workflows on the native path, use Lemonade direct on
`http://127.0.0.1:13305/v1`. OmniRouter tool definitions target Lemonade
endpoints such as `/v1/images/generations`, `/v1/images/edits`,
`/v1/audio/speech`, `/v1/audio/transcriptions`, and `/v1/chat/completions` for
vision.

## Security

These endpoints are local developer services by default. Do not expose them directly to the internet. Put authentication, TLS, and an explicit reverse proxy policy in front of any remote access.

## References

- Lemonade docs: https://lemonade-server.ai/docs/
- Lemonade API overview: https://lemonade-server.ai/docs/api/
- Lemonade OmniRouter docs: https://lemonade-server.ai/docs/omni-router/
- Lemonade server configuration: https://lemonade-server.ai/docs/server/configuration/
