# Connecting clients

The current client contract is OpenAI-compatible HTTP through the union endpoint:

```text
http://127.0.0.1:13306/v1
http://127.0.0.1:13306/api/v1
```

The proxy keeps Lemonade as the default route and diverts known FastFlowLM model families to the XDNA lane.

Direct endpoints remain available for debugging:

| Surface | URL | Use |
|---|---|---|
| 1bit union endpoint | `http://127.0.0.1:13306/v1` | Default for apps |
| Lemonade direct | `http://127.0.0.1:13305/api/v1` or `/v1` | Multimodal / OmniRouter direct path |
| FastFlowLM direct | `http://127.0.0.1:52625/v1` | NPU chat / embeddings lane |
| Open WebUI | `http://127.0.0.1:3000` | Secondary browser UI |

If a client requires an API key, use `local-no-auth` unless authentication was explicitly enabled.

## Open WebUI

Open WebUI is a secondary compatibility UI. It is allowed under Rule A only as an isolated caller/UI surface pointed at the proxy; it is not the core runtime.

### Native service path

The supported install wires Open WebUI to:

```text
OPENAI_API_BASE_URL=http://127.0.0.1:13306/v1
OPENAI_API_KEY=local-no-auth
```

Visit `http://localhost:3000` and select any model exposed by `/v1/models`.

### Docker path

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:13306/v1 \
  -e OPENAI_API_KEY=local-no-auth \
  -v openwebui-data:/app/backend/data \
  --name openwebui \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

## GAIA

GAIA is the primary agent/control surface. Point it at the proxy:

```text
http://127.0.0.1:13306/api/v1
```

## Raw HTTP

### List models

```bash
curl -s http://127.0.0.1:13306/v1/models | jq '.data[].id'
```

### One-shot completion

```bash
curl -s http://127.0.0.1:13306/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3:1.7b",
    "messages": [
      {"role":"system","content":"Be concise."},
      {"role":"user","content":"Explain ternary weights in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | jq -r '.choices[0].message.content'
```

### Embeddings

```bash
curl -s http://127.0.0.1:13306/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nomic-embed-text-v2-moe-GGUF",
    "input": "health check"
  }' | jq '.data[0].embedding | length'
```

## SDKs

Any OpenAI SDK works. These examples are caller-side code, so Rule A does not apply to them as runtime services.

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:13306/v1",
    api_key="local-no-auth",
)

response = client.chat.completions.create(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Hello."}],
)

print(response.choices[0].message.content)
```

### TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:13306/v1",
  apiKey: "local-no-auth",
});

const response = await client.chat.completions.create({
  model: "qwen3:1.7b",
  messages: [{ role: "user", content: "Hello." }],
});

console.log(response.choices[0]?.message?.content ?? "");
```

### Rust

```rust
use async_openai::{config::OpenAIConfig, types::*, Client};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OpenAIConfig::new()
        .with_api_base("http://localhost:13306/v1")
        .with_api_key("local-no-auth");
    let client = Client::with_config(config);

    let req = CreateChatCompletionRequestArgs::default()
        .model("qwen3:1.7b")
        .messages([ChatCompletionRequestUserMessageArgs::default()
            .content("Hello.")
            .build()?
            .into()])
        .build()?;

    let response = client.chat().create(req).await?;
    println!("{}", response.choices[0].message.content.as_deref().unwrap_or(""));
    Ok(())
}
```
