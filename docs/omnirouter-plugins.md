# Lemonade OmniRouter Plugins

`1bit omni` supports caller-side plugins around Lemonade OmniRouter. A plugin adds OpenAI tool definitions plus a small executor that calls Lemonade endpoints on `http://127.0.0.1:13305/v1`.

This is not a new runtime service. Plugins live on the client/operator side, so Rule A remains intact: Lemonade owns OmniRouter, FastFlowLM owns the NPU lane, and `1bit-proxy` remains a union endpoint for clients that want model routing.

## Plugin Search Path

`1bit-omni.py` loads `*.json` plugin manifests from:

```text
scripts/omni-plugins/
/usr/share/1bit-systems/omni-plugins/
/usr/local/share/1bit-systems/omni-plugins/
```

Override with:

```bash
ONEBIT_OMNI_PLUGIN_DIRS=/path/to/plugins:/another/path 1bit omni "List OmniRouter models"
```

## Manifest Shape

```json
{
  "name": "lemonade-discovery",
  "version": "0.1.0",
  "description": "Caller-side Lemonade OmniRouter discovery tools.",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "list_omni_models",
        "description": "List Lemonade models, including Collections and labels.",
        "parameters": {
          "type": "object",
          "properties": {},
          "additionalProperties": false
        }
      }
    }
  ],
  "handlers": {
    "list_omni_models": {
      "kind": "http_get",
      "endpoint": "/models?show_all=true",
      "max_chars": 6000
    }
  }
}
```

The `tools` array is standard OpenAI tool-calling JSON. The `handlers` map tells `1bit omni` how to execute a selected tool.

Supported handler kinds:

| Kind | Behavior |
|---|---|
| `http_get` | `GET {LEMONADE_BASE_URL}{endpoint}` |
| `http_json` | `POST {LEMONADE_BASE_URL}{endpoint}` with a JSON body |

For `http_json`, the `body` object can use literals, arguments, or environment defaults:

```json
{
  "kind": "http_json",
  "endpoint": "/some/lemonade/endpoint",
  "body": {
    "model": {"env": "ONEBIT_OMNI_IMAGE", "default": "SD-Turbo"},
    "prompt": {"arg": "prompt"}
  }
}
```

## Seed Plugin

The repo ships `scripts/omni-plugins/lemonade-discovery.json`. It adds `list_omni_models`, which calls:

```text
GET /v1/models?show_all=true
```

That is the Lemonade OmniRouter discovery path for Collections and labels. It lets an agent inspect which loaded models satisfy tool labels such as `image`, `edit`, `tts`, `speech`, `audio`, `transcription`, and `vision`.

## Boundary

Use plugins for caller-side orchestration around Lemonade OmniRouter:

- discovery helpers
- tool bundles
- workflow-specific modality calls
- model label checks

Do not use plugins to move OmniRouter into `1bit-proxy`. The proxy can forward ordinary OpenAI-compatible requests, but Lemonade remains the canonical OmniRouter server.
