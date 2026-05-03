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
| `process_json` | Run a local command with a JSON request on stdin and return stdout |

Handlers may also set `extract_json` to a dotted path, such as `choices.0.message.content`, to return only one field from a JSON response.

For `http_json`, the `body` object can use literals, arguments, or environment defaults. Substitution works recursively through nested objects and arrays, so plugins can build normal OpenAI-compatible request bodies:

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

## PA Assistant Plugin

The repo also ships `scripts/omni-plugins/pa-assistant.json`. It adds `ask_pa_assistant`, a local personal-assistant specialist routed through Lemonade chat completions:

```bash
1bit omni "Ask the PA assistant to turn this repo status and meeting note into a 5 item action list: ..."
```

By default it uses the downloaded `Gemma-4-E4B-it-GGUF` model because it returns ordinary assistant content reliably for short PA subcalls. Override it independently with:

```bash
ONEBIT_PA_MODEL='your-local-model-id' 1bit omni "Ask the PA assistant to plan my morning from these notes: ..."
```

The PA assistant plugin is intentionally local and caller-side. It does not read calendars, messages, or files by itself; pass that context into the prompt or pair it with another local tool that supplies context.

The plugin sets `enable_thinking: false` on its Lemonade chat completion request and extracts `choices.0.message.content` from the response before returning the tool result.

## 3D Asset Plugin

`scripts/omni-plugins/3d-assets.json` adds `create_3d_asset`, a caller-side bridge for local 3D asset generation. The default backend target is Hunyuan3D-2.1:

```bash
1bit omni "Create a draft GLB request for a low-poly neon arcade cabinet with worn stickers"
```

The default bridge script is `scripts/omni-tools/asset3d.py`. It creates a durable job folder under `/tmp/1bit-omni/assets/` and writes `request.json`. If `ONEBIT_3D_ASSET_RUNNER` is set, it delegates to that runner and expects the runner to write the generated asset into the job folder.

```bash
ONEBIT_3D_ASSET_RUNNER=/path/to/hunyuan3d-runner \
  1bit omni "Create a GLB: a toy robot with red boots"
```

Backend targets:

| Backend | Default model | Input posture |
|---|---|---|
| `hunyuan3d` | `tencent/Hunyuan3D-2.1` | Text-to-3D or image-to-3D |
| `trellis` | `microsoft/TRELLIS-text-large` | Text-to-3D |
| `trellis2` | `microsoft/TRELLIS.2-4B` | Image-to-3D |
| `sf3d` | `stabilityai/stable-fast-3d` | Image-to-3D |

Override the model or output path:

```bash
ONEBIT_3D_ASSET_MODEL=microsoft/TRELLIS-text-large \
ONEBIT_3D_ASSET_OUT=/tmp/1bit-omni/assets \
  1bit omni "Create a 3D asset request for a brass telescope"
```

A voice-first flow can chain `record_microphone`, `transcribe_audio`, and `create_3d_asset` because `1bit omni` supports bounded multi-step tool rounds.

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
