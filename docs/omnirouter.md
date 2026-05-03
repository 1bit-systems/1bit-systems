# OmniRouter Compatibility

OmniRouter belongs to Lemonade. In this stack Lemonade is the canonical OpenAI-compatible multimodal server on `http://127.0.0.1:13305/v1`, and GAIA is the primary agent/UI/control layer that consumes that capability.

The `1bit-proxy` endpoint on `http://127.0.0.1:13306/v1` is a convenience union endpoint. It keeps Lemonade as the default route, then sends targeted model families to FastFlowLM on `:52625` when that is the better lane, for example FLM chat models and `embed-*` embedding requests.

## Routing Model

| Workload | Canonical route | Notes |
|---|---|---|
| Chat, tools, images, TTS, Lemonade STT, vision | Lemonade `:13305/v1` | Lemonade owns multimodal OpenAI compatibility and OmniRouter behavior. |
| NPU chat models | FastFlowLM `:52625/v1` | Use direct FLM or the union proxy when model routing is desired. |
| Embeddings with `embed-*` models | FastFlowLM `:52625/v1` | `embed-gemma:300m` is verified locally through the proxy. |
| ASR with `whisper-v3:*` | FastFlowLM `:52625/v1` | Opt-in: pull the model and enable `--asr 1` first. |
| OpenAI clients that want one base URL | `1bit-proxy :13306/v1` | Convenience layer, not the canonical OmniRouter server. |

## How OmniRouter Works Here

Lemonade exposes standard OpenAI endpoints for model calls and modality tools. An agent loop gives the LLM tool definitions such as `generate_image`, `text_to_speech`, and `transcribe_audio`. When the LLM emits a tool call, the orchestrator calls the matching Lemonade modality endpoint:

| Tool call | Endpoint | Typical backend |
|---|---|---|
| `generate_image` | `POST /v1/images/generations` | `sd-cpp:rocm` or CPU fallback |
| `text_to_speech` | `POST /v1/audio/speech` | `kokoro` |
| `transcribe_audio` | `POST /v1/audio/transcriptions` | Lemonade Whisper backend, or FLM only for `whisper-v3:*` through the proxy |
| vision input | `POST /v1/chat/completions` with image content | Lemonade multimodal model |

`scripts/1bit-omni.py` is the local helper loop for exercising that pattern with the stack defaults.

## Useful Commands

```sh
1bit up
1bit omni "Generate an image of a clean 1bit.systems rack diagram"
1bit omni "Say '1bit systems is online' out loud"
1bit gaia cli
```

Use these base URLs deliberately:

```text
Lemonade canonical OmniRouter: http://127.0.0.1:13305/v1
Union endpoint for clients:    http://127.0.0.1:13306/v1
FastFlowLM direct NPU lane:    http://127.0.0.1:52625/v1
```

## References

- Lemonade docs: https://lemonade-server.ai/docs/
- Lemonade OmniRouter docs: https://lemonade-server.ai/docs/omni-router/
- Lemonade OpenAI API docs: https://lemonade-server.ai/docs/api/openai/
- FastFlowLM docs: https://fastflowlm.com/docs/
- FastFlowLM server docs: https://fastflowlm.com/docs/instructions/server/
- GAIA quickstart: https://amd-gaia.ai/docs/quickstart
