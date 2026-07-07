# Lemonade v10.3 — Omni-Modal Compatibility

Tracking: [1bit-systems#2](https://github.com/bong-water-water-bong/1bit-systems/issues/2)
Upstream: [lemonade-sdk/lemonade#1713](https://github.com/lemonade-sdk/lemonade/pull/1713)

## TL;DR

1bit-halo-server (`npu_server.py :8081`) speaks OpenAI-compatible `/v1/chat/completions`.
Once Lemonade v10.3 ships with omni-modal chat, any Lemonade user can point the UI at
our backend and get BitNet-1.58 text + Lemonade's whisper/kokoro/sd legs with zero
code changes on our side.

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Lemonade v10.3 (omni-modal chat)                │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────────┐ │
│  │whisper│  │kokoro│  │  sd  │  │ text (→us) │ │
│  └──────┘  └──────┘  └──────┘  └─────┬──────┘ │
│                                       │         │
└───────────────────────────────────────┼─────────┘
                                        │ /v1/chat/completions
                                        ▼
┌─────────────────────────────────────────────────┐
│ 1bit-halo-server (:8081)                        │
│  ┌──────────────┐  ┌──────────┐                │
│  │npu_engine_mt │  │ tokenize │                │
│  │BitNet 1.58b  │  │SP tokenizer│              │
│  └──────────────┘  └──────────┘                │
│  Model: Qwen3-0.6B-NPU2 (q4nx)                 │
└─────────────────────────────────────────────────┘
```

## API Contract

Our `/v1/chat/completions` endpoint accepts:

```json
{
  "model": "qwen3-npu",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

Returns standard OpenAI chat completion format.

## Version-Pin Pattern

We adopted the Lemonade `*_bin` config shape in `packages.toml`:

| Component | Version | Type |
|-----------|---------|------|
| `npu_engine` | `/path/to/build` | local build |
| `tokenizer` | `/path/to/build` | local build |
| `flm` | `0.9.44` | pinned tag |
| `triton_xdna` | `builtin` | bundled submodule |
| `models.qwen3_0_6b_npu` | `Qwen3-0.6B-NPU2` | pinned model |

Each component supports: `builtin`, `latest`, `<tag>`, or `<path>`.

## Smoke Test Plan (when v10.3 ships)

1. **Start the server:**
   ```bash
   python3 /home/bcloud/npu-sandbox/npu-infer/1bit/npu_server.py
   # Listens on :8081 by default
   ```

2. **Verify health:**
   ```bash
   curl http://localhost:8081/v1/models
   ```

3. **Test chat completion:**
   ```bash
   curl http://localhost:8081/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen3-npu","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
   ```

4. **Point Lemonade at our backend:**
   - Set `text_bin = "http://localhost:8081/v1"` in Lemonade config
   - Open omni-modal chat UI
   - Verify text responses come from NPU, audio from whisper/kokoro, images from SD

## Tool-Use / Function-Calling

If Lemonade v10.3 requires function-calling on the text backend, we'll need to add
a tool-use shim. Our current endpoint doesn't support `tools` / `tool_calls` in the
OpenAI spec. This is scoped as a post-v10.3-ship pass.

## Current Status

- [x] `packages.toml` created with `*_bin` config shape
- [x] `/v1/chat/completions` endpoint running (npu_server.py :8081)
- [x] FLM v0.9.44 installed and serving qwen3:0.6b
- [ ] Smoke test against Lemonade v10.3 (blocked on upstream ship)
- [ ] Tool-use shim (post-v10.3)
- [ ] Site + wiki integration recipe (post-smoke-test)
