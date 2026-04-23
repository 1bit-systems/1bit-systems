# Lemonade v10.3 — omni-modal integration

`1bit-halo-server` is OpenAI-compatible at `:8180/v1/chat/completions` and `/v1/models`, so Lemonade SDK (v10.3+) can treat it as an LLM backend inside the omni-modal chat UI. Once [upstream PR #1713](https://github.com/lemonade-sdk/lemonade/pull/1713) lands and v10.3 ships, this page is the operator runbook.

## What Lemonade v10.3 brings

- Omni-modal chat surface — text + voice (whisper.cpp STT, kokoro TTS) + image (sd.cpp) + tool-calling in a single UI.
- `*_bin` config keys accept `builtin` / `latest` / upstream-tag / local path per [#1713](https://github.com/lemonade-sdk/lemonade/pull/1713). Applies live, no daemon restart.
- Uses standard OpenAI `/v1/chat/completions` for the LLM leg — any OpenAI-compatible endpoint plugs in.

## Pointing Lemonade at `1bit-halo-server`

On the Lemonade box, edit the active profile (`~/.config/lemonade/lemonade.yaml`) and set the LLM endpoint to the 1bit HTTP surface:

```yaml
llm:
  backend: openai_compat
  endpoint: http://<strixhalo-host>:8180/v1
  model: halo-1bit-2b       # must match /v1/models; registry validates per request
  bearer: ""                 # optional; only set if Caddy bearer is enabled
```

Over the headscale tailnet: substitute `<strixhalo-host>` with `100.64.0.1` or the tailnet hostname.

## What we serve natively

- Text chat: `halo-1bit-2b` (2B BitNet 1.58) + any additional `.h1b` under `--models-dir`, discovered by the `ModelRegistry` and listed at `/v1/models`.
- Voice STT (planned on sliger): whisper.cpp Vulkan on Arc B580, `:8190`. See [Installation.md](./Installation.md).
- Voice TTS (planned on sliger): kokoro, `:8191`.
- Image: sd.cpp native-HIP port on gfx1151.

Until sliger audio ships, operators can leave Lemonade's built-in whisper/kokoro backends in place — `1bit-halo-server` covers the LLM slot only.

## Known limits

- `max_tokens` hard-capped at model `max_ctx - prompt_tokens`. BitNet-1.58-2B-4T = 4096 ctx.
- Model registry rejects requests whose `model` field is not in `/v1/models` with a 400 — Lemonade's `openai_compat` backend must send the correct id.
- SSE streaming supported; non-streaming path also returns full envelope.
- No function-calling / tool-use API shim today. If Lemonade v10.3 expects it, that's a future pass.

## Version pinning on our side

We mirror Lemonade's new `*_bin` pattern in our `packages.toml` post-Run-4 (see [tracking issue](https://github.com/bong-water-water-bong/1bit-systems/issues)). Until then, `./install.sh` drives the subtree pins directly.

## Feedback loop with upstream

Author of #1713 is `@jeremyfowers` (AMD). Already connected via our [amdgpu OPTC issue #1](https://github.com/bong-water-water-bong/1bit-systems/issues/1). Keep the channel open — if Lemonade needs contract changes on our `/v1` side to match v10.3's expectations, we accommodate.

---

*Draft 2026-04-22 post-Run-4. Revisit when v10.3 ships.*
