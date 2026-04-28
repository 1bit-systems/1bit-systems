# OmniRouter compatibility

[Lemonade OmniRouter](https://www.reddit.com/r/LocalLLaMA/comments/1sy54d1/lemonade_omnirouter_unifying_the_best_local_ai/) (announced by AMD's `jfowers_amd` 2026-04-28) unifies local AI engines under standard OpenAI tool-calling. The LLM gets tool defs (`generate_image`, `text_to_speech`, etc.); when it triggers a tool call, the orchestrator routes to the matching Lemonade modality endpoint:

| Tool call | Routes to | Backend |
|---|---|---|
| `generate_image` | `POST /v1/images/generations` | `sd-cpp:rocm` / `sd-cpp:cpu` |
| `text_to_speech` | `POST /v1/audio/speech` | `kokoro:cpu` |
| `transcribe_audio` | `POST /v1/audio/transcriptions` | `whispercpp:vulkan` / `whispercpp:cpu` |
| (vision input) | `POST /v1/chat/completions` w/ image content | `llamacpp:rocm` (multimodal model) |

**This stack already supports it.** The reference example at [`lemonade-sdk/lemonade/examples/lemonade_tools.py`](https://github.com/lemonade-sdk/lemonade/blob/main/examples/lemonade_tools.py) runs unmodified against our `lemond` on `:13305`.

## Verified working — 2026-04-28

Adapted the reference example to use our daily-driver model, fired two requests:

```sh
python3 omni-test.py "Generate an image of a 1bit-systems mascot — neon two-bit themed"
# → Qwen3.5-35B-A3B-GGUF picks the prompt
# → tool call: generate_image({...})
# → POST /v1/images/generations  →  SD-Turbo  →  531 KB PNG (512x512)

python3 omni-test.py "Say 'one bit systems is the two bit killer' out loud"
# → tool call: text_to_speech({input: ...})
# → POST /v1/audio/speech  →  kokoro-v1  →  303 KB WAV (24kHz mono)
```

No code changes to lemond, no patches to llama.cpp, no fork needed. The `1bit-systems` install does the legwork (pulls all the backends + their default models) and the OpenAI-compat tool-calling + modality-routing pattern uses the endpoints that come for free.

## Why no fork (this time)

Per `feedback_fork_dont_report` doctrine we fork upstream when we have a patch we want to keep as part of our edge. OmniRouter is the *opposite* — it's a public pattern that anyone running Lemonade with the modality backends installed already supports. There's nothing to patch, no edge to keep. We just point at it and say "yes, works."

The win for `1bit-systems` is the **default-models tier policy** layered on top: our preferred LLM is `Qwen3.5-35B-A3B-IQ2_XXS` (sub-2-bit MoE, daily driver) which the OmniRouter agentic loop calls into. Anyone using the OmniRouter pattern with our install gets a 35B-class model dispatching tools at half the disk + ~35% faster decode than the Q4_K_XL baseline.

## Models that work as the LLM in the loop

Anything with the `tool-calling` label in `lemonade list`. On this box right now:

- `Qwen3.5-35B-A3B-GGUF` (Q4_K_XL) — default per `LEMONADE_MODEL` env
- `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS` — Tier 2 daily driver
- (other Qwen3-Instruct variants if pulled — `Qwen3-4B-Instruct-2507-GGUF` per the example)

Embedding models, Vision-only models, and the FLM/NPU lane don't fit the OmniRouter loop role (they're tools or specific-purpose, not the orchestrator).

## See also

- Reddit announcement (2026-04-28): https://www.reddit.com/r/LocalLLaMA/comments/1sy54d1/
- Reference example: https://github.com/lemonade-sdk/lemonade/blob/main/examples/lemonade_tools.py
- Our model-priority tiering: [`docs/model-priority.md`](model-priority.md)
- Default-model env wiring: [`docs/aur.md`](aur.md) and `~/.config/fish/config.fish`
