# Model priority policy

`1bit-systems` prefers low-bit models in this order. Set in
`~/.cache/lemonade/user_models.json` via `labels` and `suggested` flags.

## Tiers

### Tier 1 — 1-bit (preferred)

Sub-1.6 bpw. The brand floor. Pick from this tier first whenever a Tier-1 variant
of the requested model class exists.

| Model | Quant | Size | Recipe | Notes |
|---|---|---|---|---|
| `Bonsai-1.7B-requantized-Bonsai-1.7B-IQ1_S.gguf` | IQ1_S (~1.6 bpw) | 385 MB | `llamacpp` | Showcase / floor — ~280 tok/s decode on `gfx1151` |

### Tier 2 — 2-bit (daily driver)

~2.0–2.5 bpw. Used when a Tier-1 variant isn't available or doesn't meet quality
requirements. The realistic daily driver for 30B-class MoE on a single APU.

| Model | Quant | Size | Recipe | Notes |
|---|---|---|---|---|
| `Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS` | IQ2_XXS (~2.0 bpw) | 11.5 GB | `llamacpp` | Default chat model; 73 tok/s decode, half disk of Q4_K_XL |
| `bitnet_b1_58-3B-TQ2_0-bitnet_b1_58-3B-TQ2_0.gguf` | TQ2_0 (~2.06 bpw) | 1.8 GB | `llamacpp` | BitNet b1.58 ternary, 71 tok/s decode |

### Tier 3 — everything else

Higher-precision GGUFs, FLM/NPU models, embeddings, vision. Loaded only when
explicitly requested or when no Tier-1 / Tier-2 variant exists.

Examples (not exhaustive):
- `Qwen3.5-35B-A3B-GGUF` (Q4_K_XL, 22 GB) — kept as quality-comparison anchor
- `LFM2-1.2B-GGUF` (Q4_K_M) — fast small chat
- `qwen3-0.6b-FLM` / `qwen3-1.7b-FLM` / `deepseek-r1-8b-FLM` — NPU lane
- `nomic-embed-text-v2-moe-GGUF` — embeddings (RAG)
- `Qwen3-VL-4B-Instruct-GGUF` — vision-language

## Default model selection

`LEMONADE_MODEL` env var is the runtime default — set it to the highest-tier
model whose quality meets your task. Currently:

```
LEMONADE_MODEL=user.Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf-UD-IQ2_XXS
```

This is **Tier 2** because no Tier-1 variant of Qwen3.5-35B-A3B exists as a
single-file GGUF on HF today. Thireus's `IQ1_S-SPECIAL_SPLIT` exists at
~35 GB / 734 per-tensor shards but requires their merge tool to assemble — not
supported by `lemonade pull` directly.

For demos / showcases of the 1-bit floor, use:

```
LEMONADE_MODEL=Bonsai-1.7B-requantized-Bonsai-1.7B-IQ1_S.gguf
```

(via `lemonade pull lilyanatia/Bonsai-1.7B-requantized:Bonsai-1.7B-IQ1_S.gguf`,
which is what `install.sh` does by default).

## How priority is enforced

- **`suggested: true`** in `user_models.json` — surfaces the model in Lemonade's
  *suggested* list and in compatible client UIs. Set on Tier-1 and Tier-2 models;
  unset on Tier-3 models that should not be a default pick.
- **`labels`** include `priority-tier-1-1bit` / `priority-tier-2-2bit` /
  `priority-tier-3-other` — searchable / filterable tags for clients and tools.
- **`LEMONADE_MODEL` env var** — runtime default the GAIA / lemonade-client picks
  when no model is specified in a request. Set in:
  - `~/.config/fish/config.fish`
  - `~/.zshenv`
  - `~/.config/systemd/user/gaia.service`

## Adding new models

When adding a model to `user_models.json`:

1. Identify its quantization (look at the file: `IQ1_*` → Tier 1, `IQ2_*` /
   `TQ2_*` / Q2 → Tier 2, others → Tier 3).
2. Add the matching `priority-tier-*` label in the `labels` array.
3. Set `suggested: true` if it's Tier 1 or Tier 2 *and* it actually loads
   correctly on the current llama.cpp build (`lemonade load <name>` to verify
   before promoting).
4. Tier 3 models default to `suggested: false` unless they fill a unique role
   (embeddings, vision, NPU-only).
