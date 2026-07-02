## Task 1 Report: Dynamic Model Discovery

**Status:** DONE

**What was implemented:**
Rewrote `/home/bcloud/npu-sandbox/npu-infer/1bit/src/models.ts` completely:
- Removed `KNOWN_MODELS` hardcoded dict and `knownDim` fallback dict
- Dynamic directory scan of `~/.config/flm/models/*/` and `~/models/*/`
- Reads `config.json` for all dimension parameters
- Derives xclbin tag from model type + hidden_size with known mapping
- Architecture flags derived from model type (q_norm, k_norm, rope_freqs, gu_split)
- Correctly handles `head_dim` override (Gemma4 uses HD=256, not H/NH)
- Reports status: "ready", "no_xclbins", "no_model_file", "no_config"
- `needsXclbins` array lists missing operation xclbins

**Test results:**
- 37 directories discovered
- 5 ready models with correct dimensions:
  - Gemma4-E2B-IT-NPU2: H=1536 IM=6144 NH=8 NKV=1 HD=256
  - Llama-3.1-8B-NPU2: H=4096 IM=14336 NH=32 NKV=8 HD=128
  - Qwen3-0.6B-NPU2: H=1024 IM=3072 NH=16 NKV=8 HD=128
  - Qwen3-VL-4B-Instruct-NPU2: H=2560 IM=9728 NH=32 NKV=8 HD=128
  - Qwen3-8B-NPU2: H=4096 IM=12288 NH=32 NKV=8 HD=128
- All 5 ready models have matching xclbins
- 32 other directories correctly marked with appropriate status
- Llama, Qwen3-8B, Qwen3-VL correctly flagged gu_split=true

**Key corrections from plan:**
- Gemma uses `gemma4_text` model_type, not `gemma2` — added to deriveTag
- `head_dim` read from config.json when present (Gemma4 needs HD=256)
- `H / NH` fallback for head_dim only when NH > 0 to avoid div-by-zero
- Removed unused `ENGINE` constant (server.ts defines its own)

**Commit:** e85d606cdd9aa3fe79dfd403d49490e972252fce

**Files changed:**
- `src/models.ts` — complete rewrite, 258 insertions
