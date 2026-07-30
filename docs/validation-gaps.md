# Validation Gaps & Engineering Blockers

> **Canonical gap tracker.** Updated 2026-07-29 after live validation on Strix Halo
> (Ryzen AI Max+ 395, Radeon 8060S, 256 GB/s, NPU firmware 1.1.2.65, ROCm 7.1).
>
#â Every claim in `docs/wiki/models.md` was either validated on real hardware,
> identified as a documentation error, or catalogued here as a genuine engineering gap.

---

## ðŸ› Confirmed Bugs

### B1. Mamba1 GGUF Metadata Key Mismatch

**Location**: `tools/test_mamba1_backend.cpp` (also `src/backend_mamba1.cpp` read path)

**Issue**: The config reader looks for GGUF metadata keys with a `mamba.`prefix
(e.g. `mamba.block_count`, `mamba.embedding_length`), but the actual GGUF
files store these keys **without** the prefix (e.g. `block_count`,
`embedding_length`).

**Affected keys**:
|Code reads|File has|Effect|
|---|---|----|
|`mamba.block_count`|`block_count`|num_layers = stack garbage |
|`mamba.embedding_length`|`embedding_length`|xidden_size = stack garbage |
|`mamba.vocab_size`|`vocab_size`|vocab_size = stack garbage |
|`mamba.ssm.state_size`|`ssm.state_size`|d_state = stack garbage |

**Impact**: Config was read as Hidden=2048, Layers=40, Vocab=262272, d_state=128
instead of the real values (Hidden=1152, Layers=30, Vocab=50304, d_state=16).
This caused `backend->init()` to fail at layer 30 (past the actual 30 layers).

**Fix applied**: Removed `mamba.`prefix from key lookups. BlackMamba 1.5B now
runs at **79.9 tok/s** (validated, matches documented 79.4).
