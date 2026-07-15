# MLIR Generator Port Blocker — July 2, 2026

## What's Done

| Component | Status | Location |
|-----------|--------|---------|
| contract_06b.py | ✅ Created | torch2aie/examples/qwen3-decode-layer/ |
| qwen3_constants_06b.h | ✅ Created | torch2aie/examples/qwen3-decode-layer/ |
| main_projection_q4nx_06b.o | ✅ Compiled (80KB) | build/qwen3_decode_layer_objects_06b/ |
| edge_attention_06b.o | ✅ Compiled (37KB) | build/qwen3_decode_layer_objects_06b/ |
| postprocess_qkv_06b.o | ✅ Compiled (34KB) | build/qwen3_decode_layer_objects_06b/ |
| full_vector_station_06b.o | ✅ Compiled (20KB) | build/qwen3_decode_layer_objects_06b/ |
| swiglu_06b.o | ✅ Compiled (7KB) | build/qwen3_decode_layer_objects_06b/ |

## What's Blocking

FLM's `full_layer_engine_generate.py` MLIR generator hardcodes `WEIGHT_PATCH_BD_IDS = ((0,2,4,6,8,10,12,14), (1,3,5,7,9,11,13,15))` — 8 BD spans per patch.

For 8B model:
- WEIGHT_SPAN_CHUNKS = QKV_BODY_WEIGHT_CHUNKS = 192
- UPGATE (768 chunks) splits into 768/192 = 4 spans
- DOWN (384 chunks) splits into 384/192 = 2 spans
- Total: 1(QKV) + 1(O) + 4 + 2 = **8 spans** ✓

For 0.6B model:
- Q_weight per column = 16 (4 body × 4 chunks), K=8 (2×4), V=8 (2×4)
- QKV_BODY_WEIGHT_CHUNKS = 32
- UPGATE (48 chunks) splits into ceil(48/32) = 2 spans
- DOWN (24 chunks) splits into ceil(24/32) = 1 span
- Total: 1 + 1 + 2 + 1 = **5 spans** ✗ (expected 8)

## Required Fix (estimated 4-6 hours)

In `cases/full_layer_engine_generate.py`:

1. **Adjust WEIGHT_PATCH_BD_IDS** for 0.6B:
   - Calculate correct span count from 0.6B dimensions
   - Re-assign BD IDs to avoid conflicts with other BD users
   - Update all hardcoded BD count checks

2. **Recalculate weight spans** in `_full_weight_spans()`:
   - Adjust WEIGHT_SPAN_CHUNKS or make `_split_weight_span` handle non-uniform spans
   - Update weight streaming BD writes to use variable span count

3. **Update reference checks** in `full_layer_engine_reference.py`:
   - All weight/patch/chunk counts change with dimensions
   - Numerical validation thresholds may need adjustment

4. **Fix link_with paths** in generated MLIR:
   - Change `edge_attention.o` → `edge_attention_06b.o`
   - Change `postprocess_qkv.o` → `postprocess_qkv_06b.o`
   - etc.

## Alternative: Direct MLIR Handoff

Instead of fixing the generator, directly hand-edit the 3407-line design.mlir
to use 0.6B dimensions and 0.6B kernel objects. Changes needed:
- Buffer sizes (HIDDEN_DWORDS, COMPACT_PACKET_DWORDS, etc.)
- BD transfer lengths
- Weight span counts and BD assignments
- Phase body records and chunk counts
- link_with paths

This is ~50 lines of changes in a 3400-line file. More tedious but more reliable
than fixing the generator.

## Fallback

v12 engine at 97 tok/s. Beats FLM Kraken Point by 46%.
Use existing 4-GEMM-per-layer dispatch until fused design is ported.
