# FastFlowLM Proprietary Secrets — Complete Extraction

## 1. Q4NX Weight Format (from `libq4_npu_eXpress.so`)

### Constants Table

| Constant | Value | Description |
|----------|-------|-------------|
| `group_size` | 32 | Elements per quantization group |
| `group_size_bytes` | 40960 (0xa000) | Bytes per group allocation |
| `row_chunk_size` | 32 | Rows per NPU chunk |
| `col_chunk_size` | 256 | Columns per NPU chunk |
| `weight_per_chunk` | 8192 (0x2000) | Bytes per chunk |
| `num_groups_per_chunk_row` | 8 | Groups per chunk row |
| `num_groups_per_row_parallel` | 2 | Parallel rows per group |
| `block_size` | 5120 (0x1400) | NPU DMA block granularity |
| `parallel_size` | 16 | Parallelism factor |

### Weight Reordering Algorithm (`_q4nx_reorder`)

The proprietary reorder transforms a flat [rows × cols] weight matrix into a tiled layout for NPU DMA:

```
Input:  [rows][cols] bf16 weights
Output: [num_chunks][padded_rows][cols] + scales + zero_points

1. num_chunks = ceil(orig_size / 32)      // 32 = row_chunk_size
2. padded_size = round_up(chunk_data, 32)  // pad to 32-element boundary
3. For each chunk:
   a. qdata ← pack 4-bit indices (2 per byte, little-endian)
   b. scales[bfloat16] ← per-group scale factors
   c. zero_points[int] ← per-group zero points (usually 0)
   d. Zero-fill padding regions
4. Result: each 32×256 tile is contiguous in NPU memory
```

### Quantization Scheme

```
Per group (32 elements):
  [4-bit index × 32] = 16 bytes packed
  [bf16 scale]        = 2 bytes
  [bf16 zero_point]   = 2 bytes
  Total: 20 bytes / 32 elements = 0.625 bytes/element

Dequant: value = (q4_index × scale) + zero_point
```

### Metadata JSON Keys

```
row_parallel_size, row_chunk_size, col_chunk_size, group_size,
scale_type ("bf16"), zero_point_type ("bf16"), _scale, _zero_point
```

### Safetensors -> Q4NX Pipeline

```
SafeTensors::_open_file() → parse JSON header
  → SafeTensors::get_metadata() → shape, dtype, data_offsets
  → Q4NX::_grap_metadata() → quantization params
  → Q4NX::_convert_to_q4nx() → quantize + pack
  → Q4NX::_q4nx_reorder() → tile for NPU
  → write /model.q4nx
```

### _convert_llama Weight Map

Iterates over each layer with SafeTensors::get_tensor_metadata() + _convert_to_q4nx():

```
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.self_attn.q_proj.weight
model.layers.N.self_attn.k_proj.weight
model.layers.N.self_attn.v_proj.weight
model.layers.N.self_attn.o_proj.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight
model.norm.weight
lm_head.weight
```

---

## 2. NPU Instruction Set (XDNA2)

### Device Parameters (NPU2)

| Parameter | Value |
|-----------|-------|
| NPU generation | 4 |
| Rows | 6 |
| Cols | 8 |
| Mem tile rows | 1 |
| Headers | `npu_seq[0] = 0x00000406`, `npu_seq[1] = 0x00000108` |

### Instruction Header Format

```
Word 0: [npu_major:8 | npu_minor:8 | npu_dev_gen:8 | npu_rows:8]
Word 1: [npu_cols:8 | npu_mem_tile_rows:8 | reserved:16]
Word 2: instruction_count (number of command objects)
Word 3: instruction_lines (total uint32_t count, << 2)
```

### Tile Layout

```
Row 0: IT0 IT1 IT2 IT3 IT4 IT5 IT6 IT7  (Shim/DMA)
Row 1: MT0 MT1 MT2 MT3 MT4 MT5 MT6 MT7  (Memory)
Row 2: CT00..CT07  (Compute row 0)
Row 3: CT10..CT17  (Compute row 1)
Row 4: CT20..CT27  (Compute row 2)
Row 5: CT30..CT37  (Compute row 3)
```

### Command Opcodes

| Value | Name | Class | Size | Purpose |
|-------|------|-------|------|---------|
| `0x00` | XAIE_IO_WRITE | `npu_write_cmd` | 6 words | Register write / Queue push |
| `0x01` | XAIE_IO_BLOCKWRITE | `npu_dma_block_cmd` | 12 words | N-dimensional DMA transfer |
| `0x03` | XAIE_IO_MASKWRITE | `npu_issue_token_cmd` | 7 words | Issue sync token |
| `0x80` | XAIE_IO_CUSTOM_OP_TCT | `npu_wait_cmd` | Variable | Wait for DMA completion |
| `0x81` | XAIE_IO_CUSTOM_OP_DDR_PATCH | `npu_ddr_cmd` | 10 words | DDR address patching |

### RTP Register Addresses (from `Gemm::generate_seq`)

| Register | Purpose |
|----------|---------|
| `0x1000` | Tile M dimension (rows of output) |
| `0x1004` | Tile K dimension (inner reduction) |
| `0x1008` | Tile N dimension (cols of output) |
| `0x100c` | Activation type (0=none, 1=GeLU, 2=SiLU) |
| `0x1010` | Bias enable flag |
| `0x1d200` | Queue/S2MM channel base |
| `0x1d204` | Queue push register (MM2S) |
| `0x1f0a0` | GEMM kernel kick-off (write 1 to execute) |

### Shim Tile Array (`Gemm::Impl::shim_tiles`)

```
shim_tiles[8] = { IT0, IT1, IT2, IT3, IT4, IT5, IT6, IT7 }
               = { 0, 1, 2, 3, 4, 5, 6, 7 }
```

Each column's SHIM tile handles DDR DMA for its corresponding compute column.

---

## 3. XRT Runtime Architecture (from `libqwen3_npu.so`)

### Components

| Component | API | Purpose |
|-----------|-----|---------|
| `xrt::device` | Core device handle | Open NPU device |
| `xrt::xclbin` | Load bitstream | Program AIE array |
| `xrt::hw_context` | Context management | Hardware execution context |
| `xrt::kernel` | Kernel handle | Reference to compiled AIE kernel |
| `xrt::module` | Module loading | Load AIE module from ELF |
| `xrt::bo` | Buffer object | Device memory allocation |
| `xrt::run` | Single execution | Execute one kernel invocation |
| `xrt::runlist` | Batch execution | Execute chained kernels |
| `npu_app` | Custom wrapper | App-level NPU management |
| `npu_app_manager` | App factory | Creates NPU apps per xclbin kernel |
| `bytes` | Buffer wrapper | Maps xrt::bo to host pointer |

### Initialization Sequence

```
1. xrt::xclbin(path) → load .xclbin file
2. device.register_xclbin(xclbin) → program FPGA
3. xclbin.get_uuid() → get unique ID
4. xrt::hw_context(device, uuid) → create context
5. xrt::module(elf) → compile AIE ELF
6. xrt::ext::kernel(context, module, name) → create kernel
7. xrt::bo(device, size) → allocate device buffers
8. npu_app::create_run(args...) → create execution
9. runlist.add(run) → add to batch
10. runlist.execute() → launch
11. runlist.wait() → wait for completion
```

### Checkpoint/Restore Sequence (Multi-turn)

```
checkpoint():
  bytes::sync_from_device() → copy KV cache from NPU to host
  bytes owns the buffer — persists across turns

restore():
  bytes::sync_to_device() → copy saved KV cache back to NPU
  Returns the context length at time of checkpoint
```

### Important XRT Call Pattern

```
npu_app::operator()<buffer<bf16>&, buffer<bf16>&, buffer<bf16>&>(input, weights, output):
  → xrt::run::set_arg_at_index(0, input_bo)
  → xrt::run::set_arg_at_index(1, weights_bo)
  → xrt::run::set_arg_at_index(2, output_bo)
  → xrt::run::start()
  → xrt::run::wait()

npu_app::_setup_kernel():
  → Set NPU instruction buffer as arg
  → Configure instruction count
  → Map kernel I/O buffers

bytes::sync_to_device():
  → xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE, size, 0)

bytes::sync_from_device():
  → xrt::bo::sync(XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
```

---

## 4. Inference Pipeline Flow

### Model Loading

```
Q4NX model directory → read config.json → extract LM_Config
  → create Q4NX instance → read /model.q4nx file
  → create model_npu instance (e.g., qwen3_npu)
  → model_npu.load_weights(q4nx) → DMA weights to NPU
  → setup_tokenizer(model_path)
  → setup_sampler(top_k=40, top_p=0.9, min_p=0.1, temp=0.8)
```

### Prefill (Multi-token)

```
prefill(token_ids):
  → _prefill_with_mv() or _prefill_with_mm()
    → For each layer:
      → gen_layer_seq(seq, L)  with L = context_length
      → Submit to NPU via xrt::runlist
    → gen_lm_head_seq(seq)
    → Run on NPU → return logits
```

Two variants:
- `_prefill_with_mv`: Use masked attention (causal mask on NPU)
- `_prefill_with_mm`: Multi-modal prefill (for VL models, includes vision tokens)

### Decode (Single token, autoregressive)

```
forward(token_id):
  → token_history.push_back(token_id)
  → For each layer:
    → gen_layer_seq(seq, current_length)
    → gen_mha_engine_seq(seq, l_begin, l_end)  // flash attention on NPU
  → gen_lm_head_seq(seq)
  → Execute on NPU → return logits
```

### Generation Loop

```
while total_tokens < MAX_L:
  logits = lm_engine.forward(last_token)
  sampled_token = sampler.sample(logits)
  output += tokenizer.decode(sampled_token)
  if is_eos(sampled_token): break
```

---

## 5. Per-Model Architecture Differences

### Dense Models (Qwen3, Llama3, Phi-4, Nanbeige)

Common `npu_sequence` pattern:
```
gen_layer_seq(seq, L):
  → _send_x(seq)           // broadcast hidden states
  → _send_rms_weights(seq) // RMS norm weights
  → _send_rope_rms_weights(seq) // RoPE + RMS combined
  → _move_weights(seq, k_off, v_off, kv_size)  // load QKV projection weights
  → gen_mha_engine_seq(seq, L_begin, L_end)  // flash attention
  → _receive_kv_cache(seq, layer)  // save K,V back to DDR
  → _move_kv_cache(seq, L)  // advance KV cache pointer
  → gen_dequant_seq(seq, d_in, d_out, w_off)  // dequant FFN weights
  → [GEMM + activation for gate/up/down projections]
  → gen_lm_head_seq(seq)  // final projection (last layer only)
```

### KV Cache Layout

| Model | Cache Structure | Getter |
|-------|-----------------|--------|
| Qwen3/Llama/Phi | k03, k47, v03, v47 (4 buffers × 2 = 8 cache segments) | `get_k03_offset()` etc. |
| Gemma3 | k01, k23, v01, v23 (2 buffers × 2 = 4 cache segments) | `get_k01_offset()` etc. |
| Gemma3 Text | k, v (1 buffer each = 2 cache segments) | `get_k_offset()` etc. |
| Gemma4e | Per-layer type cache | `gemma4e_layer_type_t` parameter |
| Qwen3.6 MoE | Standard + router-specific | `_send_router_w_and_share_exp_gate()` |

### VL Model Extras

Qwen3.5/3.6 VL models add:
- `conv3d_patch_embed()` — 3D convolution for patch embedding
- `Qwen3_5VLImageEncoder` / `Qwen3_6_MOEVLImageEncoder` — vision encoder
- `gen_seq_conv1d()` — 1D convolution for cross-attention
- `_send_linear_conv_weights()` + `_receive_linear_kv_cache()` — cross-attention
- `gen_mha_vision_attention()` — vision-specific attention
- `_gen_linear_sequence()` — linear layer sequence

### MoE Extras (Qwen3.6)

- `_send_router_w_and_share_exp_gate()` — MoE routing weights
- `send_manual_expert_up_gate_q41()` — expert up-projection (Q4.1)
- `send_manual_expert_down_gate_q41()` — expert down-projection (Q4.1)
- `setup_expert_up_gate_q41()` / `setup_expert_down_gate_q41()` — expert config
- `gen_dequant_mm()` / `gen_dequant_mm_512()` — MoE matmul dequant

### LFM2 Extras (Mamba-like)

- `gen_conv1d_engine_seq()` — 1D convolution state
- `gen_conv_layer_seq()` — convolutional layer
- `_send_conv_cache()` / `_receive_conv_cache()` — stateful conv cache

### Whisper Extras

- `WhisperEncoder::forward()` — audio encoder
- `WhisperDecoder::forward()` — autoregressive decoder
- `prepare_mm()` / `prepare_qkt()` / `prepare_sv()` / `prepare_proj()` — attention
- `load_cross_attn_kv_cache()` — cross-attention KV cache for encoder-decoder

### Gemma4e Extras

- `gen_swa_engine_seq()` — sliding window attention engine
- `Gemma4e_AudioEncoder` + `Gemma4e_ImageEncoder` — multi-modal encoders
- `compute_audio_blocked_attention()` / `compute_audio_self_attention()` — CPU fallback
- `create_sliding_window_attention_mask()` — SWA mask generation
- SIMD ops: `simd_rms_norm`, `simd_clamp`, `simd_mul`, `simd_relu`, `simd_silu`, `simd_conv2d`
- `generate_gemma4_audio_rotary_pos_emb()` / `generate_gemma4_vision_rotary_pos_emb()`

---

## 6. Attention Fusion (MHA Engine)

### MHA Variants

| Variant | Head Dim | Quant | Use Case |
|---------|----------|-------|----------|
| `d64_q4` | 64 | Q4 | Small heads (Gemma) |
| `d128_q2` | 128 | Q2 | Medium heads, aggressive compression |
| `d128_q3` | 128 | Q3 | Medium heads, balanced |
| `d128_q4` | 128 | Q4 | Medium heads, standard (most common) |
| `d256_q2` | 256 | Q2 | Large heads (Qwen3, Llama), compression |
| `d256_q4` | 256 | Q4 | Large heads, high quality |

### Attention on NPU

The MHA engine generates NPU sequences for:
1. **QK^T multiply**: Q_head × K_head → attention scores (on compute tiles)
2. **Softmax**: Online softmax on NPU (requires tile-local reduction)
3. **PV multiply**: Attention scores × V_head → output (on compute tiles)

The engine uses dedicated AIE kernel configurations for each (head_dim, quant) pair, loaded from the xclbin bitstream.

---

## 7. Build Environment & Toolchain

| Component | Version |
|-----------|---------|
| Compiler | GCC 13.3.0 (Ubuntu 24.04.1) |
| Language | C++20 |
| JSON library | nlohmann v3.12.0 |
| XRT | 2.x (libxrt_coreutil.so.2) |
| AIEBU | AMD AIE binary utilities (static .a) |
| FFmpeg | n7.1 (optional, for media decode) |
| FFTW | 3.x (for audio processing) |

### Source File Paths (from debug info)

```
gemm.cpp, dequant.cpp, dequant_detail.hpp
conv3dPatchEmbedding.cpp, aiebu_error.cpp
../../include/npu_utils/npu_instr_utils.hpp
../../include/npu_utils/instr_utils/npu_cmd_write_dma.hpp
../../include/nlohmann/json.hpp
../../include/buffer.hpp
```

---

## 8. Key Assertions & Error Messages

```
"a_block_size % vetrical_block_interleave_byte_size == 0"
"total_size == q4nx_weight.size()"
"MHA parameter check failed!"
" > 6, Should be mandatorily patched in host!!!"
"Creating checkpoint at context length "
"Failed to allocate bytes of size "
"Failed to open file: "
"AIEBU Build Version: ", "BUILD ID: ", "Build Version Branch: "
"A linear transfer, no D0"
"RTP write"
"Always 0E"
"Parameter check error: "
"!this->is_a_quantized_model"
```
