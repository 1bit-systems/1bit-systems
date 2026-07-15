# FLM Bridge Status Summary (June 28)

## What Works

### FLM Bridge (`flm_bridge.cpp` / `flm_bridge.h`)
- ✅ Resolves all FLM instruction generators via dlopen
- ✅ `gen_gemm_instrs(M,N,K)` generates correct instruction sequences:
  - q_proj (M=512,N=1024,K=1024): **5592 instrs** (22KB)
  - gate_proj (M=512,N=3072,K=1024): **5592 instrs**
  - down_proj (M=512,N=1024,K=3072): **12152 instrs**
- ✅ `test_flm_bridge` passes and saves instruction binaries to /tmp/

### npu_infer (current engine)
- ✅ Loads model (311 tensors, 28 layers)
- ✅ Loads all 4 xclbins (mm, attn, layer, dequant)
- ✅ Pre-packs all weights into BOs (1065 BOs, 361 ms)
- ✅ Runs prefill + decode pipeline
- ❌ **Tokens are deterministic WRONG** (919, 996, 185, 385...)
  - Root cause: Uses opcode=3 (IDENTITY) — kernel copies activation to output, ignoring weights
  - No instructions are submitted (instr_size=0, instr_bo=0)

## The Fundamental Problem

FLM's instruction generator (`gen_dequant_seq` + `move_weights` + `Gemm::generate_seq`) bakes DDR addresses into the instructions. When we submit with our own BOs (allocated at different physical addresses), the AIE reads/writes to the wrong memory locations.

Three paths to fix:

### Path A: Use FLM's npu_xclbin_manager (Recommended)
- FLM has an `npu_xclbin_manager` class with `register_xclbin()` method 
- `qwen3_npu::Impl` constructor takes `npu_xclbin_manager*` parameter
- If we create an npu_xclbin_manager and pass it to FLM's instruction generators, they'll allocate BOs through it and addresses will match
- Need to find the constructor symbol for npu_xclbin_manager (investigate what size/fields it has)

### Path B: Parse and Relocate Instructions
- Instructions reference DDR addresses - we'd need to parse the AIE instruction format, find address fields, and patch them with our BO addresses
- Very fragile - would break with xclbin version changes

### Path C: Full Custom via torch2aie
- torch2aie example for Qwen3 exists at `/home/bcloud/torch2aie/examples/qwen3-decode-layer/`
- Would need to: create xclbins, handle weight DMA, write instruction sequences from scratch
- Much more work but gives complete control

## Next Action

The most practical next step is **Path A**: Build a wrapper that uses `npu_xclbin_manager` to allocate BOs through FLM's infrastructure, then serve those BOs to both the instruction generator and the kernel submit call.

Alternatively: bypass instruction generation entirely and use the captured BO dumps (h8_128MB_cmds_layer0.bin has pre-generated instructions for layer 0) as a fixed program, allocating BOs at matching addresses.
