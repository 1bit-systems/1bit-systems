# Roadmap

## Phase 1: INT8 Inference ✅

- [x] INT8 K-interleaving fix (dataReuse on ObjectFifo)
- [x] 5 INT8 xclbins built and NPU-verified (QKV, O, GU, D, KV)
- [x] Per-tensor symmetric INT8 quantization
- [x] 4-live contexts (no swapping)
- [x] Pre-loaded per-layer weight BOs
- [x] NPU attention kernel compiled (not yet wired)
- [x] 243 ms/tok (4.1 tok/s) on Qwen3-0.6B

## Phase 2: Production INT8 📋

- [ ] Wire NPU attention dispatch (174 ms/tok target)
- [ ] GGUF Q8_0 model loading (eliminate Q4NX dependency)
- [ ] Multi-token prefill batching
- [ ] Top-k / top-p / temperature sampling
- [ ] Tokenizer integration (HF tokenizers via C API)

## Phase 3: 1-bit / BitNet 🔮

- [ ] BitNet b1.58 model loading
- [ ] Ternary GEMV kernel (replaces INT8 GEMM for 1-bit weights)
- [ ] Hybrid precision: attention in BF16, weights in ternary
- [ ] Target: <50 ms/tok on Strix Halo NPU

## Phase 4: Productionization

- [ ] HTTP API server (C++ civetweb or bare sockets)
- [ ] OpenAI-compatible /v1/chat/completions endpoint
- [ ] Docker image
- [ ] AUR package
- [ ] Windows support (via AMD's XDNA 2 driver)
