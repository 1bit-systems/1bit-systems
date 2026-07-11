# Roadmap

## Phase 1: INT8 Inference ✅

- [x] INT8 K-interleaving fix (dataReuse on ObjectFifo)
- [x] 5 INT8 xclbins built and NPU-verified (QKV, O, GU, D, KV)
- [x] Per-tensor symmetric INT8 quantization
- [x] 4-live contexts (no swapping)
- [x] Pre-loaded per-layer weight BOs
- [x] NPU attention kernel compiled
- [x] 244 ms/tok (4.1 tok/s) on Qwen3-0.6B
- [x] Batched prefill: 20 ms/tok (13.5× faster)
- [x] HTTP API server (OpenAI-compatible /v1/chat/completions)
- [x] Landing page + live dashboard + PR-Agent
- [x] Packaging: deb, snap, tarball, docker, ollama, AUR

## Phase 2: Speculative Decode 📋

- [ ] Draft model selection (KQV-only or 1-layer variant of Qwen3-0.6B)
- [ ] Draft loop: greedy-predict N tokens, queue for batch verify
- [ ] Batched verification: M=N draft tokens through existing INT8 xclbins
- [ ] Token acceptance: compare draft tokens against verified logits, discard mismatches
- [ ] Target: <50 ms/tok effective throughput
- [ ] Reuse DeepSpec draft training pipeline at `/home/bcloud/DeepSpec/`

### How speculative decoding works for 1bit.systems

```
Current:   token1(244ms) → token2(244ms) → token3(244ms) = 732ms for 3
Spec:      draft 4 tokens → batch verify(300ms) ≈ 75ms/tok effective
```

The draft model runs cheaply (greedy single-token), then one batched INT8 forward
pass verifies all candidates. High acceptance rates expected because the draft model
is the same architecture. No new xclbins needed — reuses existing INT8 GEMMs at M=N.

## Phase 3: GGUF + Model Agnostic 📋

- [ ] GGUF Q8_0 model loading (eliminate Q4NX dependency)
- [ ] Direct Q8_0 → INT8 BO packing (no intermediate dequant)
- [ ] Multi-model support via xclbin parameterization
- [ ] NPU attention dispatch for high-context (>32 tokens)

## Phase 4: 1-bit / BitNet 🔮

- [ ] BitNet b1.58 model loading (ternary weights)
- [ ] Ternary GEMV kernel — replaces INT8 GEMM for 1-bit
- [ ] Hybrid precision: attention in BF16, weights in ternary
- [ ] Target: <25 ms/tok on Strix Halo NPU
- [ ] Bonsai-1.7B already benchmarked at 281 tok/s on GPU

## Phase 5: Productionization ✅ (mostly done)

- [x] HTTP API server (OpenAI-compatible, pure C++ sockets)
- [x] Ollama Modelfile
- [x] OpenAI SDK / LangChain / Open WebUI compatibility
- [x] Docker image (Dockerfile ready)
- [x] AUR package (PKGBUILD ready)
- [x] Snap package (built, 36 KB)
- [x] Debian package (built, 35 KB)
- [ ] Windows support (via AMD's XDNA 2 driver)
