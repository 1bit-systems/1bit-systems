# Ollama + 1bit.systems NPU

Point Ollama at the 1bit.systems NPU server for local INT8 inference.

## Setup

```bash
# 1. Start the NPU server
./1bit-server 8081

# 2. Register the model with Ollama
ollama create qwen3-npu -f Modelfile

# 3. Run
ollama run qwen3-npu "Hello, world!"
```

The NPU server speaks `/v1/chat/completions` in OpenAI-compatible JSON.
Any client that supports custom API bases can use it — not just Ollama:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Performance

- Qwen3-0.6B INT8: 244 ms/tok (4.1 tok/s)
- 8 NPU contexts alive simultaneously
- Zero Python at runtime
- 56 KB binary

## Architecture

```
Ollama → HTTP → 1bit-server → XRT → NPU (XDNA 2)
```
