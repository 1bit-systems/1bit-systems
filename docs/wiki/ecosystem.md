# Ecosystem

The daemon speaks the **OpenAI-compatible API**. Works with:

**vLLM** · **Ollama** · **OpenAI SDK** · **LangChain** · **Open WebUI** · **curl**

Set `OPENAI_API_BASE=http://localhost:9090/v1` in any client.

### Backends

| Engine | Lang | Hardware |
|--------|------|----------|
| NPU | C++23 | XDNA 2 (XRT) |
| GPU ⭐ | Zig | Vulkan 1.3 |
| ROCm | C/Zig | AMD HIP |
| CUDA | C/Zig | NVIDIA |
| Metal | Zig | Apple Silicon |
