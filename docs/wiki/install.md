# Install & Run

```bash
curl -sL https://1bit.systems/npu-install.sh | bash
1bit pull qwen3-0.6b
1bit chat
```

Or serve (OpenAI-compatible API):
```bash
1bit serve &
curl localhost:8081/v1/chat/completions \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hello!"}]}'
```

**Build from source:**
- NPU (C++23): `cd engine/npu && cmake -B build && cmake --build build`
- GPU (Zig): `cd engine/gpu && zig build -Doptimize=ReleaseFast`
- Fused: `cd engine/fusion && zig build -Doptimize=ReleaseFast`

**Packages:** deb · snap · docker · AUR · homebrew · ollama

**Multi-modal:** See [`tools/video-lora/`](../tools/video-lora/)
