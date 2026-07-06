# capabilities

- [Chat / LLM](chat.md) — NPU-powered chat at 94 tok/s via Qwen3-0.6B on XDNA2. Streaming response via WebSocket or HTTP SSE. Conversation memory.  [Capability]
- [Configuration](configuration.md) — Environment variables, config dataclass, and all tunable parameters for JARVIS.  [Guide]
- [Open Knowledge Format](open_knowledge.md) — All knowledge stored as human-readable markdown files with YAML frontmatter. Edit with any text editor. No lock-in.  [Capability]
- [RAG (Retrieval-Augmented Generation)](rag.md) — Search through uploaded documents and knowledge entries using full-text keyword search. Results injected into LLM context.  [Capability]
- [Tool Calling](tool_calling.md) — LLM tool execution — calculator, Python code execution, file operations, and web search. All local.  [Capability]
- [Vision](vision.md) — Image analysis via Qwen3-VL-4B multimodal model on NPU. Accepts uploaded images and base64-encoded image data.  [Capability]
- [Voice I/O](voice.md) — Speech-to-text via Whisper (NPU) + Text-to-speech via Piper (CPU). Real-time bidirectional voice over WebSocket.  [Capability]
