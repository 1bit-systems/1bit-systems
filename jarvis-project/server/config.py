"""JARVIS Configuration — 1bit.systems"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JarvisConfig:
    # Unified daemon API
    api_base: str = "http://127.0.0.1:9090"
    flm_port: int = 52625

    # Model IDs (via FLM/unified daemon)
    llm_model: str = "qwen3:0.6b"
    vision_model: str = "qwen3vl-it:4b"
    embed_model: str = "embed-gemma:300m"
    asr_model: str = "whisper-v3:turbo"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True

    # Paths
    jarvis_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    ))
    voice_dir: str = field(default_factory=lambda: os.path.expanduser("~/.local/share/piper/voices/"))
    whisper_cpp_dir: str = "/home/bcloud/whisper.cpp"
    whisper_server_bin: str = "/home/bcloud/whisper.cpp/build/bin/whisper-server"
    whisper_model: str = "/home/bcloud/whisper.cpp/models/ggml-base.en.bin"

    # Conversation
    max_history: int = 50
    max_tokens: int = 2048
    temperature: float = 0.7

    # Available tools
    tools_enabled: list = field(default_factory=lambda: [
        "calculator", "web_search", "python_exec", "read_file", "list_dir",
    ])


def load_config() -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.api_base = os.environ.get("JARVIS_API_BASE", cfg.api_base)
    cfg.llm_model = os.environ.get("JARVIS_LLM_MODEL", cfg.llm_model)
    cfg.port = int(os.environ.get("JARVIS_PORT", str(cfg.port)))
    cfg.debug = os.environ.get("JARVIS_DEBUG", "1") == "1"
    return cfg
