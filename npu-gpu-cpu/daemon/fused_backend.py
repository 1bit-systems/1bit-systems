#!/usr/bin/env python3
"""
Fused Engine Backend — subprocess-based NPU+GPU hybrid inference.

Spawns the fused-engine binary as a subprocess and communicates
via stdin/stdout with newline-delimited JSON. Replaces the
separate NPU-only and GPU-only backends with one unified path.

Usage:
    engine = FusedEngine()
    engine.start()
    engine.prefill([151643, 872, 198])  # prefill tokens
    tok = engine.generate()              # decode step
    engine.reset()
    engine.stop()
"""

import json
import os
import subprocess
import time
from typing import Optional


class FusedEngine:
    """Fused NPU+GPU engine subprocess manager."""

    ENGINE_PATH = os.environ.get(
        "FUSED_ENGINE_PATH",
        "/home/bcloud/engine/fusion/zig-out/bin/fused-engine"
    )
    DEFAULT_MODEL = os.environ.get(
        "NPU_MODEL_PATH",
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
    )
    DEFAULT_POLICY = os.environ.get("FUSED_POLICY", "auto")

    def __init__(
        self,
        engine_path: str = None,
        model_path: str = None,
        policy: str = None,
        port: int = 0,
    ):
        self.engine_path = engine_path or self.ENGINE_PATH
        self.model_path = model_path or self.DEFAULT_MODEL
        self.policy = policy or self.DEFAULT_POLICY
        self.port = port
        self.proc: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.gen_count = 0
        self.total_ms = 0.0

        # Determine xclbin directory
        self.xclbin_dir = os.environ.get(
            "XCLBIN_DIR",
            "/home/bcloud/npu-sandbox/npu-infer/build/int8"
        )

        # Model tag detection
        self.model_tag = self._detect_model_tag()

    def _detect_model_tag(self) -> str:
        """Infer model tag from path (mirrors NPU engine's logic)."""
        path = self.model_path
        if "/" in path:
            parent = os.path.dirname(path)
            grandparent = os.path.dirname(parent)
            if grandparent:
                tag = os.path.basename(parent)
                for suffix in ["-npu2", "-instruct", "-it", "_npu2", "_instruct", "_it", "_it_npu2"]:
                    if tag.endswith(suffix):
                        return tag[:-len(suffix)]
                return tag
        return "qwen3"

    def start_server(self, port: int = 8080) -> bool:
        """Start the fused engine in HTTP server mode."""
        self.port = port
        if not os.path.exists(self.engine_path):
            print(f"  ⚠️  Fused engine not found at {self.engine_path}")
            print("  Build: cd engine/fusion && zig build -Doptimize=ReleaseFast")
            return False

        if not os.path.exists(self.model_path):
            print(f"  ⚠️  Model not found at {self.model_path}")
            return False

        cmd = [
            self.engine_path,
            "--model", self.model_path,
            "--xclbin-dir", self.xclbin_dir,
            "--model-tag", self.model_tag,
            "--port", str(port),
            "--policy", self.policy,
            "--kv-pages", "1024",
            "--parallel", "4",
        ]

        print(f"  Starting fused engine (policy={self.policy}): {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.start_time = time.time()
        time.sleep(1)  # Brief wait for server init
        return True

    def start_cli(self, prompt: str, max_tokens: int = 128) -> Optional[str]:
        """Run the fused engine in CLI mode (single prompt)."""
        if not os.path.exists(self.engine_path):
            print(f"  ⚠️  Fused engine not found at {self.engine_path}")
            return None

        cmd = [
            self.engine_path,
            "--model", self.model_path,
            "--xclbin-dir", self.xclbin_dir,
            "--model-tag", self.model_tag,
            "--policy", self.policy,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout

    def stop(self):
        """Stop the fused engine."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


# ── Self-test ──
if __name__ == "__main__":
    import sys

    with FusedEngine() as eng:
        if len(sys.argv) > 1 and sys.argv[1] == "server":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            eng.start_server(port)
            print(f"Fused engine server running on port {port}")
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                print("\nShutting down...")
        else:
            output = eng.start_cli("Hello, world!", 64)
            if output:
                print(output)
            else:
                print("CLI mode failed")
