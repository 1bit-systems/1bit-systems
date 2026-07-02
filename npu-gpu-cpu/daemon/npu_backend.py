#!/usr/bin/env python3
"""
NPU Backend — subprocess-based inference engine.

Spawns the NPU inference binary as a subprocess and communicates
via stdin/stdout with newline-delimited JSON.

Usage:
    engine = NPUEngine()
    engine.start()
    tok = engine.generate(151643)  # prefill
    tok = engine.generate(tok)     # continue generating
    engine.reset()
    engine.stop()
"""

import json
import os
import subprocess
import time
from typing import Optional


class NPUEngine:
    """NPU engine subprocess manager."""

    # Paths for the NPU engine binary
    ENGINE_PATH = os.environ.get(
        "NPU_ENGINE_PATH",
        "/home/bcloud/1bit-systems/engine/npu/build/npu_engine_all"
    )
    DEFAULT_MODEL = os.environ.get(
        "NPU_MODEL_PATH",
        "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
    )

    # LD_LIBRARY_PATH for XRT
    LD_LIBRARY_PATH = os.environ.get(
        "NPU_LD_PATH",
        "/home/bcloud/torch2aie/toolchain/mlir_aie.libs:"
        "/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64"
    )

    def __init__(self, engine_path: str = None, model_path: str = None):
        self.engine_path = engine_path or self.ENGINE_PATH
        self.model_path = model_path or self.DEFAULT_MODEL
        self.proc: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.gen_count = 0
        self.total_ms = 0.0

    def start(self) -> bool:
        """Start the NPU inference subprocess."""
        if not os.path.exists(self.engine_path):
            print(f"  ⚠️  NPU engine not found at {self.engine_path}")
            return False

        if not os.path.exists(self.model_path):
            print(f"  ⚠️  Model not found at {self.model_path}")
            return False

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self.LD_LIBRARY_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

        print(f"  Starting NPU engine: {self.engine_path}")
        self.proc = subprocess.Popen(
            [self.engine_path, self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        self.start_time = time.time()
        self.gen_count = 0
        self.total_ms = 0.0
        return True

    def _send_recv(self, req: dict) -> Optional[dict]:
        """Send a JSON request and receive response."""
        if not self.proc or self.proc.poll() is not None:
            print("  ⚠️  NPU engine not running")
            return None

        line = json.dumps(req) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            # Check for errors
            err = self.proc.stderr.read()
            if err:
                print(f"  ⚠️  NPU engine error: {err.strip()}")
            return None
        try:
            return json.loads(resp_line.strip())
        except json.JSONDecodeError as e:
            print(f"  ⚠️  NPU engine JSON error: {e} [{resp_line.strip()}]")
            return None

    def generate(self, token: int) -> Optional[int]:
        """Generate next token. Token < 0 means continue without new input."""
        if token < 0:
            return self._generate_continue()
        resp = self._send_recv({"token": token})
        if resp is None:
            return None
        if "token" in resp:
            self.gen_count += 1
            if "ms" in resp:
                self.total_ms += resp["ms"]
            return resp["token"]
        return None

    def _generate_continue(self) -> Optional[int]:
        """Generate without setting new input token."""
        resp = self._send_recv({"continue": True})
        if resp is None:
            return None
        if "token" in resp:
            self.gen_count += 1
            if "ms" in resp:
                self.total_ms += resp["ms"]
            return resp["token"]
        return None

    def reset(self) -> bool:
        """Reset KV cache."""
        resp = self._send_recv({"reset": True})
        return resp is not None and resp.get("ok", False)

    def stats(self) -> dict:
        """Get timing stats."""
        resp = self._send_recv({"stats": True})
        if resp:
            return resp
        return {"ms_per_token": 0, "tokens": 0}

    @property
    def ms_per_token(self) -> float:
        if self.gen_count == 0:
            return 0.0
        return self.total_ms / self.gen_count

    def stop(self):
        """Stop the NPU engine subprocess."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── Self-test ──
if __name__ == "__main__":
    with NPUEngine() as eng:
        print("NPU engine started")
        
        # Prefill tokens
        prefill = [151643, 872, 198, 11852, 151644, 198, 151643, 77091, 198]
        for i, tok in enumerate(prefill):
            result = eng.generate(tok)
            if result is None:
                print(f"  FAIL at prefill {i}")
                break
            if i % 3 == 0:
                print(f"  prefill {i+1}/{len(prefill)}")
        
        # Generate tokens — use continue (no new input)
        print("  generating...")
        for i in range(8):
            tok = eng.generate(-1)  # continue from last token
            if tok is None:
                print(f"  FAIL at gen {i}")
                break
            print(f"  [{i}] {tok}")
        
        print(f"  ms/tok: {eng.ms_per_token:.0f}")
        print("  Done!")
