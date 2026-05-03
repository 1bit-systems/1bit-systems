#!/usr/bin/env python3
"""
Queue or run a local 3D asset generation job for 1bit OmniRouter.

Default target:
  backend=hunyuan3d
  hf_model=tencent/Hunyuan3D-2.1

Set ONEBIT_3D_ASSET_RUNNER to an executable that accepts:

  --backend <name> --model <hf-id> --prompt <text> --source-image <path>
  --format <glb|obj|ply> --quality <draft|standard|high> --output-dir <dir>

Until a runner is configured, this creates a durable job folder with
request.json so voice/text OmniRouter flows can be tested without pretending
that the heavy 3D runtime is ready.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BACKEND_MODELS = {
    "hunyuan3d": "tencent/Hunyuan3D-2.1",
    "trellis": "microsoft/TRELLIS-text-large",
    "trellis2": "microsoft/TRELLIS.2-4B",
    "sf3d": "stabilityai/stable-fast-3d",
}


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:48] or "asset"


def token_status() -> str:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return "env"
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ):
        if path.is_file() and path.read_text(errors="ignore").strip():
            return str(path)
    return "missing"


def main() -> int:
    req = json.load(sys.stdin)
    prompt = str(req.get("prompt", "")).strip()
    if not prompt:
        print(json.dumps({"status": "error", "error": "missing prompt"}))
        return 2

    fmt = str(req.get("format") or "glb").lower()
    if fmt not in {"glb", "obj", "ply"}:
        fmt = "glb"
    quality = str(req.get("quality") or "draft").lower()
    if quality not in {"draft", "standard", "high"}:
        quality = "draft"
    backend = str(req.get("backend") or "hunyuan3d").lower()
    if backend not in BACKEND_MODELS:
        backend = "hunyuan3d"

    hf_model = str(req.get("hf_model") or "").strip() or BACKEND_MODELS[backend]
    out_root = Path(req.get("output_dir") or "/tmp/1bit-omni/assets").expanduser()
    digest = hashlib.sha1(f"{time.time_ns()}:{backend}:{prompt}".encode()).hexdigest()[:8]
    job_dir = out_root / f"{int(time.time())}-{backend}-{slugify(prompt)}-{digest}"
    job_dir.mkdir(parents=True, exist_ok=True)

    source_image = str(req.get("source_image") or "").strip()
    repo_path = str(req.get("repo_path") or "").strip()
    job = {
        "backend": backend,
        "hf_model": hf_model,
        "prompt": prompt,
        "source_image": source_image,
        "format": fmt,
        "quality": quality,
        "repo_path": repo_path,
        "job_dir": str(job_dir),
        "hf_token": token_status(),
    }
    (job_dir / "request.json").write_text(json.dumps(job, indent=2) + "\n")

    runner = os.environ.get("ONEBIT_3D_ASSET_RUNNER", "").strip()
    if not runner:
        note = "Hunyuan3D supports text/image-to-3D; TRELLIS.2 and SF3D are image-to-3D oriented."
        print(json.dumps({
            "status": "queued",
            "backend": backend,
            "hf_model": hf_model,
            "hf_token": job["hf_token"],
            "job_dir": str(job_dir),
            "request": str(job_dir / "request.json"),
            "next": "Set ONEBIT_3D_ASSET_RUNNER to a local Hunyuan3D/TRELLIS/SF3D wrapper command to produce the asset.",
            "note": note,
        }))
        return 0

    cmd = [
        runner,
        "--backend", backend,
        "--model", hf_model,
        "--prompt", prompt,
        "--format", fmt,
        "--quality", quality,
        "--output-dir", str(job_dir),
    ]
    if source_image:
        cmd += ["--source-image", source_image]
    if repo_path:
        cmd += ["--repo-path", repo_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (job_dir / "runner.stdout").write_text(proc.stdout)
    (job_dir / "runner.stderr").write_text(proc.stderr)
    if proc.returncode != 0:
        print(json.dumps({
            "status": "failed",
            "backend": backend,
            "hf_model": hf_model,
            "job_dir": str(job_dir),
            "exit_code": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }))
        return proc.returncode

    assets = []
    for ext in ("glb", "obj", "ply"):
        assets.extend(sorted(str(p) for p in job_dir.glob(f"*.{ext}")))
    print(json.dumps({
        "status": "complete" if assets else "runner_finished",
        "backend": backend,
        "hf_model": hf_model,
        "job_dir": str(job_dir),
        "assets": assets,
        "stdout": proc.stdout[-2000:],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
