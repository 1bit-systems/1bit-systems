"""Smoke tests for video-lora model backends."""

import io
import re
import sys
from pathlib import Path


def test_module_imports():
    """At minimum, all modules should import cleanly."""
    from video_lora import __version__
    # Check version format (semver) instead of exact string — avoids brittleness
    assert re.match(r"^\d+\.\d+\.\d+$", __version__), \
        f"__version__ should be semver, got {__version__!r}"

    from video_lora.core.pipeline import VideoPipeline
    assert VideoPipeline is not None

    from video_lora.core.lora_loader import load_lora_into_pipe
    assert load_lora_into_pipe is not None

    from video_lora.cli import main, register_models
    assert main is not None
    assert register_models is not None


def test_cli_list_models():
    """CLI should list available models without crashing and produce correct output."""
    from video_lora.cli import main

    # Monkey-patch argv for list-models and capture stdout
    old_argv = sys.argv
    old_stdout = sys.stdout
    sys.argv = ["video-lora", "list-models"]
    captured = io.StringIO()
    sys.stdout = captured
    try:
        main()
    except SystemExit as e:
        # Only allow exit code 0 (successful completion via argument parser)
        if e.code not in (None, 0):
            raise  # Re-raise non-zero exits — they indicate real errors
    finally:
        sys.stdout = old_stdout
        sys.argv = old_argv

    output = captured.getvalue()
    # Verify the output contains expected model names
    assert "wan" in output, f"'wan' not in list-models output: {output!r}"
    assert "ltx" in output, f"'ltx' not in list-models output: {output!r}"
    assert "animatediff" in output, f"'animatediff' not in list-models output: {output!r}"
