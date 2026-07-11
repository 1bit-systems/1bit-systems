"""Smoke tests for video-lora model backends."""

from pathlib import Path


def test_module_imports():
    """At minimum, all modules should import cleanly."""
    from video_lora import __version__
    assert __version__ == "0.1.0"

    from video_lora.core.pipeline import VideoPipeline
    assert VideoPipeline is not None

    from video_lora.core.lora_loader import load_lora_into_pipe
    assert load_lora_into_pipe is not None

    from video_lora.cli import main, register_models
    assert main is not None
    assert register_models is not None


def test_cli_list_models():
    """CLI should list available models without crashing."""
    import sys
    from video_lora.cli import main

    # Monkey-patch argv for list-models
    old_argv = sys.argv
    sys.argv = ["video-lora", "list-models"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
