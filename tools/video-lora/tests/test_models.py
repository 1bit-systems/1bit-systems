"""Smoke tests for video-lora — model-agnostic engine."""

from pathlib import Path


def test_module_imports():
    """At minimum, all modules should import cleanly."""
    from video_lora import __version__, AgnosticPipeline, all_known, lookup
    assert __version__ == "0.3.0"
    assert AgnosticPipeline is not None

    # Registry should have known models
    known = all_known()
    assert "Wan2.2" in known
    assert "CogVideoX" in known
    assert "HunyuanVideo" in known
    assert "LTX2" in known
    assert "Sana Video" in known
    assert "Mochi" in known
    assert "EasyAnimate" in known
    assert "Cosmos" in known
    assert "Allegro" in known
    assert "Motif Video" in known
    assert "ConsisID" in known
    assert "SkyReels V2" in known
    assert "AnimateDiff" in known
    assert "LTX-Video" in known
    assert len(known) >= 14

    # Lookup by alias
    assert lookup("wan") is not None
    assert lookup("hunyuan") is not None
    assert lookup("cogvideo") is not None
    assert lookup("sana") is not None
    assert lookup("mochi") is not None
    assert lookup("cosmos") is not None
    assert lookup("consisid") is not None

    # Lookup by HF ID prefix
    assert lookup("Wan-AI/Wan2.1-T2V-1.3B-Diffusers") is not None
    assert lookup("Tencent/HunyuanVideo") is not None
    assert lookup("Lightricks/LTX-Video") is not None

    # Audio models should be registered
    assert lookup("stable-audio") is not None
    assert lookup("audioldm2") is not None
    assert lookup("longcat") is not None

    # Audio models should have modality="audio"
    assert lookup("stable-audio").modality == "audio"
    assert lookup("audioldm2").modality == "audio"
    assert lookup("longcat").modality == "audio"

    # Video models should have modality="video"
    assert lookup("wan").modality == "video"
    assert lookup("hunyuan").modality == "video"
    assert lookup("sana").modality == "video"

    # Core modules still work
    from video_lora.core.pipeline import VideoPipeline
    assert VideoPipeline is not None

    from video_lora.core.lora_loader import load_lora_into_pipe
    assert load_lora_into_pipe is not None

    from video_lora.utils.export import export_to_wav
    assert export_to_wav is not None


def test_cli_list_models():
    """CLI should list models without crashing."""
    import sys
    from video_lora.cli import main

    old_argv = sys.argv
    sys.argv = ["video-lora", "list-models"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_cli_generate_help():
    """CLI help should show model flexibility."""
    # Verify the CLI generates help without errors
    import sys
    from video_lora.cli import main

    old_argv = sys.argv
    sys.argv = ["video-lora", "--help"]
    try:
        main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def test_consisid_requires_image():
    """ConsisID should require image_path."""
    from video_lora.engine.registry import lookup
    info = lookup("consisid")
    assert info is not None
    assert info.requires_image is True


def test_param_aliases_sana():
    """Sana Video should alias num_frames → frames."""
    from video_lora.engine.registry import lookup
    info = lookup("sana")
    assert info is not None
    assert info.param_aliases.get("num_frames") == "frames"


def test_special_setups():
    """Models with special VAEs should have setup_fn."""
    from video_lora.engine.registry import lookup
    assert lookup("allegro").setup_fn is not None
    assert lookup("skyreels").setup_fn is not None
    assert lookup("animatediff").setup_fn is not None
    # Standard models should NOT have setup_fn
    assert lookup("wan").setup_fn is None
    assert lookup("hunyuan").setup_fn is None
    assert lookup("cogvideo").setup_fn is None
