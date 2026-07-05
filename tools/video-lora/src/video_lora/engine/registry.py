"""Model registry — maps model IDs to pipeline classes, defaults, and special setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ModelInfo:
    """Metadata + setup for a known video model."""

    # Human-readable info
    name: str
    description: str

    # The diffusers pipeline class (imported lazily)
    pipeline_class: Optional[type] = None
    pipeline_import: Optional[str] = None  # e.g. "diffusers:CogVideoXPipeline"

    # Parameter name remapping (our name → pipe.__call__ name)
    param_aliases: dict[str, str] = field(default_factory=dict)

    # Default generation parameters
    defaults: dict[str, Any] = field(default_factory=lambda: {
        "num_frames": 16,
        "width": 640,
        "height": 480,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
    })

    # Modality: "video" or "audio"
    modality: str = "video"

    # Special setup: callable(pipe, device, model_id) → pipe
    setup_fn: Optional[Callable] = None

    # Whether this model requires an image input (I2V / identity)
    requires_image: bool = False

    # Example model IDs for this architecture
    example_ids: list[str] = field(default_factory=list)

    # Short aliases (e.g. "wan", "hunyuan")
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lazy pipeline class loaders — imported only when needed
# ---------------------------------------------------------------------------

def _load_class(module_path: str, class_name: str) -> type:
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _pipeline(module_path: str) -> type:
    """Import a diffusers pipeline class by direct module path."""
    return _load_class(module_path.rsplit(".", 1)[0], module_path.rsplit(".", 1)[1])


def _diffusers_pipeline(module_path: str, class_name: str) -> type:
    """Import a pipeline class from diffusers top-level (not pipelines submodule)."""
    import importlib
    mod = importlib.import_module("diffusers")
    return getattr(mod, class_name)


# ---------------------------------------------------------------------------
# Special setup functions for models with VAEs or other requirements
# ---------------------------------------------------------------------------

def _setup_allegro(pipe, device, model_id):
    """Allegro needs float32 VAE loaded separately."""
    from diffusers import AutoencoderKLAllegro
    vae = AutoencoderKLAllegro.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe.vae = vae
    return pipe


def _setup_skyreels(pipe, device, model_id):
    """SkyReels V2 needs a Wan VAE loaded at float32."""
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe.vae = vae
    return pipe


def _setup_animatediff(pipe, device, model_id):
    """AnimateDiff needs a motion adapter + LCM scheduler."""
    from diffusers import LCMScheduler, MotionAdapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe.motion_adapter = adapter
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if device != "cpu":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to(device)
    return pipe


# ---------------------------------------------------------------------------
# Master registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelInfo] = {}

def _register(info: ModelInfo) -> None:
    """Register a model by its aliases and example IDs."""
    for alias in info.aliases:
        REGISTRY[alias] = info
    for eid in info.example_ids:
        REGISTRY[eid] = info

_register(ModelInfo(
    name="Wan2.2",
    description="Alibaba, 1.3B / 14B, reward LoRAs + camera control",
    pipeline_class=_pipeline("diffusers.pipelines.wan.pipeline_wan.WanPipeline"),
    aliases=["wan", "wan2", "wan2.2"],
    example_ids=[
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ],
    defaults={"num_frames": 16, "width": 640, "height": 480, "guidance_scale": 5.0},
))

_register(ModelInfo(
    name="LTX-Video",
    description="Lightricks, 13B, IC LoRA detailer, V2V control",
    pipeline_class=_pipeline("diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline"),
    aliases=["ltx", "ltx1"],
    example_ids=["Lightricks/LTX-Video"],
    defaults={"num_frames": 49, "width": 704, "height": 480, "guidance_scale": 3.0},
))

_register(ModelInfo(
    name="LTX2",
    description="Lightricks, 2B, next-gen LTX with IC LoRA + HDR",
    pipeline_class=_pipeline("diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline"),
    aliases=["ltx2"],
    example_ids=["Lightricks/LTX-Video-2B-v0.9"],
    defaults={"num_frames": 121, "width": 768, "height": 512, "guidance_scale": 4.0},
))

_register(ModelInfo(
    name="CogVideoX",
    description="Tsinghua/THUDM, 2B / 5B, transformer-based, coherent motion",
    pipeline_class=_pipeline("diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline"),
    aliases=["cogvideo", "cogvideox", "cog"],
    example_ids=["THUDM/CogVideoX-2B", "THUDM/CogVideoX-5B"],
    defaults={"num_frames": 48, "width": 720, "height": 480, "guidance_scale": 6.0},
))

_register(ModelInfo(
    name="HunyuanVideo",
    description="Tencent, 13B, strong T2V/I2V, SkyReels compatible",
    pipeline_class=_pipeline("diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.HunyuanVideoPipeline"),
    aliases=["hunyuan", "hunyuanvideo"],
    example_ids=["Tencent/HunyuanVideo"],
    defaults={"num_frames": 61, "width": 848, "height": 480, "guidance_scale": 6.0},
))

_register(ModelInfo(
    name="AnimateDiff",
    description="Community-driven, 1.5B base, 1000+ motion/style LoRAs",
    pipeline_class=_pipeline("diffusers.pipelines.animatediff.pipeline_animatediff.AnimateDiffPipeline"),
    setup_fn=_setup_animatediff,
    aliases=["animatediff", "animate"],
    example_ids=["SG161222/Realistic_Vision_V5.1_noVAE"],
    defaults={"num_frames": 16, "width": 512, "height": 512, "guidance_scale": 7.5},
))

_register(ModelInfo(
    name="Sana Video",
    description="Efficient linear attention, 2B, very fast on CPU/GPU",
    pipeline_class=_pipeline("diffusers.pipelines.sana_video.pipeline_sana_video.SanaVideoPipeline"),
    aliases=["sana", "sanavideo"],
    param_aliases={"num_frames": "frames"},
    example_ids=["Efficient-Large-Model/SANA-Video_2B_480p_diffusers"],
    defaults={"num_frames": 81, "width": 832, "height": 480, "guidance_scale": 6.0},
))

_register(ModelInfo(
    name="Mochi",
    description="Genmo, 10B, strong prompt adherence",
    pipeline_class=_pipeline("diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline"),
    aliases=["mochi"],
    example_ids=["genmo/mochi-1-preview"],
    defaults={"num_frames": 19, "width": 848, "height": 480, "guidance_scale": 4.5},
))

_register(ModelInfo(
    name="EasyAnimate",
    description="Alibaba, 3B / 7B / 12B, control + inpaint variants",
    pipeline_class=_pipeline("diffusers.pipelines.easyanimate.pipeline_easyanimate.EasyAnimatePipeline"),
    aliases=["easyanimate", "easy"],
    example_ids=[
        "alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers",
        "alibaba-pai/EasyAnimateV5.1-12b-zh",
    ],
    defaults={"num_frames": 49, "width": 512, "height": 512, "guidance_scale": 5.0},
))

_register(ModelInfo(
    name="Cosmos",
    description="NVIDIA, 7B, world model / physics-first generation",
    pipeline_class=_pipeline("diffusers.pipelines.cosmos.pipeline_cosmos_text2world.CosmosTextToWorldPipeline"),
    aliases=["cosmos"],
    example_ids=["nvidia/Cosmos-1.0-Diffusion-7B-Text2World"],
    defaults={"num_frames": 121, "width": 1280, "height": 704, "guidance_scale": 7.0},
))

_register(ModelInfo(
    name="Allegro",
    description="Rhymes AI, custom float32 VAE, 100-step scheduler",
    pipeline_class=_pipeline("diffusers.pipelines.allegro.pipeline_allegro.AllegroPipeline"),
    setup_fn=_setup_allegro,
    aliases=["allegro"],
    example_ids=["rhymes-ai/Allegro"],
    defaults={"num_frames": 24, "width": 640, "height": 360, "guidance_scale": 7.5,
              "num_inference_steps": 100},
))

_register(ModelInfo(
    name="Motif Video",
    description="Motif Technologies, 2B, coherent long-form video",
    pipeline_class=_pipeline("diffusers.pipelines.motif_video.pipeline_motif_video.MotifVideoPipeline"),
    aliases=["motif", "motifvideo"],
    example_ids=["Motif-Technologies/Motif-Video-2B"],
    defaults={"num_frames": 121, "width": 1280, "height": 736, "guidance_scale": 6.0},
))

_register(ModelInfo(
    name="ConsisID",
    description="Identity-consistent I2V, face preservation",
    pipeline_class=_pipeline("diffusers.pipelines.consisid.pipeline_consisid.ConsisIDPipeline"),
    aliases=["consisid", "consis"],
    requires_image=True,
    example_ids=["BestWishYsh/ConsisID-preview"],
    defaults={"num_frames": 49, "width": 720, "height": 480, "guidance_scale": 6.0},
))

_register(ModelInfo(
    name="SkyReels V2",
    description="Skywork, 14B, 540P / 720P diffusion forcing",
    pipeline_class=_pipeline("diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2.SkyReelsV2Pipeline"),
    setup_fn=_setup_skyreels,
    aliases=["skyreels", "skyreelsv2"],
    example_ids=[
        "Skywork/SkyReels-V2-T2V-14B-540P-Diffusers",
        "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
    ],
    defaults={"num_frames": 97, "width": 960, "height": 544, "guidance_scale": 6.0},
))

# ---------------------------------------------------------------------------
# Audio models
# ---------------------------------------------------------------------------

_register(ModelInfo(
    name="Stable Audio Open",
    description="Stability AI, 44.1kHz stereo, up to 47s text-to-audio/SFX",
    pipeline_class=_pipeline("diffusers.pipelines.stable_audio.pipeline_stable_audio.StableAudioPipeline"),
    aliases=["stable-audio", "stableaudio", "audio"],
    example_ids=["stabilityai/stable-audio-open-1.0"],
    modality="audio",
    defaults={"guidance_scale": 7.0, "num_inference_steps": 200,
              "audio_end_in_s": 10.0, "audio_start_in_s": 0.0},
))

_register(ModelInfo(
    name="AudioLDM2",
    description="General text-to-audio (speech, music, SFX), mono/stereo",
    pipeline_class=_pipeline("diffusers.pipelines.audioldm2.pipeline_audioldm2.AudioLDM2Pipeline"),
    aliases=["audioldm", "audioldm2"],
    example_ids=["cvssp/audioldm2"],
    modality="audio",
    defaults={"guidance_scale": 3.5, "num_inference_steps": 200,
              "audio_length_in_s": 10.0},
))

_register(ModelInfo(
    name="LongCat-AudioDiT",
    description="Meituan, high-fidelity waveform diffusion TTS",
    pipeline_class=_pipeline("diffusers.pipelines.longcat_audio_dit.pipeline_longcat_audio_dit.LongCatAudioDiTPipeline"),
    aliases=["longcat", "longcataudiodit"],
    example_ids=["ruixiangma/LongCat-AudioDiT-1B-Diffusers"],
    modality="audio",
    defaults={"guidance_scale": 5.0, "num_inference_steps": 100,
              "audio_length_in_s": 10.0},
))


def lookup(model_id: str) -> Optional[ModelInfo]:
    """Look up a model by short alias, exact HF ID, or HF ID prefix."""
    # 1. Exact match (alias or HF ID)
    if model_id in REGISTRY:
        return REGISTRY[model_id]

    # 2. HF ID prefix match (e.g. "Wan-AI/Wan2.1..." matches "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    for key, info in REGISTRY.items():
        if "/" in key and model_id.startswith(key):
            return info
        # Match by org prefix
        if "/" in model_id and "/" in key:
            org = model_id.split("/")[0]
            key_org = key.split("/")[0]
            if org == key_org:
                return info

    return None


def all_known() -> dict[str, ModelInfo]:
    """Return all uniquely-named models (deduplicated by name)."""
    seen: dict[str, ModelInfo] = {}
    for info in REGISTRY.values():
        seen[info.name] = info
    return seen
