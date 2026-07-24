"""
1bit-systems ComfyUI Integration

Custom nodes that expose 1bit-systems's NPU-accelerated inference engine
to ComfyUI workflows. Supports:

- LLM text generation (via 1BP models on NPU/GPU)
- Vision-Language (image understanding via 1BP VLMs)
- Image generation (via stable-diffusion.cpp integration)
- Audio TTS/STT (via audio.cpp integration)
- LoRA hot-loading for any model

Architecture:
    ComfyUI (Python)  <->  HTTP  <->  1bit unified_server (C++)
    ComfyUI (Python)  <->  HTTP  <->  image_server / jarvis_server (C++)

Requirements:
    - 1bit-systems servers running (unified_server, image_server, jarvis_server)
    - httpx (pip install httpx)
"""
import json
import os
import base64
import io
import httpx
from PIL import Image
import numpy as np
import torch

# ─── Configuration ─────────────────────────────────────────────────
DEFAULT_UNIFIED_URL = "http://127.0.0.1:8088/v1"
DEFAULT_IMAGE_URL   = "http://127.0.0.1:8089/v1"
DEFAULT_AUDIO_URL   = "http://127.0.0.1:8090/v1"

# ─── Helper: image to base64 ──────────────────────────────────────
def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ═══════════════════════════════════════════════════════════════════
# Node: 1BP LLM Text Generator
# ═══════════════════════════════════════════════════════════════════
class OneBP_LLM_Generate:
    """Generate text using a 1BP model via the unified_server."""
    
    CATEGORY = "1bit-systems/LLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Hello, world!"}),
                "model": ("STRING", {"default": "zaya1-8b"}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "server_url": ("STRING", {"default": DEFAULT_UNIFIED_URL}),
            }
        }
    
    def generate(self, prompt, model, max_tokens, temperature,
                 system_prompt="", server_url=DEFAULT_UNIFIED_URL):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{server_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[ERROR: {e}]"
        
        return (text,)

# ═══════════════════════════════════════════════════════════════════
# Node: 1BP Vision-Language (Image Understanding)
# ═══════════════════════════════════════════════════════════════════
class OneBP_VLM_Understand:
    """Analyze an image using a 1BP VLM."""
    
    CATEGORY = "1bit-systems/Vision"
    FUNCTION = "analyze"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model": ("STRING", {"default": "zaya1-vl-8b"}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
            },
            "optional": {
                "server_url": ("STRING", {"default": DEFAULT_UNIFIED_URL}),
            }
        }
    
    def analyze(self, image, prompt, model, max_tokens,
                server_url=DEFAULT_UNIFIED_URL):
        # Convert ComfyUI tensor (B,H,W,C) to PIL
        if isinstance(image, torch.Tensor):
            img_np = image[0].cpu().numpy()
            if img_np.shape[-1] == 1:
                img_np = np.squeeze(img_np, axis=-1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            img_pil = image
        
        b64 = pil_to_base64(img_pil, "PNG")
        data_url = f"data:image/png;base64,{b64}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{server_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[ERROR: {e}]"
        
        return (text,)

# ═══════════════════════════════════════════════════════════════════
# Node: 1BP Image Generation (via stable-diffusion.cpp)
# ═══════════════════════════════════════════════════════════════════
class OneBP_Image_Generate:
    """Generate images using the 1bit diffusion server."""
    
    CATEGORY = "1bit-systems/Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "model": ("STRING", {"default": "flux.1-dev"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "lora_path": ("STRING", {"default": ""}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "seed": ("INT", {"default": -1}),
                "server_url": ("STRING", {"default": DEFAULT_IMAGE_URL}),
            }
        }
    
    def generate(self, prompt, width, height, steps, cfg_scale, model,
                 negative_prompt="", lora_path="", lora_strength=1.0,
                 seed=-1, server_url=DEFAULT_IMAGE_URL):
        params = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }
        if lora_path:
            params["lora_paths"] = [lora_path]
            params["lora_strengths"] = [lora_strength]
        
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{server_url}/images/generations",
                    json=params
                )
                resp.raise_for_status()
                data = resp.json()
                
                # Decode base64 image
                b64 = data["data"][0]["b64_json"]
                img_bytes = base64.b64decode(b64)
                img_pil = Image.open(io.BytesIO(img_bytes))
                
                # Convert to ComfyUI tensor
                img_np = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)[None, ...]
                
                return (img_tensor,)
        except Exception as e:
            print(f"[ERROR] OneBP Image Generate: {e}")
            # Return a black image on error
            dummy = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (dummy,)

# ═══════════════════════════════════════════════════════════════════
# Node: 1BP TTS (Audio Generation)
# ═══════════════════════════════════════════════════════════════════
class OneBP_TTS:
    """Generate speech from text via the jarvis_server (audio.cpp)."""
    
    CATEGORY = "1bit-systems/Audio"
    FUNCTION = "synthesize"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, I am a 1bit AI assistant."}),
                "voice": ("STRING", {"default": "default"}),
            },
            "optional": {
                "server_url": ("STRING", {"default": DEFAULT_AUDIO_URL}),
            }
        }
    
    def synthesize(self, text, voice, server_url=DEFAULT_AUDIO_URL):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{server_url}/audio/speech",
                    json={
                        "model": "tts-1",
                        "input": text,
                        "voice": voice,
                    }
                )
                resp.raise_for_status()
                
                # Return raw audio bytes with sample rate
                audio_bytes = resp.content
                sample_rate = 24000
                
                return ({"waveform": audio_bytes, "sample_rate": sample_rate},)
        except Exception as e:
            print(f"[ERROR] OneBP TTS: {e}")
            return ({"waveform": b"", "sample_rate": 24000},)

# ═══════════════════════════════════════════════════════════════════
# Node: 1BP LoRA Loader (for LLM text gen)
# ═══════════════════════════════════════════════════════════════════
class OneBP_LoRA_Loader:
    """Hot-load a LoRA adapter into the 1bit inference engine."""
    
    CATEGORY = "1bit-systems/LoRA"
    FUNCTION = "load_lora"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "/path/to/lora.gguf"}),
                "model": ("STRING", {"default": "zaya1-8b"}),
            },
            "optional": {
                "server_url": ("STRING", {"default": DEFAULT_UNIFIED_URL}),
            }
        }
    
    def load_lora(self, lora_path, model, server_url=DEFAULT_UNIFIED_URL):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{server_url}/lora/load",
                    json={
                        "model": model,
                        "lora_path": lora_path,
                    }
                )
                resp.raise_for_status()
                status = f"LoRA loaded: {lora_path}"
        except Exception as e:
            status = f"[ERROR] {e}"
        
        return (status,)

# ═══════════════════════════════════════════════════════════════════
# Node Mappings
# ═══════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "OneBP_LLM_Generate": OneBP_LLM_Generate,
    "OneBP_VLM_Understand": OneBP_VLM_Understand,
    "OneBP_Image_Generate": OneBP_Image_Generate,
    "OneBP_TTS": OneBP_TTS,
    "OneBP_LoRA_Loader": OneBP_LoRA_Loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneBP_LLM_Generate": "1BP LLM Generate",
    "OneBP_VLM_Understand": "1BP VLM Analyze Image",
    "OneBP_Image_Generate": "1BP Image Generate",
    "OneBP_TTS": "1BP Text-to-Speech",
    "OneBP_LoRA_Loader": "1BP LoRA Loader",
}
