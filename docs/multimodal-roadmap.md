# Multimodal Roadmap — Image & Video Generation

**The complete package:** text (NPU) + image (GPU) + video (GPU) + audio (GPU).
All from one machine. One API. Zero Python. Every modality runs on Strix Halo's combined NPU + RDNA 3.5 GPU.

---

## Current state (Phase 0) ✅

The stack already exists for image generation and is proven:

```
1bit-mobile ──→ Lemonade ──→ sd-server (stable-diffusion.cpp) ──→ Vulkan GPU
                                                                    │
                                                         20+ model architectures:
                                                         Flux, SD3.5, SDXL, SD1.5/2.1,
                                                         Qwen-Image, HiDream, Chroma, …
```

| Component | Status | Details |
|---|---|---|
| **Image generation endpoint** | ✅ `POST /v1/images/generations` | Full OpenAI-compatible via Lemonade sd_server |
| **Image editing** | ✅ `POST /v1/images/edits` | Multipart upload, supports masks |
| **Image variations** | ✅ `POST /v1/images/variations` | Generate from source image |
| **Upscaling** | ✅ `POST /api/v1/images/upscale` | ESRGAN via sd-cli subprocess |
| **Mobile client** | ✅ `1bit-mobile` ImagesEndpoint | generate(), edit() with b64_json or URL |
| **Model manager** | ✅ Lemonade auto-downloads | sd.cpp GGUF models from HuggingFace |
| **Backend auto-detect** | ✅ ROCm → Vulkan → Metal → CUDA → CPU | Picks best available |

**What's needed to ship:** desktop UI for image gen + image gen documentation on the site.

---

## Phase 1 Status 🎬

| Component | Status | Details |
|---|---|---|
| **Lemonade IVideoServer** | ✅ Built | Full C++ backend — video_server.h, video_server.cpp, router, HTTP handler |
| **ModelType::VIDEO** | ✅ Added | Enum + label detection + routing |
| **sd-server binary** | ✅ Installed | master-748, Vulkan build, Radeon 8060S detected |
| **Wan2.1 T2V 1.3B model** | ✅ Downloaded | safetensors format (5.4 GB diffusion, 6.3 GB UMT5, 243 MB VAE) |
| **Mobile VideoEndpoint** | ✅ Built | `POST /v1/video/generations` with full request/response types |
| **Lemonade config** | ✅ Created | `user_models.json` with Wan2.1-T2V-1.3B entry, `sd-cpp-video` recipe in config |
| **Model loading** | ✅ Diffusion + VAE load OK | safetensors format works (GGUF has 5D tensor limitation) |
| **Text encoder init** | ❌ UMT5 not recognized by sd.cpp | Known format compatibility issue — sd.cpp Wan implementation doesn't detect UMT5 FP8 safetensors as text encoder |

### Known issue: UMT5 text encoder format

sd.cpp's Wan2.1 implementation (master-748) doesn't recognize the UMT5-XXL text encoder in either GGUF or safetensors format. The diffusion model and VAE load correctly, but `text_encoders` shows 0 MB. This is likely a tensor-name-matching issue in `conditioner.hpp` that will be fixed upstream.

**Workaround:** Track https://github.com/leejet/stable-diffusion.cpp for Wan2.1 text encoder fixes. Once upstream fixes the UMT5 detection, the existing pipeline (Lemonade backend + sd-server + models) will work end-to-end.

---

## Phase 1: Video Generation 🎬 (next — sd.cpp Wan2.1)

sd.cpp (stable-diffusion.cpp) already supports **video generation natively** — Wan2.1 T2V 1.3B/14B, Wan2.2 I2V 14B, LTX-2.3 22B — all via the same `sd-server` binary using `-M vid_gen`. The gap is that Lemonade has no `IVideoServer` interface or routing for video yet.

### What needs building

```
┌──────────────────────────────────────────────────────────────┐
│  Lemonade backend: VideoServer                               │
│                                                              │
│  New files:                                                  │
│  • include/lemon/backends/video_server.h   — IVideoServer   │
│  • server/backends/video_server.cpp         — sd.cpp wrapper │
│  • include/lemon/server_capabilities.h      — add IVideoServer│
│  • include/lemon/model_types.h              — add VIDEO type │
│  • include/lemon/router.h                   — route /v1/video│
│  • server/server.cpp                        — HTTP handler   │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                    sd-server (sd.cpp) — GPU Vulkan
                    ┌──────────────────────────┐
                    │ -M vid_gen               │
                    │ args: --diffusion-model  │
                    │       --llm (T5)         │
                    │       --vae              │
                    └──────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
           Wan2.1 T2V 1.3B             Wan2.2 I2V 14B
           (2.6 GB fp16, Q8: ~3 GB)    (10 GB fp16, Q8: ~5 GB)
           ~5-8 min for 33f@480p       ~8-12 min
           Best fit for Strix Halo     With --offload-to-cpu
```

### Step-by-step

1. **Add `ModelType::VIDEO`** to `model_types.h`
2. **Add `IVideoServer`** to `server_capabilities.h`:
   ```cpp
   class IVideoServer : public virtual ICapability {
   public:
       virtual ~IVideoServer() = default;
       virtual json video_generations(const json& request) = 0;
   };
   ```
3. **Create `VideoServer` backend** — wraps sd-server with video args:
   - Same install params as sd_server (same binary)
   - Load with `--diffusion-model wan.gguf --llm umt5.gguf --vae wan_vae.gguf`
   - Proxies `/v1/video/generations` → `sd-server`'s video endpoint
   - Parameters: prompt, width, height, num_frames, fps, seed, steps, cfg_scale
4. **Add routing** in `router.h/cpp` — route model type `video` to `VideoServer`
5. **Add HTTP handler** `POST /v1/video/generations` in `server.cpp`
6. **Wire mobile client** — add `VideoEndpoint` mirroring `ImagesEndpoint`
7. **Target**: Wan2.1 1.3B at ~5 min per 4-second 480p clip on Strix Halo

### Video models ranked for Strix Halo

| Model | Params | VRAM Q8 | Est. time 480p | Notes |
|---|---|---|---|---|
| **Wan2.1 T2V 1.3B** | 1.3B | ~3 GB | 5-8 min | **Best first target** — fits VRAM, fast |
| **Wan2.2 TI2V 5B** | 5B | ~5 GB | 8-12 min | Text+image→video, good second target |
| **Wan2.1 T2V 14B** | 14B | ~14 GB | 15-25 min | Needs offload-to-cpu, Q8 marginal |
| **LTX-2.3 22B** | 22B | ~16 GB (Q4) | 10-20 min | Q4_K_M, newest architecture |

---

## Phase 2: Desktop Client 🖥️

1bit.systems has a mobile client but no desktop image/video generation UI. Build one.

- **Web UI** — extend the existing landing page or add a `/studio` route with:
  - Text-to-image panel (prompt, model, size, seed, steps, cfg)
  - Image-to-image / inpainting panel
  - Text-to-video panel (prompt, model, frames, fps)
  - Gallery view of generated images/videos
  - No Python — pure HTML/CSS/JS, talks to Lemonade API
- **Desktop app** — optionally: Flutter desktop build of 1bit-mobile (already cross-platform)

### Steps

1. Design Studio UI mockup in `site/studio.html`
2. Build image gen form → `POST /v1/images/generations` → display b64 result
3. Build video gen form → `POST /v1/video/generations` → poll job → display MP4
4. Add to site navigation, update footer
5. **Target**: Web studio live on 1bit.systems/studio

---

## Phase 3: NPU INT8 Diffusion 🔮

The NPU has 50 TOPS INT8. Diffusion U-Nets and MMDiTs are mostly matmul — same primitive the NPU already runs for text LLMs at 97 tok/s. An INT8 MMDiT forward pass through a 3B-param model should take ~60-80 ms per step.

### What this unlocks

```
Current (GPU):   28 steps × ~300ms = 8.4s per image (Flux, Vulkan)
NPU target:      28 steps × ~70ms  = 2.0s per image (INT8 xclbins)
Speedup:         ~4×
```

### What needs building

1. **INT8 diffusion xclbins** — MMDiT attention + feed-forward on NPU:
   - MMDiT has QKV + O projections (same as text transformer) + cross-attention
   - Reuses existing INT8 GEMM xclbin patterns from `n1_core_i8_v2.py`
   - New kernels: modulated double-stream block (adaLN modulation)
2. **VAE decoder on NPU** — conv2d INT8 kernels for latent→pixel decode:
   - Currently runs on GPU; porting to NPU eliminates GPU dependency
   - Chess C++ conv2d kernels (or reshape+gemm for 1×1 convs)
3. **CLIP/T5 text encoder** — INT8 on NPU, or borrow from Lemonade's LLM backend
4. **Sampling loop in C++** — DDIM / Flow-matching scheduler, CFG scaling
   - No Python — pure C++23, same pattern as the text inference loop

### Model targets

| Model | Architecture | INT8 weights | Target NPU |
|---|---|---|---|
| **FLUX.1-schnell** | 3.5B MMDiT (double-stream) | ~3.5 GB | ✅ Best — 4-step distilled |
| **SD3.5** | 2.6B MMDiT (single-stream) | ~2.6 GB | ✅ Good — standard quality |
| **Wan2.1 1.3B** | 1.3B 3D VAE + ST-DiT | ~1.3 GB | ✅ Video — UNet-type, straightforward |

### Steps

1. Export SD3.5/FLUX weights to Q4NX format (one-time conversion script)
2. Build `mmdit_i8_xclbin` — Chess C++ for INT8 MMDiT attention
3. Build `vae_conv_i8_xclbin` — Chess C++ for INT8 conv2d tiles
4. Build C++ sampling orchestrator (`int8_diffusion.cpp`)
5. Integrate as a Lemonade backend (`lmn_diffusion_server`)
6. **Target**: 2.0s per FLUX.1-schnell image on NPU

---

## Phase 4: Video Editing + Advanced Pipelines 🎥

Once video generation works on GPU and image gen is on NPU:

1. **Video-to-video (Wan2.2 I2V)** — input video + prompt → modified video
   - sd.cpp already supports this; need to expose `POST /v1/video/edits`
   - Mobile client: upload source video → edit → download result
2. **Frame interpolation** — increase FPS of generated video
   - RIFE or FILM model via sd.cpp or separate backend
3. **Video upscaling** — ESRGAN for video frames
   - Reuse sd_server upscale_via_cli, run per-frame
4. **Image-to-video with audio** — image + audio narration → talking video
   - Wan2.2 TI2V + Whisper for lip sync
   - 1bit-mobile already has audio narration upload

---

## Phase 5: Productionization 🚀

- [ ] Desktop web studio (`/studio`) on 1bit.systems
- [ ] Docker image with image gen + video gen included
- [ ] Documentation: image gen guide, video gen guide, model management
- [ ] Benchmark page: image gen tok/s, video gen s/frame across models
- [ ] Packaging update: deb/snap includes model download for image/video
- [ ] Ollama integration: Modelfile for SD models

---

## Architecture diagram (future state)

```
┌──────────────────────────────────────────────────────────────┐
│                    1bit-mobile (Flutter)                      │
│  ChatEndpoint · ImagesEndpoint · VideoEndpoint · VisionClient │
└─────┬────────────┬──────────────┬───────────────────────────┘
      │            │              │
      ▼            ▼              ▼
┌─────────┐ ┌──────────────┐ ┌──────────────┐
│ NPU     │ │  Lemonade    │ │ 1bit-vision  │
│ Engine  │ │  (OmniRouter)│ │ -server      │
│ :9090   │ │  :8180       │ │ :8787        │
│ text    │ │              │ │ product rec  │
└─────────┘ └──────┬───────┘ └──────────────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
   ┌──────────┐ ┌──────┐ ┌────────┐
   │ sd-server│ │NPU   │ │ Whisper│
   │ (sd.cpp) │ │Diff. │ │ (audio)│
   │ GPU/VK   │ │INT8  │ └────────┘
   │ image/*  │ │phase3│
   │ video/*  │ └──────┘
   └──────────┘
```

## Summary timeline

| Phase | What | When | Depends on |
|---|---|---|---|
| **0** | Image gen (already working) | ✅ Now | — |
| **1** | Video gen via sd.cpp + Lemonade | ⏳ Next | Lemonade sd_server backend extension |
| **2** | Desktop web studio | After Ph1 | Phase 1 endpoint |
| **3** | NPU INT8 diffusion | After Ph2 | Chess xclbin pipeline for MMDiT |
| **4** | Video editing pipelines | After Ph3 | Phase 1 + 3 |
| **5** | Production shipping | Ongoing | All phases |
