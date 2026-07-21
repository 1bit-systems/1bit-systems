# Vision Module — Image Preprocessing for VL Models

## Architecture

```
user uploads image (base64 / URL)
         │
         ▼
  vl_processor.h/.cpp       ← Image decode via stb_image (vendored)
  ├─ vl_load_image()           JPEG/PNG → float [0..1]
  ├─ vl_resize_normalize()     Bilinear resize + mean/std norm
  ├─ vl_load_from_memory()     Decode from raw bytes (base64 case)
  ├─ vl_decode_base64_image()  data:image/...;base64 → raw bytes
  └─ vl_download_image()       HTTP GET via curl → raw bytes
         │
         ▼ (optional GPU acceleration)
  kernels/vl_resize_norm.hip  ← HIP kernel for Radeon 8060S
  ├─ vl_resize_norm_kernel()    GPU bilinear resize + normalize
  └─ vl_resize_norm_launch()    Host-side launcher (HWC or NCHW)
         │
         ▼
  processed pixels → ViT forward → text decoder (via forward_embed)
```

## Files Created (all additive — no existing file modified)

| File | Purpose | Deps |
|------|---------|------|
| `include/vl_preprocess.h` | Lightweight image load + resize + normalize | stb_image (vendored) |
| `include/vl_processor.h` | VlProcessor class + base64 decode | vl_preprocess.h |
| `src/vl_processor.cpp` | stb_image impl + download + base64 | stb_image + curl |
| `kernels/vl_resize_norm.hip` | GPU-accelerated resize+norm kernel | ROCm HIP |
| `tools/vision_server.cpp` | Standalone VL inference HTTP server | httplib + backend_manager + vl_image |

## Files Modified (minimal, easy to cherry-pick)

| File | Change |
|------|--------|
| `CMakeLists.txt` | Added 3 lines: `vl_image` static lib target, added `vl_resize_norm.hip` to rocm_cpp, added `vision_server` target |
| `tools/unified_server.cpp` | Added 4 lines: `#include "vl_processor.h"`, `image_url` parsing in content loop, `forward_embed()` injection loop |
| `src/backend.h` | Added `virtual int forward_embed(const float*)` — already present upstream |

## Cherry-Pick Instructions

To rebase this on upstream changes:

```bash
# From the 1bit-systems repo root:
git fetch upstream main

# The additive files clobber-free cherry-pick:
git checkout upstream/main -- \
    include/vl_preprocess.h \
    include/vl_processor.h \
    src/vl_processor.cpp \
    kernels/vl_resize_norm.hip \
    tools/vision_server.cpp \
    docs/vision-module.md

# Manual merge for modified files:
#   CMakeLists.txt — 3 additions (search for "vl_image", "vl_resize_norm", "vision_server")
#   tools/unified_server.cpp — search for "vl_processor.h" and "vision_images"
```

## Integration Points

### 1. HTTP API (OpenAI-compatible)
```
POST /v1/chat/completions
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this:"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

### 2. GPU Acceleration Path
The HIP kernel (`vl_resize_norm.hip`) runs on Radeon 8060S:
1. Upload raw uint8 image to GPU
2. Launch `vl_resize_norm_kernel` (16x16 thread blocks)
3. Download processed float pixels to host OR keep on device
4. Feed to ViT (on GPU, future work)

### 3. ViT Forward Pass
The actual ViT forward is in `tools/vision_qwen2vl_poc.cpp` (CPU) and should be
extracted into a shared library for production use. See that file's header
comment for the complete reference architecture.

## Why Not OpenCV?

| Requirement | Our Approach | OpenCV Cost |
|-------------|-------------|-------------|
| JPEG/PNG decode | stb_image (96 KB header) | +20-60 MB linked |
| Bilinear resize | 30 lines of C++ | Same function, huge dep |
| Mean/std normalize | 5 lines of C++ | Same |
| Base64 decode | 40 lines of C++ | Not in OpenCV |
| GPU resize | HIP kernel (60 lines) | OpenCV HIP module |
| Build complexity | 0 external deps | cmake subproject |

OpenCV is the gold standard for computer vision research. For a minimal
inference engine that runs on one AMD laptop, the cost/benefit doesn't pencil
out — stb_image + 150 lines of C++ does everything we need.
