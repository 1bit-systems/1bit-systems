---
type: Capability
title: Vision
description: Image analysis via Qwen3-VL-4B multimodal model on NPU. Accepts uploaded images and base64-encoded image data.
tags: [vision, multimodal, npu, image]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

JARVIS supports multimodal vision: send an image with your message and it will describe, analyze, or answer questions about it.

## Model

- **Model**: Qwen3-VL-4B
- **Backend**: NPU via FLM
- **Speed**: ~11 tok/s
- **Config key**: `vision_model: "qwen3vl-it:4b"`

## How It Works

1. User sends text + image (via HTTP file upload or WebSocket base64)
2. Image is converted to base64 data URI (`data:image/jpeg;base64,...`)
3. LLM message content is a list: `[{type: "text"}, {type: "image_url"}]`
4. FLM NPU processes multimodal input
5. Response streamed back as text

## Upload Endpoints

```bash
# Via HTTP upload
curl -X POST http://localhost:8080/api/upload \
  -F "file=@photo.jpg" \
  -F "category=image"

# Via WebSocket with image
{
  "type": "message",
  "text": "What's in this image?",
  "image": "<base64-encoded-jpeg>"
}
```

## Citations

[1] [Data Flow](/architecture/data_flow.md)
[2] [Chat](/capabilities/chat.md)
