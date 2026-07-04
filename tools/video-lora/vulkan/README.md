# Vulkan Compute Backend (Pure Zig)

Low-level Vulkan compute shaders for SD1.5/AnimateDiff inference.
Zero Python, zero C — pure Zig + GLSL, following the Zinc engine's architecture.

## Quick Start

```bash
cd video-lora/vulkan
zig build run -- --prompt "a cat walking, cinematic lighting" --frames 8
zig build run -- --help
```

## Architecture

```
vulkan/
├── build.zig           # Zig build — compiles shaders + links libvulkan
├── src/
│   ├── main.zig        # CLI entry point
│   ├── vulkan/
│   │   ├── vk.zig          # Vulkan handle types
│   │   ├── instance.zig    # Device, queue, memory
│   │   ├── pipeline.zig    # SPIR-V compute pipeline
│   │   └── buffer.zig      # Device buffer + NCHW tensor
│   └── sd/
│       ├── conv.zig        # Conv2d dispatcher
│       ├── attention.zig   # Self/cross-attention dispatcher
│       └── lora.zig        # LoRA weight fusor
└── shaders/             # GLSL compute → SPIR-V (via glslc)
    ├── conv2d.comp
    ├── group_norm.comp
    ├── silu.comp
    ├── elementwise.comp
    ├── attention.comp
    └── lora_merge.comp
```

## Build Dependencies

- **Zig 0.15.2+** (`zig version`)
- **Vulkan loader** (`libvulkan-dev` or `vulkan-loader`)
- **glslc** (from Vulkan SDK or `shaderc`)

## Adding a New Shader

1. Write `shaders/my_op.comp` following the GLSL pattern below
2. Add the name to the `shader_names` array in `build.zig`
3. Create a dispatcher in `src/sd/` that wraps it

### Shader Pattern

```glsl
#version 460
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Input  { float in_data[]; };
layout(set = 0, binding = 1) buffer Output { float out_data[]; };

layout(push_constant) uniform Push {
    uint N;
    // ... per-operation params
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N) return;
    // compute...
}
```

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | GLSL shaders (conv, norm, act, attn, lora) | ✅ 6 shaders compiled to SPIR-V |
| 2 | Zig Vulkan context + buffers | ✅ instance.zig, buffer.zig |
| 3 | SPIR-V pipeline management | ✅ pipeline.zig |
| 4 | Conv2d / attention dispatchers | 🔧 scaffolded — need full descriptor wiring |
| 5 | UNet forward pass orchestration | 📋 |
| 6 | AnimateDiff temporal attention | 📋 |
| 7 | LoRA fusion at load time | 📋 |
