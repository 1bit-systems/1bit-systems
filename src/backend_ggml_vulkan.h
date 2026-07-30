#pragma once
// backend_ggml_vulkan.h — llama.cpp Vulkan backend wrapper.
// Uses ggml-vulkan (MIT License) via llama.cpp API for high-performance inference.

struct Backend;
Backend* create_ggml_vulkan_backend();
