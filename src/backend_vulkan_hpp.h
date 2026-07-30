#pragma once
// backend_vulkan_hpp.h — Vulkan-Hpp inference backend.
// Uses Vulkan-Hpp C++ bindings and ZINC SPIR-V shaders.

struct Backend;
Backend* create_vulkan_hpp_backend();
