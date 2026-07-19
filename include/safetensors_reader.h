#pragma once
// safetensors_reader.h — reader for the HuggingFace .safetensors model format.
//
// Container layout: 8-byte little-endian header length, then a JSON header
// mapping tensor name -> {dtype, shape, data_offsets}, then raw tensor data
// (byte-identical wrapper to Q4NX's container — reuses Q4nxReader for the
// low-level mmap/header read).
//
// Unlike GGUF, safetensors carries no architecture/dimension metadata of its
// own — real checkpoints ship a sibling config.json (the HuggingFace
// standard) with that information, which we prefer when present. Falls back
// to tensor-name/shape inference (same technique as q4nx_reader.h) for
// whatever config.json doesn't cover, or when it's absent entirely.

#include "common.h"
#include "q4nx_reader.h"
#include <string>

bool read_safetensors_metadata(const std::string& path, ModelConfig& cfg);
