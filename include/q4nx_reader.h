#pragma once
// q4nx_reader.h — shared reader for the Q4NX model format.
//
// Q4NX uses a safetensors-style container: 8-byte little-endian header length,
// then a JSON header mapping tensor name -> {dtype, shape, data_offsets}, then
// raw tensor data. There is no top-level architecture/config field — dimensions
// and architecture must be derived from tensor names/shapes (matching how
// engine/fusion/model_data.zig's deriveConfig() works).

#include "common.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>

// POSIX-only headers — not available on Windows
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

// ── Q4NX model reader ──
// Reads BF16 weights (widened to float32) from the mmap'd model file by
// JSON key lookup. The model file is an indexed format: JSON header with
// data_offsets, then raw BF16 weight data at those offsets.
struct Q4nxReader {
    const char* data = nullptr;
    size_t size = 0;
    // Absolute file offset where tensor data begins: 8 (header-length
    // prefix) + header_len (JSON byte length). data_offsets in the JSON
    // are relative to this, not to byte 0 of the file (#1058).
    size_t data_start = 0;

    bool open(const std::string& path);
    void close();

    // Find data offset for a JSON key in the model header
    // Uses standard C string search instead of GNU memmem extension.
    uint64_t find_offset(const char* key) const;

    // Read a BF16 array at offset, widened to float32, into a vector
    std::vector<float> read_floats(uint64_t offset, size_t count) const;

    // Parse tensor shape [rows, cols] from the JSON header by tensor name.
    // Returns empty vector if tensor or shape field is not found.
    std::vector<uint64_t> get_tensor_shape(const std::string& json_header,
                                            const std::string& tensor_name) const;

    // Convenience: check that tensor_name in json_header has shape
    // [expected_rows, expected_cols]. Returns true on match.
    bool validate_tensor_shape(const std::string& json_header,
                               const std::string& tensor_name,
                               size_t expected_rows,
                               size_t expected_cols) const;
};

// Best-effort metadata extraction for model discovery: dims from the embedding
// tensor's shape, layer count from the highest "model.layers.N." index seen,
// quantization from the embedding tensor's dtype string, architecture from the
// filename (Q4NX has no self-describing architecture field — same convention
// engine/fusion's model_tag already relies on).
bool read_q4nx_metadata(const std::string& path, ModelConfig& cfg);
