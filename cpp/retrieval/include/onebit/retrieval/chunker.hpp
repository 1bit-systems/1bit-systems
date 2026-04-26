#pragma once

// Markdown-aware chunker.
//
// Algorithm:
//  1. Walk the file line by line.
//  2. Lines starting with `#` reset the "current heading" and force a
//     chunk boundary (the accumulated section buffer is flushed first).
//  3. Inside a section, accumulate lines into a buffer. When the buffer
//     exceeds SOFT_CAP, emit it as a chunk (rolling back to the last
//     word/line boundary) and restart with a ~OVERLAP tail for context.
//
// The heading is attached to *every* chunk produced from its section.

#include <string>
#include <string_view>
#include <vector>

namespace onebit::retrieval {

inline constexpr std::size_t SOFT_CAP = 500;
inline constexpr std::size_t OVERLAP  = 50;

struct Chunk {
    std::string heading;
    std::string text;
};

[[nodiscard]] std::vector<Chunk> chunk_markdown(std::string_view src);

} // namespace onebit::retrieval
