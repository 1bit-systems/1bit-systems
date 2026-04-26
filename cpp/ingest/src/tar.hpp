#pragma once

// Minimal POSIX (ustar) tar writer. Only file entries; no directories,
// links, or extended headers. Mirrors the subset of `tar::Builder` we
// touch from the Rust port.

#include <cstdint>
#include <filesystem>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

namespace onebit::ingest::detail::tar {

class Writer {
public:
    explicit Writer(std::ostream& sink) noexcept : sink_{&sink} {}

    // Append an in-memory byte payload at `name` (mode 0644).
    void append_blob(std::string_view             name,
                     std::span<const std::uint8_t> bytes,
                     std::uint32_t                mode = 0644);

    // Append a regular file copied from `path`, stored at `name` in the
    // archive. Throws nothing — sets `error_` and returns false on any
    // I/O failure.
    [[nodiscard]] bool append_path_with_name(const std::filesystem::path& path,
                                             std::string_view             name);

    // Write the two 512-byte zero blocks that close a tarball.
    void finish();

    [[nodiscard]] const std::string& error() const noexcept { return error_; }
    [[nodiscard]] bool               ok() const noexcept { return error_.empty(); }

private:
    std::ostream* sink_;
    std::string   error_;

    void write_header(std::string_view name, std::uint64_t size, std::uint32_t mode);
    void write_zeros(std::size_t n);
    void write_padding_to_block(std::size_t bytes_written);
};

} // namespace onebit::ingest::detail::tar
