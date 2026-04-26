#include "tar.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace onebit::ingest::detail::tar {

namespace {

constexpr std::size_t BLOCK = 512;

void write_octal(char* dst, std::size_t width, std::uint64_t value)
{
    // POSIX-style: width-1 octal digits, NUL-terminated.
    if (width == 0) {
        return;
    }
    std::array<char, 32> buf{};
    std::size_t          i = 0;
    if (value == 0) {
        buf[i++] = '0';
    } else {
        while (value > 0 && i < buf.size()) {
            buf[i++] = static_cast<char>('0' + (value & 0x7));
            value >>= 3;
        }
    }
    // pad with leading zeros to width-1
    while (i < width - 1) {
        buf[i++] = '0';
    }
    // copy reversed
    for (std::size_t k = 0; k < width - 1; ++k) {
        dst[k] = buf[width - 2 - k];
    }
    dst[width - 1] = '\0';
}

} // namespace

void Writer::write_header(std::string_view name, std::uint64_t size, std::uint32_t mode)
{
    if (!ok()) {
        return;
    }
    std::array<char, BLOCK> hdr{};

    // name: 100 bytes
    const auto copy_n = std::min<std::size_t>(name.size(), 100);
    std::memcpy(hdr.data(), name.data(), copy_n);

    // mode: 8 bytes octal (zero-padded, NUL-terminated)
    write_octal(hdr.data() + 100, 8, mode & 07777u);  // 4 KiB perm mask
    // uid, gid: 8 bytes each
    write_octal(hdr.data() + 108, 8, 0);
    write_octal(hdr.data() + 116, 8, 0);
    // size: 12 bytes
    write_octal(hdr.data() + 124, 12, size);
    // mtime: 12 bytes (we leave 0 — deterministic builds)
    write_octal(hdr.data() + 136, 12, 0);

    // typeflag: '0' for regular file (offset 156)
    hdr[156] = '0';

    // ustar magic + version (offset 257..)
    std::memcpy(hdr.data() + 257, "ustar\0", 6);
    std::memcpy(hdr.data() + 263, "00", 2);

    // checksum: spaces, then computed
    for (int i = 148; i < 156; ++i) {
        hdr[i] = ' ';
    }
    std::uint32_t sum = 0;
    for (auto c : hdr) {
        sum += static_cast<std::uint8_t>(c);
    }
    write_octal(hdr.data() + 148, 7, sum);
    hdr[155] = ' ';

    sink_->write(hdr.data(), BLOCK);
    if (!sink_->good()) {
        error_ = "tar: write header failed";
    }
}

void Writer::write_padding_to_block(std::size_t bytes_written)
{
    const std::size_t r = bytes_written % BLOCK;
    if (r == 0) {
        return;
    }
    std::array<char, BLOCK> zeros{};
    sink_->write(zeros.data(), static_cast<std::streamsize>(BLOCK - r));
    if (!sink_->good()) {
        error_ = "tar: padding write failed";
    }
}

void Writer::write_zeros(std::size_t n)
{
    std::array<char, BLOCK> zeros{};
    while (n > 0) {
        const auto take = std::min(n, zeros.size());
        sink_->write(zeros.data(), static_cast<std::streamsize>(take));
        n -= take;
    }
    if (!sink_->good()) {
        error_ = "tar: zeros write failed";
    }
}

void Writer::append_blob(std::string_view             name,
                         std::span<const std::uint8_t> bytes,
                         std::uint32_t                mode)
{
    write_header(name, bytes.size(), mode);
    if (!ok()) {
        return;
    }
    sink_->write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!sink_->good()) {
        error_ = "tar: payload write failed";
        return;
    }
    write_padding_to_block(bytes.size());
}

bool Writer::append_path_with_name(const std::filesystem::path& path,
                                   std::string_view             name)
{
    std::error_code ec;
    const auto      sz = std::filesystem::file_size(path, ec);
    if (ec) {
        error_ = "tar: file_size: " + ec.message();
        return false;
    }
    write_header(name, sz, 0644);
    if (!ok()) {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error_ = "tar: open input failed";
        return false;
    }
    constexpr std::size_t       buf_sz = 64 * 1024;
    std::array<char, buf_sz>    buf{};
    std::uint64_t               written = 0;
    while (written < sz) {
        const auto remaining = sz - written;
        const auto want      = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, buf.size()));
        in.read(buf.data(), want);
        const auto got = in.gcount();
        if (got <= 0) {
            error_ = "tar: short read";
            return false;
        }
        sink_->write(buf.data(), got);
        if (!sink_->good()) {
            error_ = "tar: write payload failed";
            return false;
        }
        written += static_cast<std::uint64_t>(got);
    }
    write_padding_to_block(static_cast<std::size_t>(sz));
    return ok();
}

void Writer::finish()
{
    write_zeros(2 * BLOCK);
}

} // namespace onebit::ingest::detail::tar
