#include "onebit/core/error.hpp"

#include <cstdio>
#include <variant>

namespace onebit::core {

namespace {

std::string format_bytes(const std::array<std::uint8_t, 4>& b)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02x%02x%02x%02x", b[0], b[1], b[2], b[3]);
    return buf;
}

// F.55: exhaustive overload helper. Adding a new HaloError alternative
// without updating this struct becomes a compile-time error rather than
// the silent UB you'd get from a generic lambda.
template <class... Ts>
struct Overload : Ts... { using Ts::operator()...; };
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

} // namespace

std::string HaloError::what() const
{
    return std::visit(
        Overload{
            [](const ErrorIo& e) -> std::string {
                return "I/O error on " + e.path.string() + ": " + e.ec.message();
            },
            [](const ErrorRawIo& e) -> std::string {
                return "I/O error: " + e.ec.message();
            },
            [](const ErrorBadMagic& e) -> std::string {
                return "bad magic: expected " + format_bytes(e.expected)
                       + ", got " + format_bytes(e.got);
            },
            [](const ErrorUnsupportedVersion& e) -> std::string {
                return "unsupported format version " + std::to_string(e.version)
                       + " (supported: " + std::to_string(e.min) + ".."
                       + std::to_string(e.max) + ")";
            },
            [](const ErrorTruncated& e) -> std::string {
                return "truncated file: needed " + std::to_string(e.needed)
                       + " bytes at offset " + std::to_string(e.offset)
                       + ", only " + std::to_string(e.have) + " available";
            },
            [](const ErrorInvalidConfig& e) -> std::string {
                return std::string("invalid h1b config: ") + e.msg;
            },
            [](const ErrorUnknownBytePiece&) -> std::string {
                return "tokenizer piece is unknown";
            },
            [](const ErrorSampler& e) -> std::string {
                return std::string("sampler error: ") + e.msg;
            },
        },
        v_);
}

} // namespace onebit::core
