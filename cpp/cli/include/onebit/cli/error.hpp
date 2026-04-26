#pragma once

// Single Error struct used across the CLI's `std::expected<T, Error>` returns.
// Deliberately stringly-typed (one variant tag, one human message): the
// CLI's job is to print + bail, not to enumerate every failure mode.

#include <string>
#include <string_view>
#include <utility>

namespace onebit::cli {

enum class ErrorKind {
    Io,
    Parse,
    Schema,
    NotFound,
    InvalidArgument,
    PreconditionFailed,
    Subprocess,
    Network,
    Hash,
    Unknown,
};

[[nodiscard]] constexpr std::string_view kind_label(ErrorKind k) noexcept
{
    switch (k) {
        case ErrorKind::Io:                  return "io";
        case ErrorKind::Parse:               return "parse";
        case ErrorKind::Schema:              return "schema";
        case ErrorKind::NotFound:            return "not-found";
        case ErrorKind::InvalidArgument:     return "invalid-argument";
        case ErrorKind::PreconditionFailed:  return "precondition";
        case ErrorKind::Subprocess:          return "subprocess";
        case ErrorKind::Network:             return "network";
        case ErrorKind::Hash:                return "hash";
        case ErrorKind::Unknown:             return "unknown";
    }
    return "unknown";
}

struct Error {
    ErrorKind   kind = ErrorKind::Unknown;
    std::string message;

    Error() = default;
    Error(ErrorKind k, std::string m) noexcept : kind(k), message(std::move(m)) {}

    static Error io(std::string m)       { return {ErrorKind::Io, std::move(m)}; }
    static Error parse(std::string m)    { return {ErrorKind::Parse, std::move(m)}; }
    static Error schema(std::string m)   { return {ErrorKind::Schema, std::move(m)}; }
    static Error not_found(std::string m){ return {ErrorKind::NotFound, std::move(m)}; }
    static Error invalid(std::string m)  { return {ErrorKind::InvalidArgument, std::move(m)}; }
    static Error precondition(std::string m){ return {ErrorKind::PreconditionFailed, std::move(m)}; }
    static Error subprocess(std::string m){ return {ErrorKind::Subprocess, std::move(m)}; }
    static Error network(std::string m)  { return {ErrorKind::Network, std::move(m)}; }
    static Error hash(std::string m)     { return {ErrorKind::Hash, std::move(m)}; }
};

}  // namespace onebit::cli
