// error.cpp — labels + what() for onebit::aie::Error.

#include "onebit/aie/error.hpp"

#include <string>

namespace onebit::aie {

std::string_view label(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::XclbinNotFound:     return "xclbin_not_found";
        case ErrorKind::ShapeMismatch:      return "shape_mismatch";
        case ErrorKind::DtypeMismatch:      return "dtype_mismatch";
        case ErrorKind::Xrt:                return "xrt";
        case ErrorKind::NotYetWired:        return "not_yet_wired";
        case ErrorKind::LibraryUnavailable: return "library_unavailable";
    }
    return "unknown";
}

std::string Error::what() const {
    std::string out{label(kind_)};
    if (!detail_.empty()) {
        out += ": ";
        out += detail_;
    }
    if (kind_ == ErrorKind::Xrt && xrt_code_ != 0) {
        out += " (code ";
        out += std::to_string(xrt_code_);
        out += ")";
    }
    return out;
}

} // namespace onebit::aie
