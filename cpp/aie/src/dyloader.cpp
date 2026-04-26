// dyloader.cpp — POSIX dlopen wrapper for libxrt_coreutil.so.
//
// Idempotent open(); resolve() returns nullptr (no exception) on miss.

#include "onebit/aie/dyloader.hpp"

#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#  include <dlfcn.h>
#  define ONEBIT_AIE_HAVE_DLOPEN 1
#else
#  define ONEBIT_AIE_HAVE_DLOPEN 0
#endif

namespace onebit::aie {

DyLoader::DyLoader(DyLoader&& other) noexcept
    : handle_{other.handle_}, soname_{std::move(other.soname_)} {
    other.handle_ = nullptr;
}

DyLoader& DyLoader::operator=(DyLoader&& other) noexcept {
    if (this != &other) {
        close();
        handle_ = other.handle_;
        soname_ = std::move(other.soname_);
        other.handle_ = nullptr;
    }
    return *this;
}

DyLoader::~DyLoader() {
    close();
}

void DyLoader::close() noexcept {
#if ONEBIT_AIE_HAVE_DLOPEN
    if (handle_) {
        ::dlclose(handle_);
        handle_ = nullptr;
        soname_.clear();
    }
#else
    handle_ = nullptr;
    soname_.clear();
#endif
}

std::expected<std::string, Error> DyLoader::open() {
#if !ONEBIT_AIE_HAVE_DLOPEN
    return std::unexpected(Error{ErrorKind::LibraryUnavailable,
                                 "dlopen not available on this platform"});
#else
    if (handle_) return soname_;

    std::string last_err;
    for (auto soname : kXrtSonames) {
        // dlopen wants a null-terminated C string. Our table entries
        // are string_view literals; copy into a temp buffer.
        std::string s{soname};
        // RTLD_LAZY: we only need symbols at first call; RTLD_LOCAL:
        // don't pollute the global symbol namespace.
        void* h = ::dlopen(s.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (h) {
            handle_ = h;
            soname_ = std::move(s);
            return soname_;
        }
        if (const char* e = ::dlerror()) {
            if (!last_err.empty()) last_err += "; ";
            last_err += s;
            last_err += ": ";
            last_err += e;
        }
    }
    return std::unexpected(Error{ErrorKind::LibraryUnavailable,
                                 last_err.empty()
                                     ? std::string{"no libxrt SO found"}
                                     : last_err});
#endif
}

void* DyLoader::resolve(const char* name) const noexcept {
#if !ONEBIT_AIE_HAVE_DLOPEN
    (void)name;
    return nullptr;
#else
    if (!handle_ || !name) return nullptr;
    // Clear any stale dlerror first; we don't propagate it here
    // (callers branch on null and report NotYetWired).
    (void)::dlerror();
    void* sym = ::dlsym(handle_, name);
    return sym;
#endif
}

} // namespace onebit::aie
