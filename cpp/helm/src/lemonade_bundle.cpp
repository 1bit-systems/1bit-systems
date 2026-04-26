#include "onebit/helm/lemonade_bundle.hpp"

#include <httplib.h>

#include <cerrno>
#include <unistd.h>      // readlink
#include <linux/limits.h>

namespace onebit::helm::lemonade_bundle {

namespace {

// argv[0] resolves to the helm exe; from there `../share/1bit-helm/lemonade/bin/lemond`
// is the install layout produced by `install(DIRECTORY ... share/1bit-helm/lemonade)`.
[[nodiscard]] std::optional<std::filesystem::path> exe_dir()
{
    char buf[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return std::nullopt;
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path();
}

} // namespace

std::optional<std::filesystem::path>
bundled_lemond_path()
{
#ifndef HELM_BUNDLE_LEMONADE
    return std::nullopt;
#else
    const auto bin = exe_dir();
    if (!bin) return std::nullopt;

    // Layout: <prefix>/bin/1bit-helm  →  <prefix>/share/1bit-helm/lemonade/bin/lemond
    const auto candidate =
        bin->parent_path() / "share" / "1bit-helm" / "lemonade" / "bin" / "lemond";

    std::error_code ec;
    if (std::filesystem::exists(candidate, ec)
        && std::filesystem::is_regular_file(candidate, ec)) {
        return candidate;
    }
    return std::nullopt;
#endif
}

bool
probe_openai_endpoint(const std::string& host, int port, int timeout_ms)
{
    httplib::Client cli(host, port);
    cli.set_connection_timeout(0, timeout_ms * 1000);
    cli.set_read_timeout(0,       timeout_ms * 1000);

    auto res = cli.Get("/v1/models");
    return res && res->status >= 200 && res->status < 300;
}

} // namespace onebit::helm::lemonade_bundle
