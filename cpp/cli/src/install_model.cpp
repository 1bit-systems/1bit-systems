#include "onebit/cli/install_model.hpp"

#include "onebit/cli/paths.hpp"
#include "onebit/cli/proc.hpp"
#include "onebit/cli/update.hpp"

#include <chrono>
#include <fmt/core.h>
#include <iostream>
#include <system_error>
#include <thread>

namespace onebit::cli {

std::filesystem::path models_root()
{
    return xdg_data_home() / "1bit" / "models";
}

std::filesystem::path model_dir(std::string_view id)
{
    return models_root() / std::filesystem::path(std::string(id));
}

std::expected<void, Error>
install_model(const Model& spec, const std::vector<std::string>& engine_units)
{
    if (spec.sha256.starts_with("PENDING-")) {
        return std::unexpected(Error::precondition(fmt::format(
            "model `{}` is not yet trained (sha256 sentinel `{}`); "
            "weights will publish after the owning training run completes.",
            spec.id, spec.sha256)));
    }
    std::error_code ec;
    const auto root = models_root();
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return std::unexpected(Error::io("mkdir " + root.string()));
    }
    const auto dir = model_dir(spec.id);
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return std::unexpected(Error::io("mkdir " + dir.string()));
    }

    auto rc = run_inherit({
        "hf", "download", spec.hf_repo, spec.hf_file,
        "--local-dir", dir.string(),
    });
    if (!rc) return std::unexpected(rc.error());
    if (*rc != 0) {
        return std::unexpected(Error::subprocess(
            fmt::format("hf download exit {}", *rc)));
    }

    const auto file_path = dir / spec.hf_file;

    if (spec.sha256 != "UPSTREAM" && !spec.sha256.empty()) {
        if (auto v = verify_sha256(file_path, spec.sha256); !v) {
            return std::unexpected(v.error());
        }
    }

    const auto canonical = dir / "model.gguf";
    std::filesystem::remove(canonical, ec);  // best-effort
    std::filesystem::create_symlink(spec.hf_file, canonical, ec);
    if (ec) {
        return std::unexpected(Error::io(
            fmt::format("symlink {} -> {}: {}",
                        canonical.string(), spec.hf_file, ec.message())));
    }

    for (const auto& unit : engine_units) {
        (void)run_inherit({"systemctl", "--user", "restart", unit});
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
    return {};
}

}  // namespace onebit::cli
