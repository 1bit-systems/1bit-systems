#include "onebit/cli/install.hpp"

#include "onebit/cli/http.hpp"
#include "onebit/cli/oobe_error.hpp"
#include "onebit/cli/preflight.hpp"
#include "onebit/cli/proc.hpp"

#include <chrono>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace onebit::cli {

namespace {

std::expected<void, Error>
resolve_recursive(const Manifest& m,
                  const std::string& target,
                  std::vector<std::string>& order,
                  std::unordered_set<std::string>& seen)
{
    if (seen.contains(target)) return {};
    auto it = m.components.find(target);
    if (it == m.components.end()) {
        return std::unexpected(Error::not_found(
            "unknown component '" + target + "' (try `1bit install --list`)"));
    }
    for (const auto& d : it->second.deps) {
        if (auto rc = resolve_recursive(m, d, order, seen); !rc) return rc;
    }
    seen.insert(target);
    order.push_back(target);
    return {};
}

[[nodiscard]] std::string render_subs(std::string_view raw,
                                       const std::map<std::string, std::string>& subs)
{
    std::string out(raw);
    for (const auto& [key, raw_val] : subs) {
        const std::string needle = std::string("@") + key + "@";
        const std::string value  = expand_placeholder(raw_val);
        std::size_t pos = 0;
        while ((pos = out.find(needle, pos)) != std::string::npos) {
            out.replace(pos, needle.size(), value);
            pos += value.size();
        }
    }
    return out;
}

class RealExecutor final : public HostExecutor {
public:
    std::expected<void, Error>
    run_argv(const std::filesystem::path& cwd,
             const std::vector<std::string>& argv) override
    {
        if (argv.empty()) return std::unexpected(Error::invalid("empty argv"));
        std::vector<std::string> expanded;
        expanded.reserve(argv.size());
        for (const auto& a : argv) expanded.push_back(expand_tilde(a));
        std::cout << "    $";
        for (const auto& a : expanded) std::cout << ' ' << a;
        std::cout << '\n';
        auto rc = run_inherit(expanded, cwd);
        if (!rc) return std::unexpected(rc.error());
        if (*rc != 0) {
            return std::unexpected(Error::subprocess(
                fmt::format("{} failed (exit {})", expanded.front(), *rc)));
        }
        return {};
    }

    std::expected<void, Error> systemctl_enable_now(std::string_view unit) override
    {
        std::cout << "    $ systemctl --user enable --now " << unit << '\n';
        auto rc = run_inherit({"systemctl", "--user", "enable", "--now",
                               std::string(unit)});
        if (!rc) return std::unexpected(rc.error());
        if (*rc != 0) {
            return std::unexpected(Error::subprocess(
                fmt::format("systemctl failed for {}", unit)));
        }
        return {};
    }

    std::expected<void, Error> systemctl_restart(std::string_view unit) override
    {
        auto rc = run_inherit({"systemctl", "--user", "restart", std::string(unit)});
        if (!rc) return std::unexpected(rc.error());
        if (*rc != 0) {
            return std::unexpected(Error::subprocess(
                fmt::format("systemctl restart failed for {}", unit)));
        }
        return {};
    }

    std::expected<bool, Error>
    copy_tracked_file(const std::filesystem::path& src,
                      const std::filesystem::path& dest,
                      const std::map<std::string, std::string>& subs) override
    {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(src, ec)) {
            return std::unexpected(Error::io("tracked file not found: " + src.string()));
        }
        if (std::filesystem::exists(dest, ec)) {
            std::cout << "    skip (exists): " << dest.string() << '\n';
            return false;
        }
        std::ifstream in(src, std::ios::binary);
        if (!in) return std::unexpected(Error::io("cannot open " + src.string()));
        std::ostringstream raw;
        raw << in.rdbuf();
        const std::string rendered = render_subs(raw.str(), subs);

        if (dest.is_absolute()) {
            std::cout << "    installing (sudo) " << src.string()
                      << " → " << dest.string() << '\n';
            // popen("sudo tee <path>", "w") + write rendered.
            const std::string cmd =
                std::string("sudo tee ") + dest.string() + " > /dev/null";
            FILE* p = ::popen(cmd.c_str(), "w");
            if (p == nullptr) {
                return std::unexpected(Error::subprocess("popen sudo tee failed"));
            }
            const auto written = std::fwrite(rendered.data(), 1, rendered.size(), p);
            const int rc = ::pclose(p);
            if (written != rendered.size() || rc != 0) {
                return std::unexpected(Error::subprocess(
                    "sudo tee failed for " + dest.string()));
            }
            return true;
        }
        std::filesystem::create_directories(dest.parent_path(), ec);
        std::ofstream out(dest, std::ios::binary | std::ios::trunc);
        if (!out) return std::unexpected(Error::io("cannot write " + dest.string()));
        out.write(rendered.data(), static_cast<std::streamsize>(rendered.size()));
        std::cout << "    copied " << src.string() << " → " << dest.string() << '\n';
        return true;
    }

    bool healthcheck(std::string_view url) override
    {
        if (url.empty()) return true;
        return ::onebit::cli::healthcheck(default_http_client(), url, 3000);
    }
};

}  // namespace

std::expected<std::vector<std::string>, Error>
resolve_install_order(const Manifest& m, std::string_view target)
{
    std::vector<std::string> order;
    std::unordered_set<std::string> seen;
    if (auto rc = resolve_recursive(m, std::string(target), order, seen); !rc) {
        return std::unexpected(rc.error());
    }
    return order;
}

std::vector<std::string> render_install_list(const Manifest& m)
{
    std::vector<std::string> out;
    out.reserve(m.components.size() + 4);
    out.emplace_back("strix-ai-rs components:");
    out.emplace_back("");
    for (const auto& [name, c] : m.components) {
        out.push_back(fmt::format("  {:<16} {}", name, c.description));
        if (!c.deps.empty()) {
            std::string j;
            for (std::size_t i = 0; i < c.deps.size(); ++i) {
                if (i) j += ", ";
                j += c.deps[i];
            }
            out.push_back(fmt::format("  {:<16}   deps: {}", "", j));
        }
        if (!c.packages.empty()) {
            std::string j;
            for (std::size_t i = 0; i < c.packages.size(); ++i) {
                if (i) j += ", ";
                j += c.packages[i];
            }
            out.push_back(fmt::format("  {:<16}   packages: {}", "", j));
        }
    }
    return out;
}

std::unique_ptr<HostExecutor> make_real_executor()
{
    return std::make_unique<RealExecutor>();
}

std::expected<void, Error>
run_install(HostExecutor& host,
            const Manifest& m,
            std::string_view component,
            InstallTracker& tracker,
            const InstallContext& ctx)
{
    auto order = resolve_install_order(m, component);
    if (!order) return std::unexpected(order.error());

    std::cout << "install plan: ";
    for (std::size_t i = 0; i < order->size(); ++i) {
        if (i) std::cout << " → ";
        std::cout << (*order)[i];
    }
    std::cout << "\n\n";

    for (const auto& name : *order) {
        const auto& c = m.components.at(name);
        std::cout << "── " << name << ": " << c.description << " ──\n";

        if (!c.packages.empty()) {
            std::cout << "    required distro packages: ";
            for (std::size_t i = 0; i < c.packages.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << c.packages[i];
            }
            std::cout << " (install via install.sh / pacman)\n";
        }

        for (const auto& step : c.build) {
            if (auto rc = host.run_argv(ctx.workspace_root, step); !rc) {
                return std::unexpected(rc.error());
            }
        }

        for (const auto& f : c.files) {
            const std::filesystem::path src = ctx.workspace_root / f.src;
            const std::filesystem::path dest = std::filesystem::path(f.dst).is_absolute()
                ? std::filesystem::path(f.dst)
                : ctx.config_root / f.dst;
            const bool pre_existed = std::filesystem::exists(dest);
            auto wrote = host.copy_tracked_file(src, dest, f.substitute);
            if (!wrote) return std::unexpected(wrote.error());
            if (!pre_existed && *wrote) {
                if (dest.is_absolute()) tracker.record(ActionCopiedSudo{dest});
                else                    tracker.record(ActionCopiedFile{dest});
            }
        }

        for (const auto& unit : c.units) {
            if (auto rc = host.systemctl_enable_now(unit); !rc) {
                return std::unexpected(rc.error());
            }
            tracker.record(ActionEnabledUnit{unit});
        }

        if (!c.check.empty()) {
            std::cout << "    checking " << c.check << "…  ";
            bool ok = false;
            for (int i = 0; i < 5; ++i) {
                if (host.healthcheck(c.check)) { ok = true; break; }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            std::cout << (ok ? "ok" : "FAIL (component installed but health probe failed)") << '\n';
        }
    }

    std::cout << "\n✓ install complete\n";
    return {};
}

std::expected<void, Error>
run_oobe_full(HostExecutor& host,
              SystemProbe& probe,
              DoctorProbe& doctor,
              const Manifest& m,
              const OobeDefaults& defaults,
              const InstallContext& ctx)
{
    auto results = run_all(probe);
    std::cout << "preflight:\n";
    for (const auto& r : results) {
        const char* glyph = "?";
        const std::string note = std::visit([](const auto& o) -> std::string {
            using T = std::decay_t<decltype(o)>;
            if constexpr (std::is_same_v<T, PreflightPass>) return o.note;
            else if constexpr (std::is_same_v<T, PreflightSkip>) return o.note;
            else if constexpr (std::is_same_v<T, PreflightFail>) return o.err.what;
        }, r.outcome);
        if (std::holds_alternative<PreflightPass>(r.outcome)) glyph = "[ OK ]";
        else if (std::holds_alternative<PreflightSkip>(r.outcome)) glyph = "[WARN]";
        else                                                       glyph = "[FAIL]";
        std::cout << fmt::format("  {:6} {:8}: {}\n", glyph, r.name, note);
    }

    for (const auto& r : results) {
        if (!is_green(r.outcome)) {
            if (const auto* fail = std::get_if<PreflightFail>(&r.outcome)) {
                std::cout << "\nerror: " << r.name << '\n' << fail->err << '\n';
            }
            return std::unexpected(Error::precondition(
                std::string("preflight failed at gate '") + r.name + "'"));
        }
    }

    if (defaults.yes) {
        std::cout << "\n(--yes) non-interactive mode; all prompts auto-answered yes.\n";
    }

    if (defaults.skip_build) {
        std::cout << "\n(--skip-build) preflight green; skipping cargo build / install for '"
                  << defaults.component << "'.\n";
    } else {
        std::cout << "\npreflight green — proceeding with `1bit install "
                  << defaults.component << "`\n\n";
        InstallTracker tracker;
        auto rc = run_install(host, m, defaults.component, tracker, ctx);
        if (!rc) {
            std::cout << "\ninstall failed: " << rc.error().message << '\n';
            std::cout << "atomic revert (anchor 10):\n";
            tracker.best_effort_revert();
            const auto remaining = tracker.actions();
            if (remaining.empty()) {
                std::cout << "    left state: installer undid its own side-effects cleanly.\n";
            } else {
                std::cout << "    left state: " << remaining.size()
                          << " action(s) could not be reverted — see above.\n";
            }
            const auto oe = OobeError::install_step_failed("install");
            std::cout << "\nerror: install\n" << oe << '\n';
            return std::unexpected(rc.error());
        }
    }

    if (defaults.doctor_skip) {
        std::cout << "\n(--doctor-skip) skipping the `1bit doctor` tail probe.\n";
        return {};
    }
    std::cout << "\nrunning `1bit doctor` (OOBE anchor #7) ...\n";
    auto [warn, fail] = doctor.run();
    std::cout << "doctor summary: " << warn << " warn, " << fail << " fail\n";
    if (fail > 0) {
        const auto oe = OobeError::doctor_failed(fail);
        std::cout << "\nerror: doctor\n" << oe << '\n';
        return std::unexpected(Error::precondition(
            fmt::format("`1bit doctor` reported {} fail(s)", fail)));
    }
    return {};
}

}  // namespace onebit::cli
