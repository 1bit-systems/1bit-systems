#include "onebit/power/ryzen.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#include <spdlog/spdlog.h>

namespace onebit::power {

bool is_known_knob(std::string_view key) noexcept
{
    return std::find(KNOB_NAMES.begin(), KNOB_NAMES.end(), key) != KNOB_NAMES.end();
}

// =====================================================================
// ShelloutBackend
// =====================================================================

std::vector<std::string> ShelloutBackend::profile_to_args(const Profile& p)
{
    std::vector<std::string> args;
    args.reserve(8);
    auto push = [&](std::string_view flag, const std::optional<std::uint32_t>& v) {
        if (!v) return;
        std::ostringstream os;
        os << "--" << flag << "=" << *v;
        args.emplace_back(os.str());
    };
    push("stapm-limit",       p.stapm_limit);
    push("fast-limit",        p.fast_limit);
    push("slow-limit",        p.slow_limit);
    push("tctl-temp",         p.tctl_temp);
    push("vrm-current",       p.vrm_current);
    push("vrmmax-current",    p.vrmmax_current);
    push("vrmsoc-current",    p.vrmsoc_current);
    push("vrmsocmax-current", p.vrmsocmax_current);
    return args;
}

Status ShelloutBackend::run(const std::vector<std::string>& args) const
{
    if (dry_run_) {
        std::string joined = path_;
        for (const auto& a : args) { joined.push_back(' '); joined += a; }
        spdlog::info("dry-run: {}", joined);
        return Status::success();
    }

    // posix_spawn — preferred over fork+exec for hot paths, but here we
    // run once per invocation so plain fork/exec would be fine too.
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    // posix_spawn does not modify argv, but the API requires char* not
    // const char*. Strip const via local copies to stay strictly conforming.
    std::vector<std::string> mutable_args = args;
    std::string mutable_path = path_;
    argv.push_back(mutable_path.data());
    for (auto& a : mutable_args) argv.push_back(a.data());
    argv.push_back(nullptr);

    pid_t pid = 0;
    int rc = posix_spawn(&pid, path_.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        std::ostringstream os;
        os << "spawning " << path_ << ": " << std::strerror(rc);
        return Status::fail(Error::BackendError, os.str());
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        std::ostringstream os; os << "waitpid: " << std::strerror(errno);
        return Status::fail(Error::BackendError, os.str());
    }
    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code != 0) {
            std::ostringstream os; os << "ryzenadj exit status " << code;
            return Status::fail(Error::BackendError, os.str());
        }
        return Status::success();
    }
    return Status::fail(Error::BackendError, "ryzenadj killed by signal");
}

Status ShelloutBackend::apply_profile(const Profile& p)
{
    auto args = profile_to_args(p);
    if (args.empty()) return Status::success();
    return run(args);
}

Status ShelloutBackend::set_one(std::string_view key, std::uint32_t value)
{
    if (!is_known_knob(key)) {
        return Status::fail(Error::UnknownKnob,
            "unknown knob `" + std::string{key} + "`");
    }
    std::ostringstream os;
    os << "--" << key << "=" << value;
    return run({os.str()});
}

// =====================================================================
// LibBackend
// =====================================================================

namespace detail {

void DlHandleDeleter::operator()(void* h) const noexcept
{
    if (h) ::dlclose(h);
}

} // namespace detail

namespace {

template <class FnT>
FnT resolve(void* h, const char* name)
{
    // dlsym intentionally returns void*; cast through pointer-to-function
    // is the documented mechanism on POSIX.
    void* sym = ::dlsym(h, name);
    return reinterpret_cast<FnT>(sym);
}

} // namespace

Result<std::unique_ptr<LibBackend>> LibBackend::open(bool dry_run, std::string_view soname)
{
    // Clear existing dlerror.
    (void)::dlerror();
    void* raw = ::dlopen(std::string{soname}.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!raw) {
        const char* e = ::dlerror();
        std::ostringstream os;
        os << "dlopen(" << soname << ") failed: " << (e ? e : "unknown");
        return Status::fail(Error::NotAvailable, os.str());
    }
    detail::DlHandle h{raw};

    detail::LibSymbols s{};
    s.init_ryzenadj          = resolve<detail::init_fn>   (raw, "init_ryzenadj");
    s.cleanup_ryzenadj       = resolve<detail::cleanup_fn>(raw, "cleanup_ryzenadj");
    s.set_stapm_limit        = resolve<detail::set_u32_fn>(raw, "set_stapm_limit");
    s.set_fast_limit         = resolve<detail::set_u32_fn>(raw, "set_fast_limit");
    s.set_slow_limit         = resolve<detail::set_u32_fn>(raw, "set_slow_limit");
    s.set_tctl_temp          = resolve<detail::set_u32_fn>(raw, "set_tctl_temp");
    s.set_vrm_current        = resolve<detail::set_u32_fn>(raw, "set_vrm_current");
    s.set_vrmmax_current     = resolve<detail::set_u32_fn>(raw, "set_vrmmax_current");
    s.set_vrmsoc_current     = resolve<detail::set_u32_fn>(raw, "set_vrmsoc_current");
    s.set_vrmsocmax_current  = resolve<detail::set_u32_fn>(raw, "set_vrmsocmax_current");

    if (!s.valid()) {
        return Status::fail(Error::SymbolMissing,
            std::string{"libryzenadj symbol set incomplete in "} + std::string{soname});
    }

    detail::ryzen_access access = nullptr;
    if (!dry_run) {
        access = s.init_ryzenadj();
        if (!access) {
            return Status::fail(Error::NotAvailable,
                "init_ryzenadj() returned null (need root, kernel module, or compatible CPU?)");
        }
    }
    // ctor is private; subclass to gain access (LibBackend is no longer
    // `final`, so a thin pass-through derived class can call the protected
    // base ctor).
    struct MakeUnique : LibBackend {
        MakeUnique(detail::DlHandle h_, detail::LibSymbols s_,
                   detail::ryzen_access a_, bool d_)
            : LibBackend(std::move(h_), s_, a_, d_) {}
    };
    return std::unique_ptr<LibBackend>{
        new MakeUnique(std::move(h), s, access, dry_run)};
}

LibBackend::~LibBackend()
{
    if (access_ && sym_.cleanup_ryzenadj) {
        sym_.cleanup_ryzenadj(access_);
    }
    // dl_ closes the .so via DlHandleDeleter automatically.
}

Status LibBackend::invoke(detail::set_u32_fn fn,
                          std::string_view label,
                          std::uint32_t value) const
{
    if (dry_run_) {
        spdlog::info("dry-run: libryzenadj {}({})", label, value);
        return Status::success();
    }
    if (!access_) {
        return Status::fail(Error::NotAvailable,
            "libryzenadj handle not initialised");
    }
    int rc = fn(access_, value);
    if (rc != 0) {
        std::ostringstream os;
        os << "libryzenadj " << label << "(" << value << ") returned " << rc;
        return Status::fail(Error::BackendError, os.str());
    }
    return Status::success();
}

Status LibBackend::apply_profile(const Profile& p)
{
    if (p.stapm_limit)        if (auto s = invoke(sym_.set_stapm_limit,       "set_stapm_limit",       *p.stapm_limit);       !s) return s;
    if (p.fast_limit)         if (auto s = invoke(sym_.set_fast_limit,        "set_fast_limit",        *p.fast_limit);        !s) return s;
    if (p.slow_limit)         if (auto s = invoke(sym_.set_slow_limit,        "set_slow_limit",        *p.slow_limit);        !s) return s;
    if (p.tctl_temp)          if (auto s = invoke(sym_.set_tctl_temp,         "set_tctl_temp",         *p.tctl_temp);         !s) return s;
    if (p.vrm_current)        if (auto s = invoke(sym_.set_vrm_current,       "set_vrm_current",       *p.vrm_current);       !s) return s;
    if (p.vrmmax_current)     if (auto s = invoke(sym_.set_vrmmax_current,    "set_vrmmax_current",    *p.vrmmax_current);    !s) return s;
    if (p.vrmsoc_current)     if (auto s = invoke(sym_.set_vrmsoc_current,    "set_vrmsoc_current",    *p.vrmsoc_current);    !s) return s;
    if (p.vrmsocmax_current)  if (auto s = invoke(sym_.set_vrmsocmax_current, "set_vrmsocmax_current", *p.vrmsocmax_current); !s) return s;
    return Status::success();
}

Status LibBackend::set_one(std::string_view key, std::uint32_t value)
{
    if (!is_known_knob(key)) {
        return Status::fail(Error::UnknownKnob,
            "unknown knob `" + std::string{key} + "`");
    }
    if (key == "stapm-limit")        return invoke(sym_.set_stapm_limit,       key, value);
    if (key == "fast-limit")         return invoke(sym_.set_fast_limit,        key, value);
    if (key == "slow-limit")         return invoke(sym_.set_slow_limit,        key, value);
    if (key == "tctl-temp")          return invoke(sym_.set_tctl_temp,         key, value);
    if (key == "vrm-current")        return invoke(sym_.set_vrm_current,       key, value);
    if (key == "vrmmax-current")     return invoke(sym_.set_vrmmax_current,    key, value);
    if (key == "vrmsoc-current")     return invoke(sym_.set_vrmsoc_current,    key, value);
    if (key == "vrmsocmax-current")  return invoke(sym_.set_vrmsocmax_current, key, value);
    // is_known_knob guarantees one branch above hit.
    return Status::fail(Error::UnknownKnob, std::string{key});
}

} // namespace onebit::power
