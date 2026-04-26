#include "onebit/helm/bearer.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#if defined(ONEBIT_HELM_HAS_QT)
#  include <QtDBus/QDBusConnection>
#  include <QtDBus/QDBusConnectionInterface>
#  include <QtCore/QString>
#endif

#if defined(__unix__)
#  include <sys/stat.h>
#endif

namespace onebit::helm {

namespace {

constexpr std::string_view kFallbackDir  = "1bit-helm";
constexpr std::string_view kFallbackFile = "bearer.txt";

std::filesystem::path xdg_config_home()
{
    if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && *x) {
        return std::filesystem::path(x);
    }
    if (const char* h = std::getenv("HOME"); h && *h) {
        return std::filesystem::path(h) / ".config";
    }
    return std::filesystem::current_path();
}

std::filesystem::path default_fallback_root()
{
    return xdg_config_home() / std::string(kFallbackDir);
}

std::filesystem::path file_path(const std::filesystem::path& root)
{
    return root / std::string(kFallbackFile);
}

std::optional<std::string>
load_file(const std::filesystem::path& root)
{
    std::ifstream f(file_path(root));
    if (!f) return std::nullopt;
    std::ostringstream buf;
    buf << f.rdbuf();
    auto raw = buf.str();
    while (!raw.empty()
           && (raw.back() == '\n' || raw.back() == '\r' || raw.back() == ' ')) {
        raw.pop_back();
    }
    if (raw.empty()) return std::nullopt;
    return raw;
}

std::expected<void, std::string>
store_file(const std::filesystem::path& root, std::string_view token)
{
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        return std::unexpected("mkdir " + root.string() + ": " + ec.message());
    }
    auto p = file_path(root);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected("open " + p.string());
    f.write(token.data(), static_cast<std::streamsize>(token.size()));
    if (!f) return std::unexpected("write " + p.string());
    f.close();
#if defined(__unix__)
    if (::chmod(p.string().c_str(), 0600) != 0) {
        return std::unexpected("chmod 0600 " + p.string());
    }
#endif
    return {};
}

std::expected<void, std::string>
delete_file(const std::filesystem::path& root)
{
    auto p = file_path(root);
    std::error_code ec;
    if (std::filesystem::exists(p, ec)) {
        std::filesystem::remove(p, ec);
        if (ec) return std::unexpected("rm " + p.string() + ": " + ec.message());
    }
    return {};
}

// Linux secret-service via QtDBus.
//
// We talk to the standard org.freedesktop.secrets bus (kwallet +
// gnome-keyring both implement it). The full Secret Service spec
// (SearchItems → OpenSession → GetSecret with custom-marshalled
// `Secret` struct) is non-trivial to type-safely call from
// QDBusInterface — it requires `Q_DECLARE_METATYPE` plus operator
// `<<`/`>>` overloads for the `(o,ay,ay,s)` tuple plus DBus-side
// type registration.
//
// MVP: we *probe* the bus to detect whether secret-service is
// available, but persist via the XDG file regardless. The Rust
// build's `keyring` crate also falls back to the file when no
// agent is reachable, and the file is 0600 so the practical
// security delta on a single-user box is small. Full keyring
// marshaling lands in a follow-up commit alongside the
// QDBusArgument operators (TODO: project_helm_qt6_keyring).
#if defined(ONEBIT_HELM_HAS_QT)
[[maybe_unused]] bool keyring_bus_alive()
{
    auto bus = QDBusConnection::sessionBus();
    if (!bus.isConnected()) return false;
    auto* iface = bus.interface();
    if (!iface) return false;
    auto reply = iface->isServiceRegistered(
        QStringLiteral("org.freedesktop.secrets"));
    return reply.isValid() && reply.value();
}
#endif

inline std::optional<std::string> load_keyring()  { return std::nullopt; }
inline bool                       store_keyring(std::string_view) { return false; }
inline bool                       clear_keyring() { return false; }

} // namespace

struct Bearer::Impl {
    std::filesystem::path  fallback_root;
    bool                   try_keyring{true};
    BearerBackend          backend{BearerBackend::None};
    std::optional<std::string> cached;
};

std::string_view bearer_backend_label(BearerBackend b) noexcept
{
    switch (b) {
        case BearerBackend::Keyring: return "system keyring";
        case BearerBackend::XdgFile: return "~/.config/1bit-helm/bearer.txt (0600)";
        case BearerBackend::None:    return "unset";
    }
    return "unset";
}

Bearer::Bearer()
    : impl_(std::make_unique<Impl>())
{
    impl_->fallback_root = default_fallback_root();
    impl_->try_keyring   = true;
}

Bearer::Bearer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Bearer Bearer::with_file_only(std::filesystem::path fallback_root)
{
    auto impl = std::make_unique<Impl>();
    impl->fallback_root = std::move(fallback_root);
    impl->try_keyring   = false;
    return Bearer{std::move(impl)};
}

Bearer::Bearer(Bearer&&) noexcept            = default;
Bearer& Bearer::operator=(Bearer&&) noexcept = default;
Bearer::~Bearer()                            = default;

BearerBackend Bearer::backend() const noexcept
{
    return impl_ ? impl_->backend : BearerBackend::None;
}

std::optional<std::string> Bearer::get() const
{
    if (!impl_) return std::nullopt;
    return impl_->cached;
}

std::filesystem::path Bearer::fallback_path() const
{
    return file_path(impl_->fallback_root);
}

void Bearer::load()
{
    impl_->cached  = std::nullopt;
    impl_->backend = BearerBackend::None;
    if (impl_->try_keyring) {
        if (auto v = load_keyring()) {
            impl_->cached  = std::move(v);
            impl_->backend = BearerBackend::Keyring;
            return;
        }
    }
    if (auto v = load_file(impl_->fallback_root)) {
        impl_->cached  = std::move(v);
        impl_->backend = BearerBackend::XdgFile;
    }
}

std::expected<BearerBackend, std::string>
Bearer::store(std::string_view token)
{
    // Trim. Empty post-trim is rejected.
    while (!token.empty() && (token.front() == ' ' || token.front() == '\n' ||
                              token.front() == '\r' || token.front() == '\t')) {
        token.remove_prefix(1);
    }
    while (!token.empty() && (token.back() == ' ' || token.back() == '\n' ||
                              token.back() == '\r' || token.back() == '\t')) {
        token.remove_suffix(1);
    }
    if (token.empty()) {
        return std::unexpected("bearer token is empty");
    }
    if (impl_->try_keyring && store_keyring(token)) {
        impl_->cached  = std::string(token);
        impl_->backend = BearerBackend::Keyring;
        // Best-effort: nuke any stale file fallback so we don't drift.
        (void)delete_file(impl_->fallback_root);
        return BearerBackend::Keyring;
    }
    auto rc = store_file(impl_->fallback_root, token);
    if (!rc) return std::unexpected(rc.error());
    impl_->cached  = std::string(token);
    impl_->backend = BearerBackend::XdgFile;
    return BearerBackend::XdgFile;
}

std::expected<void, std::string> Bearer::clear()
{
    impl_->cached  = std::nullopt;
    impl_->backend = BearerBackend::None;
    if (impl_->try_keyring) {
        (void)clear_keyring();
    }
    return delete_file(impl_->fallback_root);
}

} // namespace onebit::helm
