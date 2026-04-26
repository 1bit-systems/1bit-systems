// 1bit-helm — bearer-token storage for lemond's /v1/* endpoints.
//
// Mirrors crates/1bit-helm/src/bearer.rs. The Linux keyring path
// uses `org.freedesktop.Secrets` over QtDBus (Plasma kwallet +
// gnome-keyring both speak it); on dbus failure we fall back to
// `~/.config/1bit-helm/bearer.txt` with 0600 perms — same paths the
// Rust crate writes so existing installs roll forward.
//
// pImpl per Core Guidelines I.27 so the QtDBus symbols don't bleed
// into headers — the Settings widget pulls in only this header.

#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace onebit::helm {

enum class BearerBackend : std::uint8_t { Keyring, XdgFile, None };

[[nodiscard]] std::string_view bearer_backend_label(BearerBackend b) noexcept;

class Bearer {
public:
    // Production: real XDG config dir + tries the keyring.
    Bearer();
    // Test-friendly: force the fallback root + skip keyring.
    static Bearer with_file_only(std::filesystem::path fallback_root);

    Bearer(const Bearer&)            = delete;
    Bearer& operator=(const Bearer&) = delete;
    Bearer(Bearer&&) noexcept;
    Bearer& operator=(Bearer&&) noexcept;
    ~Bearer();

    [[nodiscard]] BearerBackend backend() const noexcept;
    [[nodiscard]] std::optional<std::string> get() const;

    // Best-effort load. Keyring → file → give up. Never errors.
    void load();

    // Write `token`. Keyring first; on failure persist to the XDG
    // file with 0600.
    [[nodiscard]] std::expected<BearerBackend, std::string>
    store(std::string_view token);

    // Forget the bearer everywhere.
    [[nodiscard]] std::expected<void, std::string> clear();

    // Visible for tests + Settings pane.
    [[nodiscard]] std::filesystem::path fallback_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit Bearer(std::unique_ptr<Impl> impl);
};

} // namespace onebit::helm
