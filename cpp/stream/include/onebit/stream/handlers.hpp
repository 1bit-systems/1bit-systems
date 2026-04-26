#pragma once

// HTTP handlers + AppState for the .1bl catalog server.
//
// Routes (see crate-level doc for full shape):
//   GET  /v1/health
//   GET  /v1/catalogs
//   GET  /v1/catalogs/:slug
//   GET  /v1/catalogs/:slug/lossy
//   GET  /v1/catalogs/:slug/lossless     (premium-gated)
//   POST /internal/reindex               (admin-gated)

#include "onebit/stream/auth.hpp"
#include "onebit/stream/container.hpp"

#include <httplib.h>

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace onebit::stream {

class AppState {
public:
    AppState(std::filesystem::path catalog_dir, AuthConfig auth);
    AppState(const AppState&)            = delete;
    AppState& operator=(const AppState&) = delete;
    AppState(AppState&&) noexcept;
    AppState& operator=(AppState&&) noexcept;
    ~AppState();

    [[nodiscard]] const std::filesystem::path& catalog_dir() const noexcept;
    [[nodiscard]] const AuthConfig&            auth() const noexcept;

    // Returns (loaded_count, list_of_(path,error_msg)).
    struct ReindexReport {
        std::size_t                                       loaded{0};
        std::vector<std::pair<std::string, std::string>> errors;
    };
    [[nodiscard]] ReindexReport reindex();

    // Snapshot of catalogs under read lock.
    [[nodiscard]] std::vector<Catalog> snapshot_catalogs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Wire all routes onto `server`. `state` must outlive the server.
void build(httplib::Server& server, AppState& state);

} // namespace onebit::stream
