#include "onebit/stream/handlers.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <utility>

namespace onebit::stream {

namespace fs = std::filesystem;
using json   = nlohmann::json;

struct AppState::Impl {
    fs::path                          catalog_dir;
    AuthConfig                        auth;
    mutable std::shared_mutex         m;
    std::vector<Catalog>              catalogs;
};

AppState::AppState(fs::path catalog_dir, AuthConfig auth)
    : impl_{std::make_unique<Impl>()}
{
    impl_->catalog_dir = std::move(catalog_dir);
    impl_->auth        = std::move(auth);
}
AppState::AppState(AppState&&) noexcept            = default;
AppState& AppState::operator=(AppState&&) noexcept = default;
AppState::~AppState()                              = default;

const fs::path& AppState::catalog_dir() const noexcept
{
    return impl_->catalog_dir;
}
const AuthConfig& AppState::auth() const noexcept { return impl_->auth; }

AppState::ReindexReport AppState::reindex()
{
    ReindexReport r;
    std::vector<Catalog> loaded;

    std::error_code ec;
    if (!fs::exists(impl_->catalog_dir, ec) || !fs::is_directory(impl_->catalog_dir, ec)) {
        r.errors.emplace_back(impl_->catalog_dir.string(),
                              ec ? ec.message() : "not a directory");
        std::unique_lock lk(impl_->m);
        impl_->catalogs.clear();
        return r;
    }

    auto       it  = fs::directory_iterator(impl_->catalog_dir, ec);
    const auto end = fs::directory_iterator{};
    if (ec) {
        r.errors.emplace_back(impl_->catalog_dir.string(), ec.message());
        std::unique_lock lk(impl_->m);
        impl_->catalogs.clear();
        return r;
    }
    while (it != end) {
        std::error_code step_ec;
        if (it->is_regular_file(step_ec)) {
            const auto& p = it->path();
            if (p.extension() == ".1bl") {
                auto cat = open_catalog(p);
                if (!cat) {
                    r.errors.emplace_back(p.string(), cat.error().what());
                } else {
                    loaded.push_back(std::move(*cat));
                }
            }
        }
        it.increment(step_ec);
        if (step_ec) {
            break;
        }
    }
    r.loaded = loaded.size();
    {
        std::unique_lock lk(impl_->m);
        impl_->catalogs = std::move(loaded);
    }
    return r;
}

std::vector<Catalog> AppState::snapshot_catalogs() const
{
    std::shared_lock lk(impl_->m);
    return impl_->catalogs;
}

namespace {

[[nodiscard]] std::optional<Catalog> find_by_slug(
    const std::vector<Catalog>& cats, std::string_view slug)
{
    for (const auto& c : cats) {
        if (c.slug() == slug) {
            return c;
        }
    }
    return std::nullopt;
}

void respond_lossy(const Catalog& cat, httplib::Response& res)
{
    auto bytes = build_lossy_bytes(cat);
    if (!bytes) {
        res.status = 500;
        res.set_content("lossy build: " + bytes.error().what(), "text/plain");
        return;
    }
    const std::string filename = cat.slug() + ".lossy.1bl";
    res.status                 = 200;
    res.set_header("Content-Disposition",
                   "attachment; filename=\"" + filename + "\"");
    res.set_content(reinterpret_cast<const char*>(bytes->data()),
                    bytes->size(), "application/octet-stream");
}

void respond_lossless(const Catalog& cat, httplib::Response& res)
{
    std::ifstream in(cat.path, std::ios::binary);
    if (!in) {
        res.status = 500;
        res.set_content("open: failed", "text/plain");
        return;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    auto              bytes    = std::move(ss).str();
    const std::string filename = cat.slug() + ".1bl";
    res.status                 = 200;
    res.set_header("Content-Length", std::to_string(bytes.size()));
    res.set_header("Content-Disposition",
                   "attachment; filename=\"" + filename + "\"");
    res.set_content(bytes.data(), bytes.size(), "application/octet-stream");
}

[[nodiscard]] const char* gate_message(GateOutcome g) noexcept
{
    switch (g) {
    case GateOutcome::MissingHeader:
        return "missing authorization header";
    case GateOutcome::BadScheme:
        return "expected Bearer scheme";
    case GateOutcome::BadToken:
        return "invalid token";
    case GateOutcome::WrongTier:
        return "token does not carry premium tier";
    case GateOutcome::ServerMisconfigured:
        return "lossless gate not configured";
    case GateOutcome::Allow:
        return "ok";
    }
    return "unknown";
}

} // namespace

void build(httplib::Server& server, AppState& state)
{
    server.Get("/v1/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    server.Get("/v1/catalogs",
               [&state](const httplib::Request&, httplib::Response& res) {
                   const auto cats = state.snapshot_catalogs();
                   json       data = json::array();
                   for (const auto& c : cats) {
                       data.push_back(json{
                           {"slug",             c.slug()},
                           {"title",            c.manifest.title},
                           {"artist",           c.manifest.artist},
                           {"license",          c.manifest.license},
                           {"tier",             c.manifest.tier},
                           {"residual_present", c.manifest.residual_present},
                           {"bytes",            c.total_bytes},
                       });
                   }
                   const json body{{"object", "list"}, {"data", std::move(data)}};
                   res.status = 200;
                   res.set_content(body.dump(), "application/json");
               });

    server.Get(R"(/v1/catalogs/([^/]+))",
               [&state](const httplib::Request& req, httplib::Response& res) {
                   const auto cats = state.snapshot_catalogs();
                   const std::string slug = req.matches[1];
                   const auto        cat  = find_by_slug(cats, slug);
                   if (!cat) {
                       res.status = 404;
                       res.set_content("unknown catalog", "text/plain");
                       return;
                   }
                   json m{{"v", cat->manifest.v},
                          {"catalog", cat->manifest.catalog},
                          {"title", cat->manifest.title},
                          {"artist", cat->manifest.artist},
                          {"license", cat->manifest.license},
                          {"created", cat->manifest.created},
                          {"tier", cat->manifest.tier},
                          {"residual_present", cat->manifest.residual_present}};
                   res.status = 200;
                   res.set_content(m.dump(), "application/json");
               });

    server.Get(R"(/v1/catalogs/([^/]+)/lossy)",
               [&state](const httplib::Request& req, httplib::Response& res) {
                   const auto cats = state.snapshot_catalogs();
                   const std::string slug = req.matches[1];
                   const auto        cat  = find_by_slug(cats, slug);
                   if (!cat) {
                       res.status = 404;
                       res.set_content("unknown catalog", "text/plain");
                       return;
                   }
                   respond_lossy(*cat, res);
               });

    server.Get(R"(/v1/catalogs/([^/]+)/lossless)",
               [&state](const httplib::Request& req, httplib::Response& res) {
                   const auto gate = check_premium(state.auth(), req);
                   if (gate != GateOutcome::Allow) {
                       res.status = as_status(gate);
                       res.set_content(gate_message(gate), "text/plain");
                       return;
                   }
                   const auto cats = state.snapshot_catalogs();
                   const std::string slug = req.matches[1];
                   const auto        cat  = find_by_slug(cats, slug);
                   if (!cat) {
                       res.status = 404;
                       res.set_content("unknown catalog", "text/plain");
                       return;
                   }
                   respond_lossless(*cat, res);
               });

    server.Post("/internal/reindex",
                [&state](const httplib::Request& req, httplib::Response& res) {
                    const auto gate = check_admin(state.auth(), req);
                    if (gate != GateOutcome::Allow) {
                        res.status = as_status(gate);
                        res.set_content("admin auth failed", "text/plain");
                        return;
                    }
                    auto rep = state.reindex();
                    json errs = json::array();
                    for (const auto& [p, e] : rep.errors) {
                        errs.push_back(json{{"path", p}, {"error", e}});
                    }
                    json body{{"loaded", rep.loaded}, {"errors", std::move(errs)}};
                    res.status = 200;
                    res.set_content(body.dump(), "application/json");
                });
}

} // namespace onebit::stream
