// backend_plugin.cpp — Plugin loader for external backend modules
// ONNX Runtime EP equivalent: load backends from .so files at runtime.

#include "backend_plugin.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>

// ── Load a single plugin ──
BackendPluginLoader* BackendPluginLoader::load(const std::string& so_path, std::string* error_out) {
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        if (error_out) *error_out = std::string("dlopen failed: ") + dlerror();
        return nullptr;
    }

    auto* loader = new BackendPluginLoader();
    loader->handle_ = handle;
    loader->path_ = so_path;

    if (!loader->resolve_symbols(error_out)) {
        delete loader;
        return nullptr;
    }

    loader->valid_ = true;
    return loader;
}

// ── Scan directory for plugins ──
std::vector<BackendPluginLoader*> BackendPluginLoader::scan_directory(
    const std::string& dir, std::vector<std::string>* errors) {
    std::vector<BackendPluginLoader*> loaders;

    DIR* d = opendir(dir.c_str());
    if (!d) {
        if (errors) errors->push_back("Cannot open plugin directory: " + dir);
        return loaders;
    }

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;

        // Must be a .so file
        if (name.size() < 3 || name.substr(name.size() - 3) != ".so") continue;
        // Skip backup/versioned .so files (e.g., .so.1)
        if (name.find(".so.") != std::string::npos) continue;

        std::string full_path = dir + "/" + name;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;

        std::string error;
        auto* loader = load(full_path, &error);
        if (loader) {
            loaders.push_back(loader);
        } else if (errors) {
            errors->push_back(name + ": " + error);
        }
    }
    closedir(d);

    // Sort by id for deterministic ordering
    std::sort(loaders.begin(), loaders.end(),
        [](BackendPluginLoader* a, BackendPluginLoader* b) {
            return a->id() < b->id();
        });

    return loaders;
}

// ── Destructor ──
BackendPluginLoader::~BackendPluginLoader() {
    close();
}

// ── Resolve all required symbols ──
bool BackendPluginLoader::resolve_symbols(std::string* error) {
    auto resolve = [&](const char* name) -> void* {
        void* sym = dlsym(handle_, name);
        if (!sym && error) *error = "Missing symbol: " + std::string(name);
        return sym;
    };

    // Required symbols
    abi_.create_backend    = (Backend* (*)())resolve("create_backend");
    abi_.destroy_backend   = (void (*)(Backend*))resolve("destroy_backend");
    abi_.backend_type      = (int (*)())resolve("backend_type");
    abi_.backend_id        = (const char* (*)())resolve("backend_id");
    abi_.backend_version   = (const char* (*)())resolve("backend_version");

    if (!abi_.create_backend || !abi_.destroy_backend || !abi_.backend_type ||
        !abi_.backend_id || !abi_.backend_version) {
        if (error) *error = "Missing required ABI symbols";
        return false;
    }

    // Optional
    abi_.backend_description = (const char* (*)())resolve("backend_description");

    // Grab metadata
    type_ = static_cast<BackendType>(abi_.backend_type());
    id_ = abi_.backend_id() ? abi_.backend_id() : "unknown";
    version_ = abi_.backend_version() ? abi_.backend_version() : "0.0.0";
    description_ = abi_.backend_description ? abi_.backend_description() : id_;

    return true;
}

// ── Instantiate backend ──
Backend* BackendPluginLoader::instantiate() {
    if (!valid_ || !abi_.create_backend) return nullptr;
    return abi_.create_backend();
}

// ── Destroy backend ──
void BackendPluginLoader::destroy(Backend* backend) {
    if (valid_ && abi_.destroy_backend && backend) {
        abi_.destroy_backend(backend);
    }
}

// ── Close plugin ──
void BackendPluginLoader::close() {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
    valid_ = false;
}

// ── Convenience: find plugins ──
std::vector<BackendPluginLoader*> load_backend_plugins(const std::string& custom_dir) {
    std::vector<BackendPluginLoader*> loaders;
    std::vector<std::string> errors;

    // Search order:
    // 1. Custom directory (if specified)
    // 2. $HOME/.backends/
    // 3. /usr/lib/backends/
    // 4. ./backends/ (relative to cwd)

    std::vector<std::string> dirs;
    if (!custom_dir.empty()) dirs.push_back(custom_dir);

    const char* home = getenv("HOME");
    if (home) dirs.push_back(std::string(home) + "/.backends");

    dirs.push_back("/usr/lib/backends");
    dirs.push_back("./backends");

    for (auto& dir : dirs) {
        auto found = BackendPluginLoader::scan_directory(dir, &errors);
        loaders.insert(loaders.end(), found.begin(), found.end());
    }

    // Print errors if any
    for (auto& e : errors)
        fprintf(stderr, "  plugin: %s\n", e.c_str());

    return loaders;
}
