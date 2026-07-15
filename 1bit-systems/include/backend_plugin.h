// backend_plugin.h — Plugin loader for external backend modules
// Windows ONNX Runtime EP equivalent — backends as dynamically loaded .so files.
// A backend plugin must export these symbols:
//   extern "C" Backend*  create_backend();
//   extern "C" void      destroy_backend(Backend*);
//   extern "C" int       backend_type();  // returns BackendType enum value
//   extern "C" const char* backend_id();  // returns unique string ID
//   extern "C" const char* backend_version();

#pragma once
#include "backend.h"
#include <string>
#include <vector>

// ── Plugin ABI (v1) ──
// All symbols are C-linkage, versioned with a sentinel to detect ABI mismatch.
struct BackendABI {
    uint32_t version_major = 0;
    uint32_t version_minor = 0;
    uint32_t abi_sentinel = 0x42A1B1E5;  // "ABI" magic

    Backend* (*create_backend)() = nullptr;
    void     (*destroy_backend)(Backend*) = nullptr;
    int      (*backend_type)() = nullptr;
    const char* (*backend_id)() = nullptr;
    const char* (*backend_version)() = nullptr;
    const char* (*backend_description)() = nullptr;
};

// ── Plugin scanner ──
class BackendPluginLoader {
public:
    static constexpr uint32_t ABI_VERSION = 1;

    /// Load a single plugin .so
    static BackendPluginLoader* load(const std::string& so_path, std::string* error_out = nullptr);

    /// Scan a directory for backend plugins (*.so)
    static std::vector<BackendPluginLoader*> scan_directory(
        const std::string& dir, std::vector<std::string>* errors = nullptr);

    ~BackendPluginLoader();

    bool is_valid() const { return valid_; }
    const std::string& id() const { return id_; }
    const std::string& version() const { return version_; }
    const std::string& description() const { return description_; }
    BackendType type() const { return type_; }
    void* handle() const { return handle_; }

    /// Instantiate the backend (calls create_backend from the plugin)
    Backend* instantiate();

    /// Destroy the backend (calls destroy_backend from the plugin)
    void destroy(Backend* backend);

private:
    BackendPluginLoader() = default;
    bool resolve_symbols(std::string* error);
    void close();

    void* handle_ = nullptr;
    bool valid_ = false;
    std::string id_;
    std::string version_;
    std::string description_;
    std::string path_;
    BackendType type_ = BackendType::NONE;
    BackendABI abi_;
};

// ── Convenience: find plugins relative to the executable ──
std::vector<BackendPluginLoader*> load_backend_plugins(
    const std::string& custom_dir = "");
