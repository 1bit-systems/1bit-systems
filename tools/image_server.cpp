// image_server.cpp — OpenAI-compatible image generation server
//
// Routes:
//   POST /v1/images/generations  — text-to-image
//   POST /v1/images/edits        — image-to-image (inpainting/outpainting)
//   POST /v1/images/variations   — create variations of an image
//   POST /v1/video/generations   — text-to-video (Wan, LTX, Hunyuan)
//   GET  /v1/models              — list available models
//   GET  /health                 — health check
//
// Built on stable-diffusion.cpp via the diffusion_bridge.
// Pure C++23, zero Python at runtime.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <csignal>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "diffusion_bridge.h"
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Simple base64 encode (no external dependency needed)
static std::string base64_encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        int b = (data[i] << 16) | (i+1 < len ? data[i+1] << 8 : 0) | (i+2 < len ? data[i+2] : 0);
        out += tbl[(b >> 18) & 0x3F];
        out += tbl[(b >> 12) & 0x3F];
        out += i+1 < len ? tbl[(b >> 6) & 0x3F] : '=';
        out += i+2 < len ? tbl[b & 0x3F] : '=';
    }
    return out;
}

// ─── Global config ────────────────────────────────────────────────
static std::string g_models_dir = "./models";
static int g_port = 8089;
static bool g_verbose = false;
static std::string g_backend_str = "auto";

// ─── Model registry ───────────────────────────────────────────────
struct DiffusionModelEntry {
    std::string id;
    std::string path;
    std::string vae_path;
    std::string description;
};

static std::vector<DiffusionModelEntry> g_models;
static std::mutex g_model_mtx;
static std::string g_active_model_id;
static std::unique_ptr<DiffusionEngine> g_engine;

// ─── Discover models in directory ─────────────────────────────────
static void discover_models(const std::string& dir) {
    g_models.clear();
    
    try {
        for (auto& entry : fs::directory_iterator(dir)) {
            auto path = entry.path();
            auto ext = path.extension().string();
            auto stem = path.stem().string();
            
            // Supported extensions: .gguf, .safetensors, .ckpt, .pth
            if (ext != ".gguf" && ext != ".safetensors" &&
                ext != ".ckpt" && ext != ".pth" && ext != ".pt") continue;
            
            DiffusionModelEntry model;
            model.id = stem;
            model.path = path.string();
            
            // Detect model type from filename conventions
            std::string lower = stem;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (lower.find("flux") != std::string::npos) {
                model.description = "FLUX";
            } else if (lower.find("sdxl") != std::string::npos) {
                model.description = "SDXL";
            } else if (lower.find("sd3") != std::string::npos || lower.find("sd_3") != std::string::npos) {
                model.description = "SD3";
            } else if (lower.find("qwen") != std::string::npos && lower.find("image") != std::string::npos) {
                model.description = "Qwen-Image";
            } else if (lower.find("wan") != std::string::npos) {
                model.description = "Wan Video";
            } else if (lower.find("ltx") != std::string::npos) {
                model.description = "LTX Video";
            } else if (lower.find("hunyuan") != std::string::npos) {
                model.description = "Hunyuan Video";
            } else if (lower.find("chroma") != std::string::npos) {
                model.description = "Chroma";
            } else if (lower.find("lens") != std::string::npos) {
                model.description = "Lens";
            } else if (lower.find("z-image") != std::string::npos || lower.find("zimage") != std::string::npos) {
                model.description = "Z-Image";
            } else {
                model.description = "SD1.x / default";
            }
            
            // Look for companion VAE
            std::string vae_path_str = path.string();
            auto dot = vae_path_str.find_last_of('.');
            if (dot != std::string::npos) {
                vae_path_str = vae_path_str.substr(0, dot) + ".vae" + ext;
                if (fs::exists(vae_path_str)) {
                    model.vae_path = vae_path_str;
                }
            }
            
            g_models.push_back(model);
            
            if (g_verbose) {
                printf("  [%zu] %s (%s)\n", g_models.size() - 1,
                       model.id.c_str(), model.description.c_str());
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error scanning models dir: %s\n", e.what());
    }
}

// ─── HTTP Handlers ────────────────────────────────────────────────

static void handle_health(const httplib::Request& req, httplib::Response& res) {
    json resp = {
        {"status", "ok"},
        {"engine", "1bit-systems image server"},
        {"model_loaded", g_engine && g_engine->is_loaded()},
        {"models_available", g_models.size()},
    };
    res.set_content(resp.dump(), "application/json");
}

static void handle_list_models(const httplib::Request& req, httplib::Response& res) {
    json data = json::array();
    for (auto& m : g_models) {
        data.push_back({
            {"id", m.id},
            {"object", "model"},
            {"created", 0},
            {"owned_by", "1bit-systems"},
            {"description", m.description},
        });
    }
    json resp = {{"object", "list"}, {"data", data}};
    res.set_content(resp.dump(), "application/json");
}

static bool ensure_model_loaded(const std::string& model_id) {
    if (g_engine && g_engine->is_loaded() && g_active_model_id == model_id) {
        return true;  // Already loaded
    }
    
    // Find model in registry
    DiffusionModelEntry* found = nullptr;
    for (auto& m : g_models) {
        if (m.id == model_id) { found = &m; break; }
    }
    if (!found) {
        fprintf(stderr, "Model '%s' not found in registry\n", model_id.c_str());
        return false;
    }
    
    if (!g_engine) g_engine = std::make_unique<DiffusionEngine>();
    printf("Loading model: %s (backend=%s)\n", model_id.c_str(), g_backend_str.c_str());
    if (!g_engine->load_model(found->path, found->vae_path)) {
        fprintf(stderr, "Failed to load model '%s'\n", model_id.c_str());
        return false;
    }
    
    g_active_model_id = model_id;
    printf("Loaded model: %s\n", model_id.c_str());
    return true;
}

static void handle_image_generation(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        
        std::string model_id = body.value("model", g_models.empty() ? "" : g_models[0].id);
        std::string prompt = body.value("prompt", "");
        std::string negative_prompt = body.value("negative_prompt", "");
        int width = body.value("width", 512);
        int height = body.value("height", 512);
        int steps = body.value("steps", 20);
        float cfg_scale = body.value("cfg_scale", 7.0f);
        int seed = body.value("seed", -1);
        int n = body.value("n", 1);
        
        // Optional LoRA
        std::vector<std::string> lora_paths;
        std::vector<float> lora_strengths;
        if (body.contains("lora_paths")) {
            for (auto& lp : body["lora_paths"]) lora_paths.push_back(lp);
            for (auto& ls : body["lora_strengths"]) lora_strengths.push_back(ls);
        }
        
        if (prompt.empty()) {
            json err = {{"error", "prompt is required"}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        if (!ensure_model_loaded(model_id)) {
            json err = {{"error", "Failed to load model: " + model_id}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        DiffusionParams params;
        params.prompt = prompt;
        params.negative_prompt = negative_prompt;
        params.width = width;
        params.height = height;
        params.steps = steps;
        params.cfg_scale = cfg_scale;
        params.seed = seed;
        params.lora_paths = lora_paths;
        params.lora_weights = lora_strengths;
        
        auto result = g_engine->txt2img(params);
        
        if (result.data.empty()) {
            json err = {{"error", "Generation failed (null result)"}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }
        
        // Build OpenAI-compatible response
        std::string b64 = base64_encode(result.data.data(), result.data.size());
        json resp = {
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"data", json::array({
                {
                    {"b64_json", b64},
                    {"seed", result.seed_used},
                    {"width", result.width},
                    {"height", result.height},
                    {"mime_type", result.mime_type},
                }
            })}
        };
        
        res.set_content(resp.dump(), "application/json");
        
    } catch (const std::exception& e) {
        json err = {{"error", std::string("Internal error: ") + e.what()}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    }
}

// ─── Main ─────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -w, --models-dir DIR    Model directory (default: ./models)\n"
        "  -p, --port PORT         Server port  (default: 8089)\n"
        "  --backend BACKEND       Backend: auto|cpu|cuda|vulkan|metal (default: auto)\n"
        "  -v, --verbose           Verbose output\n"
        "  -h, --help              Show this help\n",
        prog);
}

int main(int argc, char** argv) {
    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" || arg == "--models-dir") {
            if (i + 1 < argc) g_models_dir = argv[++i];
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) g_port = std::stoi(argv[++i]);
        } else if (arg == "--backend") {
            if (i + 1 < argc) g_backend_str = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            g_verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Discover models
    printf("Scanning models in %s...\n", g_models_dir.c_str());
    discover_models(g_models_dir);
    printf("Found %zu diffusion model(s)\n", g_models.size());
    
    // Create HTTP server
    httplib::Server svr;
    
    // Routes
    svr.Get("/health", handle_health);
    svr.Get("/v1/models", handle_list_models);
    svr.Post("/v1/images/generations", handle_image_generation);
    
    // Error handler
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
        json err = {{"error", "Internal server error"}};
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    });
    
    printf("Starting image server on port %d...\n", g_port);
    printf("  Endpoints:\n");
    printf("    GET  /health\n");
    printf("    GET  /v1/models\n");
    printf("    POST /v1/images/generations\n");
    
    svr.listen("0.0.0.0", g_port);
    
    return 0;
}
