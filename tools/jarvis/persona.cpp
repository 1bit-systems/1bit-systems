// persona.cpp — PersonaManager implementation.
// Loads persona JSON files from disk, manages active persona,
// builds system prompts, and applies catchphrases to LLM output.

#include "jarvis/persona.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace jarvis {

// ── Helpers ────────────────────────────────────────────────────────────

static std::string home_dir() {
    const char* h = getenv("HOME");
    return h ? h : "/tmp";
}

// ── Implementation struct ──────────────────────────────────────────────

struct PersonaManager::Impl {
    std::mutex mutex;
    std::map<std::string, PersonaConfig, std::less<>> personas;
    std::string active_name = "Zaya";
    std::mt19937 rng{std::random_device{}()};

    // Default Zaya persona — always available
    PersonaConfig default_persona() const {
        PersonaConfig cfg;
        cfg.name = "Zaya";
        cfg.voice_pack = "zaya_default.voice";
        cfg.system_prompt =
            "You are Zaya, an enthusiastic and knowledgeable AI co-host. "
            "You love helping people with creative projects, technology discussions, "
            "and podcast conversations. Keep responses conversational and engaging. "
            "Ask follow-up questions to keep the dialogue flowing.";
        cfg.speaking_style = "chatty";
        cfg.speaking_rate = 1.0f;
        cfg.enthusiasm = 0.7f;
        cfg.formality = 0.3f;
        cfg.catchphrases = {
            "That's a great question!",
            "Here's what I think...",
            "Let me share something interesting about that."
        };
        return cfg;
    }

    Impl() {
        // Insert the default persona so it's always available
        personas["Zaya"] = default_persona();
    }
};

PersonaManager::PersonaManager() : impl_(std::make_unique<Impl>()) {}
PersonaManager::~PersonaManager() = default;

bool PersonaManager::load_persona(const std::string& path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[persona] Cannot open persona file: %s\n", path.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        fprintf(stderr, "[persona] JSON parse error in %s: %s\n", path.c_str(), e.what());
        return false;
    }

    PersonaConfig cfg;
    cfg.name = j.value("name", "Unnamed");
    cfg.voice_pack = j.value("voice_pack", "");
    cfg.system_prompt = j.value("system_prompt", "");
    cfg.speaking_style = j.value("speaking_style", "chatty");
    cfg.speaking_rate = j.value("speaking_rate", 1.0f);
    cfg.voice_pitch = j.value("voice_pitch", 1.0f);
    cfg.knowledge_domain = j.value("knowledge_domain", "");
    cfg.enthusiasm = j.value("enthusiasm", 0.7f);
    cfg.formality = j.value("formality", 0.5f);

    // Parse catchphrases
    if (j.contains("catchphrases") && j["catchphrases"].is_array()) {
        for (auto& cp : j["catchphrases"]) {
            cfg.catchphrases.push_back(cp.get<std::string>());
        }
    }

    impl_->personas[cfg.name] = std::move(cfg);
    fprintf(stderr, "[persona] Loaded persona '%s' from %s\n",
            impl_->personas[cfg.name].name.c_str(), path.c_str());
    return true;
}

const PersonaConfig& PersonaManager::active() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->personas.find(impl_->active_name);
    if (it != impl_->personas.end()) return it->second;
    // Fallback to default — always present
    static PersonaConfig default_cfg;
    default_cfg = impl_->default_persona();
    return default_cfg;
}

bool PersonaManager::set_active(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->personas.find(name);
    if (it == impl_->personas.end()) {
        fprintf(stderr, "[persona] Unknown persona '%s' — not switching\n", name.c_str());
        return false;
    }
    impl_->active_name = name;
    fprintf(stderr, "[persona] Switched to persona '%s'\n", name.c_str());
    return true;
}

std::vector<std::string> PersonaManager::list_personas() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::string> names;
    names.reserve(impl_->personas.size());
    for (auto& [name, _] : impl_->personas) {
        names.push_back(name);
    }
    return names;
}

const PersonaConfig* PersonaManager::get_persona(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->personas.find(name);
    if (it != impl_->personas.end()) return &it->second;
    return nullptr;
}

std::string PersonaManager::build_system_prompt() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->personas.find(impl_->active_name);
    if (it == impl_->personas.end()) return "";

    const PersonaConfig& cfg = it->second;
    std::ostringstream ss;

    // Start with the persona's system prompt
    if (!cfg.system_prompt.empty()) {
        ss << cfg.system_prompt << "\n\n";
    }

    // Append style instructions
    ss << "Speaking style: " << cfg.speaking_style << ".\n";

    if (cfg.enthusiasm > 0.6f) {
        ss << "Be enthusiastic and energetic in your responses.\n";
    } else if (cfg.enthusiasm < 0.3f) {
        ss << "Keep a calm, measured tone.\n";
    }

    if (cfg.formality > 0.7f) {
        ss << "Use formal language and proper structure.\n";
    } else if (cfg.formality < 0.3f) {
        ss << "Be casual and conversational. Use contractions and everyday language.\n";
    }

    if (!cfg.knowledge_domain.empty()) {
        ss << "Your knowledge domain is: " << cfg.knowledge_domain << ".\n";
    }

    // Speaking rate hint (for TTS, not LLM, but include as instruction)
    ss << "Speaking rate: " << cfg.speaking_rate << "x.\n";

    // Catchphrase injection hint
    if (!cfg.catchphrases.empty()) {
        ss << "When appropriate, use natural transitions like: ";
        for (size_t i = 0; i < cfg.catchphrases.size(); i++) {
            if (i > 0) ss << ", ";
            ss << "\"" << cfg.catchphrases[i] << "\"";
        }
        ss << ".\n";
    }

    return ss.str();
}

std::string PersonaManager::apply_catchphrases(const std::string& text) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->personas.find(impl_->active_name);
    if (it == impl_->personas.end()) return text;
    const PersonaConfig& cfg = it->second;

    if (cfg.catchphrases.empty() || text.empty()) return text;

    // Simple strategy: find natural break points and inject a catchphrase
    // at the beginning or after a sentence boundary.
    // For now, just prepend a random catchphrase if the text doesn't already
    // start with one and the text isn't too short.
    if (text.size() < 20) return text;

    // Check if text already contains a catchphrase
    for (const auto& cp : cfg.catchphrases) {
        if (text.find(cp) != std::string::npos) return text;
    }

    // 30% chance to inject one catchphrase at the start
    std::uniform_int_distribution<int> dist(0, 9);
    if (dist(impl_->rng) >= 3) return text;

    std::uniform_int_distribution<size_t> pick(0, cfg.catchphrases.size() - 1);
    const std::string& cp = cfg.catchphrases[pick(impl_->rng)];

    return cp + " " + text;
}

int PersonaManager::scan_directory(const std::string& dir) {
    std::string scan_dir = dir;
    if (scan_dir.empty()) {
        scan_dir = home_dir() + "/personas";
    }

    if (!fs::is_directory(scan_dir)) {
        std::error_code ec;
        fs::create_directories(scan_dir, ec);
        if (ec) {
            fprintf(stderr, "[persona] Cannot create personas directory '%s': %s\n",
                    scan_dir.c_str(), ec.message().c_str());
            return 0;
        }
    }

    int loaded = 0;
    for (auto& entry : fs::directory_iterator(scan_dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        fs::path p = entry.path();
        if (p.extension() != ".json") continue;

        if (load_persona(p.string())) {
            loaded++;
        }
    }

    fprintf(stderr, "[persona] Scanned '%s': loaded %d persona(s)\n", scan_dir.c_str(), loaded);
    return loaded;
}

} // namespace jarvis
