// persona.h — Co-host persona system for Zaya voice co-host.
// Defines personality, speaking style, and behavior configuration.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace jarvis {

struct PersonaConfig {
    std::string name = "Zaya";
    std::string voice_pack = "";           // which .voice pack to use
    std::string system_prompt = "";        // system prompt for LLM
    std::string speaking_style = "chatty"; // chatty, professional, concise
    float speaking_rate = 1.0f;            // 0.5-2.0
    float voice_pitch = 1.0f;              // 0.5-2.0 (via codec decoder)
    std::vector<std::string> catchphrases; // injected naturally
    std::string knowledge_domain = "";     // e.g., "tech", "gaming", "podcast"

    // Emotion/affect settings
    float enthusiasm = 0.7f;               // 0.0 (monotone) - 1.0 (excited)
    float formality = 0.5f;                // 0.0 (casual) - 1.0 (formal)
};

class PersonaManager {
public:
    PersonaManager();
    ~PersonaManager();

    // Load persona from JSON file
    bool load_persona(const std::string& path);

    // Get active persona
    const PersonaConfig& active() const;

    // Switch persona by name (must already be loaded)
    bool set_active(const std::string& name);

    // List available personas
    std::vector<std::string> list_personas() const;

    // Get a persona config by name (immutable reference)
    const PersonaConfig* get_persona(const std::string& name) const;

    // Build system prompt with persona
    std::string build_system_prompt() const;

    // Inject catchphrases into text (post-process LLM output)
    std::string apply_catchphrases(const std::string& text) const;

    // Scan a directory for persona JSON files and load them
    int scan_directory(const std::string& dir);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace jarvis
