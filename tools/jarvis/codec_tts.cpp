// codec_tts.cpp — Voice pack + TTS wrapper implementation.
//
// Voice packs are loaded from ~/voice-packs/ (configurable). Each pack
// is either a directory containing:
//   decoder.onnx       — the RVQ-VAE decoder model
//   speaker_emb.bin    — 512 float32 speaker embedding (raw binary)
//   config.json        — metadata: {"name":"...", "speaker":"...", "lang":"..."}
//
// Or a .voice zip file containing the same files. Zip support requires
// USE_MINIZ to be defined at build time.
//
// Text→codec tokens: currently a placeholder that generates random tokens.
// A real text→codec adapter (e.g., a small LM or lookup table) will be
// added in a future phase.
//
// Hot-reload: scan_voice_packs() checks mtime of each pack's files and
// reloads if modified. Call it periodically or on demand.

#include "jarvis/codec_tts.h"

#include "codec_decoder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace jarvis {

// ── Debug logging ──────────────────────────────────────────────────────
#ifndef NDEBUG
#define CODEC_LOG(fmt, ...) fprintf(stderr, "[codec-tts] " fmt "\n", ##__VA_ARGS__)
#else
#define CODEC_LOG(fmt, ...) ((void)0)
#endif

// ── Constants ──────────────────────────────────────────────────────────
static constexpr int kNumCodebooks = 8;
static constexpr int kSpeakerEmbDim = 512;
static constexpr int kDefaultVocabSize = 1024;
static constexpr int kSampleRate = 24000;

// ── Voice pack metadata (parsed from config.json) ──────────────────────
struct VoicePack {
    std::string name;
    std::string speaker_name;
    std::string language;
    std::string path;          // path to pack directory or .voice file
    int sample_rate = kSampleRate;
    fs::file_time_type mtime;  // for hot-reload detection

    // Loaded assets (filled after successful load)
    std::vector<float> speaker_emb;
    std::unique_ptr<CodecDecoder> decoder;
    bool loaded = false;
};

// ── Implementation struct ──────────────────────────────────────────────
struct CodecTtsImpl {
    std::string voice_packs_dir;
    std::mutex mutex;
    std::map<std::string, VoicePack, std::less<>> packs; // keyed by name
    std::mt19937 rng{std::random_device{}()};
};

// ── Helpers ────────────────────────────────────────────────────────────

static std::string env_or(const char* key, const std::string& def) {
    const char* v = getenv(key);
    return (v && *v) ? v : def;
}

static std::string home_dir() {
    const char* h = getenv("HOME");
    return h ? h : "/tmp";
}

// Trim whitespace from a string
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Simple JSON field extraction (no dependency — avoids pulling in nlohmann
// for the voice pack config parser). Handles the subset of JSON we need:
// {"key": "value", "key2": "value2"} with string values only.
static std::map<std::string, std::string> parse_simple_json(const std::string& text) {
    std::map<std::string, std::string> result;
    size_t pos = 0;

    // Skip past opening brace
    pos = text.find('{');
    if (pos == std::string::npos) return result;
    pos++;

    while (pos < text.size()) {
        // Skip whitespace and commas
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' ||
                                      text[pos] == '\n' || text[pos] == '\r' ||
                                      text[pos] == ','))
            pos++;
        if (pos >= text.size() || text[pos] == '}') break;

        // Parse key
        if (text[pos] != '"') break;
        pos++;
        size_t key_start = pos;
        size_t key_end = text.find('"', pos);
        if (key_end == std::string::npos) break;
        std::string key = text.substr(key_start, key_end - key_start);
        pos = key_end + 1;

        // Skip colon
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == ':'))
            pos++;

        // Parse value (string or number)
        if (pos >= text.size()) break;

        std::string value;
        if (text[pos] == '"') {
            // String value
            pos++;
            size_t val_start = pos;
            size_t val_end = text.find('"', pos);
            if (val_end == std::string::npos) break;
            value = text.substr(val_start, val_end - val_start);
            pos = val_end + 1;
        } else {
            // Number or bare word
            size_t val_end = text.find_first_of(",}\n\r", pos);
            value = trim(text.substr(pos, val_end - pos));
            pos = val_end;
        }

        result[key] = value;
    }

    return result;
}

// Read a binary file into a vector
static std::vector<uint8_t> read_binary_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    if (size <= 0) return {};
    f.seekg(0);
    std::vector<uint8_t> data((size_t)size);
    f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)size);
    return data;
}

// ── Voice pack directory loader ────────────────────────────────────────
// Loads a voice pack from a directory containing decoder.onnx,
// speaker_emb.bin, and config.json.

static bool load_pack_from_dir(VoicePack& pack) {
    fs::path dir(pack.path);
    if (!fs::is_directory(dir)) {
        CODEC_LOG("Not a directory: %s", pack.path.c_str());
        return false;
    }

    fs::path config_path = dir / "config.json";
    fs::path decoder_path = dir / "decoder.onnx";
    fs::path speaker_emb_path = dir / "speaker_emb.bin";

    // Parse config.json
    if (fs::exists(config_path)) {
        auto cfg_bytes = read_binary_file(config_path);
        if (!cfg_bytes.empty()) {
            std::string cfg_text(cfg_bytes.begin(), cfg_bytes.end());
            auto cfg = parse_simple_json(cfg_text);
            if (cfg.count("name"))       pack.name = cfg["name"];
            if (cfg.count("speaker"))    pack.speaker_name = cfg["speaker"];
            if (cfg.count("language") || cfg.count("lang"))
                pack.language = cfg.count("language") ? cfg["language"] : cfg["lang"];
        }
    }

    // Load speaker embedding
    if (fs::exists(speaker_emb_path)) {
        auto emb_bytes = read_binary_file(speaker_emb_path);
        if (emb_bytes.size() >= (size_t)kSpeakerEmbDim * sizeof(float)) {
            pack.speaker_emb.resize(kSpeakerEmbDim);
            std::memcpy(pack.speaker_emb.data(), emb_bytes.data(),
                        kSpeakerEmbDim * sizeof(float));
        } else {
            CODEC_LOG("speaker_emb.bin too small (%zu bytes, need %d)",
                      emb_bytes.size(), kSpeakerEmbDim * (int)sizeof(float));
        }
    }

    // If speaker embedding is missing, use zeros (silent but non-crashing)
    if (pack.speaker_emb.empty()) {
        pack.speaker_emb.resize(kSpeakerEmbDim, 0.0f);
        CODEC_LOG("WARNING: no speaker_emb.bin in '%s' — using zeros", pack.path.c_str());
    }

    // Load decoder model
    if (fs::exists(decoder_path)) {
        pack.decoder = std::make_unique<CodecDecoder>();
        if (!pack.decoder->load(decoder_path.string())) {
            CODEC_LOG("Failed to load decoder from '%s'", decoder_path.string().c_str());
            pack.decoder.reset();
            return false;
        }
    } else {
        CODEC_LOG("decoder.onnx not found in '%s'", pack.path.c_str());
        return false;
    }

    pack.loaded = true;
    pack.mtime = fs::last_write_time(decoder_path);

    CODEC_LOG("Loaded voice pack '%s' (speaker=%s, lang=%s)",
              pack.name.c_str(), pack.speaker_name.c_str(), pack.language.c_str());
    return true;
}

// ── Voice pack .zip loader ─────────────────────────────────────────────
// Only compiled when USE_MINIZ is defined.

#ifdef USE_MINIZ

// miniz is a single-file public-domain library. Include it here.
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_STDIO
#define MINIZ_NO_TIME
#include "miniz.h"

static bool load_pack_from_zip(VoicePack& pack) {
    // Read the .voice file into memory
    auto zip_data = read_binary_file(pack.path);
    if (zip_data.empty()) {
        CODEC_LOG("Cannot read .voice file: %s", pack.path.c_str());
        return false;
    }

    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    if (!mz_zip_reader_init_mem(&zip, zip_data.data(), zip_data.size(), 0)) {
        CODEC_LOG("Failed to open zip archive: %s", pack.path.c_str());
        return false;
    }

    // Extract config.json
    size_t config_size = 0;
    const char* config_data = (const char*)mz_zip_reader_extract_file_to_heap(
        &zip, "config.json", &config_size, 0);
    if (config_data && config_size > 0) {
        std::string cfg_text(config_data, config_size);
        auto cfg = parse_simple_json(cfg_text);
        if (cfg.count("name"))       pack.name = cfg["name"];
        if (cfg.count("speaker"))    pack.speaker_name = cfg["speaker"];
        if (cfg.count("language") || cfg.count("lang"))
            pack.language = cfg.count("language") ? cfg["language"] : cfg["lang"];
        mz_free((void*)config_data);
    }

    // Extract speaker_emb.bin
    size_t emb_size = 0;
    float* emb_data = (float*)mz_zip_reader_extract_file_to_heap(
        &zip, "speaker_emb.bin", &emb_size, 0);
    if (emb_data && emb_size >= (size_t)kSpeakerEmbDim * sizeof(float)) {
        pack.speaker_emb.assign(emb_data, emb_data + kSpeakerEmbDim);
        mz_free(emb_data);
    }
    if (pack.speaker_emb.empty()) {
        pack.speaker_emb.resize(kSpeakerEmbDim, 0.0f);
        CODEC_LOG("WARNING: no speaker_emb.bin in zip '%s' — using zeros", pack.path.c_str());
    }

    // Extract decoder.onnx to a temp file (ONNX Runtime needs a file path)
    size_t onnx_size = 0;
    void* onnx_data = mz_zip_reader_extract_file_to_heap(
        &zip, "decoder.onnx", &onnx_size, 0);
    if (onnx_data && onnx_size > 0) {
        // Write to temp file
        std::string tmp_path = fs::temp_directory_path() / ("codec_decoder_" + pack.name + ".onnx");
        {
            std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
            f.write((const char*)onnx_data, (std::streamsize)onnx_size);
        }
        mz_free(onnx_data);

        // Load the decoder
        pack.decoder = std::make_unique<CodecDecoder>();
        if (!pack.decoder->load(tmp_path)) {
            CODEC_LOG("Failed to load decoder from zip '%s'", pack.path.c_str());
            pack.decoder.reset();
            mz_zip_reader_end(&zip);
            // Clean up temp file
            std::error_code ec;
            fs::remove(tmp_path, ec);
            return false;
        }

        // Keep the temp file for the session (or clean up later)
        // For now we leave it; a more sophisticated implementation would
        // use memory-backed ONNX session loading when available.
        pack.path = tmp_path; // update path to point to extracted onnx
    } else {
        CODEC_LOG("decoder.onnx not found in zip '%s'", pack.path.c_str());
        mz_zip_reader_end(&zip);
        return false;
    }

    mz_zip_reader_end(&zip);
    pack.loaded = true;
    CODEC_LOG("Loaded voice pack from zip '%s' (speaker=%s, lang=%s)",
              pack.name.c_str(), pack.speaker_name.c_str(), pack.language.c_str());
    return true;
}

#else // !USE_MINIZ

static bool load_pack_from_zip(VoicePack& pack) {
    (void)pack;
    CODEC_LOG("Zip support disabled (USE_MINIZ not defined) — cannot load '%s'", pack.path.c_str());
    return false;
}

#endif // USE_MINIZ

// ── Text→codec tokens placeholder ──────────────────────────────────────
// Real adapter will come later (small LM or lookup table).
// For now: generates deterministic-ish random tokens derived from text hash.

static std::vector<int32_t> text_to_codec_tokens(const std::string& text, int vocab_size,
                                                  std::mt19937& rng) {
    if (text.empty()) return {};

    // Simple hash-based seed from text content
    std::hash<std::string> hasher;
    size_t seed = hasher(text);
    std::mt19937 local_rng((unsigned)seed);

    // Estimate token count: ~10 codec frames per character, at least 1
    int n_tokens = std::max(1, (int)text.size() / 2);

    // Generate [8 × n_tokens] random codec tokens
    std::vector<int32_t> tokens((size_t)(kNumCodebooks * n_tokens));
    std::uniform_int_distribution<int32_t> dist(0, vocab_size - 1);
    for (auto& t : tokens) t = dist(local_rng);

    return tokens;
}

// ── CodecTts implementation ────────────────────────────────────────────

CodecTts::CodecTts() : impl_(std::make_unique<CodecTtsImpl>()) {}
CodecTts::~CodecTts() = default;

void CodecTts::set_voice_packs_dir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->voice_packs_dir = dir;
}

bool CodecTts::scan_voice_packs() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::string dir = impl_->voice_packs_dir;
    if (dir.empty()) dir = env_or("VOICE_PACKS_DIR", home_dir() + "/voice-packs");
    impl_->voice_packs_dir = dir;

    if (!fs::is_directory(dir)) {
        // Try creating it
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            CODEC_LOG("voice packs directory '%s' does not exist and cannot be created: %s",
                      dir.c_str(), ec.message().c_str());
            return false;
        }
    }

    CODEC_LOG("Scanning voice packs in '%s'", dir.c_str());

    bool found_any = false;

    // Scan directories (unpacked packs)
    for (auto& entry : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_directory()) continue;

        fs::path pack_dir = entry.path();
        std::string name = pack_dir.filename().string();

        // Skip hidden directories
        if (name.size() > 0 && name[0] == '.') continue;

        // Check for existing pack
        auto it = impl_->packs.find(name);
        if (it != impl_->packs.end()) {
            // Hot-reload check: compare mtime of decoder.onnx
            fs::path decoder_path = pack_dir / "decoder.onnx";
            if (fs::exists(decoder_path)) {
                auto mtime = fs::last_write_time(decoder_path);
                if (mtime > it->second.mtime) {
                    CODEC_LOG("Hot-reload: '%s' changed, reloading", name.c_str());
                    VoicePack pack;
                    pack.name = name;
                    pack.path = pack_dir.string();
                    if (load_pack_from_dir(pack)) {
                        it->second = std::move(pack);
                    }
                }
            }
            found_any = true;
            continue;
        }

        // New pack
        VoicePack pack;
        pack.name = name;
        pack.path = pack_dir.string();
        if (load_pack_from_dir(pack)) {
            impl_->packs[name] = std::move(pack);
            found_any = true;
        }
    }

    // Scan .voice zip files
    for (auto& entry : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;

        fs::path file_path = entry.path();
        std::string ext = file_path.extension().string();
        if (ext != ".voice" && ext != ".voicepack") continue;

        std::string name = file_path.stem().string();

        auto it = impl_->packs.find(name);
        if (it != impl_->packs.end()) {
            // Check mtime for hot-reload
            auto mtime = fs::last_write_time(file_path);
            if (mtime > it->second.mtime) {
                CODEC_LOG("Hot-reload: '%s.voice' changed, reloading", name.c_str());
                VoicePack pack;
                pack.name = name;
                pack.path = file_path.string();
                if (load_pack_from_zip(pack)) {
                    it->second = std::move(pack);
                }
            }
            found_any = true;
            continue;
        }

        // New voice pack from zip
        VoicePack pack;
        pack.name = name;
        pack.path = file_path.string();
        if (load_pack_from_zip(pack)) {
            impl_->packs[name] = std::move(pack);
            found_any = true;
        }
    }

    return found_any;
}

std::vector<VoicePackInfo> CodecTts::list_voice_packs() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<VoicePackInfo> result;
    result.reserve(impl_->packs.size());
    for (auto& [name, pack] : impl_->packs) {
        result.push_back({
            pack.name,
            pack.speaker_name,
            pack.language,
            pack.sample_rate,
            pack.path
        });
    }
    return result;
}

bool CodecTts::load_voice_pack(const std::string& name_or_path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Already loaded?
    if (impl_->packs.count(name_or_path) && impl_->packs[name_or_path].loaded) {
        return true;
    }

    VoicePack pack;
    pack.name = name_or_path;

    // Check if it's a path to a directory
    if (fs::is_directory(name_or_path)) {
        pack.path = name_or_path;
        pack.name = fs::path(name_or_path).filename().string();
        return load_pack_from_dir(pack);
    }

    // Check if it's a .voice file
    if (fs::is_regular_file(name_or_path)) {
        pack.path = name_or_path;
        pack.name = fs::path(name_or_path).stem().string();
        return load_pack_from_zip(pack);
    }

    // Check in the voice packs directory
    std::string dir = impl_->voice_packs_dir;
    if (dir.empty()) dir = env_or("VOICE_PACKS_DIR", home_dir() + "/voice-packs");

    fs::path dir_path(dir);

    // Try as directory
    fs::path pack_dir = dir_path / name_or_path;
    if (fs::is_directory(pack_dir)) {
        pack.path = pack_dir.string();
        pack.name = name_or_path;
        bool ok = load_pack_from_dir(pack);
        if (ok) impl_->packs[name_or_path] = std::move(pack);
        return ok;
    }

    // Try as .voice file
    fs::path voice_file = dir_path / (name_or_path + ".voice");
    if (fs::is_regular_file(voice_file)) {
        pack.path = voice_file.string();
        pack.name = name_or_path;
        bool ok = load_pack_from_zip(pack);
        if (ok) impl_->packs[name_or_path] = std::move(pack);
        return ok;
    }

    CODEC_LOG("Voice pack '%s' not found in '%s'", name_or_path.c_str(), dir.c_str());
    return false;
}

void CodecTts::unload_voice_pack(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->packs.find(name);
    if (it != impl_->packs.end()) {
        CODEC_LOG("Unloaded voice pack '%s'", name.c_str());
        impl_->packs.erase(it);
    }
}

bool CodecTts::has_voice(const std::string& name) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->packs.find(name);
    return it != impl_->packs.end() && it->second.loaded;
}

std::string CodecTts::synthesize(const std::string& text, const std::string& voice) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto it = impl_->packs.find(voice);
    if (it == impl_->packs.end() || !it->second.loaded) {
        CODEC_LOG("Voice pack '%s' not loaded", voice.c_str());
        return {};
    }

    VoicePack& pack = it->second;

    if (!pack.decoder || !pack.decoder->is_loaded()) {
        CODEC_LOG("Voice pack '%s' has no loaded decoder", voice.c_str());
        return {};
    }

    // Convert text to codec tokens (placeholder — random for now)
    auto tokens = text_to_codec_tokens(text, kDefaultVocabSize, impl_->rng);
    if (tokens.empty()) {
        CODEC_LOG("No tokens generated for text");
        return {};
    }

    int n_tokens = (int)tokens.size() / kNumCodebooks;

    // Decode to PCM
    auto pcm_float = pack.decoder->decode(tokens.data(), n_tokens, pack.speaker_emb.data());
    if (pcm_float.empty()) {
        // ONNX Runtime not available — return gracefully
        CODEC_LOG("decode() returned empty — ONNX Runtime unavailable or model error");
        return {};
    }

    // Convert to WAV
    return CodecDecoder::pcm_to_wav(pcm_float.data(), pcm_float.size(), kSampleRate);
}

} // namespace jarvis
