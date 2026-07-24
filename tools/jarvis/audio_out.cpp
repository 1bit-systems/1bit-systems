#include "audio_out.h"

#include <cctype>
#include <cstdio>
#include <regex>
#include <sstream>
#include <thread>

namespace jarvis {
namespace {

const char* kOnboardHints[] = {"hd-audio generic", "hdmi"};

std::string to_lower(const std::string& s) {
    std::string out = s;
    for (auto& c : out) c = (char)std::tolower((unsigned char)c);
    return out;
}

} // namespace

std::vector<PlaybackDevice> list_playback_devices() {
    std::vector<PlaybackDevice> devices;
    FILE* pipe = popen("aplay -l 2>/dev/null", "r");
    if (!pipe) return devices;

    std::string output;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) output.append(buf, n);
    pclose(pipe);

    // "card (\d+): \S+ \[(.*?)\], device (\d+): (.*?) \["
    static const std::regex line_re(R"(card (\d+): \S+ \[(.*?)\], device (\d+): (.*?) \[)");
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        std::smatch m;
        if (!std::regex_search(line, m, line_re)) continue;
        PlaybackDevice d;
        d.card = std::stoi(m[1].str());
        d.name = m[2].str();
        d.device = std::stoi(m[3].str());
        d.device_name = m[4].str();
        std::string name_lower = to_lower(d.name);
        d.is_onboard = false;
        for (auto* hint : kOnboardHints) {
            if (name_lower.find(hint) != std::string::npos) { d.is_onboard = true; break; }
        }
        d.alsa_id = "plughw:CARD=" + std::to_string(d.card) + ",DEV=" + std::to_string(d.device);
        devices.push_back(std::move(d));
    }
    return devices;
}

bool find_external_speaker(PlaybackDevice* out) {
    for (auto& d : list_playback_devices()) {
        if (!d.is_onboard) {
            if (out) *out = d;
            return true;
        }
    }
    return false;
}

namespace {
void play_now(std::string wav_bytes, std::string alsa_id) {
    std::string cmd = "aplay -q -D " + alsa_id + " - >/dev/null 2>&1";
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) return;
    fwrite(wav_bytes.data(), 1, wav_bytes.size(), pipe);
    pclose(pipe);
}
} // namespace

bool play_wav_local(const std::string& wav_bytes, const PlaybackDevice* device, bool blocking) {
    PlaybackDevice dev;
    if (device) {
        dev = *device;
    } else if (!find_external_speaker(&dev)) {
        return false;
    }

    if (blocking) {
        play_now(wav_bytes, dev.alsa_id);
    } else {
        std::thread(play_now, wav_bytes, dev.alsa_id).detach();
    }
    return true;
}

} // namespace jarvis
