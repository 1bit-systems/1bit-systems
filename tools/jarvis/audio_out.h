// audio_out.h — local speaker playback for jarvis's TTS output.
// C++ port of jarvis/audio_out.py (recovered from git history at
// c252174aa~1). Lets the server mirror synthesized speech out through a
// physical speaker attached to the box jarvis runs on, independent of
// whatever the calling client does with the returned audio bytes.
// Entirely dormant (no-op, no error) until a non-onboard playback device
// is actually present.
#pragma once

#include <string>
#include <vector>

namespace jarvis {

struct PlaybackDevice {
    int card;
    int device;
    std::string name;
    std::string device_name;
    bool is_onboard;
    std::string alsa_id; // "plughw:CARD=<card>,DEV=<device>"
};

// Parses `aplay -l` output. Empty vector if aplay isn't available.
std::vector<PlaybackDevice> list_playback_devices();

// First non-onboard playback device (a plugged-in USB speaker), if any.
bool find_external_speaker(PlaybackDevice* out);

// Plays WAV bytes on a local ALSA device. No-op (returns false) if no
// external speaker is available. Fire-and-forget by default — runs on a
// detached thread so it never adds latency to the HTTP response carrying
// the same audio back to the caller.
bool play_wav_local(const std::string& wav_bytes, const PlaybackDevice* device = nullptr, bool blocking = false);

} // namespace jarvis
