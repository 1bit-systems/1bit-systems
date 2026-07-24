// tts.h — text-to-speech via the Piper CLI.
// C++ port of jarvis/tts.py (recovered from git history at c252174aa~1).
// Piper itself is a compiled C++ binary (not Python) — shelling out to it
// stays within this project's zero-Python-at-runtime mandate, same as
// this file's own use of ffmpeg/aplay elsewhere in the jarvis port.
//
// Scope note: the original also had a custom RVQ-VAE voice-cloning engine
// (jarvis/voice/engine.py+codec.py, PyTorch) tried before this Piper
// fallback. That's explicitly out of scope for this port (see the plan
// file) — Piper's stock voice is what this file provides.
#pragma once

#include <string>

namespace jarvis {

// Synthesizes `text` with the named Piper voice (a "<voice>.onnx" model
// under $PIPER_VOICES_DIR, default ~/piper-voices) and returns a complete
// WAV file (mono, 16-bit, 22050 Hz — Piper's raw output rate, matches the
// original's hardcoded assumption for this specific voice model). Returns
// an empty string if the voice model isn't found or piper fails/times out.
std::string synthesize_speech(const std::string& text, const std::string& voice = "en_US-lessac-medium");

} // namespace jarvis
