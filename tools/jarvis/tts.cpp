#include "tts.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace jarvis {
namespace {

std::string env_or(const char* key, const std::string& def) {
    const char* v = getenv(key);
    return (v && *v) ? v : def;
}

std::string home_dir() {
    const char* h = getenv("HOME");
    return h ? h : "/tmp";
}

// Runs `piper --model <model_path> --output-raw` with `text` piped to
// stdin, returns raw PCM (s16le, mono, 22050 Hz) from stdout. Empty on
// any failure — bidirectional pipe via fork/exec, not popen (popen is
// one-directional only).
std::string run_piper(const std::string& piper_bin, const std::string& model_path, const std::string& text) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) return "";

    pid_t pid = fork();
    if (pid < 0) return "";

    if (pid == 0) {
        // Child: stdin <- in_pipe[0], stdout -> out_pipe[1]
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp(piper_bin.c_str(), piper_bin.c_str(), "--model", model_path.c_str(), "--output-raw", (char*)nullptr);
        _exit(127); // exec failed
    }

    // Parent
    close(in_pipe[0]);
    close(out_pipe[1]);

    // Write text then close stdin so piper knows input is complete.
    // Piper's own text buffering is small enough that we don't need a
    // separate writer thread for jarvis-scale utterances.
    size_t written = 0;
    while (written < text.size()) {
        ssize_t n = write(in_pipe[1], text.data() + written, text.size() - written);
        if (n <= 0) break;
        written += (size_t)n;
    }
    close(in_pipe[1]);

    std::string pcm;
    char buf[65536];
    ssize_t n;
    while ((n = read(out_pipe[0], buf, sizeof(buf))) > 0) pcm.append(buf, (size_t)n);
    close(out_pipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return "";
    return pcm;
}

// Wraps raw s16le mono PCM as a complete WAV file (44-byte header).
std::string wrap_wav(const std::string& pcm, int sample_rate, int channels = 1, int bits_per_sample = 16) {
    uint32_t data_size = (uint32_t)pcm.size();
    uint32_t byte_rate = (uint32_t)(sample_rate * channels * bits_per_sample / 8);
    uint16_t block_align = (uint16_t)(channels * bits_per_sample / 8);
    uint32_t riff_size = 36 + data_size;

    std::string out;
    out.reserve(44 + pcm.size());
    auto put_u32 = [&](uint32_t v) { out.append((const char*)&v, 4); };
    auto put_u16 = [&](uint16_t v) { out.append((const char*)&v, 2); };

    out += "RIFF"; put_u32(riff_size); out += "WAVE";
    out += "fmt "; put_u32(16); put_u16(1) /*PCM*/; put_u16((uint16_t)channels);
    put_u32((uint32_t)sample_rate); put_u32(byte_rate); put_u16(block_align); put_u16((uint16_t)bits_per_sample);
    out += "data"; put_u32(data_size);
    out += pcm;
    return out;
}

} // namespace

std::string synthesize_speech(const std::string& text, const std::string& voice) {
    std::string voices_dir = env_or("PIPER_VOICES_DIR", home_dir() + "/piper-voices");
    std::string jarvis_venv = env_or("JARVIS_VENV", home_dir() + "/jarvis-env");

    std::filesystem::path model_path = std::filesystem::path(voices_dir) / (voice + ".onnx");
    if (!std::filesystem::exists(model_path)) return "";

    std::filesystem::path piper_bin = std::filesystem::path(jarvis_venv) / "bin" / "piper";
    if (!std::filesystem::exists(piper_bin)) piper_bin = "piper"; // fall back to PATH

    std::string pcm = run_piper(piper_bin.string(), model_path.string(), text);
    if (pcm.empty()) return "";
    return wrap_wav(pcm, 22050);
}

} // namespace jarvis
