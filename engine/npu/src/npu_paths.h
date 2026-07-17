// npu_paths.h — Centralized NPU path resolution with env var overrides.
//
// All NPU engine/tool sources should use these functions instead of
// hardcoding /home/bcloud paths. Each function checks an env var first,
// then falls back to a $HOME-based default.
//
// Usage:
//   #include "npu_paths.h"
//   const char* model = npu_model_path();  // $NPU_MODEL_PATH or ~/.../model.q4nx
//
// To override without editing code:
//   export NPU_MODEL_PATH=/my/models/Qwen3-0.6B/model.q4nx
//   export NPU_XCLBIN_DIR=/my/xclbins
//   export NPU_INSTS_DIR=/my/insts

#pragma once

#include <cstdlib>
#include <string>

inline const char* env_or(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return v ? v : fallback;
}

// ── Model paths ────────────────────────────────────────────────
// $NPU_MODEL_PATH — full path to model.q4nx
// $NPU_MODEL_DIR  — directory containing model files (used with model name)
// $NPU_TOKENIZER_PATH — full path to tokenizer.json

inline const char* npu_model_path() {
    return env_or("NPU_MODEL_PATH", nullptr);  // no default — must be set or passed via CLI
}

inline std::string npu_model_dir() {
    const char* e = getenv("NPU_MODEL_DIR");
    if (e) return e;
    const char* home = getenv("HOME");
    return std::string(home ? home : ".") + "/.config/flm/models";
}

inline const char* npu_tokenizer_path() {
    return env_or("NPU_TOKENIZER_PATH", nullptr);
}

// ── XCLBIN paths ───────────────────────────────────────────────
// $NPU_XCLBIN_DIR — directory containing compiled .xclbin files

inline std::string npu_xclbin_dir() {
    const char* e = getenv("NPU_XCLBIN_DIR");
    if (e) return e;
    return "engine/npu/xclbins";  // relative to repo root
}

// ── Instruction file paths ─────────────────────────────────────
// $NPU_INSTS_DIR — directory containing compiled .insts / .txt instruction files

inline std::string npu_insts_dir() {
    const char* e = getenv("NPU_INSTS_DIR");
    if (e) return e;
    return npu_xclbin_dir();  // default: same as xclbin dir
}

// ── Engine binary path ─────────────────────────────────────────
// $NPU_ENGINE_BIN — path to the NPU engine executable (for subprocess spawn)

inline std::string npu_engine_bin() {
    const char* e = getenv("NPU_ENGINE_BIN");
    if (e) return e;
    return "./npu_engine_universal";  // relative to cwd
}

// ── AIE toolchain paths (torch2aie) ────────────────────────────
// $AIE_TOOLS_DIR — root of the torch2aie installation

inline std::string aie_tools_dir() {
    const char* e = getenv("AIE_TOOLS_DIR");
    if (e) return e;
    const char* home = getenv("HOME");
    return std::string(home ? home : ".") + "/torch2aie";
}
