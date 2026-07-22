/// ZINC C++ — GPU inference engine (Vulkan backend)
/// Port of ZINC (engine/gpu/) from Zig to C++.
///
/// One binary: loads GGUF models, dispatches compute shaders on Vulkan,
/// serves an OpenAI-compatible HTTP API.
///
/// Build: cmake -B build && cmake --build build
/// Run:   ./build/zinc_cpp --model model.gguf --port 8080
#include "vulkan_wrapper.h"
#include "compute_engine.h"
#include "model_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <getopt.h>
#include <signal.h>
#include <atomic>

// ═══════════════════════════════════════════════════════════════════
//  Signal handler
// ═══════════════════════════════════════════════════════════════════
static std::atomic<bool> keep_running{true};
static void handle_sigint(int) { keep_running = false; }

// ═══════════════════════════════════════════════════════════════════
// ─── InferenceEngine is declared in compute_engine.h
//     and implemented in compute_engine.cpp
//
};

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);

    std::string model_path;
    int port = 8080;
    int device_idx = -1;

    // Parse args
    static struct option opts[] = {
        {"model",  required_argument, nullptr, 'm'},
        {"port",   required_argument, nullptr, 'p'},
        {"device", required_argument, nullptr, 'd'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:d:", opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 'd': device_idx = atoi(optarg); break;
        }
    }

    printf("\n╔═══════════════════════════════════════════╗\n");
    printf("║    ZINC C++ — GPU Inference Engine       ║\n");
    printf("║    Vulkan backend, GGUF models           ║\n");
    printf("╚═══════════════════════════════════════════╝\n\n");

    if (model_path.empty()) {
        fprintf(stderr, "Usage: %s --model model.gguf [--port 8080] [--device 0]\n", argv[0]);
        return 1;
    }

    // ── Init Vulkan ──
    ZincEngine engine;
    try {
        engine.init("shaders", device_idx);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to init Vulkan: %s\n", e.what());
        return 1;
    }

    // ── Load model ──
    printf("\n── Loading Model ──\n");
    ModelLoader loader(engine.device(), engine.queue(),
                        engine.queue_family(), *engine.cmd_pool());
    ModelGPU model;
    if (!loader.load(model_path, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Init compute engine ──
    printf("\n── Init Compute ──\n");
    ComputeEngine compute(engine.device(), engine.queue(),
                          engine.queue_family(), *engine.cmd_pool(),
                          *engine.pipeline_cache());
    InferenceEngine infer;
    infer.init(compute, model);

    // ── Quick benchmark ──
    printf("\n── Benchmark (10 tokens) ──\n");
    infer.reset();
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 0;  // BOS
    for (int i = 0; i < 10; i++) {
        tok = infer.generate(tok);
        if (tok < 0) break;
    }
    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("  %.1f ms/tok (%.1f tok/s)\n", ms / 10.0f, 10.0f / (ms / 1000.0f));

    printf("\n── Ready. Press Ctrl+C to stop. ──\n\n");

    // ── Main loop (idle, shuts down on signal) ──
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\nShutting down...\n");
    return 0;
}
