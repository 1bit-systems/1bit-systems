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
    auto engine = std::make_unique<ZincEngine>();
    try {
    #ifdef ZINC_SHADER_DIR
    std::string shader_dir = ZINC_SHADER_DIR;
#else
    std::string shader_dir = "shaders";
#endif
    engine->init(shader_dir, device_idx);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to init Vulkan: %s\n", e.what());
        return 1;
    }

    // ── Load model ──
    printf("\n── Loading Model ──\n");
    auto loader = std::make_unique<ModelLoader>(engine->device(), engine->queue(),
                        engine->queue_family(), *engine->cmd_pool());
    auto model = std::make_unique<ModelGPU>();
    if (!loader->load(model_path, *model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Init compute engine ──
    printf("\n── Init Compute ──\n");
    auto compute = std::make_unique<ComputeEngine>(engine->device(), engine->queue(),
                          engine->queue_family(), *engine->cmd_pool(),
                          *engine->pipeline_cache());
    auto infer = std::make_unique<InferenceEngine>();
    infer->init(*compute, *model);

    // ── Quick benchmark ──
    printf("\n── Benchmark (10 tokens) ──\n");
    compute->reset_descriptors();
    infer->reset();
    auto t0 = std::chrono::high_resolution_clock::now();
    int tok = 1;
    int last_tok = 0;
    for (int i = 0; i < 10; i++) {
        printf("  Token %d...\n", i);
        tok = infer->generate(tok);
        if (tok < 0) { printf("  Failed at token %d\n", i); break; }
        printf("  -> %d\n", tok);
        last_tok = tok;
    }
    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    printf("  %.1f ms/tok (%.1f tok/s)\n", ms / 10.0f, 10.0f / (ms / 1000.0f));

    printf("\n── Ready. Press Ctrl+C to stop. ──\n\n");

    // ── Main loop (idle, shuts down on signal) ──
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup: unique_ptrs destroy in reverse declaration order automatically
    // engine is destroyed LAST, so compute's refs to engine's cmd_pool/pipelines stay valid (fix #779)
    infer.reset();
    compute.reset();
    model.reset();
    loader.reset();
    engine.reset();

    printf("\nShutting down...\n");
    return 0;
}
