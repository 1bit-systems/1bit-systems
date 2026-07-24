/** lora_merge.cpp — Permanently merge LoRA adapters into a GGUF/1BP model.
 *
 *  Usage:
 *    ./build/lora_merge --model base.gguf --lora style.gguf:1.0 --output merged.gguf
 *    ./build/lora_merge --model base.1bp --lora instruct.gguf:0.8 --output merged.1bp
 *
 *  Multiple LoRAs can be specified with weights:
 *    --lora lora1.gguf:1.0 --lora lora2.gguf:0.5
 */

#include "lora.h"
#include "backend_manager.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

struct LoraSpec {
    std::string path;
    float weight;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --model PATH     Base model file (.gguf or .1bp)\n"
        "  --lora P:W       LoRA adapter path and weight (can specify multiple)\n"
        "  --output PATH    Output merged model path\n"
        "  -h, --help       Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --model base.gguf --lora style.gguf:1.0 --output merged.gguf\n"
        "  %s --model base.1bp --lora instruct.gguf:0.8 --lora code.gguf:0.3 --output merged.1bp\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string output_path;
    std::vector<LoraSpec> loras;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "--lora") {
            if (i + 1 < argc) {
                std::string spec = argv[++i];
                auto colon = spec.find(':');
                LoraSpec ls;
                if (colon != std::string::npos) {
                    ls.path = spec.substr(0, colon);
                    ls.weight = std::stof(spec.substr(colon + 1));
                } else {
                    ls.path = spec;
                    ls.weight = 1.0f;
                }
                loras.push_back(ls);
            }
        } else if (arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (model_path.empty() || loras.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    printf("lora_merge: merging %zu LoRA(s) into %s\n", loras.size(), model_path.c_str());
    printf("  Output: %s\n", output_path.c_str());
    
    auto t0 = std::chrono::steady_clock::now();
    
    std::vector<std::string> lora_paths;
    for (auto& ls : loras) lora_paths.push_back(ls.path);
    
    bool ok = lora_merge_gguf(model_path, lora_paths, output_path);
    
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    
    if (ok) {
        printf("lora_merge: done in %.2f seconds\n", ms / 1000.0);
    } else {
        fprintf(stderr, "lora_merge: failed\n");
        return 1;
    }
    
    return 0;
}
