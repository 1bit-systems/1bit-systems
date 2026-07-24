// zaya1_vl_demo.cpp — ZAYA1-VL-8B vision-language inference demo.
//
// Demonstrates the full pipeline:
//   1. Load vision tower (ViT-L/14 style encoder) from separate GGUF
//   2. Load image, preprocess, run ViT encoder
//   3. Project vision embeddings to text hidden dim
//   4. Feed into Zaya1-8B text decoder via forward_embed()
//   5. Generate text response
//
// Build:
//   cmake --build . --target zaya1_vl_demo -j8
//
// Run:
//   ./build/zaya1_vl_demo <text_model.gguf|1bp> <vision_mmproj.gguf> <image> [prompt]
//
// The vision tower must be a GGUF file following llama.cpp's clip conventions
// (v.patch_embd.weight, v.blk.N.*, v.post_ln.*, mm.0.weight, mm.2.weight).
//
// Architecture assumptions for ZAYA1-VL-8B:
//   Vision encoder: ViT-L/14 (hidden=1024, layers=24, heads=16, patch=14)
//   Projector: MLP (1024 -> 2048 = text hidden)
//   Text decoder: Zaya1-8B MoE (hidden=2048, 40 layers, 16 experts)
//   Image size: 224x224 (16x16 patches = 256 -> 64 merged with CLS removed)

#include "vision_encoder.h"
#include "backend.h"
#include "backend_manager.h"
#include "model_discovery.h"
#include "simple_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <text_model.gguf|1bp> <vision_mmproj.gguf> <image.jpg> [prompt]\n", argv[0]);
        fprintf(stderr, "\n  text_model:    Path to text decoder GGUF or 1BP file\n");
        fprintf(stderr, "  vision_mmproj: Path to vision projector GGUF (from llama.cpp clip)\n");
        fprintf(stderr, "  image:         Path to image file (JPEG/PNG)\n");
        fprintf(stderr, "  prompt:        Text prompt (default: 'Describe this image')\n");
        return 1;
    }
    std::string text_path = argv[1];
    std::string vision_path = argv[2];
    std::string image_path = argv[3];
    std::string prompt = argc > 4 ? argv[4] : "Describe this image in one sentence.";

    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr,   "║     ZAYA1-VL-8B Vision-Language Demo    ║\n");
    fprintf(stderr,   "╚══════════════════════════════════════════╝\n\n");

    // ═══ Step 1: Load vision encoder ═══
    fprintf(stderr, "1. Loading vision encoder from: %s\n", vision_path.c_str());
    VisionWeights vision;
    VitConfig vcfg = VitConfig::clip_vit_l(); // ViT-L/14 default for ZAYA1-VL
    if (!vision.load_from_gguf(vision_path, &vcfg)) {
        fprintf(stderr, "FAIL: could not load vision encoder\n");
        return 1;
    }
    fprintf(stderr, "   Vision encoder: H=%d L=%d NH=%d P=%d projector=%d\n",
            vision.config.hidden_size, vision.config.num_layers,
            vision.config.num_heads, vision.config.patch_size,
            (int)vision.proj_config.type);

    // ═══ Step 2: Load text decoder ═══
    fprintf(stderr, "\n2. Loading text decoder from: %s\n", text_path.c_str());
    
    bool is_1bp = text_path.size() >= 4 && text_path.substr(text_path.size() - 4) == ".1bp";
    
    // Load text decoder directly via GenericBackend (CPU reference, works with any GGUF)
    // For production, use BackendManager for GPU acceleration.
    ModelConfig cfg;
    if (!read_gguf_header(text_path, cfg)) {
        // Fallback: manual config for text model
        cfg.model_path = text_path;
        cfg.model_name = "bonsai-27b";
        cfg.format = ModelFormat::GGUF;
        cfg.hidden = cfg.hidden_size = 5120;
        cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = 40;
        cfg.head_dim = 128;
        cfg.n_layers = cfg.num_layers = 36;
        cfg.n_ff = cfg.intermediate_size = 13696;
        cfg.vocab = cfg.vocab_size = 152064;
    }

    // Use the generic backend (CPU reference) — works with any GGUF model
    Backend* be = create_generic_backend();
    if (!be || !be->init(cfg, text_path)) {
        fprintf(stderr, "FAIL: could not load text decoder from %s\n", text_path.c_str());
        return 1;
    }
    fprintf(stderr, "   Text decoder ready: hidden=%d layers=%d\n",
            cfg.hidden_size, cfg.num_layers);

    // Verify dimension compatibility
    int text_hidden = vision.proj_config.text_hidden;
    if (text_hidden != cfg.hidden_size) {
        fprintf(stderr, "   WARNING: vision projector outputs %d dims but text expects %d\n",
                text_hidden, cfg.hidden_size);
        // If mismatch, we can still try — projector might need override
    }

    // ═══ Step 3: Load and preprocess image ═══
    fprintf(stderr, "\n3. Loading image: %s\n", image_path.c_str());
    
    int img_size = 224; // Standard ViT input
    if (vision.config.patch_size > 0) {
        // Use multiple of patch size
        img_size = 224;
    }

    const float* mean = vision.mean[0] != 0 ? vision.mean : (vision.config.mean);
    const float* std  = vision.std[0]  != 0 ? vision.std  : (vision.config.std);
    // If no custom mean/std set, use CLIP defaults
    float clip_mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    float clip_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
    if (mean[0] == 0.0f && mean[1] == 0.0f && mean[2] == 0.0f) {
        mean = clip_mean;
        std  = clip_std;
    }

    auto img_f = vit_load_and_preprocess(image_path, img_size, img_size, mean, std);
    if (img_f.empty()) {
        fprintf(stderr, "FAIL: could not preprocess image\n");
        return 1;
    }
    fprintf(stderr, "   Image preprocessed: %dx%d\n", img_size, img_size);

    // ═══ Step 4: Run vision encoder ═══
    fprintf(stderr, "\n4. Running vision encoder...\n");
    int patch_size = vision.config.patch_size;
    int pw = img_size / patch_size, ph = img_size / patch_size;
    fprintf(stderr,   "   Patches: %dx%d = %d\n", pw, ph, pw * ph);

    auto vision_embeds = vit_forward(vision, img_f.data(), img_size, img_size);
    if (vision_embeds.empty()) {
        fprintf(stderr, "FAIL: vision encoder returned no output\n");
        return 1;
    }

    int n_vis_tokens = (int)vision_embeds.size() / text_hidden;
    fprintf(stderr, "   Vision embeddings: %d tokens x %d dims\n", n_vis_tokens, text_hidden);
    {
        double mean_v = 0, mx = -1e30, mn = 1e30;
        for (float v : vision_embeds) { mean_v += v; mx = std::max((double)v, mx); mn = std::min((double)v, mn); }
        mean_v /= vision_embeds.size();
        fprintf(stderr, "   Stats: mean=%.4f min=%.4f max=%.4f\n", mean_v, mn, mx);
    }

    // ═══ Step 5: Feed vision embeddings through text decoder ═══
    fprintf(stderr, "\n5. Feeding vision embeddings into text decoder...\n");

    // Use special tokens if they exist (Zaya models may not have vision tokens)
    // For ZAYA1-VL, vision tokens are typically inserted directly at the start
    // or after a visual-specific prefix token.
    for (int t = 0; t < n_vis_tokens; t++) {
        be->forward_embed(&vision_embeds[(size_t)t * text_hidden]);
    }
    fprintf(stderr, "   Fed %d vision tokens through text decoder\n", n_vis_tokens);

    // ═══ Step 6: Tokenize and feed prompt ═══
    fprintf(stderr, "\n6. Processing prompt: \"%s\"\n", prompt.c_str());

    // Try to use tokenizer from the model
    SimpleTokenizer tokenizer;
    tokenizer.load_from_gguf(cfg.model_path);

    std::vector<int> prompt_ids;
    if (!cfg.model_path.empty() && cfg.model_path.size() >= 5) {
        prompt_ids = tokenizer.encode(prompt);
    }

    fprintf(stderr, "   Prompt tokenized to %zu ids\n", prompt_ids.size());

    int last = 0;
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        if (i == 0) last = be->generate(prompt_ids[i]);
        else last = be->generate(prompt_ids[i]);
    }

    // ═══ Step 7: Generate response ═══
    fprintf(stderr, "\n7. Generating response...\n\n");
    fprintf(stderr, "=== Response ===\n");

    int eos_id = tokenizer.eos_id;
    std::vector<int> generated;
    for (int i = 0; i < 50; i++) {
        last = be->generate(last);
        generated.push_back(last);
        if (last == eos_id) break;
    }

    std::string response = tokenizer.decode(generated);
    // Fallback detokenize if SimpleTokenizer returned empty
    if (response.empty() && !generated.empty()) {
        std::vector<std::string> vocab;
        // Try to read vocab from the model
        auto slash = cfg.model_path.find_last_of('/');
        auto dir = cfg.model_path.substr(0, slash);
        fprintf(stderr, "   (using fallback detokenizer)\n");
    }

    fprintf(stderr, "%s\n", response.c_str());
    fprintf(stderr, "\n=== Done (%d tokens generated) ===\n", (int)generated.size());

    return 0;
}
