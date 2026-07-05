// video_wrapper.cpp — Zero-Python video generation using stable-diffusion.cpp
// with CacheDiT acceleration (DBCache, TaylorSeer, EasyCache).
//
// Build:
//   cd engine/video && mkdir build && cd build
//   cmake .. -DCMAKE_BUILD_TYPE=Release
//   make -j$(nproc)
//
// Usage:
//   ./video_engine -m model.gguf -p "cat" -f 16 --cache dbcache
//   ./video_engine -m model.gguf -p "cat" -f 81 --cache taylorseer --benchmark

#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <getopt.h>
#include <vector>

#include "stable-diffusion.h"

static void progress_cb(int step, int steps, float time, void* data) {
    fprintf(stderr, "\r  [%3d/%3d] %.1fs", step + 1, steps, time);
    fflush(stderr);
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "1bit.systems Video Engine — C++ Diffusion + CacheDiT\n"
        "\n"
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --model, -m     PATH       Model path (GGUF file)\n"
        "  --prompt, -p    TEXT       Text prompt\n"
        "  --negative, -n  TEXT       Negative prompt\n"
        "  --frames, -f    INT        Frames (default: 16, max: 81)\n"
        "  --steps, -s     INT        Denoising steps (default: 50)\n"
        "  --width, -w     INT        Width (default: 640)\n"
        "  --height         INT        Height (default: 480)\n"
        "  --cfg           FLOAT      CFG scale (default: 5.0)\n"
        "  --input-image   PATH       Input image for I2V\n"
        "  --lora          PATH       LoRA path(s) — comma-sep for multiple\n"
        "  --lora-scale    FLOAT      LoRA scale(s) — comma-sep, default: 0.7\n"
        "  --lora-high-noise PATH   High-noise LoRA (Wan2.2 MoE)\n"
        "  --lora-hn-scale FLOAT     High-noise LoRA scale (default: 1.0)\n"
        "  --seed          INT        Random seed\n"
        "  --output, -o    FILE       Output MP4 path\n"
        "  --backend       STR        Backend: cpu, cuda, vulkan, metal\n"
        "  --cache         STR        Cache mode:\n"
        "       off          No cache (default)\n"
        "       easy         EasyCache — reuse above threshold\n"
        "       dbcache      DBCache — dual block cache (Fn/Bn)\n"
        "       taylorseer   DBCache + TaylorSeer calibrator\n"
        "       spectrum     Spectrum cache\n"
        "  --cache-fn      INT        DBCache Fn compute blocks (default: 8)\n"
        "  --cache-bn      INT        DBCache Bn compute blocks (default: 0)\n"
        "  --cache-thresh  FLOAT      Cache reuse threshold (default: 0.12)\n"
        "  --flash-attn               Enable flash attention\n"
        "  --benchmark                Benchmark mode (no output file)\n"
        "  --threads       INT        Thread count\n"
        "  --help, -h                 Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -m model.gguf -p 'cat walking' -f 81 --cache dbcache\n"
        "  %s -m model.gguf -p 'cat' -f 16 --cache taylorseer --flash-attn\n"
        "  %s -m model.gguf -p 'dolly zoom' --benchmark --cache easy --backend cuda\n"
        "\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    std::string model_path, prompt, negative_prompt, input_image_path;
    std::string lora_path, lora_hn_path, output_path = "output.mp4", backend = "cpu";
    std::string cache_mode = "off";
    int num_frames = 16, num_steps = 50, width = 640, height = 480;
    float cfg_scale = 5.0f, lora_scale = 0.7f, lora_hn_scale = 1.0f;
    int seed = 42, threads = 0;
    bool benchmark = false, flash_attn = false;
    int cache_fn = 8, cache_bn = 0;
    float cache_thresh = 0.12f;

    static struct option opts[] = {
        {"model",       required_argument, 0, 'm'},
        {"prompt",      required_argument, 0, 'p'},
        {"negative",    required_argument, 0, 'n'},
        {"frames",      required_argument, 0, 'f'},
        {"steps",       required_argument, 0, 's'},
        {"width",       required_argument, 0, 'w'},
        {"height",      required_argument, 0, 'H'},
        {"cfg",         required_argument, 0, 256},
        {"input-image", required_argument, 0, 'i'},
        {"lora",        required_argument, 0, 257},
        {"lora-scale",  required_argument, 0, 258},
        {"lora-high-noise", required_argument, 0, 268},
        {"lora-hn-scale",   required_argument, 0, 269},
        {"seed",        required_argument, 0, 259},
        {"output",      required_argument, 0, 'o'},
        {"backend",     required_argument, 0, 260},
        {"cache",       required_argument, 0, 261},
        {"cache-fn",    required_argument, 0, 262},
        {"cache-bn",    required_argument, 0, 263},
        {"cache-thresh",required_argument, 0, 264},
        {"flash-attn",  no_argument,       0, 265},
        {"benchmark",   no_argument,       0, 266},
        {"threads",     required_argument, 0, 267},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:n:f:s:w:H:i:o:t:", opts, 0)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': negative_prompt = optarg; break;
            case 'f': num_frames = std::max(1, atoi(optarg)); if (num_frames > 81) num_frames = 81; break;
            case 's': num_steps = std::max(1, atoi(optarg)); break;
            case 'w': width = atoi(optarg); break;
            case 'H': height = atoi(optarg); break;
            case 'i': input_image_path = optarg; break;
            case 'o': output_path = optarg; break;
            case 256: cfg_scale = atof(optarg); break;
            case 257: lora_path = optarg; break;
            case 258: lora_scale = atof(optarg); break;
            case 268: lora_hn_path = optarg; break;
            case 269: lora_hn_scale = atof(optarg); break;
            case 259: seed = atoi(optarg); break;
            case 260: backend = optarg; break;
            case 261: cache_mode = optarg; break;
            case 262: cache_fn = atoi(optarg); break;
            case 263: cache_bn = atoi(optarg); break;
            case 264: cache_thresh = atof(optarg); break;
            case 265: flash_attn = true; break;
            case 266: benchmark = true; break;
            case 267: threads = atoi(optarg); break;
            case 'h': /* help */ print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (model_path.empty() || prompt.empty()) {
        fprintf(stderr, "Error: --model and --prompt are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Map cache mode string to enum
    auto str_to_cache = [](const std::string& s) -> sd_cache_mode_t {
        if (s == "easy")      return SD_CACHE_EASYCACHE;
        if (s == "ucache")    return SD_CACHE_UCACHE;
        if (s == "dbcache")   return SD_CACHE_DBCACHE;
        if (s == "taylorseer") return SD_CACHE_TAYLORSEER;
        if (s == "cachedit")  return SD_CACHE_CACHE_DIT;
        if (s == "spectrum")  return SD_CACHE_SPECTRUM;
        return SD_CACHE_DISABLED;
    };

    sd_cache_mode_t cache_enum = str_to_cache(cache_mode);

    // --- Parse LoRA config before model init ---
    std::vector<sd_lora_t> lora_list;
    auto split = [](const std::string& s, char delim) -> std::vector<std::string> {
        std::vector<std::string> parts;
        size_t start = 0, end;
        while ((end = s.find(delim, start)) != std::string::npos) {
            parts.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        parts.push_back(s.substr(start));
        return parts;
    };

    if (!lora_path.empty()) {
        auto paths = split(lora_path, ',');
        auto scales = split(std::to_string(lora_scale), ',');
        for (size_t i = 0; i < paths.size(); i++) {
            auto& p = paths[i];
            p.erase(0, p.find_first_not_of(" \t"));
            p.erase(p.find_last_not_of(" \t") + 1);
            if (p.empty()) continue;
            float scale = lora_scale;
            if (i < scales.size()) scale = atof(scales[i].c_str());
            lora_list.push_back({false, scale, p.c_str()});
        }
    }
    if (!lora_hn_path.empty()) {
        lora_list.push_back({true, lora_hn_scale, lora_hn_path.c_str()});
    }

    if (!lora_list.empty()) {
        fprintf(stderr, "[LoRA] %zu adapter(s) configured\n", lora_list.size());
        for (auto& l : lora_list)
            fprintf(stderr, "  %s%s @ %.2f\n",
                    l.path,
                    l.is_high_noise ? " (high-noise)" : "",
                    l.multiplier);
    }

    fprintf(stderr,
        "\n=== 1bit.systems Video Engine ===\n"
        "  Model:   %s\n"
        "  Prompt:  %s\n"
        "  Frames:  %d\n"
        "  Steps:   %d\n"
        "  Size:    %dx%d\n"
        "  CFG:     %.1f\n"
        "  Backend: %s\n"
        "  Cache:   %s",
        model_path.c_str(), prompt.c_str(),
        num_frames, num_steps, width, height, cfg_scale,
        backend.c_str(), cache_mode.c_str());

    if (cache_enum != SD_CACHE_DISABLED)
        fprintf(stderr, " (Fn=%d, Bn=%d, thresh=%.3f)",
                cache_fn, cache_bn, cache_thresh);
    fprintf(stderr, "\n  Flash:   %s\n  Seed:    %d\n  Output:  %s\n\n",
            flash_attn ? "yes" : "no", seed,
            benchmark ? "(benchmark)" : output_path.c_str());

    // --- Init stable-diffusion.cpp context ---
    sd_set_log_callback([](enum sd_log_level_t level, const char* text, void*) {
        if (level <= SD_LOG_WARN) fprintf(stderr, "[sd] %s", text);
    }, nullptr);
    sd_set_progress_callback(progress_cb, nullptr);

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path = model_path.c_str();
    ctx_params.n_threads = threads > 0 ? threads : sd_get_num_physical_cores();
    ctx_params.wtype = SD_TYPE_F16;
    ctx_params.rng_type = STD_DEFAULT_RNG;
    ctx_params.flash_attn = flash_attn;
    ctx_params.backend = backend.c_str();

    auto t0 = std::chrono::steady_clock::now();
    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    auto t1 = std::chrono::steady_clock::now();

    if (!sd_ctx) {
        fprintf(stderr, "Error: Failed to create SD context\n");
        return 1;
    }

    fprintf(stderr, "[Init] Model loaded in %.0f ms\n",
            std::chrono::duration<float, std::milli>(t1 - t0).count());
    fprintf(stderr, "[Init] Video support: %s\n",
            sd_ctx_supports_video_generation(sd_ctx) ? "yes" : "no");

    // --- Prepare video generation params ---
    sd_vid_gen_params_t vid_params;
    sd_vid_gen_params_init(&vid_params);

    vid_params.prompt = prompt.c_str();
    vid_params.negative_prompt = negative_prompt.empty() ? nullptr : negative_prompt.c_str();
    vid_params.width = width;
    vid_params.height = height;
    vid_params.video_frames = num_frames;
    vid_params.seed = seed;
    vid_params.strength = cfg_scale;

    // Sampling params
    sd_sample_params_t sample_params;
    sd_sample_params_init(&sample_params);
    sample_params.sample_steps = num_steps;
    sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
    sample_params.scheduler = sd_get_default_scheduler(sd_ctx, sample_params.sample_method);
    vid_params.sample_params = sample_params;

    // --- Configure CacheDiT acceleration ---
    if (cache_enum != SD_CACHE_DISABLED) {
        sd_cache_params_init(&vid_params.cache);
        vid_params.cache.mode = cache_enum;

        switch (cache_enum) {
            case SD_CACHE_EASYCACHE:
                vid_params.cache.reuse_threshold = cache_thresh;
                vid_params.cache.start_percent = 0.0f;
                vid_params.cache.end_percent = 1.0f;
                break;

            case SD_CACHE_DBCACHE:
                vid_params.cache.Fn_compute_blocks = cache_fn;
                vid_params.cache.Bn_compute_blocks = cache_bn;
                vid_params.cache.residual_diff_threshold = cache_thresh;
                vid_params.cache.max_warmup_steps = 4;
                vid_params.cache.max_cached_steps = -1; // unlimited
                vid_params.cache.max_continuous_cached_steps = 4;
                vid_params.cache.use_relative_threshold = true;
                break;

            case SD_CACHE_TAYLORSEER:
                vid_params.cache.Fn_compute_blocks = cache_fn;
                vid_params.cache.Bn_compute_blocks = 0; // TaylorSeer replaces Bn
                vid_params.cache.residual_diff_threshold = cache_thresh;
                vid_params.cache.max_warmup_steps = 4;
                vid_params.cache.max_cached_steps = -1;
                vid_params.cache.taylorseer_n_derivatives = 1;
                vid_params.cache.taylorseer_skip_interval = 2;
                vid_params.cache.use_relative_threshold = true;
                break;

            case SD_CACHE_SPECTRUM:
                vid_params.cache.spectrum_w = 0.1f;
                vid_params.cache.spectrum_m = 4;
                vid_params.cache.spectrum_lam = 0.5f;
                vid_params.cache.spectrum_window_size = 5;
                vid_params.cache.spectrum_flex_window = 0.3f;
                vid_params.cache.spectrum_warmup_steps = 3;
                vid_params.cache.spectrum_stop_percent = 0.8f;
                break;

            default:
                break;
        }

        fprintf(stderr, "[Cache] Configured: %s", cache_mode.c_str());
        if (cache_enum == SD_CACHE_DBCACHE || cache_enum == SD_CACHE_TAYLORSEER)
            fprintf(stderr, " Fn=%d Bn=%d thresh=%.3f warmup=%d",
                    vid_params.cache.Fn_compute_blocks,
                    vid_params.cache.Bn_compute_blocks,
                    vid_params.cache.residual_diff_threshold,
                    vid_params.cache.max_warmup_steps);
        fprintf(stderr, "\n");
    }

    // Assign pre-parsed LoRA config
    if (!lora_list.empty()) {
        vid_params.loras = lora_list.data();
        vid_params.lora_count = (uint32_t)lora_list.size();
    } else {
        vid_params.loras = nullptr;
        vid_params.lora_count = 0;
    }

    // --- Generate ---
    fprintf(stderr, "\n[Generate] Starting...\n");
    t0 = std::chrono::steady_clock::now();

    sd_image_t* frames_out = nullptr;
    int num_frames_out = 0;
    sd_audio_t* audio_out = nullptr;

    bool ok = generate_video(sd_ctx, &vid_params, &frames_out, &num_frames_out, &audio_out);

    t1 = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (ok && frames_out && num_frames_out > 0) {
        fprintf(stderr, "\n[Generate] %d frames generated\n", num_frames_out);

        if (!benchmark) {
            bool use_cuda = (backend == "cuda");
            const char* ffmpeg_pix_fmt = use_cuda ? "nv12" : "rgb24";
            char ffmpeg_cmd[512];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pix_fmt %s "
                "-s %dx%d -r 8 -i - "
                "-c:v libx264 -pix_fmt yuv420p %s 2>/dev/null",
                ffmpeg_pix_fmt, width, height, output_path.c_str());

            FILE* ffmpeg = popen(ffmpeg_cmd, "w");
            if (ffmpeg) {
                for (int i = 0; i < num_frames_out; i++) {
                    fwrite(frames_out[i].data, 1,
                           frames_out[i].width * frames_out[i].height * 3, ffmpeg);
                }
                int ret = pclose(ffmpeg);
                if (ret == 0)
                    fprintf(stderr, "  Output: %s\n", output_path.c_str());
                else
                    fprintf(stderr, "  ffmpeg error (code %d)\n", ret);
            } else {
                fprintf(stderr, "  ffmpeg not available — frames in memory only\n");
            }
        }

        free_sd_images(frames_out, num_frames_out);
    }

    // --- Results ---
    float fps = (total_ms > 0) ? 1000.0f * num_frames / total_ms : 0;
    fprintf(stderr,
        "\n=== Results ===\n"
        "  Total:    %.0f ms\n"
        "  Per frame: %.0f ms\n"
        "  FPS:     %.2f\n"
        "  Cache:   %s\n"
        "\n",
        total_ms, total_ms / num_frames, fps,
        cache_mode.c_str());

    free_sd_ctx(sd_ctx);
    return ok ? 0 : 1;
}
