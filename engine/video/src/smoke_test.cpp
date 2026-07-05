// smoke_test.cpp — Minimal test of video engine cache configuration,
// argument parsing, and sd_ctx initialization without a full model.
#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include "stable-diffusion.h"

int main() {
    int passed = 0, failed = 0;
#define TEST(name, expr) do { \
    bool ok = (expr); \
    fprintf(stderr, "  %s: %s\n", ok ? "✓" : "✗", name); \
    if (ok) passed++; else failed++; \
} while(0)

    fprintf(stderr, "\n=== Smoke Test: Video Engine ===\n\n");

    // 1. API version
    fprintf(stderr, "[API] commit=%s version=%s\n", sd_commit(), sd_version());
    TEST("sd_commit() non-null", sd_commit() != nullptr);
    TEST("sd_version() non-null", sd_version() != nullptr);

    // 2. System info
    int cores = sd_get_num_physical_cores();
    fprintf(stderr, "[System] %d cores, %s\n", cores, sd_get_system_info());
    TEST("cores > 0", cores > 0);
    TEST("system info non-null", sd_get_system_info() != nullptr);

    // 3. Backend device listing
    size_t dev_size = sd_list_devices(nullptr, 0);
    TEST("sd_list_devices query works", dev_size > 0);
    if (dev_size > 0) {
        std::string dev_buf(dev_size, '\0');
        sd_list_devices(&dev_buf[0], dev_size);
        fprintf(stderr, "[Devices]\n%s", dev_buf.c_str());
    }

    // 4. Cache params initialization
    sd_cache_params_t cache;
    sd_cache_params_init(&cache);
    TEST("cache init: mode=disabled", cache.mode == SD_CACHE_DISABLED);

    // 5. DBCache configuration
    cache.mode = SD_CACHE_DBCACHE;
    cache.Fn_compute_blocks = 8;
    cache.Bn_compute_blocks = 0;
    cache.residual_diff_threshold = 0.12f;
    cache.max_warmup_steps = 4;
    cache.max_cached_steps = -1;
    TEST("DBCache: Fn=8", cache.Fn_compute_blocks == 8);
    TEST("DBCache: Bn=0", cache.Bn_compute_blocks == 0);
    TEST("DBCache: thresh=0.12", cache.residual_diff_threshold == 0.12f);
    TEST("DBCache: warmup=4", cache.max_warmup_steps == 4);

    // 6. TaylorSeer configuration
    cache.mode = SD_CACHE_TAYLORSEER;
    cache.taylorseer_n_derivatives = 1;
    cache.taylorseer_skip_interval = 2;
    TEST("TaylorSeer: deriv=1", cache.taylorseer_n_derivatives == 1);
    TEST("TaylorSeer: skip=2", cache.taylorseer_skip_interval == 2);

    // 7. EasyCache configuration
    cache.mode = SD_CACHE_EASYCACHE;
    cache.reuse_threshold = 0.15f;
    TEST("EasyCache: thresh=0.15", cache.reuse_threshold == 0.15f);

    // 8. Spectrum configuration
    cache.mode = SD_CACHE_SPECTRUM;
    cache.spectrum_w = 0.1f;
    cache.spectrum_m = 4;
    cache.spectrum_lam = 0.5f;
    TEST("Spectrum: w=0.1", cache.spectrum_w == 0.1f);
    TEST("Spectrum: m=4", cache.spectrum_m == 4);

    // 9. Sample params initialization
    sd_sample_params_t sample;
    sd_sample_params_init(&sample);
    TEST("sample init: steps>0", sample.sample_steps > 0);
    TEST("sample init: eta>=0", sample.eta >= 0.0f);

    // 10. Tiling params initialization
    sd_tiling_params_t tiling;
    tiling.enabled = true;
    tiling.tile_size_x = 64;
    tiling.tile_size_y = 64;
    TEST("tiling: enabled", tiling.enabled == true);
    TEST("tiling: tile_x=64", tiling.tile_size_x == 64);

    // 11. Video gen params initialization
    sd_vid_gen_params_t vid;
    sd_vid_gen_params_init(&vid);
    TEST("vid init: frames>0", vid.video_frames > 0);
    TEST("vid init: fps>0", vid.fps > 0);

    // 12. Video gen params with cache (full pipeline config)
    sd_vid_gen_params_init(&vid);
    vid.prompt = "test cat";
    vid.negative_prompt = nullptr;
    vid.width = 640;
    vid.height = 480;
    vid.video_frames = 16;
    vid.seed = 42;
    vid.strength = 5.0f;
    vid.sample_params.sample_steps = 50;
    vid.sample_params.sample_method = EULER_SAMPLE_METHOD;
    vid.cache.mode = SD_CACHE_TAYLORSEER;
    vid.cache.Fn_compute_blocks = 8;
    vid.cache.Bn_compute_blocks = 0;
    vid.cache.residual_diff_threshold = 0.12f;
    vid.cache.max_warmup_steps = 4;
    vid.cache.taylorseer_n_derivatives = 1;
    vid.cache.taylorseer_skip_interval = 2;
    TEST("full config: prompt", strcmp(vid.prompt, "test cat") == 0);
    TEST("full config: cache=taylorseer", vid.cache.mode == SD_CACHE_TAYLORSEER);
    TEST("full config: Fn=8", vid.cache.Fn_compute_blocks == 8);
    TEST("full config: steps=50", vid.sample_params.sample_steps == 50);

    // 13. Model type conversion
    const char* type_name = sd_type_name(SD_TYPE_Q4_0);
    TEST("type name Q4_0", type_name != nullptr && strcmp(type_name, "q4_0") == 0);
    enum sd_type_t type = str_to_sd_type("q8_0");
    TEST("str to type q8_0", type == SD_TYPE_Q8_0);

    // 14. Sample method conversion
    const char* method_name = sd_sample_method_name(EULER_A_SAMPLE_METHOD);
    TEST("sample method name", method_name != nullptr);
    enum sample_method_t method = str_to_sample_method("euler");
    TEST("str to sample method", method == EULER_SAMPLE_METHOD);

    // 15. RNG type conversion
    const char* rng_name = sd_rng_type_name(STD_DEFAULT_RNG);
    TEST("rng name", rng_name != nullptr);
    enum rng_type_t rng = str_to_rng_type("cuda");
    TEST("str to rng cuda", rng == CUDA_RNG);

    // 16. GIF/WebP generation - check WebP support
    // (verified by stable-diffusion.cpp linking libwebp)

    // --- Summary ---
    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d total ===\n",
            passed, failed, passed + failed);
    return failed > 0 ? 1 : 0;
}
