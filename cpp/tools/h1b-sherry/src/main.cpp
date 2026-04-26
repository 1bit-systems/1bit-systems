// h1b-sherry CLI — offline requantizer: halo-1bit TQ1 v4 → Sherry 1.25-bit v3.
//
// Mirrors the Rust CLI (`tools/h1b-sherry/src/main.rs`) bit-for-bit:
//   --in <FILE>   input .h1b (must be v4 / TQ1)
//   --out <FILE>  output .h1b (will be v3 with H1B_FLAG_SHERRY_FP16 set)
//   --verbose     extra per-layer stats line

#include "onebit/tools/h1b_sherry_convert.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main(int argc, char** argv)
{
    using namespace onebit::tools::h1b_sherry;

    CLI::App app{
        "Offline requantizer: halo-1bit TQ1 v4 -> Sherry 1.25-bit v3 "
        "(fp16 flag).",
        "h1b-sherry"};
    app.set_version_flag("--version", std::string{"h1b-sherry 0.1.0"});

    std::filesystem::path input;
    std::filesystem::path output;
    bool                  verbose = false;
    app.add_option("--in", input,
                   "Input .h1b file (must be version 4 / TQ1 packing).")
        ->required()
        ->option_text("FILE");
    app.add_option("--out", output,
                   "Output .h1b file (will be v3 with H1B_FLAG_SHERRY_FP16 set).")
        ->required()
        ->option_text("FILE");
    app.add_flag("--verbose", verbose,
                 "Print detailed per-layer stats instead of just the summary.");

    CLI11_PARSE(app, argc, argv);

    auto r = convert_file(input, output);
    if (!r) {
        std::fprintf(stderr, "h1b-sherry: error: %s\n", r.error().what.c_str());
        return EXIT_FAILURE;
    }
    const ConvertStats& s = *r;
    const double pct = s.flip_fraction() * 100.0;
    std::fprintf(
        stderr,
        "h1b-sherry: wrote %s (layers=%u, rows=%llu, groups=%llu, "
        "forced_zero_flips=%llu = %.3f%%, hadamard_preserved=%s)\n",
        output.c_str(),
        static_cast<unsigned>(s.layers_processed),
        static_cast<unsigned long long>(s.rows_total),
        static_cast<unsigned long long>(s.groups_total),
        static_cast<unsigned long long>(s.forced_zero_flips),
        pct,
        s.hadamard_preserved ? "true" : "false");
    if (verbose) {
        std::fprintf(
            stderr,
            "h1b-sherry: sign-change upper bound is 25%% per group; "
            "observed %.3f%% avg\n",
            pct);
    }
    return EXIT_SUCCESS;
}
