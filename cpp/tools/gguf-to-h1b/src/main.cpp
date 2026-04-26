// gguf-to-h1b CLI — frame a PrismML Bonsai GGUF into a .h1b + emit .htok.
//
// Mirrors the Rust CLI (`tools/gguf-to-h1b/src/main.rs`):
//   --in <FILE>    input .gguf (Bonsai Q1_0_g128 / TQ2_0_g128)
//   --out <FILE>   output .h1b
//   --no-htok      skip sidecar .htok emission
//   --htok-only    skip .h1b framing; refresh tokenizer sidecar only
//   --verbose      per-tensor progress

#include "onebit/tools/gguf_to_h1b.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

int main(int argc, char** argv)
{
    using namespace onebit::tools::gguf_to_h1b;

    CLI::App app{
        "Frame a PrismML Bonsai GGUF (Q1_0_g128 / TQ2_0_g128) into a .h1b v2 "
        "with the Bonsai flag bits set.",
        "gguf-to-h1b"};
    app.set_version_flag("--version", std::string{"gguf-to-h1b 0.1.0"});

    std::filesystem::path input;
    std::filesystem::path output;
    bool no_htok   = false;
    bool htok_only = false;
    bool verbose   = false;
    app.add_option("--in", input,
                   "Input .gguf file — must be a PrismML Bonsai GGUF.")
        ->required()
        ->option_text("FILE");
    app.add_option("--out", output, "Output .h1b file. Overwritten if it exists.")
        ->required()
        ->option_text("FILE");
    app.add_flag("--no-htok", no_htok,
                 "Skip sidecar .htok emission. Default writes <out>.htok.");
    app.add_flag("--htok-only", htok_only,
                 "Skip .h1b framing; refresh tokenizer sidecar only.");
    app.add_flag("--verbose", verbose,
                 "Print per-tensor progress instead of just the summary.");

    CLI11_PARSE(app, argc, argv);

    if (verbose) {
        std::fprintf(stderr, "gguf-to-h1b: %s -> %s\n",
                     input.c_str(), output.c_str());
    }

    std::optional<ConvertStats> conv;
    if (!htok_only) {
        auto r = convert_file(input, output);
        if (!r) {
            std::fprintf(stderr, "gguf-to-h1b: error: %s\n", r.error().what.c_str());
            return EXIT_FAILURE;
        }
        conv = std::move(*r);
    }

    std::optional<HtokStats> htok;
    if (!no_htok) {
        std::filesystem::path htok_path = output;
        if (htok_path.extension() == ".h1b") {
            htok_path.replace_extension(".htok");
        } else {
            htok_path += ".htok";
        }
        auto h = export_htok_sidecar(input, htok_path);
        if (!h) {
            std::fprintf(stderr, "gguf-to-h1b: error: %s\n", h.error().what.c_str());
            return EXIT_FAILURE;
        }
        htok = std::move(*h);
    }

    if (conv) {
        const auto& s = *conv;
        const char* dt_str = s.dtype == BonsaiDtype::Q1G128 ? "Q1G128" : "TQ2G128";
        std::printf(
            "[gguf-to-h1b] dtype=%s layers=%u hidden=%u ff=%u heads=%u kv_heads=%u "
            "head_dim=%u vocab=%u ctx=%u rope_theta=%g eps=%.2e "
            "ternary_bytes=%llu output_bytes=%llu reserved_flags=0x%x path=%s\n",
            dt_str,
            s.num_layers, s.hidden_size, s.intermediate_size,
            s.num_heads, s.num_kv_heads, s.head_dim,
            s.vocab_size, s.context_length,
            static_cast<double>(s.rope_theta),
            static_cast<double>(s.rms_norm_eps),
            static_cast<unsigned long long>(s.ternary_bytes_carried),
            static_cast<unsigned long long>(s.output_bytes),
            static_cast<unsigned>(s.h1b_reserved_flags),
            s.output_path.c_str());
    }
    if (htok) {
        const auto& h = *htok;
        std::printf(
            "[gguf-to-h1b] htok vocab=%u merges=%u bos=%d eos=%d bytes=%llu "
            "dropped=%u path=%s\n",
            h.vocab_size, h.num_merges, h.bos_id, h.eos_id,
            static_cast<unsigned long long>(h.output_bytes),
            h.dropped_merges, h.output_path.c_str());
    }
    return EXIT_SUCCESS;
}
