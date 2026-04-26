// bitnet-to-tq2 CLI — repack microsoft/bitnet-b1.58-2B-4T-bf16 → .h1b v4.
//
// Mirrors the Rust CLI (`tools/bitnet-to-tq2/src/main.rs`):
//   --in <DIR>                input HF dir holding model.safetensors + config.json
//   --out <FILE>              output .h1b path
//   --per-tensor-scale        default ON (BitNet b1.58 native)
//   --per-block-scale         opt-in TQ2 native (mutually exclusive)

#include "onebit/tools/bitnet_to_tq2.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

int main(int argc, char** argv)
{
    using namespace onebit::tools::bitnet_to_tq2;

    CLI::App app{
        "Repack microsoft/bitnet-b1.58-2B-4T-bf16 -> .h1b v4 TQ2_0_g128",
        "bitnet-to-tq2"};
    app.set_version_flag("--version", std::string{"bitnet-to-tq2 0.1.0"});

    std::filesystem::path input;
    std::filesystem::path output;
    bool per_tensor = true;
    bool per_block  = false;
    app.add_option("--in", input,
                   "Input directory holding model.safetensors + config.json from "
                   "the HF -bf16 variant.")
        ->required()
        ->option_text("DIR");
    app.add_option("--out", output, "Output .h1b file. Overwrites if it exists.")
        ->required()
        ->option_text("FILE");
    auto* pt_opt = app.add_flag("--per-tensor-scale", per_tensor,
        "Use per-tensor absmean for the block d (BitNet training scale; "
        "duplicated into every block's fp16 slot). Default ON.");
    auto* pb_opt = app.add_flag("--per-block-scale", per_block,
        "Use per-128-block absmean for the block d (TQ2 native policy). "
        "Opt-in for A/B testing.");
    pt_opt->excludes(pb_opt);
    pb_opt->excludes(pt_opt);

    CLI11_PARSE(app, argc, argv);

    const ScaleMode mode = per_block ? ScaleMode::PerBlock : ScaleMode::PerTensor;
    std::fprintf(stderr, "[bitnet-to-tq2] scale_mode=%s\n",
                 mode == ScaleMode::PerTensor ? "PerTensor" : "PerBlock");

    auto r = convert_with_mode(input, output, mode);
    if (!r) {
        std::fprintf(stderr, "bitnet-to-tq2: error: %s\n", r.error().what.c_str());
        return EXIT_FAILURE;
    }
    const ConvertStats& s = *r;
    std::printf(
        "[bitnet-to-tq2] layers=%u hidden=%u ff=%u heads=%u kv_heads=%u vocab=%u "
        "ctx=%u rope_theta=%g eps=%.2e packed_ternary_bytes=%llu output_bytes=%llu "
        "reserved_flags=0x%x path=%s\n",
        s.config.num_hidden_layers,
        s.config.hidden_size,
        s.config.intermediate_size,
        s.config.num_attention_heads,
        s.config.num_key_value_heads,
        s.config.vocab_size,
        s.config.max_position_embeddings,
        static_cast<double>(s.config.rope_theta),
        static_cast<double>(s.config.rms_norm_eps),
        static_cast<unsigned long long>(s.packed_ternary_bytes),
        static_cast<unsigned long long>(s.output_bytes),
        static_cast<unsigned>(s.h1b_reserved_flags),
        s.output_path.c_str());
    if (!s.unmatched_tensors.empty()) {
        std::fprintf(stderr,
            "[bitnet-to-tq2] note: %zu unmatched HF tensors (diagnostic only):\n",
            s.unmatched_tensors.size());
        for (const auto& n : s.unmatched_tensors) {
            std::fprintf(stderr, "    %s\n", n.c_str());
        }
    }
    return EXIT_SUCCESS;
}
