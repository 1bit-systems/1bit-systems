// 1bit-ingest — four-verb curator CLI.
//
//   1bit-ingest prepare <src-dir> --out corpus.tar
//   1bit-ingest pack    --model trained.gguf --manifest catalog.toml --out kevin.1bl
//   1bit-ingest validate kevin.1bl
//   1bit-ingest add-residual --in kevin.1bl --residual kevin.arith \
//                            --index kevin-index.cbor --out kevin-premium.1bl

#include "onebit/ingest/ingest.hpp"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace fs = std::filesystem;
using onebit::ingest::IngestError;

namespace {

int run_prepare(const fs::path& src, const fs::path& out)
{
    auto r = onebit::ingest::prepare(src, out);
    if (!r) {
        std::fprintf(stderr, "error: %s\n", r.error().what().c_str());
        return 1;
    }
    std::fprintf(stderr,
                 "prepared corpus: %zu FLAC file(s), %llu bytes total -> %s\n",
                 r->flac_count,
                 static_cast<unsigned long long>(r->total_bytes),
                 out.string().c_str());
    return 0;
}

int run_pack(const fs::path&                model,
             const fs::path&                manifest,
             const std::optional<fs::path>& cover,
             const std::optional<fs::path>& lyrics,
             const fs::path&                out)
{
    auto r = onebit::ingest::pack(model, manifest, cover, lyrics, out);
    if (!r) {
        std::fprintf(stderr, "error: %s\n", r.error().what().c_str());
        return 1;
    }
    std::fprintf(stderr,
                 "packed %zu section(s), %llu bytes -> %s\n",
                 r->section_count,
                 static_cast<unsigned long long>(r->total_bytes),
                 out.string().c_str());
    return 0;
}

int run_validate(const fs::path& path)
{
    auto r = onebit::ingest::validate(path);
    if (!r) {
        std::fprintf(stderr, "error: %s\n", r.error().what().c_str());
        return 1;
    }
    std::cout << onebit::ingest::format_report(*r);
    return 0;
}

int run_add_residual(const fs::path& input,
                     const fs::path& residual,
                     const fs::path& index,
                     const fs::path& out)
{
    auto r = onebit::ingest::add_residual(input, residual, index, out);
    if (!r) {
        std::fprintf(stderr, "error: %s\n", r.error().what().c_str());
        return 1;
    }
    std::fprintf(stderr,
                 "appended residual (%llu bytes) + index (%llu bytes) -> %s\n",
                 static_cast<unsigned long long>(r->residual_bytes),
                 static_cast<unsigned long long>(r->index_bytes),
                 out.string().c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{"source-side packer for .1bl catalogs"};
    app.set_version_flag("--version", std::string{"0.1.0"});
    app.require_subcommand(1);

    // prepare
    auto* prepare_cmd = app.add_subcommand(
        "prepare",
        "Scan a FLAC directory, extract metadata, tar up a RunPod-ready training corpus.");
    fs::path prepare_src;
    fs::path prepare_out;
    prepare_cmd->add_option("src_dir", prepare_src, "Directory to walk")->required();
    prepare_cmd->add_option("--out", prepare_out, "Output .tar archive")->required();

    // pack
    auto*                   pack_cmd = app.add_subcommand(
        "pack",
        "Assemble a .1bl from a trained GGUF, a catalog.toml, and optional sidecars.");
    fs::path                pack_model;
    fs::path                pack_manifest;
    std::optional<fs::path> pack_cover;
    std::optional<fs::path> pack_lyrics;
    fs::path                pack_out;
    pack_cmd->add_option("--model", pack_model, "Trained ternary-LM .gguf")->required();
    pack_cmd->add_option("--manifest", pack_manifest, "Hand-written catalog.toml")
        ->required();
    pack_cmd->add_option("--cover", pack_cover, "Optional cover.webp / .png");
    pack_cmd->add_option("--lyrics", pack_lyrics, "Optional lyrics bundle (UTF-8 text)");
    pack_cmd->add_option("--out", pack_out, "Output .1bl path")->required();

    // validate
    auto*    validate_cmd = app.add_subcommand(
        "validate", "Verify footer hash, print the manifest, list TLV sections.");
    fs::path validate_path;
    validate_cmd->add_option("catalog", validate_path, "Path to a .1bl file")
        ->required();

    // add-residual
    auto*    addr_cmd = app.add_subcommand(
        "add-residual",
        "Append RESIDUAL_BLOB + RESIDUAL_INDEX to a lossy-only .1bl.");
    fs::path addr_in;
    fs::path addr_residual;
    fs::path addr_index;
    fs::path addr_out;
    addr_cmd->add_option("--in", addr_in, "Existing lossy-tier .1bl")->required();
    addr_cmd->add_option("--residual", addr_residual, "Arithmetic-coded residual blob")
        ->required();
    addr_cmd->add_option("--index", addr_index, "CBOR per-track byte index")->required();
    addr_cmd->add_option("--out", addr_out, "Output path")->required();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    if (prepare_cmd->parsed()) {
        return run_prepare(prepare_src, prepare_out);
    }
    if (pack_cmd->parsed()) {
        return run_pack(pack_model, pack_manifest, pack_cover, pack_lyrics, pack_out);
    }
    if (validate_cmd->parsed()) {
        return run_validate(validate_path);
    }
    if (addr_cmd->parsed()) {
        return run_add_residual(addr_in, addr_residual, addr_index, addr_out);
    }
    return 0;
}
