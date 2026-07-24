// whisper_ggml_to_gguf.cpp — one-off asset-prep tool, not part of the
// runtime product. Converts an upstream ggerganov/whisper.cpp legacy
// "ggml" binary model (magic "ggml", the classic pre-GGUF whisper.cpp
// format) into a real GGUF container this project's own src/whisper.cpp
// loader can read (it uses gguf_reader.h — strict "GGUF" magic, same
// container format as this project's LLM weights; the legacy ggml .bin
// format is a different, incompatible layout despite similar tensor
// naming).
//
// Source tensor names differ from what src/whisper.cpp expects in three
// places (verified against a real downloaded ggml-tiny.en.bin via
// `strings`, not assumed): "attn.out.*" -> "attn.output.*",
// "encoder.ln_post.*" -> "encoder.ln.*", and
// "decoder.positional_embedding" -> "decoder.position_embedding.weight".
// The sinusoidal encoder position embedding is computed at load time by
// this project's loader (not a stored tensor), so it's read and dropped.
// The vocab table is carried over as a "tokenizer.ggml.tokens" GGUF
// string array — the same metadata key this project already uses for
// LLM tokenizers (see model_discovery.cpp) — so no new reader code was
// needed on the GGUF-parsing side, only a new consumer in whisper.cpp.
//
// Build: cmake --build . --target whisper_ggml_to_gguf
// Run:   ./whisper_ggml_to_gguf ggml-tiny.en.bin tiny.en.gguf

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct SrcTensor {
    std::string name;
    std::vector<uint64_t> dims; // ggml order: dims[0] fastest-varying
    uint32_t ftype;             // 0 = f32, 1 = f16
    std::vector<float> data;    // always converted to f32
};

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

bool read_exact(std::ifstream& f, void* dst, size_t n) {
    f.read((char*)dst, (std::streamsize)n);
    return (size_t)f.gcount() == n;
}

int32_t read_i32(std::ifstream& f) {
    int32_t v = 0;
    read_exact(f, &v, 4);
    return v;
}

// Renames a source tensor name to what src/whisper.cpp's loader expects.
// Returns "" if this tensor should be dropped (e.g. the positional
// embedding tensors this project computes instead of loading).
std::string rename_tensor(const std::string& src) {
    if (src == "encoder.positional_embedding") return "";
    if (src == "decoder.positional_embedding") return "decoder.position_embedding.weight";
    if (src == "encoder.ln_post.weight") return "encoder.ln.weight";
    if (src == "encoder.ln_post.bias") return "encoder.ln.bias";

    std::string out = src;
    // "attn.out.weight"/"attn.out.bias" -> "attn.output.weight"/"attn.output.bias"
    // (also matches inside "cross_attn.out.*"). Do this as a targeted
    // substring replace since it recurs per-layer for both encoder and
    // decoder, self- and cross-attention.
    const std::string from = "attn.out.";
    const std::string to = "attn.output.";
    size_t pos = out.find(from);
    if (pos != std::string::npos) out.replace(pos, from.size(), to);
    return out;
}

void write_u32(std::ofstream& f, uint32_t v) { f.write((const char*)&v, 4); }
void write_u64(std::ofstream& f, uint64_t v) { f.write((const char*)&v, 8); }
void write_gguf_string(std::ofstream& f, const std::string& s) {
    write_u64(f, s.size());
    f.write(s.data(), (std::streamsize)s.size());
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <ggml-model.bin> <output.gguf>\n", argv[0]);
        return 1;
    }
    std::string in_path = argv[1], out_path = argv[2];

    std::ifstream f(in_path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }

    // ── Header ──
    char magic[4];
    read_exact(f, magic, 4);
    if (memcmp(magic, "lmgg", 4) != 0 && memcmp(magic, "ggml", 4) != 0) {
        fprintf(stderr, "not a ggml whisper file (bad magic)\n");
        return 1;
    }
    int32_t n_vocab_hdr = read_i32(f);
    int32_t n_audio_ctx = read_i32(f);
    int32_t n_audio_state = read_i32(f);
    int32_t n_audio_head = read_i32(f);
    int32_t n_audio_layer = read_i32(f);
    int32_t n_text_ctx = read_i32(f);
    int32_t n_text_state = read_i32(f);
    int32_t n_text_head = read_i32(f);
    int32_t n_text_layer = read_i32(f);
    int32_t n_mels = read_i32(f);
    int32_t model_ftype = read_i32(f);
    fprintf(stderr, "hparams: vocab=%d audio(ctx=%d,state=%d,head=%d,layer=%d) text(ctx=%d,state=%d,head=%d,layer=%d) mels=%d ftype=%d\n",
            n_vocab_hdr, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
            n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, model_ftype);

    // ── Mel filterbank (this project computes its own — skip) ──
    int32_t filt_n_mel = read_i32(f);
    int32_t filt_n_fft = read_i32(f);
    f.seekg((std::streamoff)filt_n_mel * filt_n_fft * 4, std::ios::cur);

    // ── Vocab ──
    int32_t n_vocab = read_i32(f);
    std::vector<std::string> vocab(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        int32_t len = read_i32(f);
        std::string tok(len, '\0');
        read_exact(f, tok.data(), (size_t)len);
        vocab[i] = tok;
    }
    fprintf(stderr, "vocab: %d tokens\n", n_vocab);

    // ── Tensors (read until EOF) ──
    std::vector<SrcTensor> tensors;
    while (f.peek() != EOF) {
        int32_t n_dims = read_i32(f);
        if (f.eof() || n_dims <= 0 || n_dims > 4) break;
        int32_t name_len = read_i32(f);
        int32_t ftype = read_i32(f);

        std::vector<uint64_t> dims(n_dims);
        uint64_t total = 1;
        for (int32_t d = 0; d < n_dims; d++) {
            int32_t dv = read_i32(f);
            dims[d] = (uint64_t)dv;
            total *= (uint64_t)dv;
        }

        std::string name(name_len, '\0');
        if (!read_exact(f, name.data(), (size_t)name_len)) break;

        std::vector<float> data(total);
        if (ftype == 0) { // f32
            if (!read_exact(f, data.data(), total * 4)) break;
        } else if (ftype == 1) { // f16
            std::vector<uint16_t> raw(total);
            if (!read_exact(f, raw.data(), total * 2)) break;
            for (uint64_t i = 0; i < total; i++) data[i] = f16_to_f32(raw[i]);
        } else {
            fprintf(stderr, "unsupported tensor ftype %d for %s, skipping rest of file\n", ftype, name.c_str());
            break;
        }

        std::string renamed = rename_tensor(name);
        if (renamed.empty()) continue; // dropped (computed at load time instead)
        tensors.push_back({renamed, dims, 0, std::move(data)});
    }
    fprintf(stderr, "tensors: %zu (after rename/drop)\n", tensors.size());

    // ── Write GGUF ──
    std::ofstream out(out_path, std::ios::binary);
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }

    // 8 scalar hparam KVs + general.architecture + 1 vocab array KV.
    const uint64_t nk = 10;
    out.write("GGUF", 4);
    write_u32(out, 3);
    write_u64(out, tensors.size());
    write_u64(out, nk);

    auto wkv_str = [&](const std::string& k, const std::string& v) {
        write_gguf_string(out, k);
        write_u32(out, 8); // string
        write_gguf_string(out, v);
    };
    auto wkv_u32 = [&](const std::string& k, uint32_t v) {
        write_gguf_string(out, k);
        write_u32(out, 4); // u32
        write_u32(out, v);
    };

    wkv_str("general.architecture", "whisper");
    wkv_u32("audio.embedding_length", (uint32_t)n_audio_state);
    wkv_u32("audio.block_count", (uint32_t)n_audio_layer);
    wkv_u32("audio.head_count", (uint32_t)n_audio_head);
    wkv_u32("text.embedding_length", (uint32_t)n_text_state);
    wkv_u32("text.block_count", (uint32_t)n_text_layer);
    wkv_u32("text.head_count", (uint32_t)n_text_head);
    // Use the header's n_vocab (matches the actual embedding/output
    // tensor row count) rather than the smaller string-vocab count —
    // whisper reserves extra rows for special/timestamp tokens that
    // aren't stored as explicit vocab strings.
    wkv_u32("vocab_size", (uint32_t)n_vocab_hdr);
    wkv_u32("audio.feature_length", (uint32_t)n_mels);

    // tokenizer.ggml.tokens: string array
    write_gguf_string(out, "tokenizer.ggml.tokens");
    write_u32(out, 9);          // array
    write_u32(out, 8);          // element type: string
    write_u64(out, vocab.size());
    for (auto& t : vocab) write_gguf_string(out, t);

    // Tensor infos
    uint64_t offset = 0;
    for (auto& t : tensors) {
        write_gguf_string(out, t.name);
        write_u32(out, (uint32_t)t.dims.size());
        for (auto d : t.dims) write_u64(out, d);
        write_u32(out, 0); // dtype: F32
        write_u64(out, offset);
        offset += (uint64_t)t.data.size() * 4;
    }

    // 32-byte align before data section (matches this project's own
    // GGUF writer convention, tools/hadamard_export.cpp).
    uint64_t pos = (uint64_t)out.tellp();
    uint64_t rem = pos % 32;
    if (rem) for (uint64_t i = 0; i < 32 - rem; i++) out.put(0);

    for (auto& t : tensors) out.write((const char*)t.data.data(), (std::streamsize)t.data.size() * 4);

    fprintf(stderr, "wrote %s (%zu tensors, %d vocab entries)\n", out_path.c_str(), tensors.size(), n_vocab);
    return 0;
}
