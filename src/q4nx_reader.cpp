// q4nx_reader.cpp — method bodies moved out of include/q4nx_reader.h
// to avoid recompilation cascades (issue #375).
#include "q4nx_reader.h"
#include <cstdio>
#include <cstdlib>

// ── Q4nxReader methods ─────────────────────────────────────────────────────

bool Q4nxReader::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror("Q4NX: open model"); return false; }
    struct stat st;
    fstat(fd, &st);
    data = (const char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size = st.st_size;
    ::close(fd);  // global POSIX close, not the member function
    if (data == MAP_FAILED) { perror("Q4NX: mmap model"); data = nullptr; return false; }
    return true;
}

void Q4nxReader::close() {
    if (data && size > 0) { munmap((void*)data, size); data = nullptr; size = 0; }
}

// Find data offset for a JSON key in the model header
// Uses standard C string search instead of GNU memmem extension.
uint64_t Q4nxReader::find_offset(const char* key) const {
    if (!data || !key) return 0;
    size_t kl = strlen(key);
    if (kl == 0) return 0;
    const char* p = data;
    const char* e = data + (size > 65536 ? 65536 : size); // header within first 64KB
    while (p + kl <= e) {
        // Find first character match
        const char* q = (const char*)memchr(p, key[0], e - p);
        if (!q || q + kl > e) return 0;
        // Check full key match
        if (memcmp(q, key, kl) == 0) {
            if (q > data && *(q - 1) == '"' && *(q + kl) == '"') {
                const char* o = strstr(q, "\"data_offsets\"");
                if (o) {
                    const char* a = strchr(o, '[');
                    if (a) return strtoull(a + 1, NULL, 10);
                }
            }
            p = q + 1;
        } else {
            p = q + 1;
        }
    }
    return 0;
}

// Read float32 array at offset into a vector
std::vector<float> Q4nxReader::read_floats(uint64_t offset, size_t count) const {
    std::vector<float> v;
    if (!data || offset == 0) return v;
    // Prevent integer overflow: if count * 4 wraps around, the bounds check
    // below would pass for maliciously large count values (CRITICAL).
    static constexpr size_t MAX_FLOATS = 256ULL * 1024 * 1024; // 256M floats = 1 GiB
    if (count > MAX_FLOATS) return v;
    uint64_t byte_len = (uint64_t)count * 4;
    if (offset + byte_len > size) return v;
    v.resize(count);
    memcpy(v.data(), data + offset, count * sizeof(float));
    return v;
}

// ── read_q4nx_metadata ─────────────────────────────────────────────────────

bool read_q4nx_metadata(const std::string& path, ModelConfig& cfg) {
    Q4nxReader r;
    if (!r.open(path)) return false;
    if (r.size < 16) { r.close(); return false; }

    uint64_t hdr_len = 0;
    memcpy(&hdr_len, r.data, 8);
    if (hdr_len == 0 || 8 + hdr_len > r.size || hdr_len > (r.size > 65536 ? 65536 : r.size)) {
        r.close();
        return false; // not a Q4NX/safetensors-style header
    }
    std::string header(r.data + 8, (size_t)hdr_len);

    // Embedding tensor gives [vocab, hidden] and dtype.
    auto find_shape_after = [&](const std::string& needle, int& a, int& b) -> bool {
        auto pos = header.find(needle);
        if (pos == std::string::npos) return false;
        auto shape_pos = header.find("\"shape\":[", pos);
        if (shape_pos == std::string::npos) return false;
        shape_pos += strlen("\"shape\":[");
        return sscanf(header.c_str() + shape_pos, "%d,%d", &a, &b) == 2;
    };
    int vocab = 0, hidden = 0;
    if (find_shape_after("\"model.embed_tokens.weight\"", vocab, hidden) ||
        find_shape_after("\"lm_head.weight\"", vocab, hidden)) {
        cfg.vocab = cfg.vocab_size = vocab;
        cfg.hidden = cfg.hidden_size = hidden;
    }

    // Layer count: highest "model.layers.N." index + 1.
    int max_layer = -1;
    size_t search = 0;
    const std::string marker = "\"model.layers.";
    while ((search = header.find(marker, search)) != std::string::npos) {
        int n = atoi(header.c_str() + search + marker.size());
        if (n > max_layer) max_layer = n;
        search += marker.size();
    }
    if (max_layer >= 0) cfg.n_layers = cfg.num_layers = max_layer + 1;

    // Quantization: dtype of the first tensor found.
    auto dtype_pos = header.find("\"dtype\":\"");
    if (dtype_pos != std::string::npos) {
        dtype_pos += strlen("\"dtype\":\"");
        auto end = header.find('"', dtype_pos);
        if (end != std::string::npos) cfg.quantization = header.substr(dtype_pos, end - dtype_pos);
    }

    // Architecture: best-effort from filename (no self-describing field in Q4NX).
    // Split on '-'/'_' only, not digits — real GGUF architecture tags keep the
    // version digit as part of the name (e.g. "qwen3", "zaya1"), so match that.
    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    std::string base = path.substr(slash + 1, (dot == std::string::npos ? path.size() : dot) - slash - 1);
    auto sep = base.find_first_of("-_");
    cfg.architecture = sep == std::string::npos ? base : base.substr(0, sep);
    cfg.model_name = base;
    cfg.model_path = path;
    cfg.format = ModelFormat::Q4NX;

    r.close();
    return true;
}
