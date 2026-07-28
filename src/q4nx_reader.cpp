// q4nx_reader.cpp — method bodies moved out of include/q4nx_reader.h
// to avoid recompilation cascades (issue #375).
#include "q4nx_reader.h"
#include <cstdio>
#include <cstdlib>

// ── Q4nxReader methods ─────────────────────────────────────────────────────

#ifdef _WIN32
// Windows implementation using CreateFileMapping + MapViewOfFile
#include <windows.h>

bool Q4nxReader::open(const std::string& path) {
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "Q4NX: open model failed\n"); return false; }
    LARGE_INTEGER li; GetFileSizeEx(hFile, &li);
    size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); fprintf(stderr, "Q4NX: CreateFileMapping failed\n"); return false; }
    data = (const char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    if (!data) { fprintf(stderr, "Q4NX: MapViewOfFile failed\n"); return false; }
    if (size >= 8) {
        uint64_t hdr_len = 0;
        memcpy(&hdr_len, data, 8);
        if (8 + hdr_len <= size) data_start = (size_t)(8 + hdr_len);
    }
    return true;
}

void Q4nxReader::close() {
    if (data) { UnmapViewOfFile(data); data = nullptr; size = 0; }
}
#else
// POSIX implementation using mmap
bool Q4nxReader::open(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror("Q4NX: open model"); return false; }
    struct stat st;
    fstat(fd, &st);
    data = (const char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    size = st.st_size;
    ::close(fd);
    if (data == MAP_FAILED) { perror("Q4NX: mmap model"); data = nullptr; return false; }
    if (size >= 8) {
        uint64_t hdr_len = 0;
        memcpy(&hdr_len, data, 8);
        if (8 + hdr_len <= size) data_start = (size_t)(8 + hdr_len);
    }
    return true;
}

void Q4nxReader::close() {
    if (data && size > 0) { munmap((void*)data, size); data = nullptr; size = 0; }
}
#endif

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
                    // data_offsets are relative to the end of the safetensors
                    // header (data_start), not to byte 0 of the file — adding
                    // data_start also makes 0 an unambiguous "not found"
                    // sentinel, since no valid tensor can start there (#1058).
                    if (a) return data_start + strtoull(a + 1, NULL, 10);
                }
            }
            p = q + 1;
        } else {
            p = q + 1;
        }
    }
    return 0;
}

// Read a BF16 array at offset, widened to float32, into a vector.
// Every tensor in this Q4NX format (embed table, all norm weights) is
// stored as BF16 (2 bytes/elem), confirmed against data_offsets byte
// spans — not float32, despite the name (#1058).
std::vector<float> Q4nxReader::read_floats(uint64_t offset, size_t count) const {
    std::vector<float> v;
    if (!data || offset == 0) return v;
    // Prevent integer overflow: if count * 2 wraps around, the bounds check
    // below would pass for maliciously large count values (CRITICAL).
    static constexpr size_t MAX_FLOATS = 256ULL * 1024 * 1024; // 256M elems = 512 MiB BF16
    if (count > MAX_FLOATS) return v;
    uint64_t byte_len = (uint64_t)count * 2;
    if (offset + byte_len > size) return v;
    v.resize(count);
    const uint16_t* bf16 = (const uint16_t*)(data + offset);
    for (size_t i = 0; i < count; i++) {
        uint32_t bits = (uint32_t)bf16[i] << 16;
        memcpy(&v[i], &bits, sizeof(float));
    }
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
