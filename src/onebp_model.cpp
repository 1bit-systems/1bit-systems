#include "onebp_loader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

OnebpModel::~OnebpModel() {
    if (data && data != MAP_FAILED) munmap(data, file_size);
    if (fd >= 0) close(fd);
}

bool OnebpModel::load(const char* path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    struct stat st;
    fstat(fd, &st);
    file_size = st.st_size;
    data = (uint8_t*)mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return false; }

    // File must be at least as large as the header (issue #1157).
    if (file_size < sizeof(OnebpHeader)) {
        fprintf(stderr, "1BP: file too small (%llu bytes, need %zu)\n",
                (unsigned long long)file_size, sizeof(OnebpHeader));
        return false;
    }

    memcpy(&header, data, sizeof(OnebpHeader));
    if (!header.valid()) {
        fprintf(stderr, "Invalid 1BP header (magic=0x%08X)\n", header.magic);
        return false;
    }

    static constexpr uint32_t MAX_NAME_LEN = 1024;     // 1 KiB
    static constexpr uint32_t MAX_NDIM = 8;            // reasonable max dims
    static constexpr uint32_t MAX_TENSOR_COUNT = 16384; // prevents OOM loops

    if (header.tensor_count > MAX_TENSOR_COUNT) {
        fprintf(stderr, "1BP: tensor_count %u exceeds max %u\n", header.tensor_count, MAX_TENSOR_COUNT);
        return false;
    }

    uint64_t pos = sizeof(OnebpHeader);
    for (uint32_t i = 0; i < header.tensor_count; i++) {
        // Bounds check: each entry needs at least 4(name_len) + 1(min name) + 4(ndim) + 8(offset) + 8(bytes) = 25 bytes
        if (pos + 25 > file_size) {
            fprintf(stderr, "1BP: truncated tensor index at entry %u/%u\n", i, header.tensor_count);
            return false;
        }
        OnebpTensor t;
        uint32_t name_len;
        memcpy(&name_len, data + pos, 4); pos += 4;
        if (name_len > MAX_NAME_LEN || pos + name_len + 1 > file_size) {
            fprintf(stderr, "1BP: invalid name_len=%u at entry %u\n", name_len, i);
            return false;
        }
        t.name = std::string((char*)(data + pos), name_len);
        pos += name_len + 1;  // +1 for null terminator written by converter
        if (pos + 4 > file_size) { fprintf(stderr, "1BP: truncated at ndim\n"); return false; }
        memcpy(&t.ndim, data + pos, 4); pos += 4;
        if (t.ndim > MAX_NDIM) {
            fprintf(stderr, "1BP: ndim=%d exceeds max %u at '%s'\n", t.ndim, MAX_NDIM, t.name.c_str());
            return false;
        }
        if (pos + (uint64_t)t.ndim * 4 + 16 > file_size) {
            fprintf(stderr, "1BP: truncated at dims for '%s'\n", t.name.c_str());
            return false;
        }
        t.dims.resize(t.ndim);
        for (int d = 0; d < t.ndim; d++) {
            memcpy(&t.dims[d], data + pos, 4); pos += 4;
        }
        memcpy(&t.offset, data + pos, 8); pos += 8;
        memcpy(&t.bytes, data + pos, 8); pos += 8;
        tensors.push_back(t);
    }

    // The data section starts right after the tensor index.
    // The index position (pos) now points to the start of data.
    data_section_offset = pos;

    // Validate all tensor offsets against file size (issue #1145).
    for (auto& t : tensors) {
        uint64_t abs_off = data_section_offset + t.offset;
        if (abs_off + t.bytes > file_size || abs_off < data_section_offset) {
            fprintf(stderr, "1BP: tensor '%s' offset=%lu bytes=%lu exceeds file size %lu\n",
                    t.name.c_str(), (unsigned long)t.offset, (unsigned long)t.bytes,
                    (unsigned long)file_size);
            return false;
        }
    }

    return true;
}

uint8_t* OnebpModel::tensor_data(const OnebpTensor& t) {
    // t.offset is relative to the start of the data section.
    // Guard against out-of-bounds access from a crafted 1BP file (issue #1145).
    uint64_t abs_off = data_section_offset + t.offset;
    if (abs_off + t.bytes > file_size || abs_off < data_section_offset) {
        fprintf(stderr, "1BP: OOB tensor '%s' offset=%lu bytes=%lu file=%lu\n",
                t.name.c_str(), (unsigned long)t.offset, (unsigned long)t.bytes,
                (unsigned long)file_size);
        return nullptr;
    }
    return data + abs_off;
}
