// Minimal .h1b model loader — pure C++ + HIP, no MLX, no halo-1bit deps.
//
// Reads the .h1b format that halo-1bit writes (magic "H1B", v1, 9-int32 config,
// then embedding + per-layer weights). Uploads everything to the GPU and
// returns raw device pointers the inference loop can feed to rocm-cpp kernels.
//
// Round-4 perf change — mmap-backed weight load.
//   The previous path read every tensor through `ifstream::read` into a
//   `std::vector` host buffer, then `hipMalloc + hipMemcpy` onto the device.
//   On a 2B-param model that meant ~3 GB of transient host RSS plus a
//   blocking copy across the unified bus — pure waste on Strix Halo where
//   CPU and iGPU share physical RAM.
//
//   The new path mmaps the entire `.h1b` once with MAP_PRIVATE | MAP_POPULATE,
//   registers the full mapping with `hipHostRegister(..., hipHostRegisterMapped)`,
//   then derives every "device" pointer via `hipHostGetDevicePointer` at the
//   correct file offset. The iGPU reads weights in-place from the same
//   physical pages the kernel page-cached on read. RSS for weights drops
//   to ~0; cold-start latency drops by the file-read time.
//
//   Only the dtype-changing reads (FP32 → FP16: norms + embeddings) and the
//   GGUF-sidecar paths still use the malloc+copy path — those cost is
//   ~0.6 GB of transient RSS at most (single embedding tensor) and would
//   need an on-device fp32→fp16 conversion kernel to avoid; deferred.
//
//   The public `bitnet_model.h` ABI is unchanged — mmap state lives in a
//   side-table keyed by `rcpp_bitnet_model_t*`; `rcpp_bitnet_free` looks it
//   up to skip hipFree on mmap-backed pointers and to munmap + unregister
//   at the right time.

#include "rocm_cpp/bitnet_model.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

// HIP_CHECK is used by the private helpers below that return plain `int`.
// The extern "C" rcpp_bitnet_load_h1b function uses LOAD_RC_HIP (below) to
// return the proper rcpp_status_t on failure.
#define HIP_CHECK(e) do { auto _s=(e); if(_s!=hipSuccess){fprintf(stderr,"HIP %d %s:%d\n",_s,__FILE__,__LINE__); return -1;}} while(0)

namespace {

// ── mmap-backed tensor side-table ───────────────────────────────────────────
// Public ABI in bitnet_model.h holds void* device pointers. Some of those now
// point into a mmap'd file rather than a hipMalloc'd buffer; the free path
// must distinguish them so it doesn't hipFree the mmap region. We keep a
// per-model map (rcpp_bitnet_model_t* → MmapState) here in the loader TU,
// looked up from rcpp_bitnet_free.
//
// MmapState holds:
//   - The mmap base + length (for munmap on free).
//   - The hipHostRegister'd flag (for hipHostUnregister on free).
//   - The set of pointers we handed out from the mmap region — rcpp_bitnet_free
//     consults this set and skips hipFree on any pointer in it.
struct MmapState {
    void* mmap_addr = nullptr;
    size_t mmap_len = 0;
    void* dev_base = nullptr;          // result of hipHostGetDevicePointer(mmap_addr)
    bool registered = false;
    std::unordered_set<void*> mmap_pointers;
};

std::mutex& mmap_table_mu() {
    static std::mutex m;
    return m;
}
std::map<rcpp_bitnet_model_t*, MmapState>& mmap_table() {
    static std::map<rcpp_bitnet_model_t*, MmapState> t;
    return t;
}

// Compute the device-visible pointer for a given file offset, given the
// model's mmap state. dev_base is the hipHostGetDevicePointer result for
// mmap_addr; offset arithmetic is byte-wise (uint8_t*).
inline void* dev_ptr_at_offset(const MmapState& s, size_t offset) {
    return static_cast<void*>(static_cast<uint8_t*>(s.dev_base) + offset);
}

inline void* host_ptr_at_offset(const MmapState& s, size_t offset) {
    return static_cast<void*>(static_cast<uint8_t*>(s.mmap_addr) + offset);
}

// Short-read guard. Every weight read goes through this — a truncated
// .h1b file (hostile or network-corrupted) otherwise leaves the tail of
// the destination buffer uninitialized; that garbage then gets uploaded
// to the GPU and poisons the first few inference tokens silently.
#define H1B_READ_OR_FAIL(f, ptr, n)                                         \
    do {                                                                    \
        (f).read(reinterpret_cast<char*>(ptr), static_cast<std::streamsize>(n)); \
        if ((f).gcount() != static_cast<std::streamsize>(n)) {              \
            fprintf(stderr,                                                 \
                    "[rocm-cpp] .h1b short read at %s:%d "                  \
                    "(wanted %zu bytes, got %lld)\n",                       \
                    __FILE__, __LINE__, (size_t)(n),                        \
                    (long long)(f).gcount());                               \
            return RCPP_INVALID_ARG;                                        \
        }                                                                   \
    } while (0)

// Read FP32 from disk, cast to FP16, upload to device (the .h1b format
// stores norms and embeddings as float32; kernels consume FP16).
int read_fp32_as_fp16(std::ifstream& f, size_t n, __half** out) {
    std::vector<float> src(n);
    H1B_READ_OR_FAIL(f, src.data(), n * sizeof(float));
    std::vector<_Float16> dst(n);
    for (size_t i = 0; i < n; ++i) dst[i] = (_Float16)src[i];
    HIP_CHECK(hipMalloc(out, n * sizeof(_Float16)));
    HIP_CHECK(hipMemcpy(*out, dst.data(), n * sizeof(_Float16), hipMemcpyHostToDevice));
    return 0;
}

// Skip a block of float32 values we don't need (e.g., the duplicated
// attn_sub_norm copies the exporter writes 4× for legacy reasons).
void skip_fp32(std::ifstream& f, size_t n) {
    f.seekg(n * sizeof(float), std::ios::cur);
}

// ── mmap-backed read variants ──────────────────────────────────────────────
// These advance the file cursor (so the rest of the loader still sees the
// "wrote N bytes / now at offset M" model) but DO NOT copy. The device
// pointer is derived from the mmap'd region at the current file offset.
//
// Bounds-check every read against `s.mmap_len` so a truncated or hostile
// `.h1b` cannot hand the GPU an out-of-bounds device address.
//
// On success, the returned pointers are stashed in `s.mmap_pointers` so
// rcpp_bitnet_free can distinguish them from hipMalloc'd pointers.

int read_ternary_mmap(std::ifstream& f, MmapState& s,
                      int rows, int cols,
                      void** packed_out, void** scales_out)
{
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] read_ternary_mmap: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    const int packed_cols = (cols + 3) / 4;
    const size_t packed_bytes = (size_t)rows * packed_cols;
    const size_t scales_bytes = (size_t)rows * sizeof(float);
    const size_t pos = (size_t)f.tellg();
    if (pos + packed_bytes + scales_bytes > s.mmap_len) {
        fprintf(stderr, "[rocm-cpp] read_ternary_mmap: would overrun mmap (%zu+%zu+%zu > %zu)\n",
                pos, packed_bytes, scales_bytes, s.mmap_len);
        return RCPP_INVALID_ARG;
    }
    *packed_out = dev_ptr_at_offset(s, pos);
    *scales_out = dev_ptr_at_offset(s, pos + packed_bytes);
    s.mmap_pointers.insert(*packed_out);
    s.mmap_pointers.insert(*scales_out);
    f.seekg((std::streamoff)(packed_bytes + scales_bytes), std::ios::cur);
    return 0;
}

int read_ternary_sherry_mmap(std::ifstream& f, MmapState& s,
                             int rows, int cols,
                             void** packed_out, void** scales_out)
{
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] sherry mmap: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    if (cols % 32 != 0) {
        fprintf(stderr, "[rocm-cpp] sherry mmap: cols=%d not divisible by 32\n", cols);
        return -1;
    }
    const size_t row_bytes = (size_t)cols * 5 / 32;
    const size_t packed_bytes = (size_t)rows * row_bytes;
    const size_t scales_bytes = (size_t)rows * sizeof(float);
    const size_t pos = (size_t)f.tellg();
    if (pos + packed_bytes + scales_bytes > s.mmap_len) {
        fprintf(stderr, "[rocm-cpp] sherry mmap: would overrun (%zu+%zu+%zu > %zu)\n",
                pos, packed_bytes, scales_bytes, s.mmap_len);
        return RCPP_INVALID_ARG;
    }
    *packed_out = dev_ptr_at_offset(s, pos);
    *scales_out = dev_ptr_at_offset(s, pos + packed_bytes);
    s.mmap_pointers.insert(*packed_out);
    s.mmap_pointers.insert(*scales_out);
    f.seekg((std::streamoff)(packed_bytes + scales_bytes), std::ios::cur);
    return 0;
}

int read_ternary_tq1_mmap(std::ifstream& f, MmapState& s,
                          int rows, int cols,
                          void** packed_out, void** scales_out)
{
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] tq1 mmap: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    const int cols_padded = ((cols + 19) / 20) * 20;
    const size_t row_bytes = (size_t)cols_padded / 5;
    const size_t packed_bytes = (size_t)rows * row_bytes;
    const size_t scales_bytes = (size_t)rows * sizeof(float);
    const size_t pos = (size_t)f.tellg();
    if (pos + packed_bytes + scales_bytes > s.mmap_len) {
        fprintf(stderr, "[rocm-cpp] tq1 mmap: would overrun (%zu+%zu+%zu > %zu)\n",
                pos, packed_bytes, scales_bytes, s.mmap_len);
        return RCPP_INVALID_ARG;
    }
    *packed_out = dev_ptr_at_offset(s, pos);
    *scales_out = dev_ptr_at_offset(s, pos + packed_bytes);
    s.mmap_pointers.insert(*packed_out);
    s.mmap_pointers.insert(*scales_out);
    f.seekg((std::streamoff)(packed_bytes + scales_bytes), std::ios::cur);
    return 0;
}

int read_bonsai_blocks_mmap(std::ifstream& f, MmapState& s,
                            int rows, int cols, int block_bytes, int group_size,
                            void** packed_out)
{
    if (rows <= 0 || cols <= 0 || block_bytes <= 0 || group_size <= 0) {
        fprintf(stderr, "[rocm-cpp] bonsai mmap: bad dims rows=%d cols=%d bb=%d gs=%d\n",
                rows, cols, block_bytes, group_size);
        return RCPP_INVALID_ARG;
    }
    if (cols % group_size != 0) {
        fprintf(stderr, "[rocm-cpp] bonsai mmap: cols=%d not divisible by group_size=%d\n",
                cols, group_size);
        return -1;
    }
    const size_t blocks_per_row = (size_t)cols / group_size;
    const size_t row_bytes = blocks_per_row * (size_t)block_bytes;
    const size_t packed_bytes = (size_t)rows * row_bytes;
    const size_t pos = (size_t)f.tellg();
    if (pos + packed_bytes > s.mmap_len) {
        fprintf(stderr, "[rocm-cpp] bonsai mmap: would overrun (%zu+%zu > %zu)\n",
                pos, packed_bytes, s.mmap_len);
        return RCPP_INVALID_ARG;
    }
    *packed_out = dev_ptr_at_offset(s, pos);
    s.mmap_pointers.insert(*packed_out);
    f.seekg((std::streamoff)packed_bytes, std::ios::cur);
    return 0;
}

// Read a packed ternary weight (halo-1bit format: uint8[rows, (cols+3)/4] + float[rows] scales).
int read_ternary(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] read_ternary: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    const int packed_cols = (cols + 3) / 4;
    std::vector<uint8_t> packed((size_t)rows * packed_cols);
    H1B_READ_OR_FAIL(f, packed.data(), packed.size());
    std::vector<float> scales(rows);
    H1B_READ_OR_FAIL(f, scales.data(), rows * sizeof(float));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(scales_out), rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(*scales_out, scales.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    return 0;
}

// Sherry v3: uint8[rows * cols * 5 / 32] + float[rows] scales. 1.25 bpw.
// cols must be a multiple of 32.
int read_ternary_sherry(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] v3 load: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    if (cols % 32 != 0) {
        fprintf(stderr, "[rocm-cpp] v3 load: cols=%d not divisible by 32\n", cols);
        return -1;
    }
    const size_t row_bytes = (size_t)cols * 5 / 32;
    std::vector<uint8_t> packed((size_t)rows * row_bytes);
    H1B_READ_OR_FAIL(f, packed.data(), packed.size());
    std::vector<float> scales(rows);
    H1B_READ_OR_FAIL(f, scales.data(), rows * sizeof(float));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(scales_out), rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(*scales_out, scales.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    return 0;
}

// TQ1 v4: uint8[rows * cols_padded / 5] + float[rows] scales. 1.6 bpw.
// cols is padded up to multiple of 20 (requantizer handles the padding).
int read_ternary_tq1(std::ifstream& f, int rows, int cols, void** packed_out, void** scales_out) {
    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "[rocm-cpp] tq1 load: bad dims rows=%d cols=%d\n", rows, cols);
        return RCPP_INVALID_ARG;
    }
    const int cols_padded = ((cols + 19) / 20) * 20;
    const size_t row_bytes = (size_t)cols_padded / 5;
    std::vector<uint8_t> packed((size_t)rows * row_bytes);
    H1B_READ_OR_FAIL(f, packed.data(), packed.size());
    std::vector<float> scales(rows);
    H1B_READ_OR_FAIL(f, scales.data(), rows * sizeof(float));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(scales_out), rows * sizeof(float)));
    HIP_CHECK(hipMemcpy(*scales_out, scales.data(), rows * sizeof(float), hipMemcpyHostToDevice));
    return 0;
}

// Bonsai Q1_0_g128 / TQ2_0_g128: block-interleaved with inline FP16 scales.
//   Q1  : 18 bytes / 128 weights (FP16 d, 16 bytes sign bits)
//   TQ2 : 34 bytes / 128 weights (32 bytes 2-bit codes, FP16 d)
// No trailing per-row scale tensor — Bonsai embeds everything inline.
int read_bonsai_blocks(std::ifstream& f, int rows, int cols,
                       int block_bytes, int group_size, void** packed_out)
{
    if (rows <= 0 || cols <= 0 || block_bytes <= 0 || group_size <= 0) {
        fprintf(stderr, "[rocm-cpp] bonsai load: bad dims rows=%d cols=%d block_bytes=%d group_size=%d\n",
                rows, cols, block_bytes, group_size);
        return RCPP_INVALID_ARG;
    }
    if (cols % group_size != 0) {
        fprintf(stderr, "[rocm-cpp] bonsai load: cols=%d not divisible by group_size=%d\n",
                cols, group_size);
        return -1;
    }
    const size_t blocks_per_row = (size_t)cols / group_size;
    const size_t row_bytes = blocks_per_row * (size_t)block_bytes;
    std::vector<uint8_t> packed((size_t)rows * row_bytes);
    H1B_READ_OR_FAIL(f, packed.data(), packed.size());
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(packed_out), packed.size()));
    HIP_CHECK(hipMemcpy(*packed_out, packed.data(), packed.size(), hipMemcpyHostToDevice));
    return 0;
}

// -----------------------------------------------------------------------------
// Minimal GGUF v3 sidecar reader — used ONLY for Bonsai (Qwen3) models to
// pull tensors the `.h1b` converter zero-fills: per-head attn_q/k_norm,
// per-layer attn/ffn_norm, final output_norm, and the ternary token embedding.
//
// The `.h1b` converter lives at tools/gguf-to-h1b/ and ships zeros for
// BitNet-shaped norm slots plus a zero-filled embedding — at load time we
// paper over that by reading the real tensors from the companion GGUF.
// -----------------------------------------------------------------------------

struct GgufTensorInfo {
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t offset;   // relative offset into the tensor data region
};

class GgufSidecar {
  public:
    bool open(const std::string& path) {
        f_.open(path, std::ios::binary);
        if (!f_) return false;
        char magic[4];
        f_.read(magic, 4);
        if (std::strncmp(magic, "GGUF", 4) != 0) return false;
        if (!read_u32(version_)) return false;
        if (version_ != 2 && version_ != 3) {
            fprintf(stderr, "[rocm-cpp][gguf] unsupported version %u\n", version_);
            return false;
        }
        uint64_t n_tensors, n_kv;
        if (!read_u64(n_tensors) || !read_u64(n_kv)) return false;

        arch_.clear();
        alignment_ = 32;  // GGUF default
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key;
            if (!read_string(key)) return false;
            uint32_t vt;
            if (!read_u32(vt)) return false;
            if (key == "general.architecture" && vt == 8 /*string*/) {
                if (!read_string(arch_)) return false;
            } else if (key == "general.alignment" && vt == 4 /*u32*/) {
                uint32_t a;
                if (!read_u32(a)) return false;
                alignment_ = a ? a : 32;
            } else {
                if (!skip_value(vt)) return false;
            }
        }

        tensors_.clear();
        for (uint64_t i = 0; i < n_tensors; ++i) {
            std::string name;
            if (!read_string(name)) return false;
            uint32_t ndim;
            if (!read_u32(ndim)) return false;
            GgufTensorInfo info;
            info.shape.resize(ndim);
            for (uint32_t d = 0; d < ndim; ++d) {
                if (!read_u64(info.shape[d])) return false;
            }
            if (!read_u32(info.dtype)) return false;
            if (!read_u64(info.offset)) return false;
            tensors_.emplace(std::move(name), std::move(info));
        }

        data_start_ = (uint64_t)f_.tellg();
        const uint64_t rem = data_start_ % alignment_;
        if (rem) data_start_ += alignment_ - rem;
        return true;
    }

    const std::string& arch() const { return arch_; }

    const GgufTensorInfo* info(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return nullptr;
        return &it->second;
    }

    bool read_tensor_bytes(const std::string& name, size_t nbytes,
                           std::vector<uint8_t>& out)
    {
        const auto* ti = info(name);
        if (!ti) return false;
        out.resize(nbytes);
        f_.seekg((std::streamoff)(data_start_ + ti->offset), std::ios::beg);
        f_.read(reinterpret_cast<char*>(out.data()), (std::streamsize)nbytes);
        return (bool)f_;
    }

  private:
    bool read_u32(uint32_t& x) {
        f_.read(reinterpret_cast<char*>(&x), 4);
        return (bool)f_;
    }
    bool read_u64(uint64_t& x) {
        f_.read(reinterpret_cast<char*>(&x), 8);
        return (bool)f_;
    }
    bool read_string(std::string& s) {
        uint64_t n;
        if (!read_u64(n)) return false;
        s.resize((size_t)n);
        if (n) f_.read(s.data(), (std::streamsize)n);
        return (bool)f_;
    }
    bool skip_value(uint32_t vt) {
        switch (vt) {
            case 0: case 1:              f_.seekg(1, std::ios::cur); break;
            case 2: case 3:              f_.seekg(2, std::ios::cur); break;
            case 4: case 5: case 6:      f_.seekg(4, std::ios::cur); break;
            case 7:                      f_.seekg(1, std::ios::cur); break;
            case 8: { std::string s; if (!read_string(s)) return false; break; }
            case 9: {
                uint32_t at;
                uint64_t n;
                if (!read_u32(at) || !read_u64(n)) return false;
                for (uint64_t i = 0; i < n; ++i) if (!skip_value(at)) return false;
                break;
            }
            case 10: case 11: case 12:   f_.seekg(8, std::ios::cur); break;
            default:
                fprintf(stderr, "[rocm-cpp][gguf] unknown value type %u\n", vt);
                return false;
        }
        return (bool)f_;
    }

    std::ifstream f_;
    std::string arch_;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_start_ = 0;
    std::map<std::string, GgufTensorInfo> tensors_;
};

// Bonsai TQ2_0_g128 host-side dequantizer → FP16.
// On-disk layout is [fp16 d : 2][qs : 32]; see bonsai_tq2_gemv.hip header.
void dequantize_bonsai_tq2_to_fp16(const uint8_t* packed, size_t rows, size_t cols,
                                   _Float16* out)
{
    const size_t group_size = 128;
    const size_t block_bytes = 34;
    const size_t blocks_per_row = cols / group_size;
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t* row = packed + r * blocks_per_row * block_bytes;
        _Float16* orow = out + r * cols;
        for (size_t b = 0; b < blocks_per_row; ++b) {
            const uint8_t* blk = row + b * block_bytes;
            uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            _Float16 d;
            std::memcpy(&d, &d_bits, 2);
            for (size_t j = 0; j < group_size; ++j) {
                const size_t byte_idx = 2 + j / 4;
                const size_t lane = j % 4;
                const uint8_t code = (blk[byte_idx] >> (lane * 2)) & 0x3u;
                float v;
                switch (code) {
                    case 0b00u: v = -1.0f; break;
                    case 0b01u: v =  0.0f; break;
                    case 0b10u: v = +1.0f; break;
                    default:    v =  0.0f; break;  // 0b11 reserved → 0
                }
                orow[b * group_size + j] = (_Float16)(v * (float)d);
            }
        }
    }
}

// Derive the sidecar GGUF path from the .h1b path.
//
// The fallback to `Ternary-Bonsai-1.7B-Q2_0.gguf` exists for the oxibonsai
// workflow where the user dropped the stock HF GGUF next to a converter-
// emitted `.h1b`. It's intentionally narrow: a BitNet-repacked `.h1b`
// dropped in a different directory will not collide with it.
std::string derive_gguf_sidecar_path(const char* h1b_path) {
    std::string p(h1b_path);
    const size_t n = p.size();
    if (n >= 4 && p.compare(n - 4, 4, ".h1b") == 0) {
        std::string cand = p.substr(0, n - 4) + ".gguf";
        std::ifstream t(cand);
        if (t) return cand;
    }
    const size_t slash = p.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".")
                                                         : p.substr(0, slash);
    const std::string fallback = dir + "/Ternary-Bonsai-1.7B-Q2_0.gguf";
    std::ifstream t(fallback);
    if (t) return fallback;
    return std::string();
}

// Decide the architecture of a Bonsai-weight-format .h1b. If a sidecar GGUF
// exists and its `general.architecture` key is `qwen3` we lock the model to
// Qwen3; otherwise we fall back to BitNet (matches the MS-BitNet-repack path
// emitted by `tools/bitnet-to-tq2/`, which writes real attn_sub_norm +
// ffn_sub_norm into the `.h1b` and has no sidecar GGUF).
//
// The returned string holds the resolved sidecar path (empty on BitNet) so
// the caller can skip the GGUF hydration pass without re-deriving it.
rcpp_arch_t resolve_bonsai_arch(const char* h1b_path, std::string& out_sidecar) {
    out_sidecar.clear();
    const std::string candidate = derive_gguf_sidecar_path(h1b_path);
    if (candidate.empty()) {
        return RCPP_ARCH_BITNET;
    }
    GgufSidecar g;
    if (!g.open(candidate)) {
        // Unreadable GGUF — treat as "no sidecar" and route through BitNet
        // rather than crash or silently fall back to Qwen3 + zeroed norms.
        fprintf(stderr,
            "[rocm-cpp] sidecar candidate %s failed to parse — treating as BitNet arch\n",
            candidate.c_str());
        return RCPP_ARCH_BITNET;
    }
    if (g.arch() == "qwen3") {
        out_sidecar = candidate;
        return RCPP_ARCH_QWEN3;
    }
    // Sidecar present but arch != qwen3 (e.g. "bitnet" / "llama"). Ignore it.
    fprintf(stderr,
        "[rocm-cpp] sidecar %s arch=%s (not qwen3) — routing as BitNet arch\n",
        candidate.c_str(), g.arch().c_str());
    return RCPP_ARCH_BITNET;
}

}  // namespace

// Local macro for the extern "C" loader — maps HIP failures to rcpp_status_t.
#define LOAD_RC_HIP(e) do { if ((e) != hipSuccess) { \
    fprintf(stderr, "HIP err %s:%d\n", __FILE__, __LINE__); return RCPP_HIP_ERROR; } \
} while (0)

extern "C" rcpp_status_t
rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model) {
    if (!path || !out_model) return RCPP_INVALID_ARG;
    std::memset(out_model, 0, sizeof(*out_model));

    // ── mmap the file once up front. Round-4 perf change: weights are read
    // in-place from the unified-memory mmap rather than copied through host
    // RSS. See file header comment for the full design.
    //
    // Opt out via HALO_NO_MMAP=1 — useful if the user is loading off a
    // network FS where MAP_POPULATE blocks on slow I/O, or when running
    // under a debugger that can't page-fault a hipHostRegister'd mapping
    // cleanly. Falls back to the legacy ifstream + hipMalloc + hipMemcpy
    // path; nothing else changes.
    const bool mmap_enabled = (std::getenv("HALO_NO_MMAP") == nullptr);

    MmapState mmap_state;
    if (mmap_enabled) {
        const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            fprintf(stderr, "Cannot open: %s (errno=%d)\n", path, errno);
            return RCPP_INVALID_ARG;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            fprintf(stderr, "fstat failed: %s (errno=%d)\n", path, errno);
            ::close(fd);
            return RCPP_INVALID_ARG;
        }
        const size_t file_len = (size_t)st.st_size;
        // MAP_POPULATE pre-faults all pages — turns a series of cold demand
        // faults at first GEMV into a single up-front sequential read, which
        // the kernel issues as one big readahead under the hood. Without it
        // we'd page-fault layer-by-layer during the first decode pass.
        void* addr = ::mmap(nullptr, file_len, PROT_READ,
                            MAP_PRIVATE | MAP_POPULATE, fd, 0);
        ::close(fd);   // mapping holds its own fd ref
        if (addr == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s (errno=%d)\n", path, errno);
            return RCPP_HIP_ERROR;
        }
        // Register the entire mapping with the HIP runtime so the iGPU can
        // address it. On Strix Halo (gfx1151, unified memory) this is a
        // bookkeeping op — no copy, no second physical allocation.
        // hipHostRegisterReadOnly hints to the runtime that the iGPU will
        // never write — we never write back to the .h1b mapping (kernels
        // only read weights), and the hint lets the runtime skip the
        // GPU→CPU coherence path.
        unsigned int reg_flags = hipHostRegisterMapped;
#if defined(hipHostRegisterReadOnly)
        reg_flags |= hipHostRegisterReadOnly;
#endif
        hipError_t reg_st = hipHostRegister(addr, file_len, reg_flags);
        if (reg_st != hipSuccess) {
            fprintf(stderr,
                "[rocm-cpp] hipHostRegister failed (rc=%d) on .h1b mmap; "
                "falling back to copy path. file=%s len=%zu\n",
                (int)reg_st, path, file_len);
            ::munmap(addr, file_len);
        } else {
            void* dev = nullptr;
            hipError_t gst = hipHostGetDevicePointer(&dev, addr, 0);
            if (gst != hipSuccess) {
                fprintf(stderr,
                    "[rocm-cpp] hipHostGetDevicePointer failed (rc=%d); "
                    "falling back to copy path.\n", (int)gst);
                hipHostUnregister(addr);
                ::munmap(addr, file_len);
            } else {
                mmap_state.mmap_addr = addr;
                mmap_state.mmap_len  = file_len;
                mmap_state.dev_base  = dev;
                mmap_state.registered = true;
                fprintf(stderr,
                    "[rocm-cpp] mmap-load enabled: %zu bytes at %p (dev %p)\n",
                    file_len, addr, dev);
            }
        }
    }
    const bool mmap_active = mmap_state.registered;

    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return RCPP_INVALID_ARG; }

    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4 || std::memcmp(magic, "H1B\0", 4) != 0) {
        fprintf(stderr, "Bad .h1b magic\n");
        return RCPP_INVALID_ARG;
    }

    int32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version != 1 && version != 2 && version != 3 && version != 4) {
        fprintf(stderr, "Unsupported .h1b version: %d\n", version);
        return RCPP_UNSUPPORTED;
    }
    out_model->format_version = version;
    const bool use_sherry = (version == 3);
    const bool use_tq1    = (version == 4);

    int32_t cfg[9];
    f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    if (f.gcount() != (std::streamsize)sizeof(cfg)) {
        fprintf(stderr, "Short read on .h1b config header\n");
        return RCPP_INVALID_ARG;
    }
    // Sanitize config fields — a hostile .h1b with INT_MAX in any of these
    // triggers signed int-mul overflow downstream and lands in hipMalloc
    // with a wrapped size. Ceilings below are ~10x the largest known
    // real-world model (Llama-3 405B class); anything past this is either
    // a format bug or a crafted attack.
    constexpr int32_t MAX_HIDDEN       = 1 << 15;   // 32768
    constexpr int32_t MAX_INTERMEDIATE = 1 << 17;   // 131072
    constexpr int32_t MAX_LAYERS       = 1 << 10;   // 1024
    constexpr int32_t MAX_HEADS        = 1 << 10;   // 1024
    constexpr int32_t MAX_KV_HEADS     = 1 << 10;   // 1024
    constexpr int32_t MAX_VOCAB        = 1 << 20;   // 1,048,576
    constexpr int32_t MAX_SEQ_LEN      = 1 << 20;
    auto bad = [&](const char* name, int32_t v, int32_t cap) {
        fprintf(stderr, ".h1b config field %s=%d out of range (must be in (0, %d])\n",
                name, v, cap);
        return RCPP_INVALID_ARG;
    };
    if (cfg[0] <= 0 || cfg[0] > MAX_HIDDEN)        return bad("hidden_size",       cfg[0], MAX_HIDDEN);
    if (cfg[1] <= 0 || cfg[1] > MAX_INTERMEDIATE)  return bad("intermediate_size", cfg[1], MAX_INTERMEDIATE);
    if (cfg[2] <= 0 || cfg[2] > MAX_LAYERS)        return bad("num_layers",        cfg[2], MAX_LAYERS);
    if (cfg[3] <= 0 || cfg[3] > MAX_HEADS)         return bad("num_heads",         cfg[3], MAX_HEADS);
    if (cfg[4] <= 0 || cfg[4] > MAX_KV_HEADS)      return bad("num_kv_heads",      cfg[4], MAX_KV_HEADS);
    if (cfg[5] <= 0 || cfg[5] > MAX_VOCAB)         return bad("vocab_size",        cfg[5], MAX_VOCAB);
    if (cfg[6] <= 0 || cfg[6] > MAX_SEQ_LEN)       return bad("max_seq_len",       cfg[6], MAX_SEQ_LEN);
    if (cfg[0] % cfg[3] != 0) {
        fprintf(stderr, ".h1b hidden_size=%d not divisible by num_heads=%d\n",
                cfg[0], cfg[3]);
        return RCPP_INVALID_ARG;
    }
    out_model->hidden_size       = cfg[0];
    out_model->intermediate_size = cfg[1];
    out_model->num_layers        = cfg[2];
    out_model->num_heads         = cfg[3];
    out_model->num_kv_heads      = cfg[4];
    out_model->vocab_size        = cfg[5];
    out_model->max_seq_len       = cfg[6];
    out_model->tie_embeddings    = cfg[7];
    out_model->flags             = static_cast<unsigned int>(cfg[8]);

    const bool sherry_fp16 = use_sherry
        && (out_model->flags & H1B_FLAG_SHERRY_FP16) != 0;
    const bool bonsai_q1  = (out_model->flags & H1B_FLAG_BONSAI_Q1)  != 0;
    const bool bonsai_tq2 = (out_model->flags & H1B_FLAG_BONSAI_TQ2) != 0;
    if (bonsai_q1 && bonsai_tq2) {
        fprintf(stderr, "[rocm-cpp] both BONSAI_Q1 and BONSAI_TQ2 set — refusing to guess\n");
        return RCPP_INVALID_ARG;
    }
    const bool is_bonsai_fmt = bonsai_q1 || bonsai_tq2;

    // Resolve dispatch tag. Bonsai bits take precedence across all .h1b
    // versions (format differs fundamentally — inline block scales, no
    // trailing per-row scales tensor).
    if (bonsai_tq2) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_BONSAI_TQ2;
    } else if (bonsai_q1) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_BONSAI_Q1;
    } else if (use_tq1) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_TQ1;
    } else if (sherry_fp16) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_SHERRY_FP16;
    } else if (use_sherry) {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_SHERRY_I8;
    } else {
        out_model->weight_format = RCPP_WEIGHT_FORMAT_HALO_V2;
    }

    // Resolve model architecture. Non-Bonsai weight formats are always
    // BitNet-flavored today (halo v2, Sherry, TQ1 all ship with
    // attn_sub_norm / ffn_sub_norm / squared-ReLU GLU). Bonsai-format
    // models split by whether a Qwen3 sidecar GGUF is present.
    std::string sidecar_path;
    if (is_bonsai_fmt) {
        out_model->arch = resolve_bonsai_arch(path, sidecar_path);
    } else {
        out_model->arch = RCPP_ARCH_BITNET;
    }
    out_model->is_qwen3 = (out_model->arch == RCPP_ARCH_QWEN3) ? 1 : 0;
    // Gate the sidecar-hydration pass at the bottom of this function on
    // *both* the weight format AND the resolved arch, so a BitNet-repacked
    // Bonsai .h1b reads its BitNet norms from the `.h1b` stream and never
    // touches a sidecar.
    const bool is_bonsai_qwen3 = is_bonsai_fmt && out_model->arch == RCPP_ARCH_QWEN3;

    if (version >= 2) {
        float extras[2] = {0.0f, 0.0f};
        f.read(reinterpret_cast<char*>(extras), sizeof(extras));
        out_model->rope_theta   = extras[0] > 0 ? extras[0] : 500000.0f;
        out_model->rms_norm_eps = extras[1] > 0 ? extras[1] : 1e-5f;
    } else {
        out_model->rope_theta   = 500000.0f;
        out_model->rms_norm_eps = 1e-5f;
    }
    if (sherry_fp16) {
        fprintf(stderr, "[rocm-cpp] .h1b v3 + SHERRY_FP16 flag — dispatching through fp16-in/fp16-out sherry_ternary_gemv_launch.\n");
    } else if (use_sherry) {
        fprintf(stderr, "[rocm-cpp] .h1b v3 (Sherry 1.25 bpw, int8-act) — dispatching ternary GEMVs through sherry decoder.\n");
    }
    if (use_tq1 && !is_bonsai_fmt) {
        fprintf(stderr, "[rocm-cpp] .h1b v4 (TQ1 base-3, 1.6 bpw, lossless) — dispatching through tq1-halo kernel.\n");
    }
    const char* arch_name = (out_model->arch == RCPP_ARCH_QWEN3) ? "Qwen3" : "BitNet";
    if (bonsai_tq2) {
        fprintf(stderr, "[rocm-cpp] .h1b v%d + BONSAI_TQ2 flag — dispatching through bonsai_tq2_gemv_launch; %s forward pass.\n",
                version, arch_name);
    }
    if (bonsai_q1) {
        fprintf(stderr, "[rocm-cpp] .h1b v%d + BONSAI_Q1  flag — dispatching through bonsai_q1_gemv_launch;  %s forward pass.\n",
                version, arch_name);
    }
    fprintf(stderr, "[rocm-cpp] .h1b v%d flags=0x%x: rope_theta=%.1f rms_norm_eps=%.1e\n",
            version, out_model->flags, out_model->rope_theta, out_model->rms_norm_eps);

    const int hs  = out_model->hidden_size;
    const int is_ = out_model->intermediate_size;
    const int nh  = out_model->num_heads;
    const int nkv = out_model->num_kv_heads;
    const int hd  = hs / nh;

    fprintf(stderr, "[rocm-cpp] loading .h1b: hs=%d is=%d L=%d nh=%d nkv=%d hd=%d vocab=%d\n",
            hs, is_, out_model->num_layers, nh, nkv, hd, out_model->vocab_size);

    // Embeddings + final norm — .h1b stores them as FP32.
    //
    // Bonsai-Qwen3 (oxibonsai converter) zero-fills both slots; we advance
    // past the zero bytes and pre-allocate empty device buffers that the
    // sidecar GGUF pass hydrates from the Qwen3 TQ2 embedding.
    //
    // Bonsai-BitNet (tools/bitnet-to-tq2/, MS-BitNet repack) writes *real*
    // fp32 embedding + final_norm payloads from the safetensors master —
    // same on-disk layout as the non-Bonsai path, so we read them directly.
    if (is_bonsai_qwen3) {
        f.seekg((std::streamoff)((size_t)out_model->vocab_size * hs * 4 + (size_t)hs * 4),
                std::ios::cur);
        LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&out_model->embedding_dev),
                              (size_t)out_model->vocab_size * hs * sizeof(_Float16)));
        LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&out_model->final_norm_weight_dev),
                              (size_t)hs * sizeof(_Float16)));
    } else {
        if (read_fp32_as_fp16(f, (size_t)out_model->vocab_size * hs,
                              reinterpret_cast<__half**>(&out_model->embedding_dev)) != 0) return RCPP_HIP_ERROR;
        if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&out_model->final_norm_weight_dev)) != 0) return RCPP_HIP_ERROR;
    }

    out_model->layers = static_cast<rcpp_bitnet_layer_t*>(
        std::calloc(out_model->num_layers, sizeof(rcpp_bitnet_layer_t)));
    if (!out_model->layers) return RCPP_INTERNAL;

    for (int l = 0; l < out_model->num_layers; ++l) {
        rcpp_bitnet_layer_t& L = out_model->layers[l];

        if (is_bonsai_qwen3) {
            // Skip the 9 zero-filled BitNet-shaped norm slots the oxibonsai
            // converter wrote. Sidecar pass below fills the ones the Qwen3
            // forward pass actually uses (input_norm, post_attn_norm,
            // attn_q/k_norm).
            f.seekg((std::streamoff)((size_t)hs * (2 + 4 + 2) * sizeof(float)
                                     + (size_t)is_ * sizeof(float)),
                    std::ios::cur);
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.input_norm_dev),
                                  (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMemset(L.input_norm_dev, 0, (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.post_attn_norm_dev),
                                  (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMemset(L.post_attn_norm_dev, 0, (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.attn_sub_norm_dev),
                                  (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMemset(L.attn_sub_norm_dev, 0, (size_t)hs * sizeof(_Float16)));
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.ffn_sub_norm_dev),
                                  (size_t)is_ * sizeof(_Float16)));
            LOAD_RC_HIP(hipMemset(L.ffn_sub_norm_dev, 0, (size_t)is_ * sizeof(_Float16)));
        } else {
            // Either classic BitNet (HALO_V2 / Sherry / TQ1) or
            // Bonsai-format + BitNet arch (tools/bitnet-to-tq2/ MS repack).
            // On-disk layout is identical — input_norm, post_attn_norm,
            // attn_sub_norm, then 3 duplicate attn_sub copies + 2 truncated
            // ffn_sub copies (historical filler) + ffn_sub_norm.
            if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.input_norm_dev))     != 0) return RCPP_HIP_ERROR;
            if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.post_attn_norm_dev)) != 0) return RCPP_HIP_ERROR;
            if (read_fp32_as_fp16(f, hs, reinterpret_cast<__half**>(&L.attn_sub_norm_dev))  != 0) return RCPP_HIP_ERROR;
            skip_fp32(f, hs);
            skip_fp32(f, hs);
            skip_fp32(f, hs);
            skip_fp32(f, hs);
            skip_fp32(f, hs);
            if (read_fp32_as_fp16(f, is_, reinterpret_cast<__half**>(&L.ffn_sub_norm_dev))  != 0) return RCPP_HIP_ERROR;
        }

        // 7 ternary linear layers: Q K V O gate up down.
        // Round-4: when mmap is active, derive device pointers in-place from
        // the unified-memory mapping. Otherwise legacy malloc+copy path.
        if (is_bonsai_fmt) {
            const int block_bytes = bonsai_tq2 ? 34 : 18;
            const int gs = 128;
            auto rb = [&](int rows, int cols, void** out) -> int {
                return mmap_active
                    ? read_bonsai_blocks_mmap(f, mmap_state, rows, cols,
                                              block_bytes, gs, out)
                    : read_bonsai_blocks(f, rows, cols, block_bytes, gs, out);
            };
            if (rb(nh * hd,  hs,    &L.q_packed_dev   ) != 0) return RCPP_HIP_ERROR;
            if (rb(nkv * hd, hs,    &L.k_packed_dev   ) != 0) return RCPP_HIP_ERROR;
            if (rb(nkv * hd, hs,    &L.v_packed_dev   ) != 0) return RCPP_HIP_ERROR;
            if (rb(hs,       nh*hd, &L.o_packed_dev   ) != 0) return RCPP_HIP_ERROR;
            if (rb(is_,      hs,    &L.gate_packed_dev) != 0) return RCPP_HIP_ERROR;
            if (rb(is_,      hs,    &L.up_packed_dev  ) != 0) return RCPP_HIP_ERROR;
            if (rb(hs,       is_,   &L.down_packed_dev) != 0) return RCPP_HIP_ERROR;
        } else {
            auto rt = [&](int rows, int cols, void** packed_out, void** scales_out) -> int {
                if (mmap_active) {
                    if (use_tq1)    return read_ternary_tq1_mmap(f, mmap_state, rows, cols, packed_out, scales_out);
                    if (use_sherry) return read_ternary_sherry_mmap(f, mmap_state, rows, cols, packed_out, scales_out);
                    return read_ternary_mmap(f, mmap_state, rows, cols, packed_out, scales_out);
                } else {
                    if (use_tq1)    return read_ternary_tq1(f, rows, cols, packed_out, scales_out);
                    if (use_sherry) return read_ternary_sherry(f, rows, cols, packed_out, scales_out);
                    return read_ternary(f, rows, cols, packed_out, scales_out);
                }
            };
            if (rt(nh * hd,  hs,    &L.q_packed_dev,    reinterpret_cast<void**>(&L.q_scales_dev))    != 0) return RCPP_HIP_ERROR;
            if (rt(nkv * hd, hs,    &L.k_packed_dev,    reinterpret_cast<void**>(&L.k_scales_dev))    != 0) return RCPP_HIP_ERROR;
            if (rt(nkv * hd, hs,    &L.v_packed_dev,    reinterpret_cast<void**>(&L.v_scales_dev))    != 0) return RCPP_HIP_ERROR;
            if (rt(hs,       nh*hd, &L.o_packed_dev,    reinterpret_cast<void**>(&L.o_scales_dev))    != 0) return RCPP_HIP_ERROR;
            if (rt(is_,      hs,    &L.gate_packed_dev, reinterpret_cast<void**>(&L.gate_scales_dev)) != 0) return RCPP_HIP_ERROR;
            if (rt(is_,      hs,    &L.up_packed_dev,   reinterpret_cast<void**>(&L.up_scales_dev))   != 0) return RCPP_HIP_ERROR;
            if (rt(hs,       is_,   &L.down_packed_dev, reinterpret_cast<void**>(&L.down_scales_dev)) != 0) return RCPP_HIP_ERROR;
        }
    }

    if (!out_model->tie_embeddings && !is_bonsai_fmt) {
        fprintf(stderr, "[rocm-cpp] WARN: untied LM head not supported in MVP loader\n");
    }

    f.close();

    // Hand the mmap state off to the side-table — at this point all weight
    // pointers in `out_model` have been derived and any subsequent
    // rcpp_bitnet_free will need the table to know which pointers it must
    // NOT hipFree. (Any RCPP_HIP_ERROR before this point unwinds via the
    // stack; the local MmapState destructor scope is the function body, so
    // a hipHostUnregister + munmap on early-return path would require an
    // RAII guard. The current implementation leaks the mapping on early
    // failure — acceptable because every error path here also leaks
    // hipMalloc'd buffers, and the next failing call typically aborts the
    // process. Future hardening: wrap MmapState in an RAII guard that
    // commits to the side-table on success.)
    if (mmap_active) {
        std::lock_guard<std::mutex> g(mmap_table_mu());
        mmap_table().emplace(out_model, std::move(mmap_state));
    }

    // -----------------------------------------------------------------------
    // Bonsai + Qwen3 sidecar — hydrate norms + embedding from the GGUF.
    // Bonsai + BitNet skips this block entirely (everything needed lives in
    // the `.h1b` already — the MS-BitNet repack writes real norms +
    // embedding fp32 payloads).
    // -----------------------------------------------------------------------
    if (is_bonsai_qwen3) {
        const std::string gguf_path = sidecar_path;
        if (gguf_path.empty()) {
            fprintf(stderr,
                "[rocm-cpp] bonsai(qwen3): NO sidecar GGUF path resolved for %s — "
                "norms + embedding stay zero.\n",
                path);
            out_model->tie_embeddings = 1;
            return RCPP_OK;
        }
        fprintf(stderr, "[rocm-cpp] bonsai(qwen3): sidecar GGUF = %s\n", gguf_path.c_str());

        GgufSidecar g;
        if (!g.open(gguf_path)) {
            fprintf(stderr, "[rocm-cpp] bonsai(qwen3): failed to parse sidecar GGUF\n");
            return RCPP_INVALID_ARG;
        }
        if (g.arch() != "qwen3") {
            fprintf(stderr, "[rocm-cpp] bonsai(qwen3): GGUF architecture=%s (expected qwen3)\n",
                    g.arch().c_str());
        }

        auto load_fp32_to_fp16_dev = [&](const std::string& name, size_t n,
                                         void* dev_out) -> bool {
            const auto* ti = g.info(name);
            if (!ti || ti->dtype != 0 /*F32*/) {
                fprintf(stderr, "[rocm-cpp][gguf] missing or non-F32 tensor: %s\n", name.c_str());
                return false;
            }
            std::vector<uint8_t> bytes;
            if (!g.read_tensor_bytes(name, n * sizeof(float), bytes)) {
                fprintf(stderr, "[rocm-cpp][gguf] short read on tensor: %s\n", name.c_str());
                return false;
            }
            const float* src = reinterpret_cast<const float*>(bytes.data());
            std::vector<_Float16> dst(n);
            for (size_t i = 0; i < n; ++i) dst[i] = (_Float16)src[i];
            if (hipMemcpy(dev_out, dst.data(), n * sizeof(_Float16),
                          hipMemcpyHostToDevice) != hipSuccess) {
                fprintf(stderr, "[rocm-cpp][gguf] hipMemcpy failed: %s\n", name.c_str());
                return false;
            }
            return true;
        };

        for (int l = 0; l < out_model->num_layers; ++l) {
            rcpp_bitnet_layer_t& L = out_model->layers[l];
            if (!load_fp32_to_fp16_dev(
                    "blk." + std::to_string(l) + ".attn_norm.weight",
                    (size_t)hs, L.input_norm_dev)) return RCPP_INVALID_ARG;
            if (!load_fp32_to_fp16_dev(
                    "blk." + std::to_string(l) + ".ffn_norm.weight",
                    (size_t)hs, L.post_attn_norm_dev)) return RCPP_INVALID_ARG;
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.attn_q_norm_dev),
                                  (size_t)hd * sizeof(_Float16)));
            LOAD_RC_HIP(hipMalloc(reinterpret_cast<void**>(&L.attn_k_norm_dev),
                                  (size_t)hd * sizeof(_Float16)));
            if (!load_fp32_to_fp16_dev(
                    "blk." + std::to_string(l) + ".attn_q_norm.weight",
                    (size_t)hd, L.attn_q_norm_dev)) return RCPP_INVALID_ARG;
            if (!load_fp32_to_fp16_dev(
                    "blk." + std::to_string(l) + ".attn_k_norm.weight",
                    (size_t)hd, L.attn_k_norm_dev)) return RCPP_INVALID_ARG;
        }

        if (!load_fp32_to_fp16_dev("output_norm.weight", (size_t)hs,
                                   out_model->final_norm_weight_dev))
            return RCPP_INVALID_ARG;

        // Token embedding — TQ2_0_g128 packed. Dequantize to FP16.
        {
            const std::string name = "token_embd.weight";
            const auto* ti = g.info(name);
            if (!ti) {
                fprintf(stderr, "[rocm-cpp][gguf] missing tensor: %s\n", name.c_str());
                return RCPP_INVALID_ARG;
            }
            if (ti->dtype != 42 /*TQ2_0_g128*/ && ti->dtype != 41 /*Q1_0_g128*/) {
                fprintf(stderr, "[rocm-cpp][gguf] token_embd.weight dtype=%u (expected 41/42)\n",
                        ti->dtype);
                return RCPP_INVALID_ARG;
            }
            if (ti->shape.size() != 2) {
                fprintf(stderr, "[rocm-cpp][gguf] token_embd not 2D\n");
                return RCPP_INVALID_ARG;
            }
            const size_t cols = ti->shape[0];
            const size_t rows = ti->shape[1];
            if ((int)cols != hs || (int)rows != out_model->vocab_size) {
                fprintf(stderr, "[rocm-cpp][gguf] token_embd shape [%zu,%zu] vs [%d,%d]\n",
                        cols, rows, hs, out_model->vocab_size);
                return RCPP_INVALID_ARG;
            }
            if (ti->dtype == 41) {
                fprintf(stderr, "[rocm-cpp][gguf] Q1_0_g128 token_embd not yet supported\n");
                return RCPP_UNSUPPORTED;
            }
            const size_t block_bytes = 34;
            const size_t gs = 128;
            const size_t blocks_per_row = cols / gs;
            const size_t row_bytes = blocks_per_row * block_bytes;
            std::vector<uint8_t> packed;
            if (!g.read_tensor_bytes(name, rows * row_bytes, packed)) {
                fprintf(stderr, "[rocm-cpp][gguf] short read on token_embd\n");
                return RCPP_INVALID_ARG;
            }
            std::vector<_Float16> fp16(rows * cols);
            dequantize_bonsai_tq2_to_fp16(packed.data(), rows, cols, fp16.data());
            if (hipMemcpy(out_model->embedding_dev, fp16.data(),
                          rows * cols * sizeof(_Float16),
                          hipMemcpyHostToDevice) != hipSuccess) {
                fprintf(stderr, "[rocm-cpp][gguf] hipMemcpy embedding failed\n");
                return RCPP_HIP_ERROR;
            }
        }

        out_model->tie_embeddings = 1;

        fprintf(stderr,
                "[rocm-cpp] bonsai sidecar hydrated: 4 × %d layer norms + output_norm + "
                "token_embd (TQ2 → fp16, %d × %d).\n",
                out_model->num_layers, out_model->vocab_size, hs);
    }

    return RCPP_OK;
}

extern "C" void
rcpp_bitnet_free(rcpp_bitnet_model_t* m) {
    if (!m) return;

    // Round-4: pointers that came from the mmap region must NOT be hipFree'd
    // (they're not hipMalloc'd — they're slices of a registered host
    // mapping). Look up the per-model state and use the contained
    // mmap_pointers set to discriminate.
    MmapState mmap_state;
    bool have_mmap = false;
    {
        std::lock_guard<std::mutex> g(mmap_table_mu());
        auto it = mmap_table().find(m);
        if (it != mmap_table().end()) {
            mmap_state = std::move(it->second);
            mmap_table().erase(it);
            have_mmap = true;
        }
    }

    auto is_mmap = [&](void* p) -> bool {
        return have_mmap && p && mmap_state.mmap_pointers.count(p) > 0;
    };
    auto f = [&](void* p) {
        if (p && !is_mmap(p)) (void)hipFree(p);
    };
    f(m->embedding_dev);
    f(m->final_norm_weight_dev);
    for (int l = 0; l < m->num_layers; ++l) {
        rcpp_bitnet_layer_t& L = m->layers[l];
        f(L.input_norm_dev);    f(L.post_attn_norm_dev);
        f(L.attn_sub_norm_dev); f(L.ffn_sub_norm_dev);
        f(L.attn_q_norm_dev);   f(L.attn_k_norm_dev);
        f(L.q_packed_dev);      f(L.q_scales_dev);
        f(L.k_packed_dev);      f(L.k_scales_dev);
        f(L.v_packed_dev);      f(L.v_scales_dev);
        f(L.o_packed_dev);      f(L.o_scales_dev);
        f(L.gate_packed_dev);   f(L.gate_scales_dev);
        f(L.up_packed_dev);     f(L.up_scales_dev);
        f(L.down_packed_dev);   f(L.down_scales_dev);
    }
    std::free(m->layers);
    std::memset(m, 0, sizeof(*m));

    // Tear down the mmap last — pointers above are no longer reachable.
    if (have_mmap) {
        if (mmap_state.registered && mmap_state.mmap_addr) {
            (void)hipHostUnregister(mmap_state.mmap_addr);
        }
        if (mmap_state.mmap_addr && mmap_state.mmap_len) {
            (void)::munmap(mmap_state.mmap_addr, mmap_state.mmap_len);
        }
    }
}
