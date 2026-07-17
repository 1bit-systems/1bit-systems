#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <cmath>

struct Tensor { int64_t offset, size; };
std::map<std::string, Tensor> g_tensors;

// Proper float32 → IEEE 754 fp16
static uint16_t f32_to_f16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint16_t s = (b >> 16) & 0x8000;
    int e = ((b >> 23) & 0xFF) - 127 + 15;
    uint16_t m = (b >> 13) & 0x03FF;
    if (e <= 0) return s;         // zero
    if (e >= 31) return s | 0x7C00 | (m ? 0x0200 : 0);  // inf/nan
    return s | (e << 10) | m;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.q4nx\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb"); if (!f) return 1;
    uint64_t hdr_size; fread(&hdr_size, 8, 1, f);
    std::string manifest(hdr_size, 0); fread(&manifest[0], 1, hdr_size, f);
    
    size_t p = 0;
    while ((p = manifest.find("\"model.", p)) != std::string::npos) {
        auto end = manifest.find('"', p + 1);
        auto name = manifest.substr(p + 1, end - p - 1);
        if (name.find('.') == std::string::npos) { p = end + 1; continue; }
        auto ok = manifest.find("\"data_offsets\"", end);
        auto br = manifest.find('[', ok);
        int64_t st = strtoll(&manifest[br+1], nullptr, 10);
        int64_t ev = strtoll(strchr(&manifest[br+1], ',')+1, nullptr, 10);
        g_tensors[name] = {st, ev - st};
        p = end + 1;
    }
    printf("Found %zu tensors\n", g_tensors.size());
    
    system("mkdir -p /tmp/zaya_fp16_cache");
    fseek(f, 0, SEEK_END); size_t fsz = ftell(f);
    std::vector<uint8_t> fd(fsz); fseek(f, 0, SEEK_SET); fread(fd.data(), 1, fsz, f);
    fclose(f);
    
    int64_t data_off = 8 + hdr_size, cnt = 0;
    for (auto& [name, t] : g_tensors) {
        const uint8_t* raw = fd.data() + data_off + t.offset;
        if (t.size % 5 != 0) continue;  // Skip BF16 tensors
        int groups = t.size / 5, elems = groups * 8;
        std::vector<uint16_t> fp16(elems);
        for (int g = 0; g < groups; g++) {
            const uint8_t* grp = raw + g * 5;
            uint32_t packed; memcpy(&packed, grp, 4);
            int8_t scale = (int8_t)grp[4];
            for (int i = 0; i < 8; i++) {
                int nibble = (packed >> (i*4)) & 0x0F;
                float val = (float)(nibble - 8) * (float)scale;
                fp16[g * 8 + i] = f32_to_f16(val);
            }
        }
        std::string out = "/tmp/zaya_fp16_cache/";
        for (char c : name) out += (c == '.') ? '_' : c;
        out += ".fp16";
        FILE* fo = fopen(out.c_str(), "wb");
        if (fo) { fwrite(fp16.data(), 2, elems, fo); fclose(fo); cnt++; }
        if (cnt % 100 == 0) printf("  %zu/%zu\n", cnt, g_tensors.size());
    }
    printf("✅ %zu tensors cached\n", cnt);
    return 0;
}
