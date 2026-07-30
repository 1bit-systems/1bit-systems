/** dump_gguf_meta.cpp — Dump all GGUF metadata keys and values */
#include "gguf_reader.h"
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    GgufReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "Failed to open\n"); return 1; }
    printf("Architecture: %s\n", r.architecture().c_str());
    uint32_t vocab = 0; r.get_u32("vocab_size", vocab);
    printf("Vocab count: %u\n", vocab);
    printf("\nMetadata keys:\n");
    for (auto& k : r.kv_keys()) {
        uint32_t u = 0; float f = 0; std::string s;
        if (r.get_u32(k, u)) printf("  %s = %u (u32)\n", k.c_str(), u);
        else if (r.get_f32(k, f)) printf("  %s = %f (f32)\n", k.c_str(), f);
        else if (r.get_string(k, s)) printf("  %s = \"%s\" (str)\n", k.c_str(), s.c_str());
        else printf("  %s = (other)\n", k.c_str());
    }
    printf("\nTensors: %zu\n", r.tensor_names().size());
    for (auto& n : r.tensor_names()) {
        auto* ti = r.tensor_info(n);
        if (!ti) continue;
        printf("  %s: [", n.c_str());
        for (size_t d = 0; d < ti->shape.size(); d++)
            printf("%s%llu", d ? "," : "", (unsigned long long)ti->shape[d]);
        printf("]\n");
    }
    return 0;
}