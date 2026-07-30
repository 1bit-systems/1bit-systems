#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <sys/stat.h>
#include "onebp_format.h"
#include "onebp_loader.cpp"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.1bp [iter]\n", argv[0]); return 1; }
    int num_iter = (argc > 2) ? atoi(argv[2]) : 3;
    
    // Get file size
    struct stat st;
    stat(argv[1], &st);
    double file_gb = st.st_size / 1e9;
    
    OnebpModel model;
    if (!model.open(argv[1])) { fprintf(stderr, "FAIL\n"); return 1; }
    auto& h = model.header();
    printf("Benchmark: %s\n", argv[1]);
    printf("  Architecture: %s  H=%d L=%d V=%d\n",
           h.model_tag, h.hidden_size, h.num_layers, h.vocab_size);
    printf("  Quant: %s  Tensors: %d  Size: %.2f GB\n",
           h.quant == 0 ? "Q4NX" : h.quant == 3 ? "TQ2" : "?",
           h.tensor_count, file_gb);
    
    // Dequantization benchmark
    printf("\n=== Dequantization ===\n");
    double total_mb = 0, total_t = 0;
    for (int it = 0; it < num_iter; it++) {
        double t = 0, mb = 0;
        for (int i = 0; i < model.tensor_count(); i++) {
            auto* te = model.tensor(i);
            if (!te) continue;
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> data;
            if (model.get_tensor_f32(te->name.c_str(), data)) {
                auto end = std::chrono::high_resolution_clock::now();
                t += std::chrono::duration<double>(end - start).count();
                mb += data.size() * 4.0 / (1024*1024);
            }
        }
        total_mb += mb; total_t += t;
        if (it == 0) printf("  Run: %.0f MB in %.3fs = %.0f MB/s\n", mb, t, mb/t);
    }
    if (num_iter > 1) {
        double avg_mb = total_mb/num_iter, avg_t = total_t/num_iter;
        printf("  Avg:  %.0f MB in %.3fs = %.0f MB/s\n", avg_mb, avg_t, avg_mb/avg_t);
    }
    
    printf("\nDone.\n");
    return 0;
}
