// onebp_loader.cpp — 1BP format model loader/verifier (standalone tool)
#include "onebp_loader.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.1bp> [tensor_count]\n", argv[0]);
        return 1;
    }
    OnebpModel model;
    if (!model.load(argv[1])) return 1;

    auto& h = model.header;
    printf("[1BP] %s\n", argv[1]);
    printf("  Arch=%u H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",
           h.arch, h.hidden_size, h.num_layers,
           h.num_attention_heads, h.num_kv_heads,
           h.head_dim, h.intermediate_size, h.vocab_size);
    printf("  Quant=%u Experts=%d/%d SW=%d/%d\n",
           h.quant, h.num_experts, h.n_expert_used,
           h.sliding_window, h.swa_period);
    printf("  Tensors: %zu, File: %.1f MB\n", model.tensors.size(), model.file_size / 1e6);

    int show = argc > 2 ? atoi(argv[2]) : (int)model.tensors.size();
    show = std::min(show, (int)model.tensors.size());
    printf("\nTensors (%d shown):\n", show);
    for (int i = 0; i < show; i++) {
        auto& t = model.tensors[i];
        printf("  [%d] %s: ndim=%d dims=[", i, t.name.c_str(), t.ndim);
        for (int d = 0; d < t.ndim; d++)
            printf("%s%u", d ? "," : "", t.dims[d]);
        printf("] off=%lu bytes=%lu\n", t.offset, t.bytes);
    }

    int n_moe = 0, n_sh = 0, n_dn = 0;
    for (auto& t : model.tensors) {
        if (t.ndim == 3 && t.name.find("_exp") != std::string::npos) n_moe++;
        if (t.name.find("_shexp") != std::string::npos) n_sh++;
        if (t.ndim == 2 && t.name.find("ffn_gate") != std::string::npos
            && t.name.find("_exp") == std::string::npos) n_dn++;
    }
    printf("\nStats: MoE=%d shared=%d dense=%d | Laguna=%s\n",
           n_moe, n_sh, n_dn, h.arch == 6 ? "yes" : "no");
    return 0;
}
