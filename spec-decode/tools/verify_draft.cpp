/**
 * verify_draft.cpp — Compare C++ MTPDraftModel output against Python reference.
 * Uses numpy-generated input for exact comparison.
 */
#include "draft/mtp_draft.h"
#include <cstdio>
#include <cmath>

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    
    MTPDraftConfig cfg;
    cfg.block_size = 5;  // Match Python test
    MTPDraftModel draft(cfg);

    const char* kCheckpoint = "checkpoints/eagle3_draft_npu_1k.bin";
    bool loaded = draft.load_weights(kCheckpoint);
    printf("Draft loaded: %s\n\n", loaded ? "YES" : "NO");
    if (!loaded) return 1;

    int H = cfg.hidden_size, V = cfg.vocab_size, BS = cfg.block_size;
    int NTL = cfg.num_target_layers;

    // Load numpy-generated input
    std::vector<float> target_feat(NTL * H);
    FILE* f = fopen("/tmp/py_target_feat.bin", "rb");
    if (!f) { printf("ERROR: run python first to generate /tmp/py_target_feat.bin\n"); return 1; }
    fread(target_feat.data(), sizeof(float), NTL * H, f);
    fclose(f);
    printf("Target input[:10]: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", target_feat[i]);
    printf("\n\n");

    int input_ids[5] = {100, 200, 300, 400, 500};

    std::vector<float> draft_hidden_step(H);
    MTPDraftState state;

    printf("C++ output:\n");
    for (int i = 0; i < BS; i++) {
        const float* draft_input = (i == 0) ? target_feat.data() : draft_hidden_step.data();
        std::vector<float> logits(V);
        
        draft.forward(draft_input, input_ids[i], i, state,
                      logits.data(), draft_hidden_step.data());

        // Find argmax
        int top_token = 0;
        float top_val = logits[0];
        for (int j = 1; j < V; j++) {
            if (logits[j] > top_val) { top_val = logits[j]; top_token = j; }
        }
        printf("pos=%d: top token=%d, logit=%.4f\n", i, top_token, top_val);
        printf("  hidden[:5]=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               draft_hidden_step[0], draft_hidden_step[1], draft_hidden_step[2],
               draft_hidden_step[3], draft_hidden_step[4]);
    }

    printf("\nPython reference (should match):\n");
    printf("  pos=0: top token=101736, logit=3.6447, hidden=[-0.3221, 0.1757, 0.6962, -0.2655, -2.4685]\n");
    printf("  pos=1: top token=109783, logit=2.8597, hidden=[-0.5295, -0.0187, 0.4946, -0.5449, -2.5511]\n");
    printf("  pos=2: top token=773, logit=2.2039, hidden=[-0.9223, -0.1416, 0.6424, -0.4327, -2.5475]\n");
    printf("  pos=3: top token=87, logit=5.9828, hidden=[-0.2743, -0.5558, 1.0427, -0.7128, -1.6578]\n");
    printf("  pos=4: top token=91, logit=3.5804\n");
    return 0;
}
