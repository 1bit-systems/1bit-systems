#!/usr/bin/env python3
"""Fix the #else block in npu_engine_universal.cpp for 1BP support."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

old_else = """#else
    // 1BP: load weights directly via OnebpModel
    std::unique_ptr<I8Ctx> cu_ptr;
    bool use_npu_attn = false;
    std::unique_ptr<AttnCtx> ca_ptr;
    std::vector<uint32_t> attn_instrs;
    I8Ctx cq,co,cg,cd;
    cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;
    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;
    cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;
    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));
    std::vector<float> fin_v(H);
    // 1BP: load weights directly via OnebpModel
cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;    cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
        in_n, pa_n, qn_w, kn_w, fin_v, emb_f32, lm_head_f32);
#endif"""

new_else = """#else
    // 1BP: load weights via OnebpModel
    std::unique_ptr<I8Ctx> cu_ptr;
    bool use_npu_attn = false;
    std::unique_ptr<AttnCtx> ca_ptr;
    std::vector<uint32_t> attn_instrs;
    I8Ctx cq,co,cg,cd;
    cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;
    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;
    cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;
    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));
    std::vector<float> fin_v(H);
    const float* lm_emb = lm_head_f32.empty() ? emb_f32.data() : lm_head_f32.data();
#endif"""

if old_else in content:
    content = content.replace(old_else, new_else, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Fixed #else block")
else:
    print("ERROR: Could not find old block")
    idx = content.find("#else")
    if idx >= 0:
        print("Found #else at position", idx)
        print(repr(content[idx:idx+500]))
