#!/usr/bin/env python3
"""onebp_integrate.py — Complete 1BP integration for npu_engine_universal.cpp.
Run: python3 onebp_integrate.py src/npu_engine_universal.cpp
Then: g++ ... -DONEBP_SUPPORT ... -o build/npu_engine_1bp ..."""

import sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()
    lines = content.split('\n')

# ─── 1. Add includes after model_config.h ───
for i, line in enumerate(lines):
    if '#include "model_config.h"' in line:
        if any('onebp' in l for l in lines[i:i+5]):
            print("Already patched. Run: git checkout -- " + path)
            sys.exit(0)
        lines.insert(i+1, '#ifdef ONEBP_SUPPORT')
        lines.insert(i+2, '#include "onebp_format.h"')
        lines.insert(i+3, '#include "onebp_loader.cpp"')
        lines.insert(i+4, '#include "onebp_weight_loader.cpp"')
        lines.insert(i+5, '#endif')
        break

# ─── 2. Add 1BP detection before Parse config ───
for i, line in enumerate(lines):
    if line.strip() == '// Parse config' and 'ModelConfig' in lines[i+1] if i+1 < len(lines) else '':
        lines.insert(i, '#ifdef ONEBP_SUPPORT')
        lines.insert(i+1, '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;')
        lines.insert(i+2, '    OnebpModel onebp_model;')
        lines.insert(i+3, '#endif')
        break

# ─── 3. Conditional ModelConfig ───
for i, line in enumerate(lines):
    if 'ModelConfig cfg=parse_q4nx_header' in line:
        indent = line[:len(line)-len(line.lstrip())]
        lines[i] = (
            indent + 'ModelConfig cfg;\n'
            + indent + '#ifdef ONEBP_SUPPORT\n'
            + indent + '    if (is_onebp) {\n'
            + indent + '        if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: 1BP\\n"); return 1; }\n'
            + indent + '        auto& oh = onebp_model.header();\n'
            + indent + '        cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;\n'
            + indent + '        cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;\n'
            + indent + '        cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;\n'
            + indent + '        cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n'
            + indent + '        cfg.XM = 128; cfg.has_lm_head = true;\n'
            + indent + '    } else\n'
            + indent + '#endif\n'
            + indent + '    cfg = parse_q4nx_header(mp,model_tag.c_str());\n')
        break

# ─── 4. 1BP embedding loading ───
for i, line in enumerate(lines):
    if 'emb_f32.resize((size_t)NV*H)' in line:
        # Check the surrounding context
        prev_5 = lines[i-5:i] if i >= 5 else lines[:i]
        prev_text = '\n'.join(prev_5)
        
        # Determine if the else is already there
        has_else = any('} else' in l for l in prev_5)
        
        if not has_else:
            # Add 1BP path before the Q4NX embedding code
            indent = line[:len(line)-len(line.lstrip())]
            ins = [
                indent + '#ifdef ONEBP_SUPPORT',
                indent + 'if (is_onebp) {',
                indent + '    std::vector<float> emb_buf;',
                indent + '    if (onebp_model.get_tensor_f32("token_embd.weight", emb_buf)) {',
                indent + '        emb_f32 = emb_buf;',
                indent + '        fprintf(stderr,"  1BP embeddings loaded\\n");',
                indent + '    }',
                indent + '} else {',
                indent + '#endif',
            ]
            for j, l in enumerate(reversed(ins)):
                lines.insert(i, l)
            
            # Add closing brace after the Q4NX embedding loop
            # Find the fprintf after the embedding loop
            for j in range(i+10, min(i+20, len(lines))):
                if 'fprintf(stderr,"  %.0fms' in lines[j]:
                    lines.insert(j+1, indent + '}')
                    break
        break

# ─── 5. Add 1BP NPU init + weight loading ───
# Find where the Q4NX "Dequant+pack" section starts and insert the 1BP equivalent
for i, line in enumerate(lines):
    if 'fprintf(stderr,"Dequant+pack' in line:
        # Check if we're in the #ifndef section or not
        if any('#ifndef' in l for l in lines[max(0,i-10):i]):
            indent = line[:len(line)-len(line.lstrip())]
            # Insert 1BP NPU init + weight loading before "Dequant+pack"
            onebp_block = [
                indent + '// 1BP path: NPU init + weight loading',
                indent + '#ifdef ONEBP_SUPPORT',
                indent + 'if (is_onebp) {',
                indent + '    fprintf(stderr,"Init NPU...\\n");',
                indent + '    xrt::device dev(0);',
                indent + '    const char* env_xd = getenv("NPU_XCLBIN_DIR");',
                indent + '    std::string xd = env_xd ? env_xd : "engine/npu/xclbins";',
                indent + '    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};',
                indent + '    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};',
                indent + '    I8Ctx cq,co,cg,cd;',
                indent + '    std::unique_ptr<I8Ctx> cu_ptr;',
                indent + '    bool use_npu_attn = false;',
                indent + '    std::unique_ptr<AttnCtx> ca_ptr;',
                indent + '    std::vector<uint32_t> attn_instrs;',
                indent + '    cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;',
                indent + '    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;',
                indent + '    cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;',
                indent + '    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;',
                indent + '    if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){fprintf(stderr,"FAIL QKV\\n");return 1;}',
                indent + '    if(!co.init(dev,xp("O").c_str(),ip("O").c_str(),4,NC)){fprintf(stderr,"FAIL O\\n");return 1;}',
                indent + '    if(cfg.gu_split){if(!cg.init(dev,xp("G").c_str(),ip("G").c_str(),4,NC)){fprintf(stderr,"FAIL G\\n");return 1;}}else{if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){fprintf(stderr,"FAIL GU\\n");return 1;}}',
                indent + '    if(!cd.init(dev,xp("D").c_str(),ip("D").c_str(),4,NC)){fprintf(stderr,"FAIL D\\n");return 1;}',
                indent + '    if(cfg.gu_split){cu_ptr=std::make_unique<I8Ctx>();cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){fprintf(stderr,"FAIL U\\n");return 1;}}',
                indent + '    fprintf(stderr,"  xclbins loaded\\n");',
                indent + '',
                indent + '    // 1BP weight loading',
                indent + '    onebp_load_weights(onebp_model, NC, H, NH, NKV, HD, IM,',
                indent + '        cq, co, cg, cd, qsc, osc, gsc, dsc,',
                indent + '        in_n, pa_n, qn_w, kn_w, fin_v, emb_f32, lm_head_f32);',
                indent + '    const float* lm_emb = lm_head_f32.empty() ? emb_f32.data() : lm_head_f32.data();',
                indent + '} else',
                indent + '#endif',
            ]
            for j, l in enumerate(reversed(onebp_block)):
                lines.insert(i, l)
        break

# ─── Write back ───
with open(path, 'w') as f:
    f.write('\n'.join(lines))

print(f"Patched {path}")
print("")
print("Build with:")
print("  cd " + os.path.dirname(os.path.abspath(path)))
print("  g++ -std=c++17 -O3 -mavx2 -march=native -DONEBP_SUPPORT \\")
print("      -I src -I include -I ../.. -I /usr/include -I ../../include \\")
print("      -DXRT_ENABLE \\")
print("      src/npu_engine_universal.cpp src/dequant_q4nx.c \\")
print("      -o build/npu_engine_1bp \\")
print("      -lxrt_coreutil -lxrt_core -luuid -ldl -fopenmp -laiebu")
