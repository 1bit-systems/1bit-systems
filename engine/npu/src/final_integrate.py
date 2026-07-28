#!/usr/bin/env python3
"""Final 1BP integration — replaces Q4NX weight section with OnebpModel.
Usage: python3 final_integrate.py src/npu_engine_universal.cpp"""

import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

# Step 1: Add includes
old_inc = '#include "model_config.h"'
new_inc = old_inc + '\n#ifdef ONEBP_SUPPORT\n#include "onebp_format.h"\n#include "onebp_loader.cpp"\n#include "onebp_weight_loader.cpp"\n#endif'
content = content.replace(old_inc, new_inc, 1)

# Step 2: Add detection before Parse config
old_parse = '\n    // Parse config\n    ModelConfig cfg=parse_q4nx_header'
new_parse = ('\n#ifdef ONEBP_SUPPORT\n'
             '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;\n'
             '    OnebpModel onebp_model;\n'
             '#endif\n'
             '    // Parse config\n'
             '    ModelConfig cfg;\n'
             '#ifdef ONEBP_SUPPORT\n'
             '    if (is_onebp) {\n'
             '        if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: 1BP\\n"); return 1; }\n'
             '        auto& oh = onebp_model.header();\n'
             '        cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;\n'
             '        cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;\n'
             '        cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;\n'
             '        cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n'
             '        cfg.XM = 128; cfg.has_lm_head = true;\n'
             '    } else\n'
             '#endif\n'
             '    cfg = parse_q4nx_header(mp,model_tag.c_str());')
content = content.replace(old_parse, new_parse, 1)

# Step 3: Replace embedding loading
old_emb = '    emb_f32.resize((size_t)NV*H);\n    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);'
new_emb = ('#ifdef ONEBP_SUPPORT\n'
           '    if (is_onebp) {\n'
           '        std::vector<float> emb_buf;\n'
           '        if (onebp_model.get_tensor_f32("token_embd.weight", emb_buf)) {\n'
           '            emb_f32 = emb_buf;\n'
           '            fprintf(stderr,"  1BP embeddings loaded\\n");\n'
           '        }\n'
           '    } else\n'
           '#endif\n'
           '    emb_f32.resize((size_t)NV*H);\n'
           '    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);')
content = content.replace(old_emb, new_emb, 1)

# Step 4: Replace the entire Q4NX weight section with a 1BP-aware block
# From "// Norm weights" through the end of the weight loop (right before "// RoPE")
# For Q4NX: keep existing code
# For 1BP: call onebp_load_weights()
old_block_start = '    // Norm weights\n    std::vector<uint64_t> in_off'
old_block_end = '    }free(gw);free(uw);\n        int dr2,dc2;float*dw=dequant_i8_to_float_ex(i8p(dp[l]),d_i8,DIN,&dr2,&dc2);\n        std::vector<float>wd((size_t)DIN*DOUT);transpose_pack(dw,DOUT,DIN,wd.data(),DOUT,0);\n        cd.packB(l,wd.data(),DIN,DOUT,dsc[l]);free(dw);}\n    fprintf(stderr,"  %.0fms\\n\\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());'

new_block = ('#ifndef ONEBP_SUPPORT\n'
             '    // Norm weights\n'
             '    std::vector<uint64_t> in_off(NC),pa_off(NC),qn_off(NC),kn_off(NC),qp(NC),kp(NC),vp(NC),op(NC),gp(NC),up(NC),dp(NC);\n'
             '    char bn[128];\n'
             '    for(int l=0;l<NC;l++){\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.q_proj.weight",l);qp[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.k_proj.weight",l);kp[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.v_proj.weight",l);vp[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.o_proj.weight",l);op[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.mlp.gate_proj.weight",l);gp[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.mlp.up_proj.weight",l);up[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.mlp.down_proj.weight",l);dp[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.input_layernorm.weight",l);in_off[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.post_attention_layernorm.weight",l);pa_off[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.q_norm.weight",l);qn_off[l]=jo(js,jl,bn);\n'
             '        snprintf(bn,128,"model.layers.%d.self_attn.k_norm.weight",l);kn_off[l]=jo(js,jl,bn);}\n'
             '    uint64_t no=jo(js,jl,"model.norm.weight");\n'
             '    uint64_t lo=jo(js,jl,"lm_head.weight");\n'
             '    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));\n'
             '    std::vector<float> fin_v(H);\n'
             '    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);\n'
             '        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}\n'
             '        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);for(int i=0;i<HD;i++)qn_w[l][i]=bf16g(qq[i]);}\n'
             '        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);for(int i=0;i<HD;i++)kn_w[l][i]=bf16g(kk[i]);}}\n'
             '    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}\n'
             '\n'
             '    // I8 tile rows\n'
             '    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);return r;};\n'
             '    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight"),k_i8=gi8("model.layers.0.self_attn.k_proj.weight"),v_i8=gi8("model.layers.0.self_attn.v_proj.weight");\n'
             '    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight"),g_i8=gi8("model.layers.0.mlp.gate_proj.weight"),u_i8=gi8("model.layers.0.mlp.up_proj.weight"),d_i8=gi8("model.layers.0.mlp.down_proj.weight");\n'
             '    int lm_i8=gi8("lm_head.weight");\n'
             '\n'
             '    // Load lm_head.weight separately\n'
             '    if(lo&&lm_i8>0){int lr,lc;float*lm_raw=dequant_i8_to_float_ex(i8p(lo),lm_i8,H,&lr,&lc);if(lm_raw){\n'
             '        lm_head_f32.assign(lm_raw,lm_raw+(size_t)lr*lc);free(lm_raw);\n'
             '        fprintf(stderr,"  lm_head: %dx%d (loaded from JSON), using for final logits\\n",lr,lc);\n'
             '    }else{fprintf(stderr,"  lm_head: dequant failed, falling back to emb\\n");}}\n'
             '    if(lm_head_f32.empty()){fprintf(stderr,"  lm_head: using emb_f32 (tied embeddings)\\n");}\n'
             '\n'
             '    fprintf(stderr,"Dequant+pack...\\n");auto tp=std::chrono::steady_clock::now();\n'
             '    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);\n'
             '    const int QOUT=NH*HD,KVOUT=NKV*HD;\n'
             '    const int OOUT=H,OIN=NH*HD;\n'
             '    const int GUOUT=IM;\n'
             '    const int DOUT=H,DIN=IM;\n'
             '    for(int l=0;l<NC;l++){int qr,kr,vr,unused;\n'
             '        float*qw=dequant_i8_to_float_ex(i8p(qp[l]),q_i8,H,&qr,&unused),*kw=dequant_i8_to_float_ex(i8p(kp[l]),k_i8,H,&kr,&unused),*vw=dequant_i8_to_float_ex(i8p(vp[l]),v_i8,H,&vr,&unused);\n'
             '        int t=QOUT+KVOUT+KVOUT;std::vector<float>w((size_t)H*t);\n'
             '        transpose_pack(qw,QOUT,H,w.data(),t,0); transpose_pack(kw,KVOUT,H,w.data(),t,QOUT); transpose_pack(vw,KVOUT,H,w.data(),t,QOUT+KVOUT);\n'
             '        cq.packB(l,w.data(),H,t,qsc[l]);free(qw);free(kw);free(vw);\n'
             '        int or2,oc2;float*ow=dequant_i8_to_float_ex(i8p(op[l]),o_i8,OIN,&or2,&oc2);\n'
             '        std::vector<float>wo((size_t)OIN*OOUT);transpose_pack(ow,OOUT,OIN,wo.data(),OOUT,0);\n'
             '        co.packB(l,wo.data(),OIN,OOUT,osc[l]);free(ow);\n'
             '        int gr,ur;float*gw=dequant_i8_to_float_ex(i8p(gp[l]),g_i8,H,&gr,&unused),*uw=dequant_i8_to_float_ex(i8p(up[l]),u_i8,H,&ur,&unused);\n'
             '        if(cfg.gu_split){\n'
             '            std::vector<float>wg((size_t)H*gr);transpose_pack(gw,GUOUT,H,wg.data(),gr,0);\n'
             '            cg.packB(l,wg.data(),H,gr,gsc[l]);\n'
             '            std::vector<float>wu((size_t)H*ur);transpose_pack(uw,GUOUT,H,wu.data(),ur,0);\n'
             '        }else{\n'
             '            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);\n'
             '            transpose_pack(gw,GUOUT,H,w2.data(),t2,0);transpose_pack(uw,GUOUT,H,w2.data(),t2,GUOUT);\n'
             '            cg.packB(l,w2.data(),H,t2,gsc[l]);\n'
             '        }free(gw);free(uw);\n'
             '        int dr2,dc2;float*dw=dequant_i8_to_float_ex(i8p(dp[l]),d_i8,DIN,&dr2,&dc2);\n'
             '        std::vector<float>wd((size_t)DIN*DOUT);transpose_pack(dw,DOUT,DIN,wd.data(),DOUT,0);\n'
             '        cd.packB(l,wd.data(),DIN,DOUT,dsc[l]);free(dw);}\n'
             '    fprintf(stderr,"  %.0fms\\n\\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());\n'
             '#else\n'
             '    // 1BP: load weights directly via OnebpModel\n'
             '    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);\n'
             '    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));\n'
             '    std::vector<float> fin_v(H);\n'
             '    onebp_load_weights(onebp_model, NC, H, NH, NKV, HD, IM,\n'
             '        cq, co, cg, cd, qsc, osc, gsc, dsc,\n'
             '        in_n, pa_n, qn_w, kn_w, fin_v, emb_f32, lm_head_f32);\n'
             '#endif\n')

if old_block_start in content:
    start = content.find(old_block_start)
    end = content.find(old_block_end, start)
    if end > start:
        end += len(old_block_end)
        content = content[:start] + new_block + content[end:]
        print("Replaced Q4NX weight section")
    else:
        print("ERROR: Could not find end of weight block")
        sys.exit(1)
else:
    print("ERROR: Could not find start of weight block")
    # Debug: find what looks similar
    for i, line in enumerate(content.split('\n')):
        if 'Norm weights' in line:
            print(f"  Found 'Norm weights' at line {i+1}: {line[:60]}")
            break
    sys.exit(1)

with open(path, 'w') as f:
    f.write(content)

print(f"Done. Patched {path}")
print("Build with: g++ ... -DONEBP_SUPPORT src/npu_engine_universal.cpp src/dequant_q4nx.c src/onebp_weight_loader.cpp ...")
