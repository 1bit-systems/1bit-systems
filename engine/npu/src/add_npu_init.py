#!/usr/bin/env python3
"""Add xrt device init + xclbin loading to the 1BP #else block."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

insert = """
    // Init NPU (shared between Q4NX and 1BP)
    fprintf(stderr,"Init NPU...\\n");
    xrt::device dev(0);
    const char* env_xd = getenv("NPU_XCLBIN_DIR");
    std::string xd = env_xd ? env_xd : "engine/npu/xclbins";
    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};
    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};
    if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){fprintf(stderr,"FAIL QKV\\n");return 1;}
    if(!co.init(dev,xp("O").c_str(),ip("O").c_str(),4,NC)){fprintf(stderr,"FAIL O\\n");return 1;}
    if(cfg.gu_split){if(!cg.init(dev,xp("G").c_str(),ip("G").c_str(),4,NC)){fprintf(stderr,"FAIL G\\n");return 1;}}else{if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){fprintf(stderr,"FAIL GU\\n");return 1;}}
    if(!cd.init(dev,xp("D").c_str(),ip("D").c_str(),4,NC)){fprintf(stderr,"FAIL D\\n");return 1;}
    if(cfg.gu_split){cu_ptr=std::make_unique<I8Ctx>();cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){fprintf(stderr,"FAIL U\\n");return 1;}}
    fprintf(stderr,"  xclbins loaded\\n");
"""

old = "    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;\n    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);"
new = "    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;" + insert + "    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);"

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Inserted NPU init OK")
else:
    print("Could not find insertion point")
    idx = content.find("cd.MD=XM;cd.KD=cfg.xclbin_d_k")
    if idx >= 0:
        print("Found at", idx)
        print(repr(content[idx:idx+250]))
