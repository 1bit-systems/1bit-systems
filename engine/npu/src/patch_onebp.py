#!/usr/bin/env python3
"""Patches npu_engine_universal.cpp to support 1BP format loading."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

if '.1bp' in content:
    print("Already patched")
    sys.exit(0)

# The em-dash character in the source
emdash = '\xe2\x80\x94'

old_block = '    // Parse config\n    ModelConfig cfg=parse_q4nx_header(mp,model_tag.c_str());\n    if(!cfg.valid()){fprintf(stderr,"ERR: invalid model config\\n");return 1;}\n    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;\n    fprintf(stderr,"=== NPU Engine Universal ' + emdash + ' %s ===\\n",model_tag.c_str());\n    fprintf(stderr,"H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GU_split=%d\\n",H,NC,NH,NKV,HD,IM,NV,cfg.gu_split);\n\n    // Open model\n    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);\n    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);\n    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;'

new_block = '    // Parse config (1BP or Q4NX)\n    ModelConfig cfg;\n    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;\n    OnebpModel obm;\n    if (is_onebp) {\n        if (!obm.open(mp)) { fprintf(stderr,"ERR: 1BP open fail\\n"); return 1; }\n        auto& h = obm.header();\n        cfg.H = h.hidden_size; cfg.NC = h.num_layers;\n        cfg.NH = h.num_attention_heads; cfg.NKV = h.num_kv_heads;\n        cfg.HD = h.head_dim; cfg.IM = h.intermediate_size;\n        cfg.NV = h.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n        cfg.rope_theta = h.rope_theta(); cfg.model_tag = model_tag; cfg.XM = 128;\n    } else {\n        cfg = parse_q4nx_header(mp,model_tag.c_str());\n    }\n    if(!cfg.valid()){fprintf(stderr,"ERR: invalid model config\\n");return 1;}\n    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;\n    fprintf(stderr,"=== NPU Engine Universal ' + emdash + ' %s ===\\n",model_tag.c_str());\n    fprintf(stderr,"H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GU_split=%d\\n",H,NC,NH,NKV,HD,IM,NV,cfg.gu_split);\n\n    // Open model (Q4NX mmap or 1BP loader)\n    int fd=-1; uint8_t* md=nullptr; struct stat st; uint64_t hsz=0, df=0;\n    if (!is_onebp) {\n        fd=open(mp,O_RDONLY);fstat(fd,&st);\n        md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);\n        memcpy(&hsz,md,8);df=8+hsz;\n    }'

if old_block not in content:
    print("ERROR: Block not found!")
    print("Looking around 'Parse config':")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Parse config' in line:
            for j in range(i, min(i+12, len(lines))):
                print(f"  [{j+1}] {lines[j]}")
            break
    sys.exit(1)

content = content.replace(old_block, new_block)

# Add includes
content = content.replace(
    '#include "model_config.h"',
    '#include "model_config.h"\n#include "onebp_format.h"\n#include "onebp_loader.cpp"'
)

with open(path, 'w') as f:
    f.write(content)

print("Patched successfully")
