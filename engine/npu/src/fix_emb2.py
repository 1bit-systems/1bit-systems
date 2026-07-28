#!/usr/bin/env python3
"""Fix the 1BP/else embedding block."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

old = (
    '    } else {\n'
    '#endif\n'
    '    emb_f32.resize((size_t)NV*H);\n'
    '    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);\n'
    '    fprintf(stderr,"  %.0fms\\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());\n'
    '\n'
    '#ifndef ONEBP_SUPPORT'
)

new = (
    '    } else {\n'
    '    emb_f32.resize((size_t)NV*H);\n'
    '    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);\n'
    '    fprintf(stderr,"  %.0fms\\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());\n'
    '    }\n'
    '#endif\n'
    '\n'
    '#ifndef ONEBP_SUPPORT'
)

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Fixed")
else:
    print("Not found")
