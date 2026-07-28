#!/usr/bin/env python3
"""Fix the 1BP embedding else block — move #endif to after the Q4NX code."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

# Fix: close the else block properly
old_block = """    } else {
#endif
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());

#ifndef ONEBP_SUPPORT"""

new_block = """    } else {
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());
    }
#endif

#ifndef ONEBP_SUPPORT"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Fixed embedding else block")
else:
    print("Could not find block")
    idx = content.find("emb_f32.resize")
    if idx >= 0:
        print("Found emb_f32 at", idx)
        print(repr(content[idx-100:idx+200]))
