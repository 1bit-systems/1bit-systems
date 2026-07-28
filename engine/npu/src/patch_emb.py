#!/usr/bin/env python3
"""Patch embedding loading for 1BP support."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()

# Add 1BP embedding loading before the Q4NX version
old = '    emb_f32.resize((size_t)NV*H);\n    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);'

new_indent = '    '
new = (
    new_indent + '#ifdef ONEBP_SUPPORT\n'
    + new_indent + 'if (is_onebp) {\n'
    + new_indent + '    std::vector<float> emb_buf;\n'
    + new_indent + '    if (onebp_model.get_tensor_f32("token_embd.weight", emb_buf)) {\n'
    + new_indent + '        emb_f32 = emb_buf;\n'
    + new_indent + '        fprintf(stderr,"  1BP embeddings loaded\\n");\n'
    + new_indent + '    }\n'
    + new_indent + '} else\n'
    + new_indent + '#endif\n'
    + old
)

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Patched embedding loading")
else:
    print("ERROR: Could not find embedding loading code")
    # Find what's there
    for i, line in enumerate(open(path).readlines()):
        if 'emb_f32.resize' in line:
            print(f"  Found at line {i+1}: {line.rstrip()}")
            print(f"  Next: {open(path).readlines()[i+1].rstrip()}")
            break
