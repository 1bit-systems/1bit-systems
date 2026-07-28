#!/usr/bin/env python3
"""Patch npu_engine_universal.cpp to add ONEBP_SUPPORT conditional compilation.
Run: python3 patch_1bp.py path/to/npu_engine_universal.cpp"""
import sys, os

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path or not os.path.exists(path):
    print("Usage: patch_1bp.py path/to/npu_engine_universal.cpp")
    sys.exit(1)

with open(path, 'r') as f:
    content = f.read()
    lines = content.split('\n')

changes = 0

# 1. Add includes after model_config.h
for i, line in enumerate(lines):
    if '#include "model_config.h"' in line:
        lines[i] = line
        insert = [
            '#ifdef ONEBP_SUPPORT',
            '#include "onebp_format.h"',
            '#include "onebp_loader.cpp"',
            '#endif',
        ]
        for j, ins in enumerate(insert):
            lines.insert(i + 1 + j, ins)
        changes += 1
        break

# 2. Add 1BP detection before "Parse config"
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == '// Parse config':
        next_line = lines[i+1] if i+1 < len(lines) else ''
        if 'ModelConfig' in next_line:
            insert = [
                '#ifdef ONEBP_SUPPORT',
                '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;',
                '    OnebpModel onebp_model;',
                '#endif',
            ]
            for j, ins in enumerate(insert):
                lines.insert(i, ins)
            changes += 1
            break

# 3. Replace the ModelConfig cfg=... line 
for i, line in enumerate(lines):
    if 'ModelConfig cfg=parse_q4nx_header' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_code = (
            indent + 'ModelConfig cfg;\n'
            + indent + '#ifdef ONEBP_SUPPORT\n'
            + indent + '    if (is_onebp) {\n'
            + indent + '        if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: cannot open 1BP\\n"); return 1; }\n'
            + indent + '        auto& oh = onebp_model.header();\n'
            + indent + '        cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;\n'
            + indent + '        cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;\n'
            + indent + '        cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;\n'
            + indent + '        cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n'
            + indent + '        cfg.rope_theta = oh.rope_theta(); cfg.model_tag = model_tag;\n'
            + indent + '        cfg.XM = 128; cfg.has_lm_head = true;\n'
            + indent + '    } else\n'
            + indent + '#endif\n'
            + indent + '    cfg = parse_q4nx_header(mp,model_tag.c_str());\n'
        )
        lines[i] = new_code
        changes += 1
        break

if changes == 0:
    print("ERROR: Could not find code to patch!")
    sys.exit(1)

# Flatten lines (some contain \n from multi-line inserts)
final = []
for line in lines:
    if '\n' in line:
        for sub in line.split('\n'):
            final.append(sub)
    else:
        final.append(line)

with open(path, 'w') as f:
    f.write('\n'.join(final))

print(f"Patched {path} ({changes} changes)")
print("Build with: cmake --build build -- -DCMAKE_CXX_FLAGS=-DONEBP_SUPPORT")
