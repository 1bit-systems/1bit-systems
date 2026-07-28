#!/usr/bin/env python3
"""Patch npu_engine_universal.cpp with ONEBP_SUPPORT guards."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '#include "model_config.h"' in line and not any('onebp' in l for l in lines[i:i+5]):
        for j, ins in enumerate(['#ifdef ONEBP_SUPPORT\n', '#include "onebp_format.h"\n', '#include "onebp_loader.cpp"\n', '#endif\n']):
            lines.insert(i+1+j, ins)
        break

for i, line in enumerate(lines):
    if line.strip() == '// Parse config' and not any('is_onebp' in l for l in lines[i-3:i+3]):
        for j, ins in enumerate(['#endif\n', '    OnebpModel onebp_model;\n', '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;\n', '#ifdef ONEBP_SUPPORT\n']):
            lines.insert(i, ins)
        break

for i, line in enumerate(lines):
    already = (i > 0 and 'is_onebp' in lines[i-1])
    if 'ModelConfig cfg=parse_q4nx_header' in line and not already:
        ind = line[:len(line)-len(line.lstrip())]
        lines[i] = (
            ind + 'ModelConfig cfg;\n'
            + ind + '#ifdef ONEBP_SUPPORT\n'
            + ind + '    if (is_onebp) {\n'
            + ind + '        if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: 1BP\\n"); return 1; }\n'
            + ind + '        auto& oh = onebp_model.header();\n'
            + ind + '        cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;\n'
            + ind + '        cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;\n'
            + ind + '        cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;\n'
            + ind + '        cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n'
            + ind + '        cfg.XM = 128; cfg.has_lm_head = true;\n'
            + ind + '    } else\n'
            + ind + '#endif\n'
            + ind + '    cfg = parse_q4nx_header(mp,model_tag.c_str());\n')
        break

with open(path, 'w') as f:
    f.writelines(lines)

d = 0
for line in lines:
    if line.strip() == '#ifdef ONEBP_SUPPORT': d += 1
    if line.strip() == '#endif': d -= 1
ok = 'OK' if d == 0 else 'FAIL'
print(f'Depth: {d} {ok}')
