#!/usr/bin/env python3
"""Full 1BP integration patch for npu_engine_universal.cpp."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    lines = f.readlines()

# Step 1: Add includes
for i, line in enumerate(lines):
    if '#include "model_config.h"' in line and not any('onebp' in l for l in lines[i:i+5]):
        for j, ins in enumerate(['#ifdef ONEBP_SUPPORT\n', '#include "onebp_format.h"\n', '#include "onebp_loader.cpp"\n', '#endif\n']):
            lines.insert(i+1+j, ins)
        break

# Step 2: Add detection before Parse config
for i, line in enumerate(lines):
    if line.strip() == '// Parse config' and not any('is_onebp' in l for l in lines[i-3:i+3]):
        for j, ins in enumerate(['#endif\n', '    OnebpModel onebp_model;\n', '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;\n', '#ifdef ONEBP_SUPPORT\n']):
            lines.insert(i, ins)
        break

# Step 3: Replace ModelConfig line
for i, line in enumerate(lines):
    already = (i > 0 and 'is_onebp' in lines[i-1])
    if 'ModelConfig cfg=parse_q4nx_header' in line and not already:
        ind = line[:len(line)-len(line.lstrip())]
        lines[i] = (ind + 'ModelConfig cfg;\n'
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

# Step 4: 1BP embedding loading
for i, line in enumerate(lines):
    if 'emb_f32.resize((size_t)NV*H)' in line:
        ind = line[:len(line)-len(line.lstrip())]
        ins = [
            ind + '#ifdef ONEBP_SUPPORT\n',
            ind + 'if (is_onebp) {\n',
            ind + '    std::vector<float> emb_buf;\n',
            ind + '    if (onebp_model.get_tensor_f32("token_embd.weight", emb_buf)) {\n',
            ind + '        emb_f32 = emb_buf;\n',
            ind + '        fprintf(stderr,"  1BP embeddings loaded\\n");\n',
            ind + '    }\n',
            ind + '} else\n',
            ind + '#endif\n',
        ]
        for j, l in enumerate(reversed(ins)):
            lines.insert(i, l)
        break

# Step 5: Wrap Q4NX-specific section (Norm through weight loop)
for i, line in enumerate(lines):
    if line.strip() == '// Norm weights':
        for j in range(i, min(i+200, len(lines))):
            if 'Dequant+pack' in line and lines[j].strip().startswith('fprintf'):
                indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
                lines.insert(i, indent + '#ifndef ONEBP_SUPPORT\n')
                lines.insert(j+2, indent + '#endif\n')
                break
        break

with open(path, 'w') as f:
    f.writelines(lines)

# Verify nesting
d = 0
for line in lines:
    if line.strip() == '#ifdef ONEBP_SUPPORT': d += 1
    if line.strip() == '#endif': d -= 1
    if line.strip() == '#ifndef ONEBP_SUPPORT': d += 1
status = "OK" if d == 0 else "FAIL"
print(f"Depth: {d} {status}")
