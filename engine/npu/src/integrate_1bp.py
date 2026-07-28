#!/usr/bin/env python3
"""Integrate 1BP support into npu_engine_universal.cpp"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    content = f.read()
    lines = content.split('\n')

if any('onebp' in l.lower() for l in lines[:30]):
    print("Already patched")
    sys.exit(0)

# 1. Add includes
for i, line in enumerate(lines):
    if '#include "model_config.h"' in line:
        lines.insert(i+1, '#ifdef ONEBP_SUPPORT')
        lines.insert(i+2, '#include "onebp_format.h"')
        lines.insert(i+3, '#include "onebp_loader.cpp"')
        lines.insert(i+4, '#endif')
        break

# 2. 1BP detection before Parse config
for i, line in enumerate(lines):
    if line.strip() == '// Parse config' and 'ModelConfig' in lines[i+1]:
        lines.insert(i, '#ifdef ONEBP_SUPPORT')
        lines.insert(i+1, '    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;')
        lines.insert(i+2, '    OnebpModel onebp_model;')
        lines.insert(i+3, '#endif')
        break

# 3. Conditional ModelConfig
for i, line in enumerate(lines):
    if 'ModelConfig cfg=parse_q4nx_header' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new = (indent + 'ModelConfig cfg;\n'
               + indent + '#ifdef ONEBP_SUPPORT\n'
               + indent + '    if (is_onebp) {\n'
               + indent + '        if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: 1BP\\n"); return 1; }\n'
               + indent + '        auto& oh = onebp_model.header();\n'
               + indent + '        cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;\n'
               + indent + '        cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;\n'
               + indent + '        cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;\n'
               + indent + '        cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;\n'
               + indent + '        cfg.rope_theta = oh.rope_theta(); cfg.model_tag = model_tag;\n'
               + indent + '        cfg.XM = 128; cfg.has_lm_head = true;\n'
               + indent + '    } else\n'
               + indent + '#endif\n'
               + indent + '    cfg = parse_q4nx_header(mp,model_tag.c_str());\n')
        lines[i] = new
        break

# 4. Wrap mmap with #ifndef
for i, line in enumerate(lines):
    if 'memcpy(&hsz,md,8);uint64_t df=8+hsz' in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i, indent + '    uint64_t hsz=0;uint64_t df=0;')
        lines.insert(i, indent + '#endif')
        lines.insert(i, indent + '#ifndef ONEBP_SUPPORT')
        changes = ['includes', 'detection', 'config', 'mmap']
        break

# 5. Add dequant wrapper before Dequant+pack
for i, line in enumerate(lines):
    if 'Dequant+pack' in line and 'fprintf' in line:
        indent = line[:len(line) - len(line.lstrip())]
        wrapper = (
            indent + '#ifdef ONEBP_SUPPORT\n'
            + indent + '// 1BP weight loader: wraps dequant_or_1bp\n'
            + indent + 'auto* deq_or_1bp = [&](uint64_t off, int i8b, int fin, const char* tn, int* ro, int* co) -> float* {\n'
            + indent + '    if (is_onebp) {\n'
            + indent + '        static std::vector<float> buf;\n'
            + indent + '        if (onebp_model.get_tensor_f32(tn, buf)) {\n'
            + indent + '            *ro = fin; *co = buf.size() / fin; return buf.data();\n'
            + indent + '        }\n'
            + indent + '        fprintf(stderr,"WARN: 1BP missing %s\\n",tn);\n'
            + indent + '    }\n'
            + indent + '    return dequant_i8_to_float_ex(i8p(off), i8b, fin, ro, co);\n'
            + indent + '};\n'
            + indent + '#endif\n'
        )
        lines.insert(i, wrapper)
        changes.append('wrapper')
        break

# 6. Replace dequant_i8_to_float_ex calls in weight loop
# The pattern is: dequant_i8_to_float_ex(i8p(XXX),YYY,H,...)
# Replace with: deq_or_1bp(XXX,YYY,H,"tensor_name",...)
# But we need the tensor names which vary by layer. Skip this for now.

with open(path, 'w') as f:
    f.write('\n'.join(lines))

print(f"Patched: {', '.join(changes)}")
print("Now run: sed to replace dequant_i8_to_float_ex calls, then build with -DONEBP_SUPPORT")
