#!/usr/bin/env python3
"""Wrap norm weight loading with #ifndef ONEBP_SUPPORT."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "src/npu_engine_universal.cpp"

with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '// Norm weights' in line:
        for j in range(i, min(i+100, len(lines))):
            if 'Dequant+pack' in lines[j]:
                indent = lines[i][:len(lines[i])-len(lines[i].lstrip())]
                lines.insert(i, indent + '#ifndef ONEBP_SUPPORT\n')
                lines.insert(j+2, indent + '#endif\n')
                break
        break

with open(path, 'w') as f:
    f.writelines(lines)

print("Wrapped norm loading")
