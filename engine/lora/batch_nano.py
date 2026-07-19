#!/usr/bin/env python3
"""Train final 3 adapters consistently."""
import subprocess, time, os, sys

PY = os.path.expanduser("~/spec-decode/train-venv/bin/python")
SCRIPT = os.path.expanduser("~/1bit-systems/engine/lora/train_lora.py")
BASE = os.path.expanduser("~/1bit-systems/engine/lora/adapters")

os.environ["BNB_IGNORE_EXT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

adapters = [
    ("b-mc2/sql-create-context", 0.05, "SQL code generation"),
    ("rajpurkar/squad", 0.05, "Reading comprehension"),
    ("EdinburghNLP/xsum", 0.02, "Extreme summarization"),
]

for ds, split, desc in adapters:
    name = f"{ds.replace('/', '_')}-r8-lr3e-4-s42"
    out_dir = os.path.join(BASE, name)
    
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        print(f"SKIP {name}")
        continue
    
    print(f"TRAIN {name} — {desc}")
    sys.stdout.flush()
    
    result = subprocess.run(
        [PY, SCRIPT, "--dataset", ds, "--rank", "8", "--lr", "3e-4", "--seed", "42",
         "--target", "all", "--epochs", "1", "--split", str(split),
         "--batch_size", "16", "--grad_accum", "1", "--name", name],
        capture_output=True, text=True, timeout=300
    )
    
    if '"status": "completed"' in result.stdout:
        print(f"DONE {name} ✓")
    elif '"status": "skipped"' in result.stdout:
        print(f"DONE {name} — skipped")
    else:
        err = (result.stderr or "")[-200:]
        print(f"FAIL {name} ✗ {err.strip()[:100]}")

print("All done!")
