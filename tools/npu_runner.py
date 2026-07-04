#!/usr/bin/env python3
"""Full NPU model runner using fused xclbin. Reads tokens from stdin, outputs tokens on stdout."""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/bcloud/torch2aie/examples/qwen3-decode-layer')
sys.path.insert(0, '/home/bcloud/torch2aie/toolchain/mlir_aie/python')

from qwen3_model import Qwen3Q4NXModel
from qwen3_8b_decode_layer_runner import _target_schedule, _build_schedule, _make_physical_fixture, build_kernel, _patch_capacity_instructions
from cases.qwen3_8b_decode_layer_reference import make_reference_inputs
import npu_build
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

MODEL_PATH = Path(os.environ.get('NPU_MODEL_PATH', '/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx'))

def main():
    # Read request
    line = sys.stdin.readline()
    if not line:
        return
    req = json.loads(line)
    prompt_tokens = req.get('tokens', [])
    max_new = req.get('max_new_tokens', 64)
    
    # Load model
    model_dir = MODEL_PATH.parent
    model = Qwen3Q4NXModel(model_dir)
    
    # Build kernel once (cached)
    target = _target_schedule(0)
    build = _build_schedule(target)
    xclbin_path, insts_path = build_kernel(build)
    runtime_insts_path, _ = _patch_capacity_instructions(insts_path, build, target)
    handle = npu_build.load_kernel(xclbin_path, runtime_insts_path)
    
    # Create reference inputs
    inputs = make_reference_inputs(0)
    
    # Pre-compute weight fixtures for all layers
    fixtures = [_make_physical_fixture(model, l, target, build) for l in range(28)]
    
    # Prefill: run each prompt token through all layers
    for pi, token_id in enumerate(prompt_tokens):
        # Set hidden state from embedding
        hidden = model.embed_tokens[token_id].copy().astype(np.float32)
        # For the fused xclbin, hidden goes through RMSNorm first (handled internally)
        # We pack hidden as i32
        hidden_i32 = (hidden.astype(np.float16).view(np.uint16).astype(np.uint32) 
                     + (hidden.astype(np.float16).view(np.uint16).astype(np.uint32) << 16))
        # ... this is complex. Let me use the runner's own fixtures.
        
        for l in range(28):
            # ... run layer l with fixture[l]
            pass
    
    # For now: quick smoke test using the runner's built-in test
    # The runner already validates against CPU oracle
    print(json.dumps({"tokens": [], "finished": True, "note": "runner integration pending"}))

if __name__ == '__main__':
    main()
