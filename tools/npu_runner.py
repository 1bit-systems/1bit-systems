#!/usr/bin/env python3
"""Full NPU model runner using fused xclbin. Reads JSON from stdin, outputs tokens on stdout."""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/bcloud/torch2aie/examples/qwen3-decode-layer')
sys.path.insert(0, '/home/bcloud/torch2aie/examples/qwen3-decode-layer/cases')
sys.path.insert(0, '/home/bcloud/torch2aie/toolchain/mlir_aie/python')

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import npu_build
from qwen3_model import Qwen3Q4NXModel

H, NC, NV = 1024, 28, 151936
EPS = 1e-6
XCLBIN = Path('/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127/design.xclbin')
INSTS = Path('/home/bcloud/torch2aie/examples/qwen3-decode-layer/build/qwen3-decode-layer-capacity-token127/design-token127-to-token0.bin')
MODEL_DIR = Path(os.environ.get('NPU_MODEL_DIR', os.path.expanduser('~/.config/flm/models/Qwen3-0.6B-NPU2')))
WEIGHT_DIR = Path('/tmp/npu_layer_weights')

class NPURunner:
    def __init__(self):
        print("  Loading model...", flush=True)
        self.model = Qwen3Q4NXModel(MODEL_DIR)
        
        print("  Loading kernel...", flush=True)
        self.handle = npu_build.load_kernel(XCLBIN, INSTS)
        
        print("  Loading weights...", flush=True)
        self.weights = [np.fromfile(WEIGHT_DIR / f'layer_{l}.bin', dtype=np.int32) for l in range(NC)]
        
        # Embeddings
        emb_bf16 = self.model.tensor_bf16("model.embed_tokens.weight")
        self.emb = emb_bf16.astype(np.float32)
        
        # LM head via q4nx_reference
        sys.path.insert(0, '/home/bcloud/1bit-systems')
        from tools.q4nx_reference import dequantize_weight, get_header
        h = get_header()
        if "lm_head.weight" in h:
            lm_info = h["lm_head.weight"]
            lm_raw = dequantize_weight(lm_info["data_offsets"][0], lm_info["shape"][0], H)
            self.lm_w = lm_raw.astype(np.float32)
        else:
            self.lm_w = self.emb
        
        # Final norm
        self.fin_w = self.model.final_norm_weight().astype(np.float32)
        print(f"  Ready.", flush=True)
    
    def _pack(self, h):
        return np.frombuffer(h.astype(np.float16).tobytes(), dtype=np.int32).copy()
    
    def _unpack(self, i32):
        return np.frombuffer(i32.astype(np.uint32).tobytes(), dtype=np.float16).astype(np.float32)[:H]
    
    def run_layer(self, l, h_i32, k_buf, v_buf):
        w_buf = XRTTensor(self.weights[l].copy(), dtype=np.int32)
        o_buf = XRTTensor(np.zeros(512, dtype=np.int32), dtype=np.int32)
        h_buf = XRTTensor(h_i32.copy(), dtype=np.int32)
        result = npu_build.run(self.handle, [k_buf, v_buf, w_buf, o_buf, h_buf])
        return self._unpack(o_buf.data), result.npu_time
    
    def generate(self, tokens, max_new=64):
        out_tokens, total_ns = [], 0
        k_bufs = [XRTTensor(np.zeros(8192, dtype=np.int32), dtype=np.int32) for _ in range(NC)]
        v_bufs = [XRTTensor(np.zeros(8192, dtype=np.int32), dtype=np.int32) for _ in range(NC)]
        
        for step in range(len(tokens) + max_new):
            tid = tokens[step] if step < len(tokens) else out_tokens[-1]
            h = self.emb[tid].copy()
            h_i32 = self._pack(h)
            
            for l in range(NC):
                h, ns = self.run_layer(l, h_i32, k_bufs[l], v_bufs[l])
                h_i32 = self._pack(h)
                total_ns += ns
            
            if step >= len(tokens):
                rms = np.sqrt(np.mean(h ** 2) + EPS)
                h_norm = h / rms * self.fin_w
                logits = h_norm @ self.lm_w.T
                best = int(np.argmax(logits))
                out_tokens.append(best)
                if best >= NV - 10:
                    break
        
        return out_tokens, total_ns / 1e6

def main():
    line = sys.stdin.readline()
    if not line: return
    req = json.loads(line)
    tokens, max_new = req.get('tokens', []), req.get('max_new_tokens', 64)
    runner = NPURunner()
    out_tokens, ms = runner.generate(tokens, max_new)
    sys.stdout.write(json.dumps({"tokens": out_tokens, "finished": True}) + "\n")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
