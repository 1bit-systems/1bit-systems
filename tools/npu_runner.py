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
        
        print("  Pre-computing instruction patches...", flush=True)
        self.xclbin_path = XCLBIN
        self.base_insts_path = INSTS.parent / 'design.bin'
        # Pre-create patched instruction files for positions 0..127
        from cases.qwen3_8b_decode_layer_runner import _target_schedule, _build_schedule
        from cases.decode_instruction_patch import patch_instruction_stream
        from cases.decode_cache_reference import DecodeSchedule
        self.insts_by_pos = {}
        for pos in range(128):
            out_path = Path('/tmp') / f'npu_insts_pos{pos}.bin'
            if not out_path.exists():
                base_sched = DecodeSchedule(current_token=127)
                target_sched = DecodeSchedule(current_token=pos)
                patch_instruction_stream(self.base_insts_path, out_path, base_sched, target_sched)
            self.insts_by_pos[pos] = out_path
        print(f"  Pre-computed {len(self.insts_by_pos)} positions", flush=True)
        
        print("  Loading weights...", flush=True)
        self.weights = [np.fromfile(WEIGHT_DIR / f'layer_{l}.bin', dtype=np.int32) for l in range(NC)]
        
        # Embeddings
        emb_bf16 = self.model.tensor_bf16("model.embed_tokens.weight")
        self.emb = emb_bf16.astype(np.float32).reshape(NV, H)
        
        # LM head from pre-computed weights (fast load)
        lm_path = Path('/tmp/lm_head_f32.npy')
        if lm_path.exists():
            self.lm_w = np.load(lm_path)
            print(f"  LM head loaded from cache: {self.lm_w.shape}", flush=True)
        else:
            # Fallback: use embeddings (tied)
            self.lm_w = self.emb
            print(f"  WARNING: LM head cache not found, using tied embeddings", flush=True)
        
        # Final norm
        self.fin_w = self.model.final_norm_weight().astype(np.float32)
        print(f"  Ready.", flush=True)
    
    def _pack(self, h):
        return np.frombuffer(h.astype(np.float16).tobytes(), dtype=np.int32).copy()
    
    def _unpack(self, i32):
        return np.frombuffer(i32.astype(np.uint32).tobytes(), dtype=np.float16).astype(np.float32)[:H]
    
    def run_layer(self, l, h_i32, k_buf, v_buf, pos=0):
        import npu_build
        handle = npu_build.load_kernel(self.xclbin_path, self.insts_by_pos.get(pos, self.insts_by_pos[0]))
        w_buf = XRTTensor(self.weights[l].copy(), dtype=np.int32)
        o_buf = XRTTensor(np.zeros(512, dtype=np.int32), dtype=np.int32)
        h_buf = XRTTensor(h_i32.copy(), dtype=np.int32)
        result = npu_build.run(handle, [k_buf, v_buf, w_buf, o_buf, h_buf])
        import aie.utils as _au; _au.DefaultNPURuntime.cleanup()
        import numpy as _np; h_out = self._unpack(o_buf.data); has_nan = _np.any(~_np.isfinite(h_out)); print(f'    Layer {l}: range=[{h_out.min():.2f},{h_out.max():.2f}] nan={has_nan}', flush=True)
        return self._unpack(o_buf.data), result.npu_time
    
    def generate(self, tokens, max_new=64):
        out_tokens, total_ns = [], 0
        k_bufs = [XRTTensor(np.zeros(8192, dtype=np.int32), dtype=np.int32) for _ in range(NC)]
        v_bufs = [XRTTensor(np.zeros(8192, dtype=np.int32), dtype=np.int32) for _ in range(NC)]
        
        # Prefill: process all prompt tokens sequentially
        h = self.emb[tokens[0]].copy()
        for pi in range(len(tokens)):
            if pi > 0:
                h = self.emb[tokens[pi]].copy()
            h_i32 = self._pack(h)
            for l in range(NC):
                h, ns = self.run_layer(l, h_i32, k_bufs[l], v_bufs[l], pos=pi)
                h_i32 = self._pack(h)
                total_ns += ns
        
        # Decode
        for step in range(max_new):
            h_i32 = self._pack(h)
            for l in range(NC):
                h, ns = self.run_layer(l, h_i32, k_bufs[l], v_bufs[l], pos=len(tokens)+step)
                h_i32 = self._pack(h)
                total_ns += ns
            
            rms = np.sqrt(np.mean(h ** 2) + EPS)
            h_norm = h / rms * self.fin_w
            logits = h_norm @ self.lm_w.T
            top5 = np.argsort(logits)[-5:][::-1]
            print(f"  Token {step}: top5={top5.tolist()} logits={logits[top5].round(2).tolist()}", flush=True)
            best = int(np.argmax(logits))
            out_tokens.append(best)
            
            if best >= NV - 10:
                break
            h = self.emb[best].copy()
        
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
