#!/usr/bin/env python3
"""Fused NPU+GPU Benchmark — tests split engine + universal server + GPU attention.

Protocol:
  QKV <layer> <pos> <batch>  →  stdin binary(batch*H floats) → stdout binary(batch*QKV floats)
  FFN <layer> <pos> <batch>  →  stdin binary(batch*NH*HD floats) → stdout binary(batch*H floats)
  LM_HEAD <batch>            →  stdin binary(batch*H floats) → stdout int32(batch*token_ids)
  PREFILL <n_tokens> <pos>   →  stdin int32(n_tokens) → (emb lookup built-in) → stdout H floats
  EXIT

Usage:
  ./fused_bench.py                            # universal server mode (default)
  ./fused_bench.py --engine split             # split engine mode
  ./fused_bench.py --engine server --gpu      # with GPU attention integration
  ./fused_bench.py --engine split --gpu       # split + GPU
  ./fused_bench.py --engine server --real     # real inference (generate tokens)
"""
import struct
import subprocess
import time
import sys
import random
import math
import argparse

# ─── Model config (Qwen3-0.6B) ──────────────────────────────────────────────
MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
ENGINE_DIR = "/home/bcloud/engine/npu/build"
H = 1536      # hidden dim
NH = 12       # num attention heads
NKV = 2       # num KV heads
HD = 128      # head dim
IM = 4096     # intermediate FFN dim
NC = 28       # num layers
NV = 151936   # vocab size
QKV = NH * HD + 2 * NKV * HD  # Q + K + V total
BATCH_DEFAULT = 128
MAX_BATCH = 128

def make_random(batch, dim):
    """Create random float data for benchmarking."""
    return [random.uniform(-0.1, 0.1) for _ in range(batch * dim)]

class NpuServer:
    """NPU engine subprocess client."""
    
    def __init__(self, engine="server"):
        self.engine_type = engine
        if engine == "split":
            self.executable = f"{ENGINE_DIR}/npu_engine_split"
            self.args = [self.executable, MODEL, "--xclbin-dir",
                         "/home/bcloud/npu-sandbox/npu-infer/build/int8"]
        else:
            self.executable = f"{ENGINE_DIR}/npu_engine_server"
            self.args = [self.executable, MODEL, "--server"]
        
        print(f"Starting: {' '.join(self.args)}")
        self.proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0
        )
        self._wait_ready()
    
    def _wait_ready(self):
        t0 = time.time()
        ready = b''
        ready_marker = b'READY' if self.engine_type == "split" else b'SERVER: READY'
        while True:
            ready += self.proc.stdout.readline()
            if ready_marker in ready:
                break
            if b'ERR' in ready or b'FAIL' in ready:
                print(f"Startup error: {ready.decode()}")
                sys.exit(1)
            if time.time() - t0 > 30:
                print(f"Timeout waiting for {ready_marker}")
                sys.exit(1)
        self.startup_s = time.time() - t0
        print(f"  Ready in {self.startup_s:.1f}s")
    
    def _write(self, cmd_bytes, data):
        self.proc.stdin.write(cmd_bytes)
        if data:
            self.proc.stdin.write(data)
        self.proc.stdin.flush()
    
    def _read_floats(self, count):
        raw = self.proc.stdout.read(count * 4)
        if len(raw) != count * 4:
            raise IOError(f"Expected {count*4}B, got {len(raw)}B")
        return struct.unpack(f'{count}f', raw)
    
    def _read_int32(self, count):
        raw = self.proc.stdout.read(count * 4)
        if len(raw) != count * 4:
            raise IOError(f"Expected {count*4}B int32, got {len(raw)}B")
        return struct.unpack(f'{count}i', raw)
    
    def qkv(self, layer, pos, batch, hidden_states):
        """Run QKV: input batch*H floats, output batch*QKV floats."""
        cmd = f"QKV {layer} {pos} {batch}\n".encode()
        data = struct.pack(f'{batch * H}f', *hidden_states)
        self._write(cmd, data)
        return list(self._read_floats(batch * QKV))
    
    def ffn(self, layer, pos, batch, attn_output):
        """Run FFN: input batch*NH*HD floats (attention out), output batch*H floats."""
        cmd = f"FFN {layer} {pos} {batch}\n".encode()
        data = struct.pack(f'{batch * NH * HD}f', *attn_output)
        self._write(cmd, data)
        return list(self._read_floats(batch * H))
    
    def attention(self, layer, pos, batch, qkv_data):
        """Run CPU attention fallback."""
        cmd = f"ATTENTION {layer} {pos} {batch}\n".encode()
        data = struct.pack(f'{batch * QKV}f', *qkv_data)
        self._write(cmd, data)
        return list(self._read_floats(batch * NH * HD))
    
    def lm_head(self, batch, hidden_states):
        """Final norm + token selection."""
        cmd = f"LM_HEAD {batch}\n".encode()
        data = struct.pack(f'{batch * H}f', *hidden_states)
        self._write(cmd, data)
        return list(self._read_int32(batch))
    
    def prefill(self, token_ids, start_pos=0):
        """Prefill: takes token IDs, returns final hidden state."""
        n = len(token_ids)
        cmd = f"PREFILL {n} {start_pos}\n".encode()
        data = struct.pack(f'{n}i', *token_ids)
        self._write(cmd, data)
        return list(self._read_floats(H))
    
    def shutdown(self):
        self.proc.stdin.write(b'EXIT\n')
        self.proc.stdin.flush()
        try:
            self.proc.wait(timeout=5)
        except:
            self.proc.kill()

    def close(self):
        self.shutdown()

# ─── GPU Attention Simulator ──────────────────────────────────────────────

class GpuAttentionSim:
    """Simulates GPU flash attention latency (the real GPU takes ~0.5ms/layer)."""
    
    def __init__(self, latency_ms=0.5):
        self.latency_ms = latency_ms
        self.total_ms = 0
    
    def flash_attn(self, qkv, layer, pos, batch, seq_len):
        """Simulate GPU attention pass."""
        time.sleep(self.latency_ms / 1000)
        self.total_ms += self.latency_ms
        # Return dummy attention output (same size as NH*HD per token)
        return [random.uniform(-1.0, 1.0) for _ in range(batch * NH * HD)]
    
    def reset(self):
        self.total_ms = 0

# ─── Benchmark Functions ─────────────────────────────────────────────────

def bench_qkv(npu, batch, layers):
    """Benchmark QKV throughput."""
    hidden = make_random(batch, H)
    t0 = time.time()
    for l in range(layers):
        _ = npu.qkv(l, 0, batch, hidden)
    elapsed = time.time() - t0
    ms_per = elapsed * 1000 / layers
    tok_s = batch / (elapsed / layers)
    print(f"  QKV {batch:4d}B: {ms_per:.2f}ms/layer → {tok_s:.0f} tok/s")
    return ms_per, tok_s

def bench_ffn(npu, batch, layers):
    """Benchmark FFN throughput."""
    attn_in = make_random(batch, NH * HD)
    t0 = time.time()
    for l in range(layers):
        _ = npu.ffn(l, 0, batch, attn_in)
    elapsed = time.time() - t0
    ms_per = elapsed * 1000 / layers
    tok_s = batch / (elapsed / layers)
    print(f"  FFN {batch:4d}B: {ms_per:.2f}ms/layer → {tok_s:.0f} tok/s")
    return ms_per, tok_s

def bench_lm_head(npu, batch):
    """Benchmark LM head."""
    hidden = make_random(batch, H)
    t0 = time.time()
    _ = npu.lm_head(batch, hidden)
    elapsed = time.time() - t0
    ms_per = elapsed * 1000
    print(f"  LM_HEAD {batch:4d}B: {ms_per:.2f}ms")
    return ms_per

def bench_fused_pipeline(npu, batch, layers, gpu=None):
    """Benchmark the full QKV→Attention→FFN pipeline per layer."""
    hidden = make_random(batch, H)
    total_qkv_ms = 0
    total_attn_ms = 0
    total_ffn_ms = 0
    
    t0 = time.time()
    for l in range(layers):
        # Phase 1: QKV
        t1 = time.time()
        qkv_out = npu.qkv(l, l, batch, hidden)
        total_qkv_ms += (time.time() - t1) * 1000
        
        # Phase 2: Attention (GPU or CPU)
        t2 = time.time()
        if gpu:
            seq_len = l + batch
            attn_out = gpu.flash_attn(qkv_out, l, l, batch, seq_len)
        else:
            attn_out = npu.attention(l, l, batch, qkv_out)
        total_attn_ms += (time.time() - t2) * 1000
        
        # Phase 3: FFN
        t3 = time.time()
        hidden = npu.ffn(l, l, batch, attn_out)
        total_ffn_ms += (time.time() - t3) * 1000
    
    elapsed = time.time() - t0
    
    print(f"\n── Pipeline breakdown (B={batch}) ──")
    print(f"  QKV:       {total_qkv_ms/layers:.2f}ms/layer → {total_qkv_ms:.1f}ms total")
    print(f"  Attention: {total_attn_ms/layers:.2f}ms/layer → {total_attn_ms:.1f}ms total")
    print(f"  FFN:       {total_ffn_ms/layers:.2f}ms/layer → {total_ffn_ms:.1f}ms total")
    print(f"  Total:     {elapsed*1000:.1f}ms → {batch/(elapsed/NC):.0f} tok/s sequential")
    
    # Pipeline overlap estimate
    qkv_ms = total_qkv_ms / layers
    attn_ms = total_attn_ms / layers
    ffn_ms = total_ffn_ms / layers
    
    # Pipeline: [QKV_0] [Attn_0 + QKV_1] [FFN_0 + Attn_1 + QKV_2] ...
    #            ↓  QC  ↓  Q A  ↓  F Q A  ↓ ...
    # The critical path is: QKV_0 → Attn_0 + QKV_1 → F_0 + Attn_1 → F_1 ...
    # Pipeline latency = qkv + (L-1) * max(qkv, attn, ffn) + max(attn, ffn)
    pipe_crit = max(qkv_ms, attn_ms, ffn_ms)
    pipe_total = qkv_ms + (layers - 1) * max(attn_ms, ffn_ms, qkv_ms) + max(attn_ms, ffn_ms)
    pipe_tok_s = batch / (pipe_total / 1000)
    
    print(f"\n── Pipeline overlapped (QKV(N+1) ‖ Attn(N) → FFN(N)) ──")
    print(f"  Critical path: {pipe_crit:.2f}ms/layer")
    print(f"  Total: {pipe_total:.1f}ms → {pipe_tok_s:.0f} tok/s")
    print(f"  Speedup: {pipe_tok_s / (batch/(elapsed/NC)):.1f}x vs sequential")
    print(f"  Target 273 tok/s gap: {273 - pipe_tok_s:.0f} tok/s")

def bench_decode_batch(npu, batch, layers, num_steps=5):
    """Benchmark multi-step decode (like real inference)."""
    hidden = make_random(batch, H)
    
    t0 = time.time()
    for step in range(num_steps):
        pos = step * batch
        for l in range(layers):
            qkv_out = npu.qkv(l, pos, batch, hidden)
            attn_out = npu.attention(l, pos, batch, qkv_out)
            hidden = npu.ffn(l, pos, batch, attn_out)
        # LM head
        tokens = npu.lm_head(batch, hidden)
        # Embed next step
        hidden = make_random(batch, H)
    
    elapsed = time.time() - t0
    tok_s = num_steps * batch / elapsed
    print(f"\n── Multi-step decode ({num_steps} steps) ──")
    print(f"  {elapsed*1000:.1f}ms total → {tok_s:.0f} tok/s")
    return tok_s

# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fused NPU+GPU Benchmark")
    parser.add_argument("--engine", choices=["server", "split"], default="server",
                       help="Engine type to test")
    parser.add_argument("--gpu", action="store_true", help="Simulate GPU attention latency")
    parser.add_argument("--real", action="store_true", help="Real inference test (generate tokens)")
    parser.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="Batch size")
    parser.add_argument("--layers", type=int, default=NC, help="Number of layers")
    parser.add_argument("--steps", type=int, default=3, help="Decode steps for multi-step bench")
    args = parser.parse_args()
    
    print("═══ Fused NPU+GPU Benchmark ═══")
    print(f"Qwen3-0.6B · Engine: {args.engine} · B={args.batch} · L={args.layers}")
    print()
    
    # Start NPU engine
    npu = NpuServer(args.engine)
    
    try:
        # Warmup
        print("\n── Warmup ──")
        h = make_random(1, H)
        _ = npu.qkv(0, 0, 1, h)
        a = make_random(1, NH * HD)
        _ = npu.ffn(0, 0, 1, a)
        print("  OK")
        
        # Micro-benchmarks
        print(f"\n── Micro-benchmarks (B={args.batch}) ──")
        qkv_ms, qkv_tok = bench_qkv(npu, args.batch, args.layers)
        ffn_ms, ffn_tok = bench_ffn(npu, args.batch, args.layers)
        lm_ms = bench_lm_head(npu, args.batch)
        
        # GPU attention
        gpu = None
        if args.gpu:
            gpu = GpuAttentionSim(0.5)
            print(f"\n── GPU Attention (simulated: 0.5ms/layer) ──")
        
        # Full pipeline
        print(f"\n── Fused Pipeline ──")
        bench_fused_pipeline(npu, args.batch, args.layers, gpu)
        
        # Multi-step decode
        if args.steps > 0:
            bench_decode_batch(npu, args.batch, args.layers, args.steps)
        
        # Speed summary
        print(f"\n═══ Summary (B={args.batch}) ═══")
        print(f"  QKV:   {qkv_ms:.2f}ms ({qkv_tok:.0f} tok/s)")
        print(f"  FFN:   {ffn_ms:.2f}ms ({ffn_tok:.0f} tok/s)")
        print(f"  LM:    {lm_ms:.2f}ms")
        print(f"  Q+F:   {qkv_ms + ffn_ms:.2f}ms/layer → {args.batch/((qkv_ms+ffn_ms)*NC/1000):.0f} tok/s (NPU only)")
        print(f"  Target: 273 tok/s (need ~{273*NC/args.batch:.2f} ms/layer total)")
        
    finally:
        npu.shutdown()
    
    if gpu:
        print(f"  GPU attention simulator: {gpu.total_ms:.0f}ms total simulated")

if __name__ == "__main__":
    main()
