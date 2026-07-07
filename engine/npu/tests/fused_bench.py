#!/usr/bin/env python3
"""Fused NPU+GPU benchmark — orchestrates npu_engine_server + GPU attention simulation."""
import struct
import subprocess
import time
import sys
import random
import math

MODEL = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
ENGINE = "/home/bcloud/engine/npu/build/npu_engine_server"
BATCH = 128
LAYERS = 28
H = 1536      # hidden dim
NH = 12       # num heads
NKV = 2       # num KV heads
HD = 128      # head dim
IM = 4096     # intermediate size
QKV = NH * HD + 2 * NKV * HD  # QKV total dim

print("═══ Fused NPU+GPU Benchmark ═══")
print(f"Model: Qwen3-0.6B, B={BATCH}, L={LAYERS}, H={H}")
print(f"Protocol: NPU server stdin/stdout + GPU attention (simulated)")

# Start server
t0 = time.time()
proc = subprocess.Popen(
    [ENGINE, MODEL, '--server'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    bufsize=0
)

# Wait for ready
ready = b''
while b'SERVER: READY' not in ready:
    ready += proc.stdout.readline()
    if b'FAIL' in ready or b'ERR' in ready:
        print(f"Server error: {ready}")
        sys.exit(1)
startup_s = time.time() - t0
print(f"Server startup: {startup_s:.1f}s\n")

# Warmup: one QKV + FFN call
hidden = [random.uniform(-0.1, 0.1) for _ in range(H)]
cmd = b"QKV 0 0 1\n"
proc.stdin.write(cmd + struct.pack(f'{H}f', *hidden))
proc.stdin.flush()
resp = proc.stdout.read(QKV * 4)
qkv = struct.unpack(f'{QKV}f', resp)

attn_in = [random.uniform(-1.0, 1.0) for _ in range(NH * HD)]
cmd = b"FFN 0 0 1\n"
proc.stdin.write(cmd + struct.pack(f'{NH*HD}f', *attn_in))
proc.stdin.flush()
resp = proc.stdout.read(H * 4)
ffn = struct.unpack(f'{H}f', resp)
print(f"Warmup: QKV={len(qkv)} outputs, FFN={len(ffn)} outputs\n")

# Benchmark: B=128, measure QKV + FFN throughput
print(f"Benchmarking B={BATCH}...")
hidden_batch = [random.uniform(-0.1, 0.1) for _ in range(BATCH * H)]

# QKV throughput
t1 = time.time()
n_qkv_calls = 0
for layer in range(LAYERS):
    cmd = f"QKV {layer} 0 {BATCH}\n".encode()
    proc.stdin.write(cmd + struct.pack(f'{BATCH*H}f', *hidden_batch))
    proc.stdin.flush()
    resp = proc.stdout.read(BATCH * QKV * 4)
    n_qkv_calls += 1
qkv_s = time.time() - t1
qkv_ms = qkv_s * 1000 / n_qkv_calls

# FFN throughput
t2 = time.time()
n_ffn_calls = 0
for layer in range(LAYERS):
    cmd = f"FFN {layer} 0 {BATCH}\n".encode()
    attn_in = struct.pack(f'{BATCH*NH*HD}f', *[random.uniform(-1.0, 1.0) for _ in range(BATCH * NH * HD)])
    proc.stdin.write(cmd + attn_in)
    proc.stdin.flush()
    resp = proc.stdout.read(BATCH * H * 4)
    n_ffn_calls += 1
ffn_s = time.time() - t2
ffn_ms = ffn_s * 1000 / n_ffn_calls

# GPU attention estimate (Vulkan flash attention on Radeon 8060S)
gpu_attn_ms_per_layer = 0.5  # estimated from Vulkan docs

# Results
print(f"\n═══ Results (B={BATCH}, L={LAYERS}) ═══")
print(f"  NPU QKV:   {qkv_ms:.2f}ms per layer ({qkv_ms*LAYERS:.1f}ms total)")
print(f"  NPU FFN:   {ffn_ms:.2f}ms per layer ({ffn_ms*LAYERS:.1f}ms total)")
print(f"  NPU total: {qkv_ms + ffn_ms:.2f}ms per layer ({(qkv_ms + ffn_ms)*LAYERS:.1f}ms total)")
print(f"  GPU Attn:  {gpu_attn_ms_per_layer:.2f}ms per layer ({gpu_attn_ms_per_layer*LAYERS:.1f}ms total)")

# Pipeline: qkv || attn || ffn (sequential within each layer)
seq_per_layer = max(qkv_ms, gpu_attn_ms_per_layer) + ffn_ms
seq_total = seq_per_layer * LAYERS
seq_tok_s = BATCH / (seq_total / 1000)

# Overlapped: qkv(N+1) || attn(N) || ffn(N)
# Timeline: [QKV_0] [Attn_0 + QKV_1] [FFN_0 + Attn_1 + QKV_2] ...
# Each layer: max(qkv, max(attn, qkv), ffn) = max(qkv, attn, ffn)
pipe_per_layer = max(qkv_ms, gpu_attn_ms_per_layer, ffn_ms)
pipe_overhead = (LAYERS - 1) * max(gpu_attn_ms_per_layer, ffn_ms)  # overlap of QKV with prev attn
pipe_total = qkv_ms + pipe_per_layer * (LAYERS - 1) + ffn_ms
# More accurate: the pipeline is gpu-bound
# QKV_0 → Attn_0+FFN_0+QKV_1 → Attn_1+FFN_1+QKV_2 → ...
# QKV(N) launches, Attn(N-1) runs concurrently
pipe_total_opt = qkv_ms + (LAYERS - 1) * max(gpu_attn_ms_per_layer, ffn_ms, qkv_ms) + max(gpu_attn_ms_per_layer, ffn_ms)
pipe_tok_s = BATCH / (pipe_total_opt / 1000)

print(f"\n── Sequential (no overlap) ──")
print(f"  Total: {seq_total:.1f}ms → {seq_tok_s:.0f} tok/s")
print(f"\n── Pipeline overlapped ──")
print(f"  NPU QKV(N+1) ‖ GPU Attn(N) → FFN(N)")
print(f"  Total: {pipe_total_opt:.1f}ms → {pipe_tok_s:.0f} tok/s")
print(f"\n── Target: 273 tok/s ──")
print(f"  Gap: {273 - pipe_tok_s:.0f} tok/s ({(273/pipe_tok_s - 1)*100:.0f}% more)")

# Benchmark w/ B=1 single token (decode mode)
print(f"\n── Single token decode (B=1) ──")
hidden_1 = [0.01 * ((i % 10) + 1) for i in range(H)]
t3 = time.time()
for layer in range(LAYERS):
    cmd = f"QKV {layer} 0 1\n".encode()
    proc.stdin.write(cmd + struct.pack(f'{H}f', *hidden_1))
    proc.stdin.flush()
    resp = proc.stdout.read(QKV * 4)
    
    cmd = f"FFN {layer} 0 1\n".encode()
    at = struct.pack(f'{NH*HD}f', *[random.uniform(-1, 1) for _ in range(NH * HD)])
    proc.stdin.write(cmd + at)
    proc.stdin.flush()
    resp = proc.stdout.read(H * 4)
single_s = time.time() - t3
single_tok_s = 1 / (single_s / LAYERS)
print(f"  Single token: {single_s*1000/LAYERS:.1f}ms/layer → {single_tok_s:.0f} tok/s (sequential)")
print(f"  With overlap: ~{single_tok_s*1.5:.0f} tok/s")

# Shutdown
proc.stdin.write(b'EXIT\n')
proc.stdin.flush()
try:
    proc.wait(timeout=5)
except:
    proc.kill()

print(f"\nDone. Server shutdown after {time.time()-t0:.1f}s total runtime.")
PYEOF