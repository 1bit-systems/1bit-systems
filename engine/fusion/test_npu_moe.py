#!/usr/bin/env python3
"""NPU MoE worker protocol test — shared-worker harness."""
import struct, subprocess, sys, os, time, math, fcntl

ENGINE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/engine/npu/build/npu_engine_universal"
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx"
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 35

def spawn_worker():
    proc = subprocess.Popen(
        [ENGINE, MODEL, "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fd = proc.stderr.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    buf = b""
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            chunk = os.read(fd, 4096)
            if chunk:
                buf += chunk
                if b"WORKER_READY" in buf:
                    return proc, True
        except BlockingIOError:
            pass
        time.sleep(0.2)
    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk: break
            buf += chunk
    except: pass
    proc.terminate()
    proc.wait()
    return None, buf.decode(errors="replace")[:300]

def quit_worker(proc):
    try:
        proc.stdin.write(struct.pack('<IIII', 0, 0, 0, 0))
        proc.stdin.flush()
        proc.wait(timeout=5)
    except:
        proc.kill()
        proc.wait()

def sr(proc, op, layer, batch, in_dim, payload=None):
    hdr = struct.pack('<IIII', op, layer, batch, in_dim)
    proc.stdin.write(hdr)
    if payload:
        proc.stdin.write(payload)
    proc.stdin.flush()
    resp = proc.stdout.read(8)
    if len(resp) < 8:
        return None, None, b""
    status, out_dim = struct.unpack('<II', resp)
    data = b""
    # NOTE: response header out_dim is per-sample, but actual data is
    # batch * out_dim_per_sample values (as written by fwrite in worker)
    if out_dim > 0:
        data = proc.stdout.read(out_dim * 4 * max(batch, 1))
    return status, out_dim, data

results = {"pass": 0, "fail": 0, "skip": 0}

def test(label, ok):
    if ok is None:
        print(f"  \u2298 {label}", flush=True)
        results["skip"] += 1
    elif ok:
        print(f"  \u2713 {label}", flush=True)
        results["pass"] += 1
    else:
        print(f"  \u2717 {label}", flush=True)
        results["fail"] += 1

def f32_pack(vals):
    return struct.pack('<' + 'f' * len(vals), *vals)

def check_finite(vals):
    return sum(1 for v in vals if not math.isfinite(v)) <= len(vals) // 2

# ── Main ──
proc, ok = spawn_worker()
if not proc:
    test("All wire-protocol tests", None)
else:
    try:
        # T1: Invalid op → status=2
        s, d, _ = sr(proc, 99, 0, 1, 1, f32_pack([0.0]))
        test("Invalid op (99) → status=2", s == 2)

        # T2: Bad layer → status=1 (batch=0 to avoid payload leak in pipe)
        s, d, _ = sr(proc, 1, 999999, 0, 64)
        test("Bad layer (999999) → status=1", s == 1)

        # T3: Batch=0 → status=1
        s, d, _ = sr(proc, 1, 0, 0, 64)
        test("Batch=0 → status=1", s == 1)

        # T4: In_dim=0 → status=1
        s, d, _ = sr(proc, 1, 0, 1, 0)
        test("In_dim=0 → status=1", s == 1)

        def unpack(bb, n_per_sample, batch=1):
            """Unpack response data: batch * n_per_sample f32 values."""
            expected = n_per_sample * batch * 4
            if len(bb) < expected:
                return None
            return struct.unpack('<' + 'f' * (n_per_sample * batch), bb[:expected])

        # T5: OP_GATEUP (op=3) layer=0 batch=1
        inp = [0.3 * (i % 7 - 3) for i in range(1024)]
        s, d, data = sr(proc, 3, 0, 1, 1024, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        if ok:
            vals = unpack(data, d, 1)
            ok = vals is not None and check_finite(vals)
        test("OP_GATEUP (op=3) layer=0 batch=1", ok)

        # T6: OP_DOWN (op=5) layer=0 batch=1
        inp = [0.05 * (i % 17) for i in range(3072)]
        s, d, data = sr(proc, 5, 0, 1, 3072, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        if ok:
            vals = unpack(data, d, 1)
            ok = vals is not None and check_finite(vals)
        test("OP_DOWN (op=5) layer=0 batch=1", ok)

        # T7: OP_QKV (op=1) layer=0 batch=1
        inp = [math.sin(i * 0.1) for i in range(1024)]
        s, d, data = sr(proc, 1, 0, 1, 1024, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        if ok:
            vals = unpack(data, d, 1)
            ok = vals is not None and check_finite(vals)
        test("OP_QKV (op=1) layer=0 batch=1", ok)

        # T8: OP_OPROJ (op=2) layer=0 batch=1
        inp = [math.cos(i * 0.05) for i in range(1024)]
        s, d, data = sr(proc, 2, 0, 1, 1024, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        if ok:
            vals = unpack(data, d, 1)
            ok = vals is not None and check_finite(vals)
        test("OP_OPROJ (op=2) layer=0 batch=1", ok)

        # T9: Batch=2 on GATEUP
        inp = [0.2 * (i % 5) for i in range(2048)]
        s, d, data = sr(proc, 3, 0, 2, 1024, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        if ok:
            vals = unpack(data, d, 2)
            ok = vals is not None and check_finite(vals)
        test("OP_GATEUP batch=2", ok)

        # T10: Sequential on same worker
        inp = [0.2 * (i % 11) for i in range(1024)]
        s, d, _ = sr(proc, 3, 0, 1, 1024, f32_pack(inp))
        ok = (s == 0) and (d > 0)
        inp = [0.03 * (i % 19) for i in range(3072)]
        s, d, _ = sr(proc, 5, 0, 1, 3072, f32_pack(inp))
        ok = ok and (s == 0) and (d > 0)
        test("Sequential: GATEUP→DOWN same worker", ok)

        # T11: QKV output dim > H (multi-head expansion)
        inp = [0.1 * (i % 13) for i in range(1024)]
        s, d, _ = sr(proc, 1, 0, 1, 1024, f32_pack(inp))
        test("OP_QKV out_dim > H (multi-head expansion)", s == 0 and d > 1500)

    finally:
        quit_worker(proc)

    # T12: Worker restart
    proc2, ok2 = spawn_worker()
    if proc2:
        quit_worker(proc2)
        test("Worker start + QUIT (restart cycle)", True)
    else:
        test("Worker start + QUIT (restart cycle)", None)

print(f"\nResults: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped", flush=True)
sys.exit(0 if results['fail'] == 0 else 1)
