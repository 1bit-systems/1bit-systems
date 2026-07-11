# 1bit CLI ↔ NPU Daemon Integration — Fixes (2026-07-02)

## Critical fixes applied

### 1. NPU Backend: Custom INT8 Engine → FLM Proxy
**Problem:** Custom `npu_engine_stdio.cpp` INT8 GEMM produced garbled tokens.
Weight packing layout mismatched pre-generated MLIR instructions.

**Fix:** Replaced `NPUBackend` in `daemon/npu-gpu-cpud.py` with FLM proxy.
Daemon starts `flm serve qwen3:0.6b` as a subprocess on port 52625 and
proxies `/v1/chat/completions` requests to it.

FLM uses BF16 dequant-on-NPU with dynamically generated instructions — produces
correct, coherent output at ~78 tok/s.

### 2. Port Mismatch Fixes
**Problem:** Multiple port mismatch issues between 1bit CLI, daemon, and FLM.

| Component | Before | After |
|-----------|--------|-------|
| 1bit CLI banner check | 127.0.0.1:9090 | 127.0.0.1:9090 (unchanged) |
| models.json baseUrl | localhost:8081/v1 | 127.0.0.1:9090/v1 |
| Daemon gateway | 8081 | 9090 |
| FLM (internal) | n/a | 52625 |

**Key detail:** `127.0.0.1` (not `localhost`) avoids Node.js IPv6 resolution issues.

### 3. Streaming Handling
**Problem:** 1bit CLI sends `stream: true` by default, daemon returned 400.

**Fix:** Daemon silently ignores `stream: true` and returns non-streaming response.

### 4. model name mismatch
**Problem:** FLM returns `model: "qwen3:0.6b"`, client expects `Qwen3-0.6B-NPU2`.

**Fix:** `NPUBackend.chat()` rewrites `resp["model"] = model` to preserve
client-requested model name.

### 5. /v1/models Endpoint
**Problem:** Was proxying to NPU engine port (52625) which has no models endpoint.

**Fix:** Returns static model list (4 NPU models) directly.

### 6. Systemd Service (Stability)
**Problem:** Daemon kept dying when started from shell background.

**Fix:** systemd service at `/etc/systemd/system/npu-daemon.service`:
- `LimitMEMLOCK=infinity` — required by FLM
- `Environment=HOME=/home/bcloud` — FLM needs HOME
- `Restart=always` + `RestartSec=5` — auto-restart

```ini
[Unit]
Description=1bit NPU Daemon (FLM-powered)
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /home/bcloud/npu-gpu-cpu/daemon/npu-gpu-cpud.py --port 9090
Restart=always
RestartSec=5
LimitMEMLOCK=infinity
Environment=HOME=/home/bcloud
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Architecture (final state)

```
1bit CLI (1bit-agent + 1bit-systems extension)
  │
  │ models.json → http://127.0.0.1:9090/v1
  ▼
npu-daemon (:9090, systemd)
  ├── NPU: FLM proxy → flm serve :52625 → NPU (BF16 dequant + GEMM)
  ├── GPU: lemond :13305 → ROCm
  └── CPU: lemond fallback
```

## Banner (ASCII Art)
Located in `1bit-systems/src/commands/chat.ts` (loaded by 1bit-agent as extension).
Multiple copies exist; all must be synced:
- `/home/bcloud/1bit-systems/src/commands/chat.ts` (source)
- `/home/bcloud/1bit-systems/dist/commands/chat.js` (compiled)
- `/home/bcloud/src/commands/chat.ts` (copy)
- `/home/bcloud/dist/commands/chat.js` (copy)
- `/home/bcloud/1bit-systems-new/src/commands/chat.ts` (submodule)
- `/home/bcloud/1bit-systems-new/dist/commands/chat.js` (submodule)

## Useful commands

```bash
# Check daemon status
sudo systemctl status npu-daemon

# View logs
sudo journalctl -u npu-daemon -f

# Test chat endpoint
curl -s -X POST http://127.0.0.1:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B-NPU2","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'

# Restart daemon
sudo systemctl restart npu-daemon
```
