---
type: Guide
title: Quick Start
description: Get JARVIS running in 3 steps — start NPU backend, activate environment, launch server.
tags: [setup, quickstart, guide]
timestamp: 2026-07-06T00:00:00Z
---

# Quick Start

## Prerequisites

- 1bit.systems AMD Strix Halo laptop with NPU
- FLM server installed
- Python 3.14+ with venv

## Step 1: Start NPU Backend

```bash
sudo flm serve qwen3:0.6b --port 52625 --pmode turbo
```

This starts the NPU inference engine on port 52625.

## Step 2: Activate Environment

```bash
source ~/jarvis-env/bin/activate
```

Or the project venv:

```bash
cd ~/jarvis
source venv/bin/activate
```

## Step 3: Launch JARVIS

```bash
cd ~/jarvis
python3 server/server.py
```

Open [http://localhost:8080/chat](http://localhost:8080/chat) in any browser.

## Verify

```bash
curl http://localhost:8080/api/status
```

Expected response includes `fused_engine.ok: true`, `flm_npu.ok: true`, and knowledge stats.

## Citations

[1] [Configuration](/capabilities/configuration.md)
[2] [Deployment](/guides/deployment.md)
[3] [Demo Script](https://github.com/1bit-systems/jarvis/blob/main/scripts/demo.sh)
