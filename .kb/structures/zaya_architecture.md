---
type: Structure
title: Zaya Model Architecture
description: Hybrid CCA-attention + Mixture-of-Experts with EDA router. Supported in zaya-llama.cpp with custom ROCm kernels.
tags: [zaya, architecture, moe, cca, eda, router, ternary]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

Zaya is a hybrid architecture combining **CCA-attention** (Cross-Computer Attention) with **Mixture-of-Experts (MoE)** and an **EDA router** (Expert-Dependent Attention routing).

## Architecture

```
Input
  │
  ▼
┌──────────────┐
│   EDA Router  │  Expert-Dependent Attention routing
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  CCA Attend  │────▶│   Expert 1   │
│  (shared)    │     ├──────────────┤
└──────────────┘     │   Expert 2   │
                     ├──────────────┤
                     │   Expert N   │
                     └──────────────┘
       │
       ▼
┌──────────────┐
│   Merged Out │
└──────────────┘
```

## Key Components

### CCA Attention
Cross-Computer Attention enables efficient processing across distributed memory. Each attention head can attend to tokens across sub-batches.

### EDA Router
Expert-Dependent Attention routing determines which experts should process each token based on attention patterns rather than simple top-k routing.

### MoE Layers
Multiple expert feed-forward networks, sparsely activated per token. Ternary weights enable extreme compression.

## ROCm Support

Custom ROCm kernels support Zaya inference on AMD Strix Halo (gfx1151):
- Ternary GEMV for MoE layers
- CCA attention kernels
- EDA router kernels

## Citations

[1] [ROCm Backend](/engines/rocm_backend.md)
[2] [zaya-llama.cpp fork](https://github.com/1bit-systems/zaya-llama.cpp)
