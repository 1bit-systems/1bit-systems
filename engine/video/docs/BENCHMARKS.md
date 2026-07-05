# Video Engine Benchmarks

## T2V: Wan2.2-1.3B Q4_0 (GGUF)

| Backend | Hardware | Resolution | Frames | Steps | Time (s) | FPS |
|---------|----------|-----------|-------|-------|---------|-----|
| CPU | AMD Strix Halo (16C/32T) | 640×480 | 16 | 50 | TBD | TBD |
| CPU | AMD Strix Halo (16C/32T) | 640×480 | 8 | 10 | TBD | TBD |
| CUDA | RTX 4090 | 640×480 | 16 | 50 | TBD | TBD |
| Vulkan | Radeon 8060S | 640×480 | 16 | 50 | TBD | TBD |

## I2V: Wan2.2-A14B Q4_0 (GGUF)

| Backend | Hardware | Resolution | Frames | Steps | Time (s) | FPS |
|---------|----------|-----------|-------|-------|---------|-----|
| CUDA | RTX 4090 | 640×480 | 16 | 50 | TBD | TBD |
| CUDA | RTX 4090 | 640×480 | 81 | 50 | TBD | TBD |

## Memory Usage

| Model | Quant | GPU VRAM | RAM |
|-------|-------|---------|-----|
| Wan2.2 T2V 1.3B | Q4_0 | ~1.5 GB | ~2 GB |
| Wan2.2 I2V A14B | Q4_0 | ~8 GB | ~10 GB |

Run your own benchmarks:
```bash
# CPU benchmark
./video_engine -m model.gguf -p test -f 8 -s 10 --benchmark

# CUDA benchmark
./video_engine -m model.gguf -p test -f 16 -s 50 --benchmark --backend cuda

# Flash attention benchmark
./video_engine -m model.gguf -p test -f 16 -s 50 --benchmark --flash-attn
```
