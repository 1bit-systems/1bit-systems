# Welcome to the 1bit.systems wiki

## Architecture
- [Fused NPU+GPU Engine](https://github.com/bong-water-water-bong/1bit-systems/blob/main/engine/fusion/fused_execute.zig)
- [Model Loaders](https://github.com/bong-water-water-bong/1bit-systems/tree/main/src)
- [Agent Watchdog](https://github.com/bong-water-water-bong/1bit-systems/blob/main/src/agent_watchdog.cpp)
- [Strategy Engine](https://github.com/bong-water-water-bong/1bit-systems/blob/main/src/strategy_engine.cpp)

## Benchmarks
- GPU ternary GEMV: 426 tok/s
- GPU 1-bit GEMV: 417 tok/s
- Prefill I8-APRE: 42.2 TFLOPS

## Building
```bash
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
```

## The Story
- [Reverse Engineering AMD's NPU](https://github.com/bong-water-water-bong/1bit-systems/blob/main/site/blog/reverse-engineered-amd-npu-4-days.html)
- [72x Speedup Sprint](https://github.com/bong-water-water-bong/1bit-systems/blob/main/site/blog/npu-optimization-sprint-72x-speedup.html)
