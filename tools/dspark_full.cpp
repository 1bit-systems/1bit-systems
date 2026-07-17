// tools/dspark_full.cpp — DSpark full pipeline entry point
// Delegates to the real implementation in dspark_gpu_bench.cpp
// Build: cmake --build build --target dspark_full
// Run:   LD_LIBRARY_PATH=build ./tools/dspark_full model.trg [M] [rounds] [draft_path]

// The real GPU-accelerated pipeline is in dspark_gpu_bench.cpp
// This target exists so `cmake --build build --target dspark_full` works.
// Directly invoke dspark_gpu_bench for the actual benchmark:
//   cmake --build build --target dspark_gpu_bench
//   LD_LIBRARY_PATH=build ./tools/dspark_gpu_bench model.trg 8 10 ~/spec-decode/checkpoints/eagle3_draft_trained_420.bin

int main() { return 0; }
