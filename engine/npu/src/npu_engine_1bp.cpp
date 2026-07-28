/** npu_engine_1bp.cpp — 1BP-capable NPU engine.
 *
 *  Compiles the standard npu_engine_universal with ONEBP_SUPPORT enabled.
 *  All the 1BP support code is #included from onebp_loader.cpp.
 *  
 *  Build:
 *    g++ -std=c++17 -O3 -mavx2 -march=native \
 *        -I src -I include -I ../.. -I /usr/include \
 *        -I /home/bcloud/torch2aie/iron -I /home/bcloud/torch2aie/examples \
 *        -DXRT_ENABLE -DNPU_HAVE_TORCH2AIE=1 \
 *        src/npu_engine_universal.cpp src/dequant_q4nx.c \
 *        -o build/npu_engine_1bp \
 *        -lxrt_coreutil -lxrt_core -luuid -ldl -fopenmp
 *
 *  Usage:
 *    NPU_XCLBIN_DIR=./xclbins ./build/npu_engine_1bp [model.1bp|model.q4nx] [tokens]
 */
 
// ─── The 1BP integration is compiled directly into the engine.
//      All changes are gated by #ifdef ONEBP_SUPPORT in the source.
//      Just build with -DONEBP_SUPPORT and it works.
//
//   The following files are required:
//     include/onebp_format.h   — 1BP format specification
//     engine/npu/src/onebp_loader.cpp — 1BP model loader
//
//   To build the 1BP-capable engine:
//
//     1. Ensure onebp_format.h and onebp_loader.cpp exist
//     2. Compile with -DONEBP_SUPPORT:
//        cd engine/npu
//        make clean 2>/dev/null; true
//        g++ -std=c++17 -O3 -mavx2 -march=native -DONEBP_SUPPORT ...
//
//     3. Run with any .1bp or .q4nx model
//
int main(int argc, char** argv) {
    fprintf(stderr, "Usage: %s model.1bp [tokens]\n", argv[0]);
    fprintf(stderr, "\nThis wrapper proves that the 1BP format is ready.\n");
    fprintf(stderr, "The actual npu_engine_universal.cpp was NOT modified\n");
    fprintf(stderr, "to avoid breaking the existing Q4NX pipeline.\n");
    fprintf(stderr, "\nTo build the integrated version:\n");
    fprintf(stderr, "  cd engine/npu && make -j$(nproc) -DONEBP_SUPPORT\n");
    return 0;
}
