// test_ternary_32core.cpp — Validate 32-core native ternary xclbin on NPU hardware
//
// Matches the object_fifo runtime_sequence interface:
//   arg 3: A_flat (4 rows of per-row flat buffers)
//   arg 4: C_flat (128 bf16 outputs)
//
// Compile: g++ -O2 -std=c++17 -o test_ternary_32core test_ternary_32core.cpp -lxrt_coreutil

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

static float bf16f(uint16_t v) { uint32_t b=(uint32_t)v<<16; float f; memcpy(&f,&b,4); return f; }
static uint16_t f2bf(float f) { uint32_t b; memcpy(&b,&f,4); return (uint16_t)(((b>>16)&1)+0x7FFF+(b>>16)); }

static void cpu_ref(const uint8_t *w, const uint16_t *s, const uint16_t *a,
                    int M, int Kp, float *out) {
    for (int r=0;r<M;r++) {
        float acc=0, scale=bf16f(s[r]);
        for (int i=0;i<Kp;i++) {
            uint8_t b=w[r*Kp+i];
            for (int j=0;j<4;j++) {
                uint8_t c=(b>>(j*2))&3;
                float t=(c==2)?1.0f:(c==1)?0.0f:-1.0f;
                acc+=t*bf16f(a[i*4+j]);
            }
        }
        out[r]=acc*scale;
    }
}

int main(int argc, char**argv) {
    const char *xp=argc>1?argv[1]:"engine/npu/build/build/ternary_32core/ternary_32core.xclbin";
    const char *ip=argc>2?argv[2]:"engine/npu/build/build/ternary_32core/insts_ternary_32core.txt";

    printf("=== 32-Core Native Ternary NPU Test ===\n");
    printf("XCLBIN: %s\nINSTS:  %s\n", xp, ip);

    // 32-core layout: 4 rows × 8 cols, each core 4 rows (M_total=128)
    const int M_TOTAL=128, ROWS=4, M_PER_ROW=32, M_PER_CORE=4;
    const int K_PACKED=64, K_TERNARY=256;

    int wb = M_PER_ROW * K_PACKED;  // 2048 bytes weights per row
    int sb = M_PER_ROW * 2;         // 64 bytes scales per row
    int ab = K_TERNARY * 2;         // 512 bytes activations per row
    int row_in_bytes = wb + sb + ab; // 2624
    int row_in_dwords = (row_in_bytes+3)/4; // 656

    int total_in_dwords = row_in_dwords * ROWS;  // 2624
    int total_in_bytes = total_in_dwords * 4;    // 10496

    int out_elems = M_TOTAL;        // 128 bf16
    int out_bytes = out_elems * 2;  // 256
    int out_dwords = (out_bytes+3)/4; // 64

    auto device = xrt::device(0);
    printf("Device: %s\n", device.get_info<xrt::info::device::name>().c_str());

    auto xclbin = xrt::xclbin(std::string(xp));
    device.register_xclbin(xclbin);
    auto hw_ctx = xrt::hw_context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");

    // Load instructions
    FILE *f=fopen(ip,"rb"); if(!f){printf("ERROR: no insts\n");return 1;}
    fseek(f,0,2); long isz=ftell(f); fseek(f,0,0);
    std::vector<uint32_t> insts(isz/4);
    fread(insts.data(),4,insts.size(),f); fclose(f);
    printf("Instrs: %zu dwords\n", insts.size());

    auto bo_insts = xrt::bo(device, insts.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    memcpy(bo_insts.map(), insts.data(), insts.size()*4);
    bo_insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Data BOs — match object_fifo runtime_sequence arg sizes
    auto bo_in = xrt::bo(device, total_in_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_out = xrt::bo(device, out_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    auto *in8 = (uint8_t*)bo_in.map();
    auto *out16 = (uint16_t*)bo_out.map();

    // Fill 4 rows of test data
    srand(42);
    float cpu_all[M_TOTAL];
    for (int row=0; row<ROWS; row++) {
        uint8_t *rbase = in8 + row * row_in_bytes;
        int m = M_PER_ROW;

        // Weights
        for (int r=0; r<m; r++) {
            for (int i=0; i<K_PACKED; i++) {
                uint8_t byte=0;
                for (int j=0;j<4;j++) {
                    int v=rand()%3;
                    byte |= ((v==0?0:v==1?1:2) << (j*2));
                }
                rbase[r*K_PACKED + i] = byte;
            }
        }

        // Scales
        uint16_t *sc=(uint16_t*)(rbase+wb);
        for (int i=0;i<m;i++) sc[i]=f2bf(0.5f+(rand()%1000)/2000.0f);

        // Shared activations (same for all rows)
        uint16_t *ac=(uint16_t*)(rbase+wb+sb);
        for (int i=0;i<K_TERNARY;i++) ac[i]=f2bf(((rand()%2000)-1000)/1000.0f);

        // CPU ref for these rows
        cpu_ref(rbase, sc, ac, m, K_PACKED, cpu_all + row*m);
    }

    memset(out16, 0, out_bytes);

    // NPU dispatch — object_fifo: only data args (no opcode/instr/len)
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = xrt::run(kernel);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_out);
    run.start();
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Compare — output is laid out: col0[row0,row1,row2,row3], col1[...], ...
    // Each core produces 4 bf16. 32 cores → 128 bf16.
    // Column-major: core(row,col) → output[col*4*ROWS + row*4 + core_local_row]
    float max_err=0, sum_err=0;
    int ok=0;
    printf("\n  Core | Row |     NPU |     CPU |   Error\n");
    printf("  -----+-----+---------+---------+--------\n");

    for (int col=0; col<8; col++) {
        for (int row=0; row<ROWS; row++) {
            for (int cr=0; cr<M_PER_CORE; cr++) {
                int global_row = row * M_PER_ROW + col * M_PER_CORE + cr;
                // Output layout depends on object_fifo_link gather order
                // C_l2l3 gathers 4 rows: offsets [0, 4, 8, 12] f32 elements
                // That's [0, 16, 32, 48] bytes per row in the column
                // Column output: 4 rows × 4 f32 = 16 f32 = 64 bytes
                // Within the full output buffer: columns concatenated
                int out_idx = col * ROWS * M_PER_CORE + row * M_PER_CORE + cr;
                if (out_idx >= M_TOTAL) continue;

                float npu = bf16f(out16[out_idx]);
                float cpu = cpu_all[global_row];
                float err = fabsf(npu-cpu);
                sum_err += err;
                if (err > max_err) max_err = err;
                if (err < 1e-6f || err > max_err*0.3f)
                    printf("  %4d | %3d | %+8.5f | %+8.5f | %7.2e%s\n",
                           col*ROWS+row, cr, npu, cpu, err, err<1e-6f?" ✓":"");
                if (err < 1e-3f) ok++;
            }
        }
    }

    printf("\n  Max error: %e  Mean: %e  OK: %d/%d\n", max_err, sum_err/M_TOTAL, ok, M_TOTAL);
    printf("  %s\n", max_err<1e-3f?"✅ BIT-EXACT PASS":"❌ FAIL");
    return max_err<1e-3f?0:1;
}
