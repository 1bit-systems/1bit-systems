// npu_dims.h — Model dimension constants for multi-variant NPU engine builds.
// Each variant is compiled with -DMODEL_<tag> to select dimensions.

#ifndef NPU_DIMS_H
#define NPU_DIMS_H

// Per-model: H, NC, NH, NKV, HD, IM, NV, GQA, rope_theta
// Split GU if 2*IM > 14336 (AIE tile limit)
// Each model also defines I8 row counts for dequant:
//   Q_I8R = H * (NH*HD) / 8192  (Q output dim = NH*HD, input = H)
//   KV_I8R = H * (NKV*HD) / 8192 (K/V output dim = NKV*HD, input = H)
//   O_I8R = (NH*HD) * H / 8192  (O output dim = H, input = NH*HD)
//   GU_I8R = H * IM / 8192       (G/U output dim = IM, input = H)
//   D_I8R = IM * H / 8192        (D output dim = H, input = IM)
//   LM_I8R = NV * H / 8192       (lm_head output dim = NV, input = H)

// Qwen3-0.6B: tag=qwen3_0_6b
#ifdef MODEL_qwen3_0_6b
  #define MODEL_TAG "qwen3_0_6b"
  #define H 1024
  #define NC 28
  #define NH 16
  #define NKV 8
  #define HD 128
  #define IM 3072
  #define NV 151936
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3_0_6b"
  #define GU_FUSED 1  // 2*IM=6144 <= 14336
  #define BOS 151643
  #define EOS 151645
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 256
  #define KV_I8R 128
  #define O_I8R 256
  #define GU_I8R 384
  #define D_I8R 384
  #define LM_I8R 18992
#endif

// Qwen3-8B: tag=qwen3_8b
#ifdef MODEL_qwen3_8b
  #define MODEL_TAG "qwen3_8b"
  #define H 4096
  #define NC 36
  #define NH 32
  #define NKV 8
  #define HD 128
  #define IM 12288
  #define NV 151936
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3_8b"
  #define GU_FUSED 0  // 2*IM=24576 > 14336
  #define BOS 151643
  #define EOS 151645
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 2048
  #define KV_I8R 512
  #define O_I8R 2048
  #define GU_I8R 6144
  #define D_I8R 6144
  #define LM_I8R 75968
#endif

// Qwen3-VL-4B: tag=qwen3_vl_4b
#ifdef MODEL_qwen3_vl_4b
  #define MODEL_TAG "qwen3_vl_4b"
  #define H 2560
  #define NC 36
  #define NH 32
  #define NKV 8
  #define HD 128
  #define IM 9728
  #define NV 151936
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3_vl_4b"
  #define GU_FUSED 0  // 2*IM=19456 > 14336
  #define BOS 151643
  #define EOS 151645
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 1280
  #define KV_I8R 320
  #define O_I8R 1280
  #define GU_I8R 3040
  #define D_I8R 3040
  #define LM_I8R 47480
#endif

// Llama-3.1-8B: tag=llama
#ifdef MODEL_llama
  #define MODEL_TAG "llama"
  #define H 4096
  #define NC 32
  #define NH 32
  #define NKV 8
  #define HD 128
  #define IM 14336
  #define NV 128256
  #define GQA (NH/NKV)
  #define ROPE_THETA 500000.0f
  #define XCLBIN_SUFFIX "llama"
  #define GU_FUSED 0  // 2*IM=28672 > 14336
  #define BOS 128000
  #define EOS 128001
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 2048
  #define KV_I8R 512
  #define O_I8R 2048
  #define GU_I8R 7168
  #define D_I8R 7168
  #define LM_I8R 64128
#endif

// ZR1-1.5B: tag=zr1 (Qwen2 arch, H=1536, reasoning-tuned)
#ifdef MODEL_zr1
  #define MODEL_TAG "zr1"
  #define H 1536
  #define NC 28
  #define NH 12
  #define NKV 2
  #define HD 128
  #define IM 8960
  #define NV 151936
  #define GQA (NH/NKV)
  #define ROPE_THETA 10000.0f
  #define XCLBIN_SUFFIX "v"
  #define GU_FUSED 0  // 2*IM=17920 > 14336
  #define BOS 151643
  #define EOS 151645
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 288
  #define KV_I8R 48
  #define O_I8R 288
  #define GU_I8R 1680
  #define D_I8R 1680
  #define LM_I8R 28488
#endif

// Qwen3.6-MoE-35B-A3B: tag=qwen3.6_moe_35b (MoE, 256 experts, 40 layers)
// Q4_K_S quant, 262k context, linear+full attention alternating
// Note: MoE expert projections use separate G/U/D xclbins with per-expert IM=512
#ifdef MODEL_qwen3_6_moe_35b
  #define MODEL_TAG "qwen3.6_moe_35b"
  #define H 2048
  #define NC 40
  #define NH 16
  #define NKV 2
  #define HD 256
  #define IM 512            // per-expert intermediate
  #define NV 248320
  #define N_EXPERTS 256
  #define TOP_K 8           // top-8 experts per token
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3.6-moe_35b"
  #define GU_FUSED 1        // 2*IM=1024 <= 14336
  #define BOS 248044
  #define EOS 248044
  #define DEF_MP NULL
  #define Q_I8R 1024
  #define KV_I8R 128
  #define O_I8R 1024
  #define GU_I8R 128         // per-expert
  #define D_I8R 128          // per-expert
  #define LM_I8R 62080
#endif

// Qwen3.5-4B-VL: tag=qwen3.5_4b (VLM, 32 layers)
#ifdef MODEL_qwen3_5_4b
  #define MODEL_TAG "qwen3.5_4b"
  #define H 2560
  #define NC 32
  #define NH 16
  #define NKV 4
  #define HD 256
  #define IM 9216
  #define NV 248320
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3.5_4b"
  #define GU_FUSED 0        // 2*IM=18432 > 14336
  #define BOS 248044
  #define EOS 248044
  #define DEF_MP NULL
  #define Q_I8R 1280
  #define KV_I8R 320
  #define O_I8R 1280
  #define GU_I8R 2880
  #define D_I8R 2880
  #define LM_I8R 77600
#endif

// Gemma4-E4B: tag=gemma4_e4b (8B, 26 layers, larger gemma4)
#ifdef MODEL_gemma4_e4b
  #define MODEL_TAG "gemma4_e4b"
  #define H 2560
  #define NC 26
  #define NH 16
  #define NKV 4
  #define HD 256
  #define IM 12288
  #define NV 262144
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "gemma4_e4b"
  #define GU_FUSED 0        // 2*IM=24576 > 14336
  #define BOS 2
  #define EOS 1
  #define DEF_MP NULL
  #define Q_I8R 1280
  #define KV_I8R 320
  #define O_I8R 1280
  #define GU_I8R 3840
  #define D_I8R 3840
  #define LM_I8R 81920
#endif

// Phi4-mini-Instruct-4B: tag=phi4_mini_4b (dense, 32 layers, GQA=3)
// Verified from Q4NX header: H=3072, NV=200064, IM=8192, NH=24, NKV=8, HD=128
#ifdef MODEL_phi4_mini_4b
  #define MODEL_TAG "phi4_mini_4b"
  #define H 3072
  #define NC 32
  #define NH 24
  #define NKV 8           // GQA: 3 KV heads per group
  #define HD 128
  #define IM 8192
  #define NV 200064
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "phi4-mini_4b"
  #define GU_FUSED 0      // 2*IM=16384 > 14336
  #define BOS 100257
  #define EOS 100257
  #define DEF_MP NULL
  #define Q_I8R 1920      // H*QKV_N/8192 = 3072*5120/8192
  #define KV_I8R 384       // H*NKV*HD/8192 = 3072*1024/8192
  #define O_I8R 1152      // NH*HD*H/8192 = 3072*3072/8192
  #define GU_I8R 3072     // H*IM/8192 = 3072*8192/8192
  #define D_I8R 3072      // IM*H/8192 = 8192*3072/8192
  #define LM_I8R 75024    // NV*H/8192 = 200064*3072/8192
#endif

// Nanbeige4.1-3B: tag=nanbeige4.1_3b (reasoning model)
#ifdef MODEL_nanbeige4_1_3b
  #define MODEL_TAG "nanbeige4.1_3b"
  #define H 2560
  #define NC 32
  #define NH 32
  #define NKV 8
  #define HD 80
  #define IM 8192
  #define NV 152064       // Nanbeige vocab
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "nanbeige4.1_3b"
  #define GU_FUSED 0      // 2*IM=16384 > 14336
  #define BOS 1
  #define EOS 2
  #define DEF_MP NULL
  #define Q_I8R 800       // 2560*(32*80)/8192 = 800
  #define KV_I8R 200      // 2560*(8*80)/8192 = 200
  #define O_I8R 800       // (32*80)*2560/8192 = 800
  #define GU_I8R 2560     // 2560*8192/8192 = 2560
  #define D_I8R 2560      // 8192*2560/8192 = 2560
  #define LM_I8R 47520    // 152064*2560/8192 = 47520
#endif

// Gemma4-E2B: tag=gemma4_e2b
// Note: actual Q4NX file has H=1536, NC=35, NH=8, NKV=1, HD=256, IM=6144
#ifdef MODEL_gemma4_e2b
  #define MODEL_TAG "gemma4_e2b"
  #define H 1536
  #define NC 35
  #define NH 8
  #define NKV 1
  #define HD 256
  #define IM 6144
  #define NV 262144
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "gemma4_e2b"
  #define GU_FUSED 1  // 2*IM=12288 <= 14336
  #define BOS 2
  #define EOS 1
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 384
  #define KV_I8R 48
  #define O_I8R 384
  #define GU_I8R 1152
  #define D_I8R 1152
  #define LM_I8R 49152  // 262144*1536/8192 = 49152
#endif

// Default (Qwen3-0.6B) when no model defined
#ifndef MODEL_TAG
  #define MODEL_TAG "qwen3_0_6b"
  #define H 1024
  #define NC 28
  #define NH 16
  #define NKV 8
  #define HD 128
  #define IM 3072
  #define NV 151936
  #define GQA (NH/NKV)
  #define ROPE_THETA 1000000.0f
  #define XCLBIN_SUFFIX "qwen3_0_6b"
  #define GU_FUSED 1
  #define BOS 151643
  #define EOS 151645
  #define DEF_MP NULL /* set $NPU_MODEL_PATH */
  #define Q_I8R 256
  #define KV_I8R 128
  #define O_I8R 256
  #define GU_I8R 384
  #define D_I8R 384
  #define LM_I8R 18992
#endif

// Derived constants
#define NNP (NH/GQA)                   // number of kv heads per group = NKV
#define MAX_POS 4096                   // KV cache size (max context)
#define XM 128                         // GEMM tile size
#define AW 4                           // attention workers
#define WQH (NH/AW)
#define WKVH (NKV/AW)

// XCLBIN directory
#define XCLBIN_DIR "int8" /* set $NPU_XCLBIN_DIR to override */

// xclbin name builder
// QKV: final_i8_QKV_<suffix>.xclbin
// O:   final_i8_O_<suffix>.xclbin
// GU:  final_i8_GU_<suffix>.xclbin or final_i8_G_<suffix>.xclbin + final_i8_U_<suffix>.xclbin
// D:   final_i8_D_<suffix>.xclbin

#endif // NPU_DIMS_H
