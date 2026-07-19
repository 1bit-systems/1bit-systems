// zaya1_dims.h — Zaya1-8B model dimension constants for HIP kernel compilation
//
// This header defines the compile-time dimensions that HIP kernels expect.
// The kernels in ../../kernels/ use these constants (or redefine them with
// matching values) for shared-memory sizes, loop bounds, and grid dimensions.
//
// For other model architectures, create a different dims header (e.g.,
// qwen_dims.h) and include that instead. The runtime ModelConfig is still
// validated in load_model() against the compile-time contract defined here.
//
// Each constant is individually guarded with #ifndef so the build system
// can override any dimension via compiler flags (-DH=4096, etc.).

#ifndef ZAYA1_DIMS_H_
#define ZAYA1_DIMS_H_

#ifndef ZAYA_H
#define ZAYA_H     2048
#endif
#ifndef ZAYA_NQ
#define ZAYA_NQ    8
#endif
#ifndef ZAYA_NKV
#define ZAYA_NKV   2
#endif
#ifndef ZAYA_HD
#define ZAYA_HD    128
#endif
#ifndef ZAYA_QD
#define ZAYA_QD    1024     // ZAYA_NQ * ZAYA_HD
#endif
#ifndef ZAYA_KD
#define ZAYA_KD    256      // ZAYA_NKV * ZAYA_HD
#endif
#ifndef ZAYA_QKV
#define ZAYA_QKV   1280     // ZAYA_QD + ZAYA_KD
#endif
#ifndef ZAYA_N_LAYERS
#define ZAYA_N_LAYERS 40
#endif
#ifndef ZAYA_VOCAB
#define ZAYA_VOCAB 262272
#endif
#ifndef ZAYA_N_EXP
#define ZAYA_N_EXP 16
#endif
#ifndef ZAYA_ROUTER_TOP_K
#define ZAYA_ROUTER_TOP_K 2
#endif
#ifndef ZAYA_N_FF
#define ZAYA_N_FF  2048
#endif
#ifndef ZAYA_RTR_H
#define ZAYA_RTR_H 256
#endif

// Legacy aliases — kernels in ../../kernels/ may use the short names.
// Define them only if not already set, so a dims header for a different
// model can override them directly.
#ifndef H
#define H    ZAYA_H
#endif
#ifndef NQ
#define NQ   ZAYA_NQ
#endif
#ifndef NKV
#define NKV  ZAYA_NKV
#endif
#ifndef HD
#define HD   ZAYA_HD
#endif
#ifndef QD
#define QD   ZAYA_QD
#endif
#ifndef KD
#define KD   ZAYA_KD
#endif
#ifndef QKV
#define QKV  ZAYA_QKV
#endif
#ifndef N_LAYERS
#define N_LAYERS ZAYA_N_LAYERS
#endif
#ifndef VOCAB
#define VOCAB ZAYA_VOCAB
#endif
#ifndef N_EXP
#define N_EXP ZAYA_N_EXP
#endif
#ifndef ROUTER_TOP_K
#define ROUTER_TOP_K ZAYA_ROUTER_TOP_K
#endif
#ifndef N_FF
#define N_FF  ZAYA_N_FF
#endif
#ifndef RTR_H
#define RTR_H ZAYA_RTR_H
#endif

#endif // ZAYA1_DIMS_H_
