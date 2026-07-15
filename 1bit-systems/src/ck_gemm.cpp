// rocm-cpp C API implementation — wraps CK's DeviceGemm_Wmma_CShuffleV3 and
// exposes a C interface for consumers (halo-1bit, lemond, external).
//
// The CK surface is fully contained in this TU.
// CK-independent ternary GEMV launchers live in ternary_gemv_launchers.hip.

#include "rocm_cpp/ck_gemm.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace {

using ck::half_t;
using ck::pk_i4_t;
using ck::index_t;

template <index_t... Is> using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;

// Winning instance from tile-tuning sweep: BlockSize=256, 128x128x32 tile,
// Interwave v1 pipeline, PermuteB=true. 0.96x rocBLAS FP16 at half B memory
// on BitNet FFN shapes (see docs/10-ck-integration-path.md).
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemm_Wmma_CShuffleV3<
    Row, Col, Row,
    half_t, pk_i4_t, half_t, float, half_t,
    PassThrough, PassThrough, PassThrough, GemmDefault,
    256,
    128, 128, 32,
    8, 8,
    16, 16,
    4, 2,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>,
    2, 8, 8, 1,
    1, 1, S<1, 32, 1, 8>, 8,
    ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1,
    half_t, half_t, /*PermuteA=*/false, /*PermuteB=*/true>;

constexpr int KPerBlock = 32;

}  // namespace

struct rcpp_ck_gemm_handle {
    int M, N, K;
    DeviceGemmInstance gemm{};
    DeviceGemmInstance::Invoker invoker = gemm.MakeInvoker();
    std::string instance_str;
};

extern "C" {

rcpp_status_t
rcpp_ck_gemm_create(int M, int N, int K, rcpp_ck_gemm_handle_t** handle_out) {
    if(!handle_out || M <= 0 || N <= 0 || K <= 0) return RCPP_INVALID_ARG;
    if(K % KPerBlock != 0)                        return RCPP_INVALID_ARG;

    auto* h = new(std::nothrow) rcpp_ck_gemm_handle{M, N, K};
    if(!h) return RCPP_INTERNAL;

    auto dummy_arg = h->gemm.MakeArgument(
        /*A*/ static_cast<const half_t*>(nullptr),
        /*B*/ static_cast<const pk_i4_t*>(nullptr),
        /*C*/ static_cast<half_t*>(nullptr),
        M, N, K,
        /*StrideA*/ K,
        /*StrideB*/ K,
        /*StrideC*/ N,
        /*KBatch*/ 1,
        PassThrough{}, PassThrough{}, PassThrough{});
    if(!h->gemm.IsSupportedArgument(dummy_arg)) {
        delete h;
        return RCPP_UNSUPPORTED;
    }

    h->instance_str = h->gemm.GetTypeString();
    *handle_out = h;
    return RCPP_OK;
}

void
rcpp_ck_gemm_destroy(rcpp_ck_gemm_handle_t* h) {
    delete h;
}

rcpp_status_t
rcpp_ck_gemm_run(rcpp_ck_gemm_handle_t* h,
                 const void* A_dev, const void* B_dev_packed, void* C_dev,
                 void* stream) {
    if(!h || !A_dev || !B_dev_packed || !C_dev) return RCPP_INVALID_ARG;

    auto arg = h->gemm.MakeArgument(
        static_cast<const half_t*>(A_dev),
        static_cast<const pk_i4_t*>(B_dev_packed),
        static_cast<half_t*>(C_dev),
        h->M, h->N, h->K,
        h->K, h->K, h->N,
        /*KBatch*/ 1,
        PassThrough{}, PassThrough{}, PassThrough{});

    if(!h->gemm.IsSupportedArgument(arg)) return RCPP_UNSUPPORTED;

    StreamConfig cfg{static_cast<hipStream_t>(stream), /*time_kernel=*/false, 0};
    h->invoker.Run(arg, cfg);
    return RCPP_OK;
}

const char*
rcpp_ck_gemm_instance_string(const rcpp_ck_gemm_handle_t* h) {
    return h ? h->instance_str.c_str() : "";
}

}  // extern "C"
