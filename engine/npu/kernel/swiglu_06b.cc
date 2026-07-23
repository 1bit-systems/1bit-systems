#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {

constexpr int32_t kSwigluLanes = 512;
constexpr int32_t kMylmSwigluSegments = 64;

#define SWIGLU_LUT_QUAD(s0, o0, s1, o1, s2, o2, s3, o3) \
    s0, o0, s1, o1, s2, o2, s3, o3, s0, o0, s1, o1, s2, o2, s3, o3
#define SWIGLU_LINEAR_APPROX_LUT_VALUES \
    SWIGLU_LUT_QUAD(-0.0000000000e+00f, -0.0000000000e+00f, -8.0871582031e-04f, -2.8416510671e-02f, -9.9945068359e-04f, -3.4123960882e-02f, -1.2359619141e-03f, -4.0880300105e-02f), \
    SWIGLU_LUT_QUAD(-1.5182495117e-03f, -4.8848759383e-02f, -1.8615722656e-03f, -5.8208428323e-02f, -2.2888183594e-03f, -6.9152273238e-02f, -2.7923583984e-03f, -8.1883266568e-02f), \
    SWIGLU_LUT_QUAD(-3.4027099609e-03f, -9.6608236432e-02f, -4.1503906250e-03f, -1.1352804303e-01f, -5.0048828125e-03f, -1.3282361627e-01f, -6.0729980469e-03f, -1.5463614464e-01f), \
    SWIGLU_LUT_QUAD(-7.2937011719e-03f, -1.7904028296e-01f, -8.7280273438e-03f, -2.0600858331e-01f, -1.0314941406e-02f, -2.3536635935e-01f, -1.2145996094e-02f, -2.6673564315e-01f), \
    SWIGLU_LUT_QUAD(-1.4221191406e-02f, -2.9946959019e-01f, -1.6479492188e-02f, -3.3258017898e-01f, -1.8676757812e-02f, -3.6466687918e-01f, -2.0996093750e-02f, -3.9386045933e-01f), \
    SWIGLU_LUT_QUAD(-2.2949218750e-02f, -4.1780564189e-01f, -2.4414062500e-02f, -4.3371707201e-01f, -2.4902343750e-02f, -4.3855389953e-01f, -2.3925781250e-02f, -4.2936223745e-01f), \
    SWIGLU_LUT_QUAD(-2.0629882812e-02f, -4.0381985903e-01f, -1.4526367188e-02f, -3.6097243428e-01f, -4.7302246094e-03f, -3.0205962062e-01f, 9.4604492188e-03f, -2.3120641708e-01f), \
    SWIGLU_LUT_QUAD(2.8320312500e-02f, -1.5563963354e-01f, 5.1757812500e-02f, -8.5079051554e-02f, 7.9101562500e-02f, -3.0141420662e-02f, 1.0937500000e-01f, 0.0000000000e+00f), \
    SWIGLU_LUT_QUAD(1.4062500000e-01f, 0.0000000000e+00f, 1.7089843750e-01f, -3.0141420662e-02f, 1.9824218750e-01f, -8.5079051554e-02f, 2.2167968750e-01f, -1.5563963354e-01f), \
    SWIGLU_LUT_QUAD(2.4023437500e-01f, -2.3120641708e-01f, 2.5390625000e-01f, -3.0205962062e-01f, 2.6367187500e-01f, -3.6097243428e-01f, 2.7148437500e-01f, -4.0381985903e-01f), \
    SWIGLU_LUT_QUAD(2.7343750000e-01f, -4.2936223745e-01f, 2.7539062500e-01f, -4.3855389953e-01f, 2.7343750000e-01f, -4.3371707201e-01f, 2.7343750000e-01f, -4.1780564189e-01f), \
    SWIGLU_LUT_QUAD(2.7148437500e-01f, -3.9386045933e-01f, 2.6953125000e-01f, -3.6466687918e-01f, 2.6562500000e-01f, -3.3258017898e-01f, 2.6367187500e-01f, -2.9946959019e-01f), \
    SWIGLU_LUT_QUAD(2.6171875000e-01f, -2.6673564315e-01f, 2.5976562500e-01f, -2.3536635935e-01f, 2.5781250000e-01f, -2.0600858331e-01f, 2.5781250000e-01f, -1.7904028296e-01f), \
    SWIGLU_LUT_QUAD(2.5585937500e-01f, -1.5463614464e-01f, 2.5585937500e-01f, -1.3282361627e-01f, 2.5390625000e-01f, -1.1352804303e-01f, 2.5390625000e-01f, -9.6608236432e-02f), \
    SWIGLU_LUT_QUAD(2.5195312500e-01f, -8.1883266568e-02f, 2.5195312500e-01f, -6.9152273238e-02f, 2.5195312500e-01f, -5.8208428323e-02f, 2.5195312500e-01f, -4.8848759383e-02f), \
    SWIGLU_LUT_QUAD(2.5195312500e-01f, -4.0880300105e-02f, 2.5195312500e-01f, -3.4123960882e-02f, 2.5000000000e-01f, -2.8416510671e-02f, 2.5000000000e-01f, -0.0000000000e+00f)

alignas(aie::vector_decl_align) static const float swiglu_linear_lut_ab[kMylmSwigluSegments * 4] = {
    SWIGLU_LINEAR_APPROX_LUT_VALUES
};
alignas(aie::vector_decl_align) static const float swiglu_linear_lut_cd[kMylmSwigluSegments * 4] = {
    SWIGLU_LINEAR_APPROX_LUT_VALUES
};
#undef SWIGLU_LINEAR_APPROX_LUT_VALUES
#undef SWIGLU_LUT_QUAD

} // namespace

extern "C" {

void ffn_swiglu_slice_bf16_inputs(
    int32_t *input,
    bfloat16 *output,
    int32_t dwords,
    int32_t slice
) {
    (void)dwords;
    (void)slice;
    bfloat16 *__restrict values = reinterpret_cast<bfloat16 *>(input);
    bfloat16 *__restrict up_values = values;
    bfloat16 *__restrict gate_values = values + kSwigluLanes;

    using Lut = aie::lut<4, float, bfloat16>;
    Lut silu_lut(kMylmSwigluSegments, swiglu_linear_lut_ab, swiglu_linear_lut_cd);
    aie::linear_approx<bfloat16, Lut> silu_approx(silu_lut, 0, 32, 0);
    const aie::vector<bfloat16, 16> scale4 = aie::broadcast<bfloat16, 16>(4.0f);
    const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);

    for (int32_t idx = 0; idx < kSwigluLanes; idx += 16)
        chess_prepare_for_pipelining chess_loop_range(kSwigluLanes / 16, kSwigluLanes / 16) {
        const aie::vector<bfloat16, 16> up_vec = aie::load_v<16>(up_values + idx);
        const aie::vector<bfloat16, 16> gate_vec = aie::load_v<16>(gate_values + idx);
        const aie::vector<bfloat16, 16> gate_scaled =
            aie::mul(gate_vec, scale4).template to_vector<bfloat16>();
        const aie::vector<float, 16> silu_vec = silu_approx.compute(gate_scaled).template to_vector<float>();
        const auto up_float = aie::mul(up_vec, one);
        const auto output_acc = aie::mul(up_float.template to_vector<float>(), silu_vec);
        aie::store_v(output + idx, output_acc.template to_vector<bfloat16>());
    }
}

} // extern "C"
