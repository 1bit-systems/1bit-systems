#ifndef ZINC_DP4A_COMPAT_GLSL
#define ZINC_DP4A_COMPAT_GLSL

// Older glslangValidator builds used on the RDNA node do not recognize
// GL_EXT_integer_dot_product as a required extension, even though the shader
// source should still compile for fallback validation. Enable native dot4 when
// the compiler exposes it; otherwise provide the same signed 4x8-bit dot
// semantics with scalar byte extraction so clean builds do not fail.
#extension GL_EXT_integer_dot_product : enable

#ifndef GL_EXT_integer_dot_product
int zinc_i8_lane(uint packed, uint lane) {
    const uint byte_value = (packed >> (lane * 8u)) & 0xFFu;
    return int(byte_value) - (byte_value >= 128u ? 256 : 0);
}

int dotPacked4x8AccSatEXT(int a, int b, int acc) {
    const uint au = uint(a);
    const uint bu = uint(b);
    return acc
        + zinc_i8_lane(au, 0u) * zinc_i8_lane(bu, 0u)
        + zinc_i8_lane(au, 1u) * zinc_i8_lane(bu, 1u)
        + zinc_i8_lane(au, 2u) * zinc_i8_lane(bu, 2u)
        + zinc_i8_lane(au, 3u) * zinc_i8_lane(bu, 3u);
}
#endif

#endif
