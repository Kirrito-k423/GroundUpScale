#include <stdint.h>

static volatile float groundupscale_scalar_sink = 0.0f;

float groundupscale_scalar_fma(uint64_t iterations) {
    float a0 = 0.101f;
    float a1 = 0.202f;
    float a2 = 0.303f;
    float a3 = 0.404f;
    float a4 = 0.505f;
    float a5 = 0.606f;
    float a6 = 0.707f;
    float a7 = 0.808f;
    const float multiplier = 0.999999f;
    const float increment = 0.000001f;
    for (uint64_t index = 0; index < iterations; ++index) {
        __asm__ volatile(
            "fmadd %s0, %s0, %s8, %s9\n\t"
            "fmadd %s1, %s1, %s8, %s9\n\t"
            "fmadd %s2, %s2, %s8, %s9\n\t"
            "fmadd %s3, %s3, %s8, %s9\n\t"
            "fmadd %s4, %s4, %s8, %s9\n\t"
            "fmadd %s5, %s5, %s8, %s9\n\t"
            "fmadd %s6, %s6, %s8, %s9\n\t"
            "fmadd %s7, %s7, %s8, %s9\n\t"
            : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3),
              "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7)
            : "w"(multiplier), "w"(increment));
    }
    groundupscale_scalar_sink =
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return groundupscale_scalar_sink;
}
