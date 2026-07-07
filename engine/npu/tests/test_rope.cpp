// test_rope.cpp — Unit tests for rope_apply (catches the c-instead-of-s bug)
// Build: g++ -std=c++23 -O2 -o test_rope test_rope.cpp -lm && ./test_rope
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Exact copy of the fixed rope_apply from npu_engine_split.cpp
static std::vector<float> rc, rs;

static void rope_init(int hd, float th, int mp) {
    int hd2 = hd / 2;
    rc.resize(mp * hd);
    rs.resize(mp * hd);
    float tb = 1.0f / th;
    for (int p = 0; p < mp; p++) {
        int pp = p;
        for (int d = 0; d < hd2; d++) {
            float a = pp * powf(tb, -(float)d / (float)hd2);
            rc[p * hd + d] = cosf(a); rs[p * hd + d] = sinf(a);
        }
    }
}

static inline void rope_apply(float* x, int hd, int p) {
    int hd2 = hd / 2;
    for (int d = 0; d < hd2; d++) {
        float a = x[d], b = x[d + hd2];
        float c = rc[p * hd + d], s = rs[p * hd + d];
        x[d] = a * c - b * s;
        x[d + hd2] = b * c + a * s;
    }
}

// Reference: explicit 2D rotation matrix
static void rope_apply_ref(float* x, int hd, int p) {
    int hd2 = hd / 2;
    for (int d = 0; d < hd2; d++) {
        float a = x[d], b = x[d + hd2];
        float c = rc[p * hd + d], s = rs[p * hd + d];
        x[d] = a * c - b * s;
        x[d + hd2] = a * s + b * c;
    }
}

int main() {
    int HD = 128;
    float theta = 1000000.0f;
    int max_pos = 4096;
    int errors = 0;
    rope_init(HD, theta, max_pos);

    // Test 1: cos^2 + sin^2 = 1
    for (int p = 0; p < 10 && errors < 5; p++)
        for (int d = 0; d < HD/2 && errors < 5; d++)
            if (fabsf(rc[p*HD+d]*rc[p*HD+d] + rs[p*HD+d]*rs[p*HD+d] - 1.0f) > 1e-5f)
                { printf("FAIL t1 pos=%d d=%d\n", p, d); errors++; }
    printf("t1 (norm):      %s\n", errors ? "FAIL" : "PASS");

    // Test 2: matches reference
    int e2 = 0;
    for (int p = 0; p < 10 && e2 < 5; p++) {
        float x1[128], x2[128];
        for (int i = 0; i < HD; i++) { float v = (float)rand()/RAND_MAX*2-1; x1[i]=v; x2[i]=v; }
        rope_apply(x1, HD, p);
        rope_apply_ref(x2, HD, p);
        for (int i = 0; i < HD && e2 < 5; i++)
            if (fabsf(x1[i]-x2[i]) > 1e-6f) { printf("FAIL t2 p=%d i=%d\n", p, i); e2++; }
    }
    printf("t2 (ref match): %s\n", e2 ? "FAIL" : "PASS"); errors += e2;

    // Test 3: preserves vector norm (rotation is isometric)
    int e3 = 0;
    for (int p = 0; p < 10 && e3 < 5; p++) {
        float x[128], nb=0, na=0;
        for (int i = 0; i < HD; i++) { x[i] = (float)rand()/RAND_MAX*2-1; nb += x[i]*x[i]; }
        rope_apply(x, HD, p);
        for (int i = 0; i < HD; i++) na += x[i]*x[i];
        if (fabsf(nb-na)/nb > 1e-4f) { printf("FAIL t3 p=%d\n", p); e3++; }
    }
    printf("t3 (isometry):  %s\n", e3 ? "FAIL" : "PASS"); errors += e3;

    // Test 4: position 0 is identity (cos(0)=1, sin(0)=0)
    int e4 = 0;
    for (int t = 0; t < 5 && e4 < 5; t++) {
        float x[128], orig[128];
        for (int i = 0; i < HD; i++) { x[i] = (float)rand()/RAND_MAX*2-1; orig[i]=x[i]; }
        rope_apply(x, HD, 0);
        for (int i = 0; i < HD && e4 < 5; i++)
            if (fabsf(x[i]-orig[i]) > 1e-5f) { printf("FAIL t4 i=%d\n", i); e4++; }
    }
    printf("t4 (identity):  %s\n", e4 ? "FAIL" : "PASS"); errors += e4;

    printf("\n%d errors total\n", errors);
    return errors > 0 ? 1 : 0;
}
