/*
 * phase2_math3d.c — 3D Math Library
 *
 * vec3, vec4, mat4, quaternion types and operations.
 * Vector and matrix math, transform builders, unit tests.
 *
 * Deliverable: Unit tests that print PASS/FAIL for each operation.
 *
 * Build: gcc -O2 -o phase2_math3d phase2_math3d.c -lm
 * Run:   ./phase2_math3d
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * TYPES
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float x, y, z, w; } vec4;
typedef struct { float w, x, y, z; } quat;  /* w is scalar part */

/* Column-major 4x4 matrix (OpenGL convention) */
typedef struct { float m[16]; } mat4;

/* ══════════════════════════════════════════════════════════════════
 * VEC3 OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

static inline vec3 v3(float x, float y, float z) {
    return (vec3){x, y, z};
}

static inline vec3 v3_add(vec3 a, vec3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline vec3 v3_sub(vec3 a, vec3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline vec3 v3_scale(vec3 v, float s) {
    return v3(v.x * s, v.y * s, v.z * s);
}

static inline vec3 v3_neg(vec3 v) {
    return v3(-v.x, -v.y, -v.z);
}

static inline float v3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

static inline float v3_len(vec3 v) {
    return sqrtf(v3_dot(v, v));
}

static inline float v3_len_sq(vec3 v) {
    return v3_dot(v, v);
}

static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    if (len < EPSILON) return v3(0, 0, 0);
    return v3_scale(v, 1.0f / len);
}

static inline vec3 v3_lerp(vec3 a, vec3 b, float t) {
    return v3_add(v3_scale(a, 1.0f - t), v3_scale(b, t));
}

static inline vec3 v3_reflect(vec3 v, vec3 n) {
    return v3_sub(v, v3_scale(n, 2.0f * v3_dot(v, n)));
}

static inline int v3_eq(vec3 a, vec3 b, float eps) {
    return fabsf(a.x - b.x) < eps &&
           fabsf(a.y - b.y) < eps &&
           fabsf(a.z - b.z) < eps;
}

/* ══════════════════════════════════════════════════════════════════
 * VEC4 OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

static inline vec4 v4(float x, float y, float z, float w) {
    return (vec4){x, y, z, w};
}

static inline vec4 v4_from_v3(vec3 v, float w) {
    return v4(v.x, v.y, v.z, w);
}

static inline vec3 v4_to_v3(vec4 v) {
    return v3(v.x, v.y, v.z);
}

static inline vec4 v4_add(vec4 a, vec4 b) {
    return v4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static inline vec4 v4_scale(vec4 v, float s) {
    return v4(v.x * s, v.y * s, v.z * s, v.w * s);
}

static inline float v4_dot(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* ══════════════════════════════════════════════════════════════════
 * MAT4 OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

/* Access element at row, column (0-indexed) */
#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

static inline mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static inline mat4 m4_zero(void) {
    mat4 m = {0};
    return m;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += M4(a, row, k) * M4(b, k, c);
            }
            M4(r, row, c) = sum;
        }
    }
    return r;
}

static mat4 m4_transpose(mat4 m) {
    mat4 r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M4(r, i, j) = M4(m, j, i);
        }
    }
    return r;
}

static vec4 m4_mul_v4(mat4 m, vec4 v) {
    return v4(
        M4(m, 0, 0) * v.x + M4(m, 0, 1) * v.y + M4(m, 0, 2) * v.z + M4(m, 0, 3) * v.w,
        M4(m, 1, 0) * v.x + M4(m, 1, 1) * v.y + M4(m, 1, 2) * v.z + M4(m, 1, 3) * v.w,
        M4(m, 2, 0) * v.x + M4(m, 2, 1) * v.y + M4(m, 2, 2) * v.z + M4(m, 2, 3) * v.w,
        M4(m, 3, 0) * v.x + M4(m, 3, 1) * v.y + M4(m, 3, 2) * v.z + M4(m, 3, 3) * v.w
    );
}

static vec3 m4_mul_point(mat4 m, vec3 p) {
    vec4 r = m4_mul_v4(m, v4_from_v3(p, 1.0f));
    if (fabsf(r.w) > EPSILON) {
        return v3(r.x / r.w, r.y / r.w, r.z / r.w);
    }
    return v4_to_v3(r);
}

static vec3 m4_mul_dir(mat4 m, vec3 d) {
    vec4 r = m4_mul_v4(m, v4_from_v3(d, 0.0f));
    return v4_to_v3(r);
}

/* Inverse of a 4x4 matrix (general case) */
static mat4 m4_inverse(mat4 m) {
    mat4 inv;
    float *o = inv.m;
    float *i = m.m;

    o[0]  =  i[5]*i[10]*i[15] - i[5]*i[11]*i[14] - i[9]*i[6]*i[15] + i[9]*i[7]*i[14] + i[13]*i[6]*i[11] - i[13]*i[7]*i[10];
    o[4]  = -i[4]*i[10]*i[15] + i[4]*i[11]*i[14] + i[8]*i[6]*i[15] - i[8]*i[7]*i[14] - i[12]*i[6]*i[11] + i[12]*i[7]*i[10];
    o[8]  =  i[4]*i[9]*i[15]  - i[4]*i[11]*i[13] - i[8]*i[5]*i[15] + i[8]*i[7]*i[13] + i[12]*i[5]*i[11] - i[12]*i[7]*i[9];
    o[12] = -i[4]*i[9]*i[14]  + i[4]*i[10]*i[13] + i[8]*i[5]*i[14] - i[8]*i[6]*i[13] - i[12]*i[5]*i[10] + i[12]*i[6]*i[9];

    o[1]  = -i[1]*i[10]*i[15] + i[1]*i[11]*i[14] + i[9]*i[2]*i[15] - i[9]*i[3]*i[14] - i[13]*i[2]*i[11] + i[13]*i[3]*i[10];
    o[5]  =  i[0]*i[10]*i[15] - i[0]*i[11]*i[14] - i[8]*i[2]*i[15] + i[8]*i[3]*i[14] + i[12]*i[2]*i[11] - i[12]*i[3]*i[10];
    o[9]  = -i[0]*i[9]*i[15]  + i[0]*i[11]*i[13] + i[8]*i[1]*i[15] - i[8]*i[3]*i[13] - i[12]*i[1]*i[11] + i[12]*i[3]*i[9];
    o[13] =  i[0]*i[9]*i[14]  - i[0]*i[10]*i[13] - i[8]*i[1]*i[14] + i[8]*i[2]*i[13] + i[12]*i[1]*i[10] - i[12]*i[2]*i[9];

    o[2]  =  i[1]*i[6]*i[15] - i[1]*i[7]*i[14] - i[5]*i[2]*i[15] + i[5]*i[3]*i[14] + i[13]*i[2]*i[7] - i[13]*i[3]*i[6];
    o[6]  = -i[0]*i[6]*i[15] + i[0]*i[7]*i[14] + i[4]*i[2]*i[15] - i[4]*i[3]*i[14] - i[12]*i[2]*i[7] + i[12]*i[3]*i[6];
    o[10] =  i[0]*i[5]*i[15] - i[0]*i[7]*i[13] - i[4]*i[1]*i[15] + i[4]*i[3]*i[13] + i[12]*i[1]*i[7] - i[12]*i[3]*i[5];
    o[14] = -i[0]*i[5]*i[14] + i[0]*i[6]*i[13] + i[4]*i[1]*i[14] - i[4]*i[2]*i[13] - i[12]*i[1]*i[6] + i[12]*i[2]*i[5];

    o[3]  = -i[1]*i[6]*i[11] + i[1]*i[7]*i[10] + i[5]*i[2]*i[11] - i[5]*i[3]*i[10] - i[9]*i[2]*i[7] + i[9]*i[3]*i[6];
    o[7]  =  i[0]*i[6]*i[11] - i[0]*i[7]*i[10] - i[4]*i[2]*i[11] + i[4]*i[3]*i[10] + i[8]*i[2]*i[7] - i[8]*i[3]*i[6];
    o[11] = -i[0]*i[5]*i[11] + i[0]*i[7]*i[9]  + i[4]*i[1]*i[11] - i[4]*i[3]*i[9]  - i[8]*i[1]*i[7] + i[8]*i[3]*i[5];
    o[15] =  i[0]*i[5]*i[10] - i[0]*i[6]*i[9]  - i[4]*i[1]*i[10] + i[4]*i[2]*i[9]  + i[8]*i[1]*i[6] - i[8]*i[2]*i[5];

    float det = i[0]*o[0] + i[1]*o[4] + i[2]*o[8] + i[3]*o[12];
    if (fabsf(det) < EPSILON) {
        return m4_identity();  /* singular matrix, return identity */
    }

    float inv_det = 1.0f / det;
    for (int j = 0; j < 16; j++) o[j] *= inv_det;

    return inv;
}

/* ══════════════════════════════════════════════════════════════════
 * TRANSFORM BUILDERS
 * ══════════════════════════════════════════════════════════════════ */

static mat4 m4_translate(vec3 t) {
    mat4 m = m4_identity();
    M4(m, 0, 3) = t.x;
    M4(m, 1, 3) = t.y;
    M4(m, 2, 3) = t.z;
    return m;
}

static mat4 m4_scale(vec3 s) {
    mat4 m = m4_identity();
    M4(m, 0, 0) = s.x;
    M4(m, 1, 1) = s.y;
    M4(m, 2, 2) = s.z;
    return m;
}

static mat4 m4_rotate_x(float rad) {
    mat4 m = m4_identity();
    float c = cosf(rad), s = sinf(rad);
    M4(m, 1, 1) = c;  M4(m, 1, 2) = -s;
    M4(m, 2, 1) = s;  M4(m, 2, 2) = c;
    return m;
}

static mat4 m4_rotate_y(float rad) {
    mat4 m = m4_identity();
    float c = cosf(rad), s = sinf(rad);
    M4(m, 0, 0) = c;  M4(m, 0, 2) = s;
    M4(m, 2, 0) = -s; M4(m, 2, 2) = c;
    return m;
}

static mat4 m4_rotate_z(float rad) {
    mat4 m = m4_identity();
    float c = cosf(rad), s = sinf(rad);
    M4(m, 0, 0) = c;  M4(m, 0, 1) = -s;
    M4(m, 1, 0) = s;  M4(m, 1, 1) = c;
    return m;
}

/* Rotation around arbitrary axis (Rodrigues' formula) */
static mat4 m4_rotate_axis(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float c = cosf(rad), s = sinf(rad);
    float t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;

    mat4 m = m4_identity();
    M4(m, 0, 0) = t*x*x + c;     M4(m, 0, 1) = t*x*y - s*z;   M4(m, 0, 2) = t*x*z + s*y;
    M4(m, 1, 0) = t*x*y + s*z;   M4(m, 1, 1) = t*y*y + c;     M4(m, 1, 2) = t*y*z - s*x;
    M4(m, 2, 0) = t*x*z - s*y;   M4(m, 2, 1) = t*y*z + s*x;   M4(m, 2, 2) = t*z*z + c;
    return m;
}

static mat4 m4_perspective(float fov_rad, float aspect, float near, float far) {
    mat4 m = m4_zero();
    float tan_half_fov = tanf(fov_rad / 2.0f);
    M4(m, 0, 0) = 1.0f / (aspect * tan_half_fov);
    M4(m, 1, 1) = 1.0f / tan_half_fov;
    M4(m, 2, 2) = -(far + near) / (far - near);
    M4(m, 2, 3) = -(2.0f * far * near) / (far - near);
    M4(m, 3, 2) = -1.0f;
    return m;
}

static mat4 m4_ortho(float l, float r, float b, float t, float n, float f) {
    mat4 m = m4_identity();
    M4(m, 0, 0) = 2.0f / (r - l);
    M4(m, 1, 1) = 2.0f / (t - b);
    M4(m, 2, 2) = -2.0f / (f - n);
    M4(m, 0, 3) = -(r + l) / (r - l);
    M4(m, 1, 3) = -(t + b) / (t - b);
    M4(m, 2, 3) = -(f + n) / (f - n);
    return m;
}

static mat4 m4_look_at(vec3 eye, vec3 target, vec3 up) {
    vec3 f = v3_norm(v3_sub(target, eye));
    vec3 r = v3_norm(v3_cross(f, up));
    vec3 u = v3_cross(r, f);

    mat4 m = m4_identity();
    M4(m, 0, 0) = r.x;  M4(m, 0, 1) = r.y;  M4(m, 0, 2) = r.z;
    M4(m, 1, 0) = u.x;  M4(m, 1, 1) = u.y;  M4(m, 1, 2) = u.z;
    M4(m, 2, 0) = -f.x; M4(m, 2, 1) = -f.y; M4(m, 2, 2) = -f.z;
    M4(m, 0, 3) = -v3_dot(r, eye);
    M4(m, 1, 3) = -v3_dot(u, eye);
    M4(m, 2, 3) = v3_dot(f, eye);
    return m;
}

/* ══════════════════════════════════════════════════════════════════
 * QUATERNION OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

static inline quat q_identity(void) {
    return (quat){1, 0, 0, 0};
}

static inline quat q_from_axis_angle(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float half = rad * 0.5f;
    float s = sinf(half);
    return (quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}

static inline quat q_mul(quat a, quat b) {
    return (quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline quat q_conj(quat q) {
    return (quat){q.w, -q.x, -q.y, -q.z};
}

static inline float q_len(quat q) {
    return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

static inline quat q_norm(quat q) {
    float len = q_len(q);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f / len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}

static vec3 q_rotate(quat q, vec3 v) {
    /* q * v * q^-1 (optimized) */
    vec3 qv = v3(q.x, q.y, q.z);
    vec3 uv = v3_cross(qv, v);
    vec3 uuv = v3_cross(qv, uv);
    return v3_add(v, v3_scale(v3_add(v3_scale(uv, q.w), uuv), 2.0f));
}

static mat4 q_to_mat4(quat q) {
    q = q_norm(q);
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;

    mat4 m = m4_identity();
    M4(m, 0, 0) = 1 - 2*(yy + zz);
    M4(m, 0, 1) = 2*(xy - wz);
    M4(m, 0, 2) = 2*(xz + wy);
    M4(m, 1, 0) = 2*(xy + wz);
    M4(m, 1, 1) = 1 - 2*(xx + zz);
    M4(m, 1, 2) = 2*(yz - wx);
    M4(m, 2, 0) = 2*(xz - wy);
    M4(m, 2, 1) = 2*(yz + wx);
    M4(m, 2, 2) = 1 - 2*(xx + yy);
    return m;
}

static quat q_slerp(quat a, quat b, float t) {
    float dot = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;

    /* If dot < 0, negate one quaternion to take shorter path */
    if (dot < 0) {
        b = (quat){-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }

    if (dot > 0.9995f) {
        /* Linear interpolation for very close quaternions */
        quat r = {
            a.w + t*(b.w - a.w),
            a.x + t*(b.x - a.x),
            a.y + t*(b.y - a.y),
            a.z + t*(b.z - a.z)
        };
        return q_norm(r);
    }

    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    float wa = sinf((1-t)*theta) / sin_theta;
    float wb = sinf(t*theta) / sin_theta;

    return (quat){
        wa*a.w + wb*b.w,
        wa*a.x + wb*b.x,
        wa*a.y + wb*b.y,
        wa*a.z + wb*b.z
    };
}

/* ══════════════════════════════════════════════════════════════════
 * UNIT TESTS
 * ══════════════════════════════════════════════════════════════════ */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name, cond) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  PASS: %s\n", name); \
    } else { \
        printf("  FAIL: %s\n", name); \
    } \
} while(0)

static void test_vec3(void) {
    printf("\n=== vec3 tests ===\n");

    vec3 a = v3(1, 2, 3);
    vec3 b = v3(4, 5, 6);

    TEST("v3 creation", a.x == 1 && a.y == 2 && a.z == 3);

    vec3 sum = v3_add(a, b);
    TEST("v3_add", sum.x == 5 && sum.y == 7 && sum.z == 9);

    vec3 diff = v3_sub(b, a);
    TEST("v3_sub", diff.x == 3 && diff.y == 3 && diff.z == 3);

    vec3 scaled = v3_scale(a, 2);
    TEST("v3_scale", scaled.x == 2 && scaled.y == 4 && scaled.z == 6);

    float dot = v3_dot(a, b);
    TEST("v3_dot", fabsf(dot - 32.0f) < EPSILON);

    vec3 c1 = v3(1, 0, 0);
    vec3 c2 = v3(0, 1, 0);
    vec3 cross = v3_cross(c1, c2);
    TEST("v3_cross", v3_eq(cross, v3(0, 0, 1), EPSILON));

    vec3 unit = v3_norm(v3(3, 0, 0));
    TEST("v3_norm", v3_eq(unit, v3(1, 0, 0), EPSILON));

    TEST("v3_len", fabsf(v3_len(v3(3, 4, 0)) - 5.0f) < EPSILON);

    vec3 lerped = v3_lerp(v3(0,0,0), v3(10,10,10), 0.5f);
    TEST("v3_lerp", v3_eq(lerped, v3(5,5,5), EPSILON));

    vec3 incident = v3(1, -1, 0);
    vec3 normal = v3(0, 1, 0);
    vec3 refl = v3_reflect(incident, normal);
    TEST("v3_reflect", v3_eq(refl, v3(1, 1, 0), EPSILON));
}

static void test_mat4(void) {
    printf("\n=== mat4 tests ===\n");

    mat4 id = m4_identity();
    TEST("m4_identity diagonal", M4(id, 0, 0) == 1 && M4(id, 1, 1) == 1 &&
                                  M4(id, 2, 2) == 1 && M4(id, 3, 3) == 1);
    TEST("m4_identity off-diagonal", M4(id, 0, 1) == 0 && M4(id, 1, 0) == 0);

    /* Test identity * identity = identity */
    mat4 id2 = m4_mul(id, id);
    TEST("m4_mul identity", M4(id2, 0, 0) == 1 && M4(id2, 3, 3) == 1);

    /* Test transpose */
    mat4 m = m4_identity();
    M4(m, 0, 1) = 5;
    M4(m, 1, 0) = 10;
    mat4 mt = m4_transpose(m);
    TEST("m4_transpose", M4(mt, 1, 0) == 5 && M4(mt, 0, 1) == 10);

    /* Test translation */
    mat4 t = m4_translate(v3(10, 20, 30));
    vec3 p = m4_mul_point(t, v3(0, 0, 0));
    TEST("m4_translate point", v3_eq(p, v3(10, 20, 30), EPSILON));

    /* Direction should not be affected by translation */
    vec3 d = m4_mul_dir(t, v3(1, 0, 0));
    TEST("m4_translate direction", v3_eq(d, v3(1, 0, 0), EPSILON));

    /* Test scale */
    mat4 s = m4_scale(v3(2, 3, 4));
    vec3 sp = m4_mul_point(s, v3(1, 1, 1));
    TEST("m4_scale", v3_eq(sp, v3(2, 3, 4), EPSILON));

    /* Test rotation around Z by 90 degrees */
    mat4 rz = m4_rotate_z((float)M_PI / 2);
    vec3 rotated = m4_mul_dir(rz, v3(1, 0, 0));
    TEST("m4_rotate_z", v3_eq(rotated, v3(0, 1, 0), EPSILON));

    /* Test inverse */
    mat4 tinv = m4_inverse(t);
    vec3 back = m4_mul_point(tinv, v3(10, 20, 30));
    TEST("m4_inverse translate", v3_eq(back, v3(0, 0, 0), EPSILON));

    /* Test M * M^-1 = I */
    mat4 prod = m4_mul(t, tinv);
    TEST("m4_inverse identity", fabsf(M4(prod, 0, 0) - 1) < EPSILON &&
                                 fabsf(M4(prod, 0, 3)) < EPSILON);
}

static void test_quat(void) {
    printf("\n=== quaternion tests ===\n");

    quat id = q_identity();
    TEST("q_identity", id.w == 1 && id.x == 0 && id.y == 0 && id.z == 0);

    /* Rotate around Z by 90 degrees */
    quat rz = q_from_axis_angle(v3(0, 0, 1), (float)M_PI / 2);
    vec3 rotated = q_rotate(rz, v3(1, 0, 0));
    TEST("q_rotate Z90", v3_eq(rotated, v3(0, 1, 0), EPSILON));

    /* Quaternion conjugate */
    quat conj = q_conj(rz);
    TEST("q_conj", conj.w == rz.w && conj.x == -rz.x && conj.y == -rz.y && conj.z == -rz.z);

    /* q * q^-1 should give identity rotation */
    quat prod = q_mul(rz, q_conj(rz));
    prod = q_norm(prod);
    TEST("q_mul inverse", fabsf(prod.w) > 0.999f && fabsf(prod.x) < EPSILON);

    /* Test slerp */
    quat q1 = q_identity();
    quat q2 = q_from_axis_angle(v3(0, 0, 1), (float)M_PI);
    quat half = q_slerp(q1, q2, 0.5f);
    vec3 half_rot = q_rotate(half, v3(1, 0, 0));
    TEST("q_slerp", v3_eq(half_rot, v3(0, 1, 0), EPSILON));

    /* Test quaternion to matrix conversion */
    mat4 qmat = q_to_mat4(rz);
    vec3 mat_rot = m4_mul_dir(qmat, v3(1, 0, 0));
    TEST("q_to_mat4", v3_eq(mat_rot, v3(0, 1, 0), EPSILON));
}

static void test_transforms(void) {
    printf("\n=== transform tests ===\n");

    /* Test combined transforms */
    mat4 t = m4_translate(v3(5, 0, 0));
    mat4 r = m4_rotate_z((float)M_PI / 2);
    mat4 tr = m4_mul(t, r);  /* first rotate, then translate */
    vec3 p = m4_mul_point(tr, v3(1, 0, 0));
    TEST("combined rotate then translate", v3_eq(p, v3(5, 1, 0), EPSILON));

    mat4 rt = m4_mul(r, t);  /* first translate, then rotate */
    p = m4_mul_point(rt, v3(0, 0, 0));
    TEST("combined translate then rotate", v3_eq(p, v3(0, 5, 0), EPSILON));

    /* Test look_at creates proper view matrix */
    mat4 view = m4_look_at(v3(0, 0, 5), v3(0, 0, 0), v3(0, 1, 0));
    vec3 origin_in_view = m4_mul_point(view, v3(0, 0, 0));
    TEST("m4_look_at origin", fabsf(origin_in_view.z + 5) < EPSILON);

    /* Test perspective preserves origin direction */
    mat4 persp = m4_perspective((float)M_PI / 4, 1.0f, 0.1f, 100.0f);
    vec4 center = m4_mul_v4(persp, v4(0, 0, -1, 1));
    TEST("m4_perspective center", fabsf(center.x) < EPSILON && fabsf(center.y) < EPSILON);
}

static void test_axis_angle(void) {
    printf("\n=== axis-angle tests ===\n");

    /* Rotate (1,0,0) around (1,1,1) axis by 120 degrees -> should get (0,1,0) */
    vec3 axis = v3_norm(v3(1, 1, 1));
    mat4 r = m4_rotate_axis(axis, 2.0f * (float)M_PI / 3.0f);
    vec3 result = m4_mul_dir(r, v3(1, 0, 0));
    TEST("m4_rotate_axis 120deg", v3_eq(result, v3(0, 1, 0), EPSILON));

    /* Same test with quaternion */
    quat q = q_from_axis_angle(axis, 2.0f * (float)M_PI / 3.0f);
    result = q_rotate(q, v3(1, 0, 0));
    TEST("q_from_axis_angle 120deg", v3_eq(result, v3(0, 1, 0), EPSILON));
}

int main(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║  MujoCol Phase 2: 3D Math Unit Tests   ║\n");
    printf("╚════════════════════════════════════════╝\n");

    test_vec3();
    test_mat4();
    test_quat();
    test_transforms();
    test_axis_angle();

    printf("\n════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("════════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
