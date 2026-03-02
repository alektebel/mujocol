/*
 * phase2_math3d.c — 3D Math Library
 *
 * EXERCISE: Implement fundamental 3D math operations.
 *
 * Build: gcc -O2 -o phase2_math3d phase2_math3d.c -lm
 * Run:   ./phase2_math3d
 *
 * LEARNING GOALS:
 * - Vector operations: dot product, cross product, normalization
 * - Matrix multiplication and inverse
 * - Quaternion rotation
 * - Transform builders (translate, rotate, perspective, look-at)
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

/* Access element at row, column (0-indexed) */
#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

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

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement dot product
 *
 * The dot product of two vectors a·b = ax*bx + ay*by + az*bz
 * Returns a scalar representing how "aligned" the vectors are
 * ══════════════════════════════════════════════════════════════════ */
static inline float v3_dot(vec3 a, vec3 b) {
    /* TODO: Implement dot product */
    (void)a; (void)b;
    return 0.0f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement cross product
 *
 * The cross product a×b gives a vector perpendicular to both a and b
 * Formula: (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_cross(vec3 a, vec3 b) {
    /* TODO: Implement cross product */
    (void)a; (void)b;
    return v3(0, 0, 0);
}

static inline float v3_len(vec3 v) {
    return sqrtf(v3_dot(v, v));
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement vector normalization
 *
 * Return a unit vector (length = 1) in the same direction
 * Handle the case where length is near zero (return zero vector)
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_norm(vec3 v) {
    /* TODO: Normalize vector to unit length */
    (void)v;
    return v3(0, 0, 0);
}

static inline vec3 v3_lerp(vec3 a, vec3 b, float t) {
    return v3_add(v3_scale(a, 1.0f - t), v3_scale(b, t));
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement reflection
 *
 * Reflect vector v around normal n
 * Formula: v - 2*(v·n)*n
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_reflect(vec3 v, vec3 n) {
    /* TODO: Implement reflection */
    (void)v; (void)n;
    return v3(0, 0, 0);
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

static inline float v4_dot(vec4 a, vec4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/* ══════════════════════════════════════════════════════════════════
 * MAT4 OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

static inline mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement matrix multiplication
 *
 * Multiply two 4x4 matrices: result = a * b
 * Use the M4(mat, row, col) macro to access elements
 * Remember: column-major order, result[row,col] = sum of a[row,k]*b[k,col]
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_mul(mat4 a, mat4 b) {
    /* TODO: Implement 4x4 matrix multiplication */
    (void)a; (void)b;
    return m4_identity();
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

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement matrix inverse (HARD - can skip initially)
 *
 * Compute the inverse of a 4x4 matrix using cofactor expansion
 * This is complex - see solution for reference or use online resources
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_inverse(mat4 m) {
    /* TODO: Implement 4x4 matrix inverse (complex!) */
    (void)m;
    return m4_identity();
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

/* ══════════════════════════════════════════════════════════════════
 * TODO #7: Implement rotation around Z axis
 *
 * Rotation matrix for angle θ around Z:
 * | cos(θ)  -sin(θ)  0  0 |
 * | sin(θ)   cos(θ)  0  0 |
 * |   0        0     1  0 |
 * |   0        0     0  1 |
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_rotate_z(float rad) {
    /* TODO: Implement rotation around Z axis */
    (void)rad;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #8: Implement rotation around arbitrary axis
 *
 * Use Rodrigues' rotation formula. Given normalized axis (x,y,z) and angle θ:
 * c = cos(θ), s = sin(θ), t = 1 - cos(θ)
 * | t*x*x+c    t*x*y-s*z  t*x*z+s*y  0 |
 * | t*x*y+s*z  t*y*y+c    t*y*z-s*x  0 |
 * | t*x*z-s*y  t*y*z+s*x  t*z*z+c    0 |
 * |    0          0          0      1 |
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_rotate_axis(vec3 axis, float rad) {
    /* TODO: Implement rotation around arbitrary axis */
    (void)axis; (void)rad;
    return m4_identity();
}

static mat4 m4_perspective(float fov_rad, float aspect, float near, float far) {
    mat4 m = {0};
    float tan_half_fov = tanf(fov_rad / 2.0f);
    M4(m, 0, 0) = 1.0f / (aspect * tan_half_fov);
    M4(m, 1, 1) = 1.0f / tan_half_fov;
    M4(m, 2, 2) = -(far + near) / (far - near);
    M4(m, 2, 3) = -(2.0f * far * near) / (far - near);
    M4(m, 3, 2) = -1.0f;
    return m;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #9: Implement look-at matrix
 *
 * Creates a view matrix looking from 'eye' toward 'target' with 'up' direction
 * 1. Compute forward = normalize(target - eye)
 * 2. Compute right = normalize(cross(forward, up))
 * 3. Compute up = cross(right, forward)
 * 4. Build rotation and translation matrix
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_look_at(vec3 eye, vec3 target, vec3 up) {
    /* TODO: Implement look-at view matrix */
    (void)eye; (void)target; (void)up;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * QUATERNION OPERATIONS
 * ══════════════════════════════════════════════════════════════════ */

static inline quat q_identity(void) {
    return (quat){1, 0, 0, 0};
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #10: Implement quaternion from axis-angle
 *
 * Given axis (normalized) and angle θ:
 * w = cos(θ/2)
 * x = axis.x * sin(θ/2)
 * y = axis.y * sin(θ/2)
 * z = axis.z * sin(θ/2)
 * ══════════════════════════════════════════════════════════════════ */
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    /* TODO: Create quaternion from axis and angle */
    (void)axis; (void)rad;
    return q_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #11: Implement quaternion multiplication
 *
 * Quaternion multiplication (Hamilton product):
 * (a.w + a.x*i + a.y*j + a.z*k) * (b.w + b.x*i + b.y*j + b.z*k)
 * w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
 * x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
 * y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
 * z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
 * ══════════════════════════════════════════════════════════════════ */
static inline quat q_mul(quat a, quat b) {
    /* TODO: Implement quaternion multiplication */
    (void)a; (void)b;
    return q_identity();
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

/* ══════════════════════════════════════════════════════════════════
 * TODO #12: Implement quaternion rotation of a vector
 *
 * Rotate vector v by quaternion q using: q * v * q^(-1)
 * Optimized formula using vector math:
 *   qv = (q.x, q.y, q.z)
 *   uv = cross(qv, v)
 *   uuv = cross(qv, uv)
 *   return v + 2 * (q.w * uv + uuv)
 * ══════════════════════════════════════════════════════════════════ */
static vec3 q_rotate(quat q, vec3 v) {
    /* TODO: Rotate vector by quaternion */
    (void)q; (void)v;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #13: Implement quaternion to rotation matrix
 *
 * Convert unit quaternion to 3x3 rotation matrix (in 4x4 form)
 * ══════════════════════════════════════════════════════════════════ */
static mat4 q_to_mat4(quat q) {
    /* TODO: Convert quaternion to rotation matrix */
    (void)q;
    return m4_identity();
}

static quat q_slerp(quat a, quat b, float t) {
    float dot = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
    if (dot < 0) {
        b = (quat){-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }
    if (dot > 0.9995f) {
        quat r = { a.w + t*(b.w - a.w), a.x + t*(b.x - a.x),
                   a.y + t*(b.y - a.y), a.z + t*(b.z - a.z) };
        return q_norm(r);
    }
    float theta = acosf(dot);
    float sin_theta = sinf(theta);
    float wa = sinf((1-t)*theta) / sin_theta;
    float wb = sinf(t*theta) / sin_theta;
    return (quat){ wa*a.w + wb*b.w, wa*a.x + wb*b.x,
                   wa*a.y + wb*b.y, wa*a.z + wb*b.z };
}

/* ══════════════════════════════════════════════════════════════════
 * UNIT TESTS - DO NOT MODIFY
 * ══════════════════════════════════════════════════════════════════ */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name, cond) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

static void test_vec3(void) {
    printf("\n=== vec3 tests ===\n");
    vec3 a = v3(1, 2, 3);
    vec3 b = v3(4, 5, 6);

    TEST("v3 creation", a.x == 1 && a.y == 2 && a.z == 3);
    TEST("v3_add", v3_eq(v3_add(a, b), v3(5, 7, 9), EPSILON));
    TEST("v3_sub", v3_eq(v3_sub(b, a), v3(3, 3, 3), EPSILON));
    TEST("v3_scale", v3_eq(v3_scale(a, 2), v3(2, 4, 6), EPSILON));
    TEST("v3_dot", fabsf(v3_dot(a, b) - 32.0f) < EPSILON);
    TEST("v3_cross", v3_eq(v3_cross(v3(1,0,0), v3(0,1,0)), v3(0,0,1), EPSILON));
    TEST("v3_norm", v3_eq(v3_norm(v3(3,0,0)), v3(1,0,0), EPSILON));
    TEST("v3_len", fabsf(v3_len(v3(3,4,0)) - 5.0f) < EPSILON);
    TEST("v3_lerp", v3_eq(v3_lerp(v3(0,0,0), v3(10,10,10), 0.5f), v3(5,5,5), EPSILON));
    TEST("v3_reflect", v3_eq(v3_reflect(v3(1,-1,0), v3(0,1,0)), v3(1,1,0), EPSILON));
}

static void test_mat4(void) {
    printf("\n=== mat4 tests ===\n");
    mat4 id = m4_identity();

    TEST("m4_identity diagonal", M4(id,0,0)==1 && M4(id,1,1)==1 && M4(id,2,2)==1 && M4(id,3,3)==1);
    TEST("m4_identity off-diagonal", M4(id,0,1)==0 && M4(id,1,0)==0);

    mat4 id2 = m4_mul(id, id);
    TEST("m4_mul identity", M4(id2,0,0)==1 && M4(id2,3,3)==1);

    mat4 m = m4_identity(); M4(m,0,1) = 5; M4(m,1,0) = 10;
    mat4 mt = m4_transpose(m);
    TEST("m4_transpose", M4(mt,1,0)==5 && M4(mt,0,1)==10);

    mat4 t = m4_translate(v3(10, 20, 30));
    vec3 p = m4_mul_point(t, v3(0, 0, 0));
    TEST("m4_translate point", v3_eq(p, v3(10, 20, 30), EPSILON));

    vec3 d = m4_mul_dir(t, v3(1, 0, 0));
    TEST("m4_translate direction", v3_eq(d, v3(1, 0, 0), EPSILON));

    mat4 s = m4_scale(v3(2, 3, 4));
    TEST("m4_scale", v3_eq(m4_mul_point(s, v3(1,1,1)), v3(2,3,4), EPSILON));

    mat4 rz = m4_rotate_z((float)M_PI / 2);
    TEST("m4_rotate_z", v3_eq(m4_mul_dir(rz, v3(1,0,0)), v3(0,1,0), EPSILON));

    mat4 tinv = m4_inverse(t);
    TEST("m4_inverse translate", v3_eq(m4_mul_point(tinv, v3(10,20,30)), v3(0,0,0), EPSILON));

    mat4 prod = m4_mul(t, tinv);
    TEST("m4_inverse identity", fabsf(M4(prod,0,0)-1) < EPSILON && fabsf(M4(prod,0,3)) < EPSILON);
}

static void test_quat(void) {
    printf("\n=== quaternion tests ===\n");
    quat id = q_identity();
    TEST("q_identity", id.w==1 && id.x==0 && id.y==0 && id.z==0);

    quat rz = q_from_axis_angle(v3(0, 0, 1), (float)M_PI / 2);
    TEST("q_rotate Z90", v3_eq(q_rotate(rz, v3(1,0,0)), v3(0,1,0), EPSILON));

    quat conj = q_conj(rz);
    TEST("q_conj", conj.w==rz.w && conj.x==-rz.x && conj.y==-rz.y && conj.z==-rz.z);

    quat qprod = q_mul(rz, q_conj(rz));
    qprod = q_norm(qprod);
    TEST("q_mul inverse", fabsf(qprod.w) > 0.999f && fabsf(qprod.x) < EPSILON);

    quat q1 = q_identity();
    quat q2 = q_from_axis_angle(v3(0,0,1), (float)M_PI);
    quat half = q_slerp(q1, q2, 0.5f);
    TEST("q_slerp", v3_eq(q_rotate(half, v3(1,0,0)), v3(0,1,0), EPSILON));

    mat4 qmat = q_to_mat4(rz);
    TEST("q_to_mat4", v3_eq(m4_mul_dir(qmat, v3(1,0,0)), v3(0,1,0), EPSILON));
}

static void test_transforms(void) {
    printf("\n=== transform tests ===\n");
    mat4 t = m4_translate(v3(5, 0, 0));
    mat4 r = m4_rotate_z((float)M_PI / 2);

    mat4 tr = m4_mul(t, r);
    TEST("combined rotate then translate", v3_eq(m4_mul_point(tr, v3(1,0,0)), v3(5,1,0), EPSILON));

    mat4 rt = m4_mul(r, t);
    TEST("combined translate then rotate", v3_eq(m4_mul_point(rt, v3(0,0,0)), v3(0,5,0), EPSILON));

    mat4 view = m4_look_at(v3(0,0,5), v3(0,0,0), v3(0,1,0));
    vec3 origin_in_view = m4_mul_point(view, v3(0,0,0));
    TEST("m4_look_at origin", fabsf(origin_in_view.z + 5) < EPSILON);

    mat4 persp = m4_perspective((float)M_PI/4, 1.0f, 0.1f, 100.0f);
    vec4 center = m4_mul_v4(persp, v4(0,0,-1,1));
    TEST("m4_perspective center", fabsf(center.x) < EPSILON && fabsf(center.y) < EPSILON);
}

static void test_axis_angle(void) {
    printf("\n=== axis-angle tests ===\n");
    vec3 axis = v3_norm(v3(1, 1, 1));

    mat4 r = m4_rotate_axis(axis, 2.0f * (float)M_PI / 3.0f);
    TEST("m4_rotate_axis 120deg", v3_eq(m4_mul_dir(r, v3(1,0,0)), v3(0,1,0), EPSILON));

    quat q = q_from_axis_angle(axis, 2.0f * (float)M_PI / 3.0f);
    TEST("q_from_axis_angle 120deg", v3_eq(q_rotate(q, v3(1,0,0)), v3(0,1,0), EPSILON));
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
