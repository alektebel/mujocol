/*
 * phase2c_quat.c — Quaternion Math and Combined Transforms
 *
 * EXERCISE (sub-phase 2c): Implement quaternion rotation and combined
 * transform pipelines.
 *
 * Build: gcc -O2 -o phase2c_quat phase2c_quat.c -lm
 * Run:   ./phase2c_quat
 *
 * LEARNING GOALS:
 * - Understand quaternions as a compact representation of 3D rotations
 * - Implement quaternion multiplication (Hamilton product)
 * - Convert between quaternions, axis-angle, and rotation matrices
 * - Use SLERP for smooth quaternion interpolation
 * - Build combined transform pipelines (TRS = Translate * Rotate * Scale)
 *
 * ──────────────────────────────────────────────────────────────────
 * TECHNIQUE OVERVIEW
 * ──────────────────────────────────────────────────────────────────
 *
 * 1. Quaternions
 *    A quaternion q = w + xi + yj + zk has a scalar part w and a 3D
 *    vector part (x,y,z). The imaginary units satisfy
 *    i²=j²=k²=ijk=-1. A unit quaternion (|q|=1) represents a 3D
 *    rotation: rotating by angle θ around unit axis n gives
 *    q = (cos(θ/2), sin(θ/2)*n). Note: in this code the struct is
 *    defined as {w, x, y, z} — w first.
 *
 * 2. Why quaternions over matrices/Euler?
 *    Three reasons: (a) 4 floats vs 16 (compact); (b) no gimbal lock
 *    (a problem with Euler angles where one rotational degree of
 *    freedom can be lost); (c) smooth SLERP interpolation between
 *    rotations.
 *
 * 3. Gimbal lock
 *    When using Euler angles (yaw/pitch/roll), if pitch=90° the yaw
 *    and roll axes align, losing one degree of freedom. Quaternions
 *    don't have this problem because they live in a 4D space with no
 *    such degeneracies.
 *
 * 4. Hamilton product
 *    Quaternion multiplication is non-commutative (q1*q2 ≠ q2*q1).
 *    Multiplying two unit quaternions q1*q2 gives a unit quaternion
 *    that represents the combined rotation: "first rotate by q2, then
 *    by q1" (right-to-left convention, like matrix multiplication).
 *
 * 5. SLERP
 *    Spherical Linear intERPolation: interpolates between two
 *    quaternions along the great-circle arc on the unit 4-sphere.
 *    Formula: slerp(q1,q2,t) = q1*(sin((1-t)*Ω)/sin(Ω)) +
 *    q2*(sin(t*Ω)/sin(Ω)), where Ω = acos(q1·q2). Falls back to
 *    linear interpolation when Ω≈0 (quaternions nearly identical).
 *    Essential for smooth animation blending.
 *
 * 6. TRS = Translate * Rotate * Scale
 *    The standard decomposition of a rigid body transform. "Scale
 *    first" means the geometry is scaled in local space before any
 *    rotation or translation is applied.
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
 * VEC3 / VEC4 — provided, do not modify
 * ══════════════════════════════════════════════════════════════════ */

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }

static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x*s, v.y*s, v.z*s); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }

static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    return len < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f / len);
}

static inline int v3_eq(vec3 a, vec3 b, float eps) {
    return fabsf(a.x-b.x) < eps && fabsf(a.y-b.y) < eps && fabsf(a.z-b.z) < eps;
}

static inline vec4 v4(float x, float y, float z, float w) { return (vec4){x, y, z, w}; }
static inline vec4 v4_from_v3(vec3 v, float w) { return v4(v.x, v.y, v.z, w); }
static inline vec3 v4_to_v3(vec4 v) { return v3(v.x, v.y, v.z); }

/* ══════════════════════════════════════════════════════════════════
 * MAT4 — provided, do not modify (m4_mul is fully implemented here)
 * ══════════════════════════════════════════════════════════════════ */

static inline mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            for (int k = 0; k < 4; k++)
                M4(r, row, col) += M4(a, row, k) * M4(b, k, col);
    return r;
}

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

static vec4 m4_mul_v4(mat4 m, vec4 v) {
    return v4(
        M4(m,0,0)*v.x + M4(m,0,1)*v.y + M4(m,0,2)*v.z + M4(m,0,3)*v.w,
        M4(m,1,0)*v.x + M4(m,1,1)*v.y + M4(m,1,2)*v.z + M4(m,1,3)*v.w,
        M4(m,2,0)*v.x + M4(m,2,1)*v.y + M4(m,2,2)*v.z + M4(m,2,3)*v.w,
        M4(m,3,0)*v.x + M4(m,3,1)*v.y + M4(m,3,2)*v.z + M4(m,3,3)*v.w
    );
}

static vec3 m4_mul_point(mat4 m, vec3 p) {
    vec4 r = m4_mul_v4(m, v4_from_v3(p, 1.0f));
    if (fabsf(r.w) > EPSILON) return v3(r.x/r.w, r.y/r.w, r.z/r.w);
    return v4_to_v3(r);
}

static vec3 m4_mul_dir(mat4 m, vec3 d) {
    return v4_to_v3(m4_mul_v4(m, v4_from_v3(d, 0.0f)));
}

/* ══════════════════════════════════════════════════════════════════
 * QUATERNION — provided operations, do not modify
 * ══════════════════════════════════════════════════════════════════ */

static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }

static inline quat q_conj(quat q) { return (quat){q.w, -q.x, -q.y, -q.z}; }

static inline float q_len(quat q) {
    return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
}

static inline quat q_norm(quat q) {
    float len = q_len(q);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f / len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}

static quat q_slerp(quat a, quat b, float t) {
    float dot = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
    /* dot < 0 means the two quaternions are on opposite hemispheres of
     * the 4-sphere. Negating one of them (q2 = -q2) gives the equivalent
     * rotation but ensures we take the short path (< 180°), avoiding
     * spinning the 'long way around'. */
    if (dot < 0) {
        b = (quat){-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }
    if (dot > 0.9995f) {
        quat r = { a.w + t*(b.w - a.w), a.x + t*(b.x - a.x),
                   a.y + t*(b.y - a.y), a.z + t*(b.z - a.z) };
        return q_norm(r);
    }
    float theta     = acosf(dot);
    float sin_theta = sinf(theta);
    float wa = sinf((1.0f - t) * theta) / sin_theta;
    float wb = sinf(t * theta) / sin_theta;
    return (quat){ wa*a.w + wb*b.w, wa*a.x + wb*b.x,
                   wa*a.y + wb*b.y, wa*a.z + wb*b.z };
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement quaternion from axis-angle
 *
 * Given a (normalized) axis and angle θ:
 *   w = cos(θ/2)
 *   x = axis.x * sin(θ/2)
 *   y = axis.y * sin(θ/2)
 *   z = axis.z * sin(θ/2)
 * Normalize the axis first if it is not already a unit vector.
 *
 * A unit quaternion for rotation θ around normalized axis n is:
 * w = cos(θ/2), (x,y,z) = sin(θ/2) * n. The half-angle is key:
 * going from angle to quaternion halves the angle. This is why
 * rotating by 360° gives q = (-1,0,0,0) (not identity): you need
 * 720° to return to identity in quaternion space — but both q and
 * -q represent the same rotation.
 * ══════════════════════════════════════════════════════════════════ */
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    /* TODO: create quaternion from axis and angle */
    (void)axis; (void)rad;
    return q_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement quaternion multiplication (Hamilton product)
 *
 * w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
 * x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y
 * y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x
 * z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
 *
 * The Hamilton product of q1=(w1,x1,y1,z1) and q2=(w2,x2,y2,z2):
 *   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
 *   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
 *   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
 *   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
 * The negative terms come from the rules i²=-1, j²=-1, k²=-1,
 * ij=k, jk=i, ki=j, ji=-k, kj=-i, ik=-j.
 * ══════════════════════════════════════════════════════════════════ */
static inline quat q_mul(quat a, quat b) {
    /* TODO: implement Hamilton product */
    (void)a; (void)b;
    return q_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement quaternion rotation of a vector
 *
 * Efficient formula (avoids the full q*v*q⁻¹ chain):
 *   qv  = (q.x, q.y, q.z)
 *   uv  = cross(qv, v)
 *   uuv = cross(qv, uv)
 *   return v + 2 * (q.w * uv + uuv)
 *
 * To rotate vector v by unit quaternion q: treat v as a pure
 * quaternion p=(0,v.x,v.y,v.z), then compute q*p*q_conj, where
 * q_conj=(w,-x,-y,-z). The xyz part of the result is the rotated
 * vector. An alternative (faster) formula using the cross product:
 *   result = v + 2*q.w*(q_vec × v) + 2*(q_vec × (q_vec × v)),
 * where q_vec=(q.x,q.y,q.z).
 * ══════════════════════════════════════════════════════════════════ */
static vec3 q_rotate(quat q, vec3 v) {
    /* TODO: rotate vector v by quaternion q */
    (void)q; (void)v;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement quaternion → 4×4 rotation matrix
 *
 * For unit quaternion (w,x,y,z) the rotation matrix is:
 * | 1-2(y²+z²)   2(xy-wz)    2(xz+wy)   0 |
 * |  2(xy+wz)   1-2(x²+z²)   2(yz-wx)   0 |
 * |  2(xz-wy)    2(yz+wx)   1-2(x²+y²)  0 |
 * |     0           0           0        1 |
 *
 * Converting a unit quaternion (w,x,y,z) to a 3×3 rotation matrix:
 *   m[0][0]=1-2(y²+z²), m[0][1]=2(xy-wz),   m[0][2]=2(xz+wy)
 *   m[1][0]=2(xy+wz),   m[1][1]=1-2(x²+z²), m[1][2]=2(yz-wx)
 *   m[2][0]=2(xz-wy),   m[2][1]=2(yz+wx),   m[2][2]=1-2(x²+y²)
 * The fourth row and column are [0,0,0,1] (no translation). Embed
 * this 3×3 into a 4×4 using M4() and return.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 q_to_mat4(quat q) {
    /* TODO: convert quaternion to 4×4 rotation matrix */
    (void)q;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement TRS transform matrix
 *
 * Build a combined Translate × Rotate × Scale matrix:
 *   TRS = m4_translate(t) × q_to_mat4(r) × m4_scale(s)
 * This is the standard model-to-world transform in 3D engines.
 *
 * TRS = m4_translate(t) * q_to_mat4(r) * m4_scale(s). Order matters:
 * scale in local space first (s), then rotate to world orientation
 * (r), then translate to world position (t). Using m4_mul from
 * phase2b to combine them.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_trs(vec3 t, quat r, vec3 s) {
    /* TODO: return m4_translate(t) * q_to_mat4(r) * m4_scale(s) */
    (void)t; (void)r; (void)s;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement quaternion from Euler angles (XYZ order)
 *
 * Build individual axis quaternions then compose in XYZ order:
 *   q = q_z * q_y * q_x
 * where q_x = q_from_axis_angle((1,0,0), rx), etc.
 *
 * Euler angles (roll=X, pitch=Y, yaw=Z) are applied in XYZ order:
 * create three quaternions:
 *   qx = q_from_axis_angle({1,0,0}, roll)
 *   qy = q_from_axis_angle({0,1,0}, pitch)
 *   qz = q_from_axis_angle({0,0,1}, yaw)
 * then multiply: q = qz * qy * qx (right-to-left: X applied first).
 * The order is critical — different orders give different orientations.
 * ══════════════════════════════════════════════════════════════════ */
static quat q_from_euler(float rx, float ry, float rz) {
    /* TODO: compose Euler angles into a single quaternion */
    (void)rx; (void)ry; (void)rz;
    return q_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * UNIT TESTS — DO NOT MODIFY
 * ══════════════════════════════════════════════════════════════════ */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name, cond) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else { printf("  FAIL: %s\n", name); } \
} while(0)

static void test_provided_mat4(void) {
    printf("\n=== provided mat4 operations ===\n");
    /* m4_mul is fully implemented in this file — verify it works */
    mat4 t  = m4_translate(v3(1,2,3));
    mat4 s  = m4_scale(v3(2,2,2));
    mat4 ts = m4_mul(t, s);
    TEST("T×S: (1,0,0) → (3,2,3)",   v3_eq(m4_mul_point(ts, v3(1,0,0)), v3(3,2,3), EPSILON));
    mat4 st = m4_mul(s, t);
    TEST("S×T: (0,0,0) → (2,4,6)",   v3_eq(m4_mul_point(st, v3(0,0,0)), v3(2,4,6), EPSILON));
}

static void test_q_from_axis_angle(void) {
    printf("\n=== TODO #1: q_from_axis_angle ===\n");
    quat qz = q_from_axis_angle(v3(0,0,1), (float)M_PI / 2.0f);
    float expected_w = cosf((float)M_PI / 4.0f);
    float expected_z = sinf((float)M_PI / 4.0f);
    TEST("w = cos(θ/2)",       fabsf(qz.w - expected_w) < EPSILON);
    TEST("z = sin(θ/2)",       fabsf(qz.z - expected_z) < EPSILON);
    TEST("x = y = 0",          fabsf(qz.x) < EPSILON && fabsf(qz.y) < EPSILON);
    TEST("result is unit",     fabsf(q_len(qz) - 1.0f) < EPSILON);
}

static void test_q_mul(void) {
    printf("\n=== TODO #2: q_mul ===\n");
    quat id = q_identity();
    quat q  = q_from_axis_angle(v3(0,0,1), (float)M_PI / 2.0f);
    TEST("identity left",      fabsf(q_mul(id, q).w - q.w) < EPSILON &&
                               fabsf(q_mul(id, q).z - q.z) < EPSILON);
    TEST("identity right",     fabsf(q_mul(q, id).w - q.w) < EPSILON &&
                               fabsf(q_mul(q, id).z - q.z) < EPSILON);

    quat prod = q_norm(q_mul(q, q_conj(q)));
    TEST("q × conj(q) = I",    fabsf(prod.w - 1.0f) < EPSILON && fabsf(prod.x) < EPSILON);

    /* Two successive 90° Z rotations equal one 180° rotation */
    quat q180 = q_mul(q, q);
    TEST("90°+90°=180°: x→-x", v3_eq(q_rotate(q180, v3(1,0,0)), v3(-1,0,0), EPSILON));
}

static void test_q_rotate(void) {
    printf("\n=== TODO #3: q_rotate ===\n");
    quat rz = q_from_axis_angle(v3(0,0,1), (float)M_PI / 2.0f);
    TEST("Z90°: x→y",         v3_eq(q_rotate(rz, v3(1,0,0)), v3(0,1,0), EPSILON));
    TEST("Z90°: y→-x",        v3_eq(q_rotate(rz, v3(0,1,0)), v3(-1,0,0), EPSILON));
    TEST("Z90°: z unchanged",  v3_eq(q_rotate(rz, v3(0,0,1)), v3(0,0,1), EPSILON));

    quat rx = q_from_axis_angle(v3(1,0,0), (float)M_PI / 2.0f);
    TEST("X90°: y→z",         v3_eq(q_rotate(rx, v3(0,1,0)), v3(0,0,1), EPSILON));
}

static void test_q_to_mat4(void) {
    printf("\n=== TODO #4: q_to_mat4 ===\n");
    quat rz = q_from_axis_angle(v3(0,0,1), (float)M_PI / 2.0f);
    mat4 mz = q_to_mat4(rz);
    TEST("Z90°: x→y",         v3_eq(m4_mul_dir(mz, v3(1,0,0)), v3(0,1,0), EPSILON));
    TEST("Z90°: y→-x",        v3_eq(m4_mul_dir(mz, v3(0,1,0)), v3(-1,0,0), EPSILON));

    mat4 mid = q_to_mat4(q_identity());
    TEST("identity quat → I",  fabsf(M4(mid,0,0)-1)<EPSILON && fabsf(M4(mid,0,1))<EPSILON);

    /* 120° around (1,1,1)/√3 cyclically permutes basis vectors */
    quat qa = q_from_axis_angle(v3_norm(v3(1,1,1)), 2.0f*(float)M_PI/3.0f);
    mat4 ma = q_to_mat4(qa);
    TEST("120° (1,1,1): x→y", v3_eq(m4_mul_dir(ma, v3(1,0,0)), v3(0,1,0), EPSILON));
}

static void test_m4_trs(void) {
    printf("\n=== TODO #5: m4_trs ===\n");
    quat rz90 = q_from_axis_angle(v3(0,0,1), (float)M_PI / 2.0f);

    /* Translate(5,0,0) × RotZ90 × Scale(1,1,1): (1,0,0) → rotate → (0,1,0) → translate → (5,1,0) */
    mat4 trs = m4_trs(v3(5,0,0), rz90, v3(1,1,1));
    TEST("TRS: (1,0,0) → (5,1,0)",   v3_eq(m4_mul_point(trs, v3(1,0,0)), v3(5,1,0), EPSILON));
    TEST("TRS: (0,0,0) → (5,0,0)",   v3_eq(m4_mul_point(trs, v3(0,0,0)), v3(5,0,0), EPSILON));

    /* Scale only */
    mat4 s_trs = m4_trs(v3(0,0,0), q_identity(), v3(2,3,4));
    TEST("TRS scale: (1,1,1) → (2,3,4)", v3_eq(m4_mul_point(s_trs, v3(1,1,1)), v3(2,3,4), EPSILON));
}

static void test_q_from_euler(void) {
    printf("\n=== TODO #6: q_from_euler ===\n");
    /* Pure Z rotation */
    quat qe_z = q_from_euler(0.0f, 0.0f, (float)M_PI / 2.0f);
    TEST("euler Z90: x→y",     v3_eq(q_rotate(qe_z, v3(1,0,0)), v3(0,1,0), EPSILON));

    /* Pure X rotation */
    quat qe_x = q_from_euler((float)M_PI / 2.0f, 0.0f, 0.0f);
    TEST("euler X90: y→z",     v3_eq(q_rotate(qe_x, v3(0,1,0)), v3(0,0,1), EPSILON));

    /* Zero angles → identity */
    quat qe_id = q_from_euler(0.0f, 0.0f, 0.0f);
    TEST("euler zeros = I",    fabsf(qe_id.w - 1.0f) < EPSILON && fabsf(qe_id.x) < EPSILON);

    /* Euler X90 then Z90 must match manual composition q_z * q_y * q_x */
    quat q_manual = q_mul(q_from_axis_angle(v3(0,0,1), (float)M_PI/2.0f),
                    q_mul(q_from_axis_angle(v3(0,1,0), 0.0f),
                          q_from_axis_angle(v3(1,0,0), (float)M_PI/2.0f)));
    quat qe_xz    = q_from_euler((float)M_PI/2.0f, 0.0f, (float)M_PI/2.0f);
    TEST("euler XZ compose matches manual",
         v3_eq(q_rotate(qe_xz, v3(1,0,0)), q_rotate(q_manual, v3(1,0,0)), EPSILON));
}

static void test_slerp(void) {
    printf("\n=== q_slerp (provided) ===\n");
    quat q1   = q_identity();
    quat q2   = q_from_axis_angle(v3(0,0,1), (float)M_PI);
    quat half = q_slerp(q1, q2, 0.5f);
    TEST("slerp half-way = 90° Z", v3_eq(q_rotate(half, v3(1,0,0)), v3(0,1,0), EPSILON));
}

int main(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║  MujoCol Phase 2c: Quaternion Tests    ║\n");
    printf("╚════════════════════════════════════════╝\n");

    test_provided_mat4();
    test_q_from_axis_angle();
    test_q_mul();
    test_q_rotate();
    test_q_to_mat4();
    test_m4_trs();
    test_q_from_euler();
    test_slerp();

    printf("\n════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("════════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
