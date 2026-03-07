/*
 * phase2b_mat4.c — 4x4 Matrix Operations
 *
 * EXERCISE (sub-phase 2b): Implement 4x4 matrix operations for 3D transforms.
 *
 * Build: gcc -O2 -o phase2b_mat4 phase2b_mat4.c -lm
 * Run:   ./phase2b_mat4
 *
 * LEARNING GOALS:
 * - Understand column-major matrix storage (OpenGL convention)
 * - Implement matrix multiplication using the dot product of rows and columns
 * - Build rotation matrices from Rodrigues' formula
 * - Construct look-at view matrices for cameras
 * - Compute the inverse of affine transform matrices
 *
 * ──────────────────────────────────────────────────────────────────
 * TECHNIQUE OVERVIEW
 * ──────────────────────────────────────────────────────────────────
 *
 * 1. Column-major matrix storage
 *    A 4×4 matrix has 16 floats. OpenGL (and this code) stores them
 *    in column-major order: column 0 first, then column 1, etc.
 *    The flat array index for element at row r, column c is: c*4 + r.
 *    The macro M4(mat, row, col) encodes this: (mat).m[(col)*4 + (row)].
 *    This differs from row-major (C's native 2D array) where the index
 *    would be r*4 + c. When you read M4(m, row, col) it looks natural,
 *    but the memory layout is column-first.
 *
 * 2. Homogeneous coordinates
 *    We use 4D vectors (x, y, z, w) to represent 3D points and
 *    directions in a unified way. Points have w=1; directions/vectors
 *    have w=0. This allows translation to be expressed as a matrix
 *    multiply (not possible with pure 3×3 matrices). The fourth row
 *    [0,0,0,1] ensures w stays 1 after multiplying by an affine matrix.
 *
 * 3. vec4
 *    New in this phase: a 4-tuple (x,y,z,w). v4_from_v3(p, 1.0f) turns
 *    a 3D point into homogeneous form. v4_to_v3 drops the w component.
 *    m4_mul_point divides by w after the multiply (perspective divide)
 *    to handle projective transforms.
 *
 * 4. m4_identity
 *    The identity matrix has 1s on the diagonal and 0s elsewhere. Any
 *    vector multiplied by identity is unchanged. Note:
 *    m.m[0]=m.m[5]=m.m[10]=m.m[15]=1.0f exploits column-major layout:
 *    index 0 is [row=0,col=0], index 5 is [row=1,col=1], etc.
 *
 * 5. Transform composition
 *    The power of matrices: T*R*S (translate, rotate, scale) means
 *    "scale first, then rotate, then translate" when we write M * point.
 *    Matrix multiplication is associative: (A*B)*C = A*(B*C).
 *    Non-commutative: A*B ≠ B*A in general.
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

/* Column-major 4x4 matrix (OpenGL convention) */
typedef struct { float m[16]; } mat4;

/* Access element at row, column (0-indexed) */
#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

/* ══════════════════════════════════════════════════════════════════
 * VEC3 / VEC4 OPERATIONS — provided, do not modify
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
 * MAT4 — provided operations, do not modify
 * ══════════════════════════════════════════════════════════════════ */

static inline mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static mat4 m4_transpose(mat4 m) {
    mat4 r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            M4(r, i, j) = M4(m, j, i);
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

/* Perspective projection — provided for reference */
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
 * TODO #1: Implement matrix multiplication
 *
 * Multiply two 4×4 matrices: result = a * b
 * Use the M4(mat, row, col) macro to access elements.
 * result[row, col] = sum over k of a[row, k] * b[k, col]
 * Remember: column-major storage — M4 handles the index math for you.
 *
 * Matrix multiplication C=A×B means: C[row,col] = sum of
 * A[row,k]*B[k,col] for k=0..3. Think of it as: each element of the
 * result is the dot product of one row of A with one column of B.
 * We use three nested loops: outer=row, middle=col, inner=k.
 * The M4 macro handles the column-major index arithmetic.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_mul(mat4 a, mat4 b) {
    /* TODO: implement 4×4 matrix multiplication */
    (void)a; (void)b;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement rotation around Z axis
 *
 * Rotation matrix for angle θ around Z:
 * | cos(θ)  -sin(θ)   0   0 |
 * | sin(θ)   cos(θ)   0   0 |
 * |   0        0      1   0 |
 * |   0        0      0   1 |
 *
 * A pure rotation leaves vectors in the rotation plane, while the
 * perpendicular axis is unchanged. For Z-rotation: vectors in the XY
 * plane rotate by θ, while Z is fixed. The cos/sin entries come from
 * the unit circle: a vector (1,0) rotated by θ becomes (cosθ, sinθ).
 * The negative sin in position [0,1] comes from the right-hand rule
 * (counter-clockwise positive when looking down +Z).
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_rotate_z(float rad) {
    /* TODO: build Z-rotation matrix */
    (void)rad;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement rotation around X axis
 *
 * Rotation matrix for angle θ around X:
 * | 1    0       0    0 |
 * | 0  cos(θ)  -sin(θ)  0 |
 * | 0  sin(θ)   cos(θ)  0 |
 * | 0    0       0    1 |
 *
 * Same pattern as Z-rotation but now X is the fixed axis, and the YZ
 * plane rotates. When implementing: start from m4_identity(), then
 * set the four elements that change:
 *   [1,1]=cos, [1,2]=-sin, [2,1]=sin, [2,2]=cos.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_rotate_x(float rad) {
    /* TODO: build X-rotation matrix */
    (void)rad;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement rotation around arbitrary axis (Rodrigues)
 *
 * Normalize axis first, then with c=cos(θ), s=sin(θ), t=1-cos(θ):
 * | t*x*x+c    t*x*y-s*z  t*x*z+s*y  0 |
 * | t*x*y+s*z  t*y*y+c    t*y*z-s*x  0 |
 * | t*x*z-s*y  t*y*z+s*x  t*z*z+c    0 |
 * |    0          0          0        1 |
 *
 * Rodrigues' rotation formula generalizes to ANY axis. The formula
 * decomposes as:
 *   (1) component of v along axis (unchanged):  t*dot(axis,v)*axis
 *   (2) component perpendicular to axis
 *       (scaled by cosθ):                       c*v_perp
 *   (3) component perpendicular to both
 *       (cross product, scaled by sinθ):         s*(axis × v)
 * Expanding these into matrix form gives the 3×3 block shown above.
 * Must normalize the axis first: a non-unit axis would scale the
 * vector.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_rotate_axis(vec3 axis, float rad) {
    /* TODO: implement Rodrigues' rotation formula */
    (void)axis; (void)rad;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement look-at view matrix
 *
 * Creates a camera view matrix looking from 'eye' toward 'target'.
 * 1. forward = normalize(target - eye)
 * 2. right   = normalize(cross(forward, up))
 * 3. up      = cross(right, forward)
 * 4. Rotation rows are right, up, -forward; translation is -dot(row, eye)
 *
 * The look-at matrix builds a view coordinate frame.
 * Step 1: forward = normalize(target-eye) — the camera's -Z axis.
 * Step 2: right = normalize(forward × up) — the camera's +X axis;
 *         must normalize again because forward and up may not be
 *         perpendicular.
 * Step 3: true_up = right × forward — recomputes up perpendicular to
 *         both forward and right (handles non-perpendicular input up).
 * Step 4: The rotation rows are right, true_up, -forward (we negate
 *         forward because cameras look down -Z by convention).
 * Step 5: The translation part is -dot(right,eye), -dot(true_up,eye),
 *         +dot(forward,eye) — this moves the world so the eye is at
 *         the origin.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_look_at(vec3 eye, vec3 target, vec3 up) {
    /* TODO: build look-at view matrix */
    (void)eye; (void)target; (void)up;
    return m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement 4×4 matrix inverse (HARD — can skip initially)
 *
 * Compute the inverse using cofactor/adjugate expansion.
 * Reference: https://semath.info/src/inverse-cofactor-ex4.html
 * Hint: compute all 16 cofactors, transpose to get adjugate,
 *       then divide each element by the determinant.
 *       Return identity if det ≈ 0 (singular matrix).
 *
 * The general 4×4 inverse uses cofactor expansion.
 * A cofactor C_ij = (-1)^(i+j) * det(minor), where minor is the 3×3
 * submatrix obtained by deleting row i and col j.
 * The adjugate matrix is the transpose of the cofactor matrix.
 * The inverse is: adjugate / det.
 * For affine transforms (translation + rotation + scale), there is a
 * faster method: inverse = [R^T, -R^T*t; 0,0,0,1], but the full
 * general method works for all cases.
 * ══════════════════════════════════════════════════════════════════ */
static mat4 m4_inverse(mat4 m) {
    /* TODO: implement full 4×4 matrix inverse */
    (void)m;
    return m4_identity();
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

static void test_m4_mul(void) {
    printf("\n=== TODO #1: m4_mul ===\n");
    mat4 id = m4_identity();
    mat4 id2 = m4_mul(id, id);
    TEST("I×I diagonal",     M4(id2,0,0)==1 && M4(id2,1,1)==1 && M4(id2,2,2)==1 && M4(id2,3,3)==1);
    TEST("I×I off-diagonal", M4(id2,0,1)==0 && M4(id2,1,0)==0);

    /* translate(1,2,3) × scale(2,2,2): point (1,0,0) → scale first → (2,0,0) → translate → (3,2,3) */
    mat4 t  = m4_translate(v3(1,2,3));
    mat4 s  = m4_scale(v3(2,2,2));
    mat4 ts = m4_mul(t, s);
    TEST("T×S: (1,0,0) → (3,2,3)", v3_eq(m4_mul_point(ts, v3(1,0,0)), v3(3,2,3), EPSILON));

    /* scale(2,2,2) × translate(1,2,3): point (0,0,0) → translate → (1,2,3) → scale → (2,4,6) */
    mat4 st = m4_mul(s, t);
    TEST("S×T: (0,0,0) → (2,4,6)", v3_eq(m4_mul_point(st, v3(0,0,0)), v3(2,4,6), EPSILON));
}

static void test_rotate_z(void) {
    printf("\n=== TODO #2: m4_rotate_z ===\n");
    mat4 rz = m4_rotate_z((float)M_PI / 2.0f);
    TEST("Z90°: x→y",         v3_eq(m4_mul_dir(rz, v3(1,0,0)), v3(0,1,0), EPSILON));
    TEST("Z90°: y→-x",        v3_eq(m4_mul_dir(rz, v3(0,1,0)), v3(-1,0,0), EPSILON));
    TEST("Z90°: z unchanged",  v3_eq(m4_mul_dir(rz, v3(0,0,1)), v3(0,0,1), EPSILON));
}

static void test_rotate_x(void) {
    printf("\n=== TODO #3: m4_rotate_x ===\n");
    mat4 rx = m4_rotate_x((float)M_PI / 2.0f);
    TEST("X90°: y→z",         v3_eq(m4_mul_dir(rx, v3(0,1,0)), v3(0,0,1), EPSILON));
    TEST("X90°: z→-y",        v3_eq(m4_mul_dir(rx, v3(0,0,1)), v3(0,-1,0), EPSILON));
    TEST("X90°: x unchanged",  v3_eq(m4_mul_dir(rx, v3(1,0,0)), v3(1,0,0), EPSILON));
}

static void test_rotate_axis(void) {
    printf("\n=== TODO #4: m4_rotate_axis ===\n");
    /* 120° around (1,1,1)/√3 cyclically permutes basis vectors: x→y, y→z, z→x */
    vec3 axis = v3_norm(v3(1,1,1));
    mat4 r    = m4_rotate_axis(axis, 2.0f * (float)M_PI / 3.0f);
    TEST("axis-angle 120° (1,1,1): x→y", v3_eq(m4_mul_dir(r, v3(1,0,0)), v3(0,1,0), EPSILON));
    TEST("axis-angle 120° (1,1,1): y→z", v3_eq(m4_mul_dir(r, v3(0,1,0)), v3(0,0,1), EPSILON));
    TEST("axis-angle 120° (1,1,1): z→x", v3_eq(m4_mul_dir(r, v3(0,0,1)), v3(1,0,0), EPSILON));
}

static void test_look_at(void) {
    printf("\n=== TODO #5: m4_look_at ===\n");
    /* Camera at (0,0,5) looking at origin: origin maps to (0,0,-5) in view space */
    mat4 view = m4_look_at(v3(0,0,5), v3(0,0,0), v3(0,1,0));
    vec3 o    = m4_mul_point(view, v3(0,0,0));
    TEST("look_at from +Z: origin at z=-5",     fabsf(o.z + 5.0f) < EPSILON);
    TEST("look_at from +Z: origin centered xy", fabsf(o.x) < EPSILON && fabsf(o.y) < EPSILON);

    /* Camera at (5,0,0) looking at origin */
    mat4 view2 = m4_look_at(v3(5,0,0), v3(0,0,0), v3(0,1,0));
    vec3 o2    = m4_mul_point(view2, v3(0,0,0));
    TEST("look_at from +X: origin at z=-5", fabsf(o2.z + 5.0f) < EPSILON);
}

static void test_inverse(void) {
    printf("\n=== TODO #6: m4_inverse ===\n");
    mat4 t    = m4_translate(v3(10,20,30));
    mat4 tinv = m4_inverse(t);
    TEST("T⁻¹ maps translated point back", v3_eq(m4_mul_point(tinv, v3(10,20,30)), v3(0,0,0), EPSILON));

    mat4 prod = m4_mul(t, tinv);
    TEST("T × T⁻¹ diagonal = 1",    fabsf(M4(prod,0,0)-1)<EPSILON && fabsf(M4(prod,3,3)-1)<EPSILON);
    TEST("T × T⁻¹ off-diag = 0",    fabsf(M4(prod,0,3))<EPSILON);

    mat4 s    = m4_scale(v3(2,3,4));
    mat4 sinv = m4_inverse(s);
    mat4 sp   = m4_mul(s, sinv);
    TEST("S × S⁻¹ diagonal = 1",    fabsf(M4(sp,0,0)-1)<EPSILON && fabsf(M4(sp,1,1)-1)<EPSILON);
}

static void test_provided(void) {
    printf("\n=== provided operations ===\n");
    mat4 m = m4_identity();
    M4(m, 0, 1) = 5;
    M4(m, 1, 0) = 10;
    mat4 mt = m4_transpose(m);
    TEST("m4_transpose",         M4(mt,1,0) == 5 && M4(mt,0,1) == 10);

    mat4 t = m4_translate(v3(3,0,0));
    TEST("m4_translate point",   v3_eq(m4_mul_point(t, v3(0,0,0)), v3(3,0,0), EPSILON));
    TEST("m4_translate dir",     v3_eq(m4_mul_dir(t, v3(1,0,0)),   v3(1,0,0), EPSILON));

    mat4 s = m4_scale(v3(2,3,4));
    TEST("m4_scale",             v3_eq(m4_mul_point(s, v3(1,1,1)), v3(2,3,4), EPSILON));

    mat4 persp  = m4_perspective((float)M_PI/4.0f, 1.0f, 0.1f, 100.0f);
    vec4 center = m4_mul_v4(persp, v4(0,0,-1,1));
    TEST("m4_perspective x=0",   fabsf(center.x) < EPSILON);
    TEST("m4_perspective y=0",   fabsf(center.y) < EPSILON);
}

int main(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║  MujoCol Phase 2b: mat4 Unit Tests     ║\n");
    printf("╚════════════════════════════════════════╝\n");

    test_m4_mul();
    test_rotate_z();
    test_rotate_x();
    test_rotate_axis();
    test_look_at();
    test_inverse();
    test_provided();

    printf("\n════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("════════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
