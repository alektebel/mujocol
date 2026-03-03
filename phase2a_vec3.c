/*
 * phase2a_vec3.c — 3D Vector Math
 *
 * EXERCISE (sub-phase 2a): Implement fundamental 3D vector operations.
 *
 * Build: gcc -O2 -o phase2a_vec3 phase2a_vec3.c -lm
 * Run:   ./phase2a_vec3
 *
 * LEARNING GOALS:
 * - Understand vectors as 3-tuples of floats
 * - Implement dot product, cross product, and normalization
 * - Learn reflection formula for ray tracing
 * - Practice unit testing with tolerance-based comparisons
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * TYPES
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;

/* ══════════════════════════════════════════════════════════════════
 * VEC3 OPERATIONS — provided, do not modify
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

static inline vec3 v3_lerp(vec3 a, vec3 b, float t) {
    return v3_add(v3_scale(a, 1.0f - t), v3_scale(b, t));
}

static inline int v3_eq(vec3 a, vec3 b, float eps) {
    return fabsf(a.x - b.x) < eps &&
           fabsf(a.y - b.y) < eps &&
           fabsf(a.z - b.z) < eps;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement dot product
 *
 * The dot product a·b = ax*bx + ay*by + az*bz
 * Returns a scalar: positive when vectors point the same way,
 * zero when perpendicular, negative when opposite.
 * ══════════════════════════════════════════════════════════════════ */
static inline float v3_dot(vec3 a, vec3 b) {
    /* TODO: return ax*bx + ay*by + az*bz */
    (void)a; (void)b;
    return 0.0f;
}

/* v3_len uses v3_dot internally — implement TODO #1 first */
static inline float v3_len(vec3 v) {
    return sqrtf(v3_dot(v, v));
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement cross product
 *
 * The cross product a×b gives a vector perpendicular to both a and b.
 * Formula: (ay*bz - az*by,  az*bx - ax*bz,  ax*by - ay*bx)
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_cross(vec3 a, vec3 b) {
    /* TODO: return the cross product vector */
    (void)a; (void)b;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement normalization
 *
 * Return a unit vector (length = 1) in the same direction as v.
 * Handle the near-zero case: if len < EPSILON, return zero vector.
 * Hint: divide each component by v3_len(v).
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_norm(vec3 v) {
    /* TODO: return unit vector, or zero if v is near-zero */
    (void)v;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement reflection
 *
 * Reflect vector v around surface normal n.
 * Formula: v - 2*(v·n)*n
 * Use case: ray tracing — computing the bounce direction of a ray.
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_reflect(vec3 v, vec3 n) {
    /* TODO: return v - 2*(v·n)*n */
    (void)v; (void)n;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement component-wise clamp
 *
 * Clamp each component of v to the range [lo, hi].
 * Useful for keeping colour values in [0, 1] after shading.
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_clamp(vec3 v, float lo, float hi) {
    /* TODO: clamp v.x, v.y, v.z each to [lo, hi] */
    (void)v; (void)lo; (void)hi;
    return v3(0, 0, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement vector projection
 *
 * Project v onto 'onto': the component of v in the direction of 'onto'.
 * Formula: (v·onto / onto·onto) * onto
 * Handle the case where 'onto' is near-zero (return zero vector).
 * ══════════════════════════════════════════════════════════════════ */
static inline vec3 v3_project(vec3 v, vec3 onto) {
    /* TODO: return (v·onto / onto·onto) * onto */
    (void)v; (void)onto;
    return v3(0, 0, 0);
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

static void test_provided(void) {
    printf("\n=== provided operations ===\n");
    vec3 a = v3(1, 2, 3);
    vec3 b = v3(4, 5, 6);

    TEST("v3 creation",    a.x == 1 && a.y == 2 && a.z == 3);
    TEST("v3_add",         v3_eq(v3_add(a, b), v3(5, 7, 9), EPSILON));
    TEST("v3_sub",         v3_eq(v3_sub(b, a), v3(3, 3, 3), EPSILON));
    TEST("v3_scale",       v3_eq(v3_scale(a, 2), v3(2, 4, 6), EPSILON));
    TEST("v3_neg",         v3_eq(v3_neg(a), v3(-1, -2, -3), EPSILON));
    TEST("v3_lerp half",   v3_eq(v3_lerp(v3(0,0,0), v3(10,10,10), 0.5f), v3(5,5,5), EPSILON));
    TEST("v3_lerp start",  v3_eq(v3_lerp(a, b, 0.0f), a, EPSILON));
    TEST("v3_lerp end",    v3_eq(v3_lerp(a, b, 1.0f), b, EPSILON));
}

static void test_dot(void) {
    printf("\n=== TODO #1: v3_dot ===\n");
    TEST("dot (1,2,3)·(4,5,6) = 32", fabsf(v3_dot(v3(1,2,3), v3(4,5,6)) - 32.0f) < EPSILON);
    TEST("dot self (3,4,0) = 25",    fabsf(v3_dot(v3(3,4,0), v3(3,4,0)) - 25.0f) < EPSILON);
    TEST("dot perpendicular = 0",    fabsf(v3_dot(v3(1,0,0), v3(0,1,0))) < EPSILON);
    TEST("v3_len via dot: len(3,4,0) = 5", fabsf(v3_len(v3(3,4,0)) - 5.0f) < EPSILON);
}

static void test_cross(void) {
    printf("\n=== TODO #2: v3_cross ===\n");
    TEST("x × y = z",         v3_eq(v3_cross(v3(1,0,0), v3(0,1,0)), v3(0,0,1), EPSILON));
    TEST("y × z = x",         v3_eq(v3_cross(v3(0,1,0), v3(0,0,1)), v3(1,0,0), EPSILON));
    TEST("z × x = y",         v3_eq(v3_cross(v3(0,0,1), v3(1,0,0)), v3(0,1,0), EPSILON));
    TEST("anti-commutative",   v3_eq(v3_cross(v3(1,0,0), v3(0,1,0)),
                                     v3_neg(v3_cross(v3(0,1,0), v3(1,0,0))), EPSILON));
}

static void test_norm(void) {
    printf("\n=== TODO #3: v3_norm ===\n");
    TEST("norm (3,0,0) = (1,0,0)",  v3_eq(v3_norm(v3(3,0,0)), v3(1,0,0), EPSILON));
    TEST("norm (0,5,0) = (0,1,0)",  v3_eq(v3_norm(v3(0,5,0)), v3(0,1,0), EPSILON));
    TEST("unit length after norm",  fabsf(v3_len(v3_norm(v3(3,4,0))) - 1.0f) < EPSILON);
    TEST("norm zero vector = zero", v3_eq(v3_norm(v3(0,0,0)), v3(0,0,0), EPSILON));
}

static void test_reflect(void) {
    printf("\n=== TODO #4: v3_reflect ===\n");
    TEST("reflect off Y plane",        v3_eq(v3_reflect(v3(1,-1,0), v3(0,1,0)), v3(1,1,0), EPSILON));
    TEST("reflect off X plane",        v3_eq(v3_reflect(v3(-1,1,0), v3(1,0,0)), v3(1,1,0), EPSILON));
    TEST("vertical bounce: down→up",   v3_eq(v3_reflect(v3(0,-1,0), v3(0,1,0)), v3(0,1,0), EPSILON));
}

static void test_clamp(void) {
    printf("\n=== TODO #5: v3_clamp ===\n");
    TEST("clamp within range", v3_eq(v3_clamp(v3(0.5f,0.5f,0.5f), 0.0f, 1.0f), v3(0.5f,0.5f,0.5f), EPSILON));
    TEST("clamp below lo",     v3_eq(v3_clamp(v3(-1,-2,-3), 0.0f, 1.0f), v3(0,0,0), EPSILON));
    TEST("clamp above hi",     v3_eq(v3_clamp(v3(2,3,4), 0.0f, 1.0f), v3(1,1,1), EPSILON));
    TEST("clamp mixed",        v3_eq(v3_clamp(v3(-1.0f, 0.5f, 2.0f), 0.0f, 1.0f), v3(0.0f, 0.5f, 1.0f), EPSILON));
}

static void test_project(void) {
    printf("\n=== TODO #6: v3_project ===\n");
    TEST("project onto x-axis",  v3_eq(v3_project(v3(3,4,0), v3(1,0,0)), v3(3,0,0), EPSILON));
    TEST("project onto y-axis",  v3_eq(v3_project(v3(3,4,0), v3(0,1,0)), v3(0,4,0), EPSILON));
    TEST("project onto self",    v3_eq(v3_project(v3(2,0,0), v3(2,0,0)), v3(2,0,0), EPSILON));
    TEST("project zero onto",    v3_eq(v3_project(v3(1,1,0), v3(0,0,0)), v3(0,0,0), EPSILON));
}

int main(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║  MujoCol Phase 2a: 3D Vector Tests     ║\n");
    printf("╚════════════════════════════════════════╝\n");

    test_provided();
    test_dot();
    test_cross();
    test_norm();
    test_reflect();
    test_clamp();
    test_project();

    printf("\n════════════════════════════════════════\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("════════════════════════════════════════\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
