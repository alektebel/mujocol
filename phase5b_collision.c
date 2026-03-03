/*
 * phase5b_collision.c — Rigid Body Collision Detection
 *
 * EXERCISE (sub-phase 5b): Implement collision detection between
 * rigid bodies and the ground plane, and between pairs of spheres.
 *
 * Build: gcc -O2 -o phase5b_collision phase5b_collision.c -lm
 * Run:   ./phase5b_collision
 *
 * LEARNING GOALS:
 * - Implement sphere-ground collision using geometric overlap test
 * - Implement box-ground collision by testing all 8 corners
 * - Implement sphere-sphere collision using distance comparison
 * - Understand Contact struct: point, normal, penetration depth
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * 3D MATH (phases 1-2 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float w, x, y, z; } quat;

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x*s, v.y*s, v.z*s); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }
static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    return len < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f / len);
}

static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float half = rad * 0.5f, s = sinf(half);
    return (quat){cosf(half), axis.x*s, axis.y*s, axis.z*s};
}
static inline float q_len(quat q) { return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); }
static inline quat q_norm(quat q) {
    float len = q_len(q);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f / len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}
static vec3 q_rotate(quat q, vec3 v) {
    vec3 qv = v3(q.x, q.y, q.z);
    vec3 uv = v3_cross(qv, v);
    vec3 uuv = v3_cross(qv, uv);
    return v3_add(v, v3_scale(v3_add(v3_scale(uv, q.w), uuv), 2.0f));
}

/* ══════════════════════════════════════════════════════════════════
 * RIGID BODY DATA STRUCTURES (already provided)
 * ══════════════════════════════════════════════════════════════════ */

typedef enum { SHAPE_SPHERE, SHAPE_BOX } ShapeType;

typedef struct {
    ShapeType type;
    float radius;
    vec3  half_extents;
} Shape;

typedef struct {
    vec3  pos;
    vec3  vel;
    quat  orientation;
    vec3  angular_vel;

    float mass;
    float inv_mass;
    float inertia;
    float inv_inertia;

    Shape shape;
    int   color_id;
    int   active;
} RigidBody;

/* ══════════════════════════════════════════════════════════════════
 * PHYSICS CONSTANTS (already provided)
 * ══════════════════════════════════════════════════════════════════ */

static const float RESTITUTION = 0.6f;
static const float FRICTION    = 0.3f;
static const float GROUND_Y    = -2.0f;

/* ══════════════════════════════════════════════════════════════════
 * body_init_sphere / body_init_box (solution from phase 5a)
 * ══════════════════════════════════════════════════════════════════ */

static void body_init_sphere(RigidBody *b, vec3 pos, float radius,
                              float mass, int color_id) {
    b->pos         = pos;
    b->vel         = v3(0, 0, 0);
    b->angular_vel = v3(0, 0, 0);
    b->orientation = q_identity();
    b->mass        = mass;
    b->inv_mass    = mass > 0 ? 1.0f / mass : 0.0f;
    b->inertia     = 0.4f * mass * radius * radius;
    b->inv_inertia = b->inertia > 0 ? 1.0f / b->inertia : 0.0f;
    b->shape.type   = SHAPE_SPHERE;
    b->shape.radius = radius;
    b->color_id    = color_id;
    b->active      = 1;
}

static void body_init_box(RigidBody *b, vec3 pos, vec3 half_ext,
                           float mass, int color_id) {
    b->pos         = pos;
    b->vel         = v3(0, 0, 0);
    b->angular_vel = v3(0, 0, 0);
    b->orientation = q_identity();
    b->mass        = mass;
    b->inv_mass    = mass > 0 ? 1.0f / mass : 0.0f;
    float w = 2.0f * half_ext.x, h = 2.0f * half_ext.y, d = 2.0f * half_ext.z;
    b->inertia     = mass * (w*w + h*h + d*d) / 12.0f;
    b->inv_inertia = b->inertia > 0 ? 1.0f / b->inertia : 0.0f;
    b->shape.type          = SHAPE_BOX;
    b->shape.half_extents  = half_ext;
    b->color_id = color_id;
    b->active   = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * CONTACT STRUCT (already provided)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   collided;
    vec3  point;       /* contact point in world space */
    vec3  normal;      /* contact normal pointing from surface into body */
    float penetration;
} Contact;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Sphere vs ground plane collision
 *
 * The ground is at y = GROUND_Y.
 * A sphere at position b->pos with radius r overlaps the ground when:
 *   dist = b->pos.y - GROUND_Y - radius  < 0
 *
 * If colliding:
 *   c.collided    = 1
 *   c.normal      = (0, 1, 0)                   [pointing up into the body]
 *   c.penetration = -dist                        [positive overlap amount]
 *   c.point       = (b->pos.x, GROUND_Y, b->pos.z)
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_sphere_ground(RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect sphere-ground collision
     * float dist = b->pos.y - GROUND_Y - b->shape.radius;
     * if (dist < 0) { ... fill c ... } */
    (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Box vs ground plane collision
 *
 * Enumerate all 8 corners of the oriented box (i,j,k ∈ {-1, +1}):
 *   local = (i * he.x,  j * he.y,  k * he.z)
 *   world = b->pos + q_rotate(b->orientation, local)
 *
 * Track the corner with the minimum world.y (deepest below ground).
 * If min_y < GROUND_Y:
 *   c.collided    = 1
 *   c.normal      = (0, 1, 0)
 *   c.penetration = GROUND_Y - min_y
 *   c.point       = that deepest corner
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_box_ground(RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect box-ground collision via 8-corner enumeration
     * Hint: use a nested loop over signs[] = {-1, +1} for i, j, k */
    (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Sphere vs sphere collision
 *
 * Two spheres collide when:
 *   dist     = length(b->pos - a->pos)
 *   min_dist = a->shape.radius + b->shape.radius
 *   dist < min_dist  AND  dist > EPSILON  (avoids divide-by-zero)
 *
 * If colliding:
 *   c.normal      = (b->pos - a->pos) / dist    [unit vector from a → b]
 *   c.penetration = min_dist - dist
 *   c.point       = a->pos + normal * a->shape.radius
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_sphere_sphere(RigidBody *a, RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect sphere-sphere collision */
    (void)a; (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Sphere vs AABB box collision (HARD / optional)
 *
 * Simplified version — ignores box rotation (treats box as axis-aligned).
 * Steps:
 *   1. Compute the sphere center relative to the box center
 *   2. Clamp each component to [-he.x, he.x], [-he.y, he.y], [-he.z, he.z]
 *      to find the closest point on (or inside) the AABB
 *   3. Compute distance from sphere center to that closest point
 *   4. If distance < sphere radius → collision
 *      normal = normalize(sphere_pos - closest_point)
 *      penetration = sphere_radius - distance
 *      point = closest_point (in world space)
 *
 * NOTE: For the full rotated-box version you would transform the sphere
 *       into box-local space first.
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_sphere_box(RigidBody *sphere, RigidBody *box) {
    Contact c = {0};
    /* TODO (HARD): Sphere vs axis-aligned box collision
     * Clamp sphere center to box extents to find closest point */
    (void)sphere; (void)box;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Find the pair of bodies with the deepest penetration
 *
 * Scan all (i, j) pairs where i < j and both bodies are active.
 * Test sphere-sphere (both SHAPE_SPHERE) or sphere-box (mixed).
 * Return the Contact with the largest penetration depth.
 * If no collisions found, return Contact with collided = 0.
 *
 * Useful for debugging — lets you confirm the worst collision first.
 * ══════════════════════════════════════════════════════════════════ */
static Contact find_deepest_penetration(RigidBody *bodies, int n) {
    Contact worst = {0};
    /* TODO: Iterate all pairs, call collide_sphere_sphere or collide_sphere_box,
     * keep the Contact with the greatest penetration depth */
    (void)bodies; (void)n;
    return worst;
}

/* ══════════════════════════════════════════════════════════════════
 * UNIT TESTS — run and verify expected output
 * ══════════════════════════════════════════════════════════════════ */

static int approx_eq(float a, float b, float tol) { return fabsf(a - b) < tol; }
static int vec3_approx(vec3 a, vec3 b, float tol) {
    return approx_eq(a.x,b.x,tol) && approx_eq(a.y,b.y,tol) && approx_eq(a.z,b.z,tol);
}

static int tests_run = 0, tests_passed = 0;
static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); }
    else       { printf("  FAIL: %s\n", name); }
}

int main(void) {
    printf("=== Phase 5b: Collision Detection Tests ===\n\n");

    /* ── Sphere below ground: should collide ── */
    printf("-- Sphere below ground (should collide) --\n");
    RigidBody s1 = {0};
    body_init_sphere(&s1, v3(1, GROUND_Y + 0.3f, 2), 0.5f, 1.0f, 0);
    /* center y = GROUND_Y+0.3, radius=0.5 → dist = 0.3-0.5 = -0.2 → collide */
    Contact c1 = collide_sphere_ground(&s1);
    check("sphere-ground: collided",         c1.collided == 1);
    check("sphere-ground: penetration≈0.2",  approx_eq(c1.penetration, 0.2f, 1e-4f));
    check("sphere-ground: normal up",        vec3_approx(c1.normal, v3(0,1,0), 1e-4f));
    check("sphere-ground: point at GROUND_Y",approx_eq(c1.point.y, GROUND_Y, 1e-4f));
    printf("\n");

    /* ── Sphere above ground: should NOT collide ── */
    printf("-- Sphere well above ground (no collision) --\n");
    RigidBody s2 = {0};
    body_init_sphere(&s2, v3(0, GROUND_Y + 2.0f, 0), 0.5f, 1.0f, 0);
    Contact c2 = collide_sphere_ground(&s2);
    check("sphere-ground: no collision above", c2.collided == 0);
    printf("\n");

    /* ── Sphere exactly touching ground: boundary case ── */
    printf("-- Sphere exactly on ground surface (no collision) --\n");
    RigidBody s3 = {0};
    body_init_sphere(&s3, v3(0, GROUND_Y + 0.5f, 0), 0.5f, 1.0f, 0);
    Contact c3 = collide_sphere_ground(&s3);
    check("sphere-ground: touching = no overlap", c3.collided == 0);
    printf("\n");

    /* ── Box below ground: should collide ── */
    printf("-- Box with corner below ground (should collide) --\n");
    RigidBody b1 = {0};
    /* half_ext=(0.5,0.5,0.5), pos.y=GROUND_Y+0.3 → bottom corner at GROUND_Y-0.2 */
    body_init_box(&b1, v3(0, GROUND_Y + 0.3f, 0), v3(0.5f, 0.5f, 0.5f), 2.0f, 0);
    Contact cb = collide_box_ground(&b1);
    check("box-ground: collided",         cb.collided == 1);
    check("box-ground: penetration≈0.2",  approx_eq(cb.penetration, 0.2f, 1e-4f));
    check("box-ground: normal up",        vec3_approx(cb.normal, v3(0,1,0), 1e-4f));
    printf("\n");

    /* ── Box fully above ground: should NOT collide ── */
    printf("-- Box well above ground (no collision) --\n");
    RigidBody b2 = {0};
    body_init_box(&b2, v3(0, GROUND_Y + 2.0f, 0), v3(0.5f, 0.5f, 0.5f), 2.0f, 0);
    Contact cb2 = collide_box_ground(&b2);
    check("box-ground: no collision above", cb2.collided == 0);
    printf("\n");

    /* ── Rotated box: one corner dips below ground ── */
    printf("-- Rotated box (45 deg, one corner below ground) --\n");
    RigidBody b3 = {0};
    body_init_box(&b3, v3(0, GROUND_Y + 0.5f, 0), v3(0.5f, 0.5f, 0.5f), 2.0f, 0);
    b3.orientation = q_from_axis_angle(v3(0,0,1), (float)M_PI / 4.0f);
    b3.orientation = q_norm(b3.orientation);
    /* corner along +x,+y reaches y ≈ 0.5*sqrt(2) ≈ 0.707 above center,
       corner along +x,-y dips to y ≈ center - 0.707 ≈ GROUND_Y - 0.207 */
    Contact cb3 = collide_box_ground(&b3);
    check("rotated-box-ground: collided", cb3.collided == 1);
    printf("\n");

    /* ── Sphere-sphere overlapping ── */
    printf("-- Two overlapping spheres --\n");
    RigidBody sa = {0}, sb = {0};
    body_init_sphere(&sa, v3(0, 0, 0), 0.5f, 1.0f, 0);
    body_init_sphere(&sb, v3(0.8f, 0, 0), 0.5f, 1.0f, 1);
    /* dist=0.8, min_dist=1.0 → penetration=0.2, normal=(1,0,0) */
    Contact css = collide_sphere_sphere(&sa, &sb);
    check("sphere-sphere: collided",            css.collided == 1);
    check("sphere-sphere: penetration≈0.2",     approx_eq(css.penetration, 0.2f, 1e-4f));
    check("sphere-sphere: normal x≈1",          approx_eq(css.normal.x, 1.0f, 1e-4f));
    check("sphere-sphere: normal y≈0",          approx_eq(css.normal.y, 0.0f, 1e-4f));
    check("sphere-sphere: contact point on sa", approx_eq(css.point.x, 0.5f, 1e-4f));
    printf("\n");

    /* ── Sphere-sphere far apart: no collision ── */
    printf("-- Two separate spheres (no collision) --\n");
    RigidBody sc = {0}, sd = {0};
    body_init_sphere(&sc, v3(-5, 0, 0), 0.5f, 1.0f, 0);
    body_init_sphere(&sd, v3( 5, 0, 0), 0.5f, 1.0f, 1);
    Contact css2 = collide_sphere_sphere(&sc, &sd);
    check("sphere-sphere: no collision when far", css2.collided == 0);
    printf("\n");

    /* ── Sphere-sphere exactly touching: no penetration ── */
    printf("-- Two spheres exactly touching --\n");
    RigidBody se = {0}, sf = {0};
    body_init_sphere(&se, v3(0,0,0), 0.5f, 1.0f, 0);
    body_init_sphere(&sf, v3(1.0f,0,0), 0.5f, 1.0f, 1);
    Contact css3 = collide_sphere_sphere(&se, &sf);
    check("sphere-sphere: touching = no collision", css3.collided == 0);
    printf("\n");

    /* ── find_deepest_penetration ── */
    printf("-- find_deepest_penetration --\n");
    RigidBody pool[3] = {0};
    body_init_sphere(&pool[0], v3(0, 0, 0), 0.5f, 1.0f, 0);  /* overlaps pool[1] by 0.2 */
    body_init_sphere(&pool[1], v3(0.8f, 0, 0), 0.5f, 1.0f, 1);
    body_init_sphere(&pool[2], v3(10, 0, 0), 0.5f, 1.0f, 2); /* far away */
    Contact deep = find_deepest_penetration(pool, 3);
    check("deepest: collided",    deep.collided == 1);
    check("deepest: depth≈0.2",   approx_eq(deep.penetration, 0.2f, 1e-4f));
    printf("\n");

    printf("=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
