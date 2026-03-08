/*
 * phase5a_body_init.c — Rigid Body Data Structures and Initialization
 *
 * EXERCISE (sub-phase 5a): Implement rigid body initialization for
 * sphere and box shapes, including inertia tensor computation.
 *
 * Build: gcc -O2 -o phase5a_body_init phase5a_body_init.c -lm
 * Run:   ./phase5a_body_init
 *
 * LEARNING GOALS:
 * - Understand rigid body state: position, velocity, orientation, angular velocity
 * - Compute moment of inertia for common shapes:
 *     Sphere: I = (2/5) * mass * radius^2
 *     Box:    I = mass * (w^2 + h^2 + d^2) / 12
 * - Understand why we store inv_mass and inv_inertia (multiplication is faster than division)
 * - Handle static bodies (mass = 0) with inv_mass = 0
 *
 * TECHNIQUE OVERVIEW:
 *
 * 1. RIGID BODY STATE
 *    A rigid body in 3D needs six degrees of freedom to fully describe its
 *    motion.  We track these as two groups of state:
 *      - Linear:  pos (vec3, world-space position of center of mass)
 *                 vel (vec3, linear velocity in m/s)
 *      - Angular: orientation (quat, rotation from body-local to world frame)
 *                 angular_vel (vec3, angular velocity in radians/second)
 *    The direction of angular_vel is the instantaneous rotation axis; its
 *    magnitude is how fast the body spins.  Quaternions avoid gimbal lock
 *    (covered in phase 2c — not repeated here).
 *
 * 2. MOMENT OF INERTIA
 *    Mass tells us how hard it is to change linear velocity.  The moment of
 *    inertia I tells us how hard it is to change angular velocity — it is the
 *    rotational equivalent of mass.  I depends on BOTH the mass distribution
 *    AND the geometry:
 *      Solid sphere (mass m, radius r):     I = (2/5) * m * r²
 *      Solid box (mass m, dims w × h × d):  I_x = m*(h²+d²)/12
 *                                            I_y = m*(w²+d²)/12
 *                                            I_z = m*(w²+h²)/12
 *    Larger or more spread-out objects have larger I and resist spinning more.
 *
 * 3. INERTIA TENSOR
 *    In 3D the moment of inertia is really a 3×3 matrix because resistance to
 *    rotation varies with the axis of rotation.  For symmetric shapes
 *    (sphere, axis-aligned box) the off-diagonal terms are zero, so the tensor
 *    is diagonal — we only need to store three values (Ixx, Iyy, Izz).
 *    This file uses a scalar simplification (one combined I) suitable for the
 *    demo; a full simulator would store a vec3 diagonal or full mat3.
 *
 * 4. WHY STORE inv_mass AND inv_inertia?
 *    Every physics step we compute:
 *      linear acceleration:  a = F / m   = F * inv_mass
 *      angular acceleration: α = τ / I   = τ * inv_inertia
 *    Floating-point division is ~4–20× slower than multiplication on modern
 *    CPUs (including aarch64/Apple Silicon).  Pre-computing 1/m and 1/I means
 *    every integration step only needs fast multiplications.
 *
 * 5. STATIC BODIES (mass = 0)
 *    Setting inv_mass = 0 makes a body "infinitely heavy" — no impulse can
 *    move it.  Ground planes and walls are static.  In the impulse formula,
 *    any term multiplied by inv_mass_A or inv_mass_B simply evaluates to zero,
 *    so the code naturally skips applying velocity changes to static bodies.
 *
 * 6. ShapeType ENUM
 *    SHAPE_SPHERE and SHAPE_BOX select:
 *      - Which SDF primitive to evaluate during ray marching (phase 3).
 *      - Which inertia formula to use during body_init.
 *      - Which collision geometry to test in phase 5b.
 *    The enum is stored alongside the shape parameters (radius / half_extents)
 *    in the Shape sub-struct of RigidBody.
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
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x*s, v.y*s, v.z*s); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }

static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
static inline float q_len(quat q) { return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); }
static inline quat q_norm(quat q) {
    float len = q_len(q);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f / len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
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
    float inertia;      /* simplified scalar inertia */
    float inv_inertia;

    Shape shape;
    int   color_id;
    int   active;
} RigidBody;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Initialize sphere rigid body
 *
 * Set position, zero vel/angular_vel, identity orientation.
 * For a solid sphere, the moment of inertia is:
 *   I = (2/5) * mass * radius^2  =  0.4 * mass * radius * radius
 * Store both inertia and its inverse (1/inertia).
 * If mass == 0 (static body), inv_mass = inv_inertia = 0.
 *
 * Implementation steps:
 *   b->pos         = pos;
 *   b->vel         = v3(0,0,0);        // starts at rest
 *   b->angular_vel = v3(0,0,0);        // no initial spin
 *   b->orientation = q_identity();     // identity = no rotation
 *   b->mass        = mass;
 *   b->inertia     = 0.4f * mass * radius * radius;   // sphere formula
 *   b->inv_mass    = (mass    > 0) ? 1.0f / mass    : 0.0f;
 *   b->inv_inertia = (inertia > 0) ? 1.0f / inertia : 0.0f;
 *   b->shape.type   = SHAPE_SPHERE;
 *   b->shape.radius = radius;
 *   b->color_id    = color_id;
 *   b->active      = 1;
 * ══════════════════════════════════════════════════════════════════ */
static void body_init_sphere(RigidBody *b, vec3 pos, float radius,
                              float mass, int color_id) {
    /* TODO: Initialize rigid body for a sphere
     * - Set b->pos = pos
     * - Zero b->vel and b->angular_vel (use v3(0,0,0))
     * - Set b->orientation = q_identity()
     * - Set b->mass = mass
     * - Compute b->inertia = 0.4f * mass * radius * radius
     * - Set b->inv_mass   = mass    > 0 ? 1.0f / mass    : 0.0f
     * - Set b->inv_inertia= inertia > 0 ? 1.0f / inertia : 0.0f
     * - Set b->shape.type = SHAPE_SPHERE, b->shape.radius = radius
     * - Set b->color_id = color_id, b->active = 1 */
    (void)b; (void)pos; (void)radius; (void)mass; (void)color_id;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Initialize box rigid body
 *
 * For a solid box with dimensions w × h × d where:
 *   w = 2 * half_ext.x,  h = 2 * half_ext.y,  d = 2 * half_ext.z
 * The scalar moment of inertia is:
 *   I = mass * (w^2 + h^2 + d^2) / 12
 *
 * This is an isotropic simplification: a real box has a different I around
 * each axis (I_x, I_y, I_z), but for the demo we average them into one
 * scalar that captures the overall rotational inertia.
 *
 * Implementation:
 *   float w = 2.0f * half_ext.x;
 *   float h = 2.0f * half_ext.y;
 *   float d = 2.0f * half_ext.z;
 *   b->inertia = mass * (w*w + h*h + d*d) / 12.0f;
 *   b->shape.type          = SHAPE_BOX;
 *   b->shape.half_extents  = half_ext;
 *   (rest is identical to sphere init — same pos/vel/orientation/inv_mass pattern)
 * ══════════════════════════════════════════════════════════════════ */
static void body_init_box(RigidBody *b, vec3 pos, vec3 half_ext,
                           float mass, int color_id) {
    /* TODO: Initialize rigid body for a box
     * - Same pos/vel/angular_vel/orientation/mass/inv_mass setup as sphere
     * - Compute w=2*half_ext.x, h=2*half_ext.y, d=2*half_ext.z
     * - Compute b->inertia = mass * (w*w + h*h + d*d) / 12.0f
     * - Set inv_inertia as usual
     * - Set b->shape.type = SHAPE_BOX, b->shape.half_extents = half_ext
     * - Set b->color_id = color_id, b->active = 1 */
    (void)b; (void)pos; (void)half_ext; (void)mass; (void)color_id;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Print body state for debugging
 *
 * Print label, position, velocity, mass, and inertia to stdout.
 * Example output:
 *   [MySphere] pos=(0.00, 5.00, 0.00) vel=(0.00, 0.00, 0.00)
 *              mass=1.00 inertia=0.10 inv_mass=1.00 inv_inertia=10.00
 *
 * Implementation hint:
 *   printf("[%s] pos=(%.2f, %.2f, %.2f) vel=(%.2f, %.2f, %.2f)\n",
 *          label, b->pos.x, b->pos.y, b->pos.z,
 *          b->vel.x, b->vel.y, b->vel.z);
 *   printf("     mass=%.2f inertia=%.4f inv_mass=%.4f inv_inertia=%.4f\n",
 *          b->mass, b->inertia, b->inv_mass, b->inv_inertia);
 *   Use a ternary to print shape type:
 *     (b->shape.type == SHAPE_SPHERE) ? "sphere" : "box"
 * ══════════════════════════════════════════════════════════════════ */
static void body_print(const RigidBody *b, const char *label) {
    /* TODO: Print body state using printf
     * - Print label, b->pos (x,y,z), b->vel (x,y,z)
     * - Print b->mass, b->inertia, b->inv_mass, b->inv_inertia
     * - Print shape type ("sphere" or "box") and relevant dimension */
    (void)b; (void)label;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Apply a force for one timestep
 *
 * Newton's second law: F = m * a  =>  a = F / m = F * inv_mass
 * Semi-implicit: accumulate velocity change directly:
 *   vel += force * inv_mass * dt
 * Static bodies (inv_mass == 0) are not affected because any term
 * scaled by inv_mass=0 evaluates to zero — no special branch needed.
 *
 * Implementation (one line):
 *   b->vel = v3_add(b->vel, v3_scale(force, b->inv_mass * dt));
 * ══════════════════════════════════════════════════════════════════ */
static void body_apply_force(RigidBody *b, vec3 force, float dt) {
    /* TODO: b->vel += force * b->inv_mass * dt
     * Hint: v3_add(b->vel, v3_scale(force, b->inv_mass * dt)) */
    (void)b; (void)force; (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Apply a torque for one timestep
 *
 * Angular analogue of force application:
 *   angular_vel += torque * inv_inertia * dt
 * Static bodies (inv_inertia == 0) are not affected — same reason as
 * inv_mass: the scale factor is zero so the update is a no-op.
 * Torque is a vec3 whose direction is the rotation axis (right-hand rule)
 * and whose magnitude is Newton·metres.
 *
 * Implementation (one line, identical structure to body_apply_force):
 *   b->angular_vel = v3_add(b->angular_vel,
 *                           v3_scale(torque, b->inv_inertia * dt));
 * ══════════════════════════════════════════════════════════════════ */
static void body_apply_torque(RigidBody *b, vec3 torque, float dt) {
    /* TODO: b->angular_vel += torque * b->inv_inertia * dt
     * Hint: same pattern as body_apply_force */
    (void)b; (void)torque; (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * UNIT TESTS — run and verify expected output
 * ══════════════════════════════════════════════════════════════════ */

/* Simple float comparison within tolerance */
static int approx_eq(float a, float b, float tol) {
    return fabsf(a - b) < tol;
}

static int tests_run = 0, tests_passed = 0;
static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); }
    else       { printf("  FAIL: %s\n", name); }
}

int main(void) {
    printf("=== Phase 5a: Rigid Body Initialization Tests ===\n\n");

    /* ── Sphere inertia ── */
    printf("-- Sphere (mass=1, radius=0.5) --\n");
    RigidBody sphere = {0};
    body_init_sphere(&sphere, v3(0, 5, 0), 0.5f, 1.0f, 0);
    body_print(&sphere, "Sphere");
    /* I = 0.4 * 1.0 * 0.5^2 = 0.1 */
    check("sphere inertia == 0.1",   approx_eq(sphere.inertia,     0.1f,  1e-4f));
    check("sphere inv_inertia==10",  approx_eq(sphere.inv_inertia, 10.0f, 1e-3f));
    check("sphere inv_mass == 1.0",  approx_eq(sphere.inv_mass,    1.0f,  1e-4f));
    check("sphere pos.y == 5.0",     approx_eq(sphere.pos.y,       5.0f,  1e-4f));
    check("sphere vel zeroed",       approx_eq(v3_len(sphere.vel), 0.0f,  1e-6f));
    check("sphere active == 1",      sphere.active == 1);
    check("sphere shape SPHERE",     sphere.shape.type == SHAPE_SPHERE);
    check("sphere shape radius",     approx_eq(sphere.shape.radius, 0.5f, 1e-4f));
    printf("\n");

    /* ── Box inertia ── */
    printf("-- Box (mass=2, half_ext=(0.5,0.5,0.5)) --\n");
    RigidBody box = {0};
    body_init_box(&box, v3(1, 2, 3), v3(0.5f, 0.5f, 0.5f), 2.0f, 1);
    body_print(&box, "Box");
    /* w=h=d=1, I = 2*(1+1+1)/12 = 0.5 */
    check("box inertia == 0.5",    approx_eq(box.inertia,     0.5f,  1e-4f));
    check("box inv_inertia == 2",  approx_eq(box.inv_inertia, 2.0f,  1e-4f));
    check("box inv_mass == 0.5",   approx_eq(box.inv_mass,    0.5f,  1e-4f));
    check("box pos correct",       approx_eq(box.pos.x, 1.0f, 1e-4f) &&
                                   approx_eq(box.pos.z, 3.0f, 1e-4f));
    check("box shape BOX",         box.shape.type == SHAPE_BOX);
    check("box half_ext.x",        approx_eq(box.shape.half_extents.x, 0.5f, 1e-4f));
    printf("\n");

    /* ── Static body (mass = 0) ── */
    printf("-- Static sphere (mass=0) --\n");
    RigidBody ground = {0};
    body_init_sphere(&ground, v3(0, -2, 0), 5.0f, 0.0f, 2);
    body_print(&ground, "StaticSphere");
    check("static inv_mass == 0",    approx_eq(ground.inv_mass,    0.0f, 1e-6f));
    check("static inv_inertia == 0", approx_eq(ground.inv_inertia, 0.0f, 1e-6f));
    printf("\n");

    /* ── body_apply_force ── */
    printf("-- Apply force (gravity for 1s) --\n");
    RigidBody b = {0};
    body_init_sphere(&b, v3(0,10,0), 0.4f, 1.0f, 0);
    vec3 gravity = v3(0, -9.81f, 0);
    body_apply_force(&b, gravity, 1.0f);   /* vel.y should be -9.81 */
    check("force: vel.y == -9.81", approx_eq(b.vel.y, -9.81f, 1e-3f));
    check("force: vel.x == 0",     approx_eq(b.vel.x,   0.0f, 1e-6f));
    printf("\n");

    /* ── body_apply_torque ── */
    printf("-- Apply torque --\n");
    body_apply_torque(&b, v3(0, 10.0f, 0), 0.1f);  /* angular_vel.y += 10 * inv_inertia * 0.1 */
    /* I = 0.4*1*0.16 = 0.064, inv_I ≈ 15.625, angular_vel.y = 10*15.625*0.1 = 15.625 */
    float expected_av = 10.0f * b.inv_inertia * 0.1f;
    check("torque: angular_vel.y correct", approx_eq(b.angular_vel.y, expected_av, 1e-3f));
    printf("\n");

    /* ── Static body ignores forces ── */
    printf("-- Static body ignores force/torque --\n");
    RigidBody st = {0};
    body_init_sphere(&st, v3(0,0,0), 1.0f, 0.0f, 0);
    body_apply_force(&st,  v3(100,100,100), 1.0f);
    body_apply_torque(&st, v3(100,100,100), 1.0f);
    check("static: vel unchanged",         approx_eq(v3_len(st.vel),         0.0f, 1e-6f));
    check("static: angular_vel unchanged", approx_eq(v3_len(st.angular_vel), 0.0f, 1e-6f));
    printf("\n");

    /* ── Non-cubic box ── */
    printf("-- Non-cubic box (mass=3, half_ext=(1,0.5,0.25)) --\n");
    RigidBody nb = {0};
    body_init_box(&nb, v3(0,0,0), v3(1.0f, 0.5f, 0.25f), 3.0f, 0);
    body_print(&nb, "NonCubicBox");
    /* w=2, h=1, d=0.5; I = 3*(4+1+0.25)/12 = 3*5.25/12 = 1.3125 */
    check("non-cubic box inertia", approx_eq(nb.inertia, 1.3125f, 1e-4f));
    printf("\n");

    printf("=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
