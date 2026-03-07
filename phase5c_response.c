/*
 * phase5c_response.c — Collision Response and Physics Integration
 *
 * EXERCISE (sub-phase 5c): Implement impulse-based collision response
 * and semi-implicit Euler integration for the full physics simulation.
 *
 * Build: gcc -O2 -o phase5c_response phase5c_response.c -lm
 * Run:   ./phase5c_response
 * Controls: [Space] add body, [R] reset, [P] pause, [WASD] camera, [Q] quit
 *
 * LEARNING GOALS:
 * - Apply impulse-based collision response (includes rotational effects)
 * - Understand the impulse formula: j = -(1+e)*vn / (1/m1 + 1/m2 + rotational terms)
 * - Implement position correction to prevent body overlap (Baumgarte stabilization)
 * - Add friction impulse to simulate surface friction
 * - Integrate equations of motion with semi-implicit Euler
 *
 * TECHNIQUE OVERVIEW:
 *
 * 1. IMPULSE-BASED COLLISION RESPONSE
 *    Rather than applying continuous forces (which need time-integration and
 *    can go unstable), we apply an instantaneous velocity change (impulse J).
 *    For body A at contact point P, moment arm r_A = P - pos_A:
 *      Δv_A  =  J / m_A             (linear velocity change)
 *      Δω_A  =  (r_A × J) / I_A    (angular velocity change)
 *    The cross product r_A × J is the torque arm; dividing by I_A gives the
 *    angular acceleration integrated over an infinitesimal timestep.
 *
 * 2. IMPULSE MAGNITUDE FORMULA
 *    For bodies A and B, restitution e (0 = inelastic, 1 = elastic):
 *
 *      j = -(1+e) * v_rel·n
 *          ─────────────────────────────────────────────────────────────
 *          1/m_A + 1/m_B + (r_A×n)·(I_A⁻¹*(r_A×n)) + (r_B×n)·(I_B⁻¹*(r_B×n))
 *
 *    where v_rel = v_A + ω_A×r_A − (v_B + ω_B×r_B)  at the contact point.
 *    Numerator:   -(1+e)*v_rel·n  — relative approach speed, scaled by bounce.
 *    Denominator: sum of linear (1/m) and rotational (cross-product) responses.
 *    If v_rel·n > 0 the bodies are already separating — skip the impulse.
 *
 * 3. BAUMGARTE STABILIZATION
 *    Impulses correct velocities but not positions.  Floating-point errors
 *    accumulate each step, causing bodies to slowly drift into each other.
 *    Baumgarte adds a "correction velocity" (bias) proportional to the
 *    penetration depth directly into the impulse numerator:
 *      bias = BAUMGARTE_FRACTION * depth / dt
 *    This gently pushes overlapping bodies apart over several frames.
 *    Typical fraction: 0.2–0.8.  Too large → jittery; too small → slow drift.
 *    Here we use 0.8 (80 % correction per step).
 *
 * 4. FRICTION IMPULSE
 *    After the normal impulse, we compute the tangential (friction) impulse.
 *    Tangent direction: t = normalize(v_rel − (v_rel·n)*n)
 *      — the relative velocity component perpendicular to the contact normal.
 *    Friction impulse magnitude (Coulomb model):
 *      j_t = −dot(v_rel, t) / total_inv_mass
 *      j_t = clamp(j_t, −mu*|j|, mu*|j|)   ← friction cone
 *    mu = FRICTION constant.  Clamping ensures friction can only oppose
 *    relative tangential motion, never reverse it.
 *
 * 5. SEMI-IMPLICIT EULER INTEGRATION
 *    Two ways to step position forward:
 *      Explicit Euler:       v_new = v + a*dt;  pos_new = pos + v*dt
 *      Semi-implicit Euler:  v_new = v + a*dt;  pos_new = pos + v_new*dt  ← this one
 *    Using the NEW velocity for the position update (symplectic / semi-implicit)
 *    conserves energy much better for oscillating/bouncing systems.  It is only
 *    one line different from explicit Euler but dramatically more stable.
 *
 * 6. QUATERNION INTEGRATION
 *    To update orientation from angular velocity ω over timestep dt:
 *      q' = q + 0.5 * q ⊗ (0, ω_x, ω_y, ω_z) * dt
 *    (0, ω_x, ω_y, ω_z) is a "pure quaternion" (zero scalar part).
 *    The Hamilton product q ⊗ ω_quat converts the body-space angular velocity
 *    into a quaternion derivative.  After adding, renormalise q' to prevent
 *    length drift (accumulated floating-point errors would make q non-unit).
 *    In practice we convert ω to an axis-angle and compose rotations:
 *      dq = q_from_axis_angle(normalize(ω), |ω|*dt)
 *      q' = q_norm(dq ⊗ q)
 */

#define _POSIX_C_SOURCE 199309L
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <time.h>
#include <poll.h>
#include <math.h>
#include <signal.h>

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
static inline vec3 v3_neg(vec3 v) { return v3(-v.x, -v.y, -v.z); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }
static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    return len < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f / len);
}
static inline vec3 v3_abs(vec3 v) { return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
static inline vec3 v3_max(vec3 a, vec3 b) {
    return v3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
static inline float v3_max_comp(vec3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float half = rad * 0.5f, s = sinf(half);
    return (quat){cosf(half), axis.x*s, axis.y*s, axis.z*s};
}
static inline quat q_mul(quat a, quat b) {
    return (quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
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
 * TERMINAL (phase 1 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;
static volatile sig_atomic_t got_resize = 0;

typedef struct { char ch; uint8_t fg; uint8_t bg; } Cell;
static Cell *screen = NULL;
static char *out_buf = NULL;
static int out_cap = 0, out_len = 0;

static void out_flush(void) { if (out_len > 0) { write(STDOUT_FILENO, out_buf, out_len); out_len = 0; } }
static void out_append(const char *s, int n) {
    while (out_len + n > out_cap) { out_cap = out_cap ? out_cap*2 : 65536; out_buf = realloc(out_buf, out_cap); }
    memcpy(out_buf + out_len, s, n); out_len += n;
}
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) { term_w = ws.ws_col; term_h = ws.ws_row; }
}
static void handle_sigwinch(int sig) { (void)sig; got_resize = 1; }
static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = 0;
    }
}
static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios); raw_mode_enabled = 1; atexit(disable_raw_mode);
    signal(SIGWINCH, handle_sigwinch);
    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT|ICRNL|INPCK|ISTRIP|IXON); raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8); raw.c_lflag &= ~(ECHO|ICANON|IEXTEN|ISIG);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
}
static void present_screen(void) {
    char seq[64]; int prev_fg = -1, prev_bg = -1;
    out_append("\033[H", 3);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            Cell *c = &screen[y * term_w + x];
            if (c->fg != prev_fg || c->bg != prev_bg) {
                int n = snprintf(seq, sizeof(seq), "\033[38;5;%dm\033[48;5;%dm", c->fg, c->bg);
                out_append(seq, n); prev_fg = c->fg; prev_bg = c->bg;
            }
            out_append(&c->ch, 1);
        }
    }
    out_flush();
}
#define KEY_NONE 0
#define KEY_ESC  27
#define KEY_UP   1000
#define KEY_DOWN 1001
#define KEY_LEFT 1002
#define KEY_RIGHT 1003
static int read_key(void) {
    struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
    if (poll(&pfd, 1, 0) <= 0) return KEY_NONE;
    char c; if (read(STDIN_FILENO, &c, 1) != 1) return KEY_NONE;
    if (c == '\033') {
        char seq[3];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return KEY_ESC;
        if (seq[0] == '[') { switch (seq[1]) { case 'A': return KEY_UP; case 'B': return KEY_DOWN; case 'C': return KEY_RIGHT; case 'D': return KEY_LEFT; } }
        return KEY_ESC;
    }
    return (int)(unsigned char)c;
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

#define MAX_BODIES 32
static RigidBody bodies[MAX_BODIES];
static int num_bodies = 0;

/* ══════════════════════════════════════════════════════════════════
 * PHYSICS CONSTANTS (already provided)
 * ══════════════════════════════════════════════════════════════════ */

static const vec3  GRAVITY         = {0, -9.81f, 0};
static const float RESTITUTION     = 0.6f;
static const float FRICTION        = 0.3f;
static const float LINEAR_DAMPING  = 0.99f;
static const float ANGULAR_DAMPING = 0.98f;
static const float GROUND_Y        = -2.0f;

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
 * COLLISION DETECTION (solution from phase 5b)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   collided;
    vec3  point;
    vec3  normal;
    float penetration;
} Contact;

static Contact collide_sphere_ground(RigidBody *b) {
    Contact c = {0};
    float dist = b->pos.y - GROUND_Y - b->shape.radius;
    if (dist < 0) {
        c.collided    = 1;
        c.normal      = v3(0, 1, 0);
        c.penetration = -dist;
        c.point       = v3(b->pos.x, GROUND_Y, b->pos.z);
    }
    return c;
}

static Contact collide_box_ground(RigidBody *b) {
    Contact c = {0};
    vec3 he = b->shape.half_extents;
    float min_y = 1e10f;
    vec3  min_pt = v3(0, 0, 0);
    int signs[2] = {-1, 1};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                vec3 local = v3(signs[i]*he.x, signs[j]*he.y, signs[k]*he.z);
                vec3 world = v3_add(b->pos, q_rotate(b->orientation, local));
                if (world.y < min_y) { min_y = world.y; min_pt = world; }
            }
    if (min_y < GROUND_Y) {
        c.collided    = 1;
        c.normal      = v3(0, 1, 0);
        c.penetration = GROUND_Y - min_y;
        c.point       = min_pt;
    }
    return c;
}

static Contact collide_sphere_sphere(RigidBody *a, RigidBody *b) {
    Contact c = {0};
    vec3  diff     = v3_sub(b->pos, a->pos);
    float dist     = v3_len(diff);
    float min_dist = a->shape.radius + b->shape.radius;
    if (dist < min_dist && dist > EPSILON) {
        c.collided    = 1;
        c.normal      = v3_scale(diff, 1.0f / dist);
        c.penetration = min_dist - dist;
        c.point       = v3_add(a->pos, v3_scale(c.normal, a->shape.radius));
    }
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Impulse-based collision response
 *
 * Resolves a collision between `body` and `other` (NULL = static ground).
 * Calls apply_friction() internally for the tangential impulse.
 *
 * Steps:
 *
 * 1. POSITION CORRECTION (Baumgarte stabilization — push overlapping bodies apart):
 *      The factor 0.8 is the Baumgarte correction fraction: correcting 80% of
 *      penetration per step avoids overshooting while still converging quickly.
 *      correction = penetration * 0.8
 *      total_inv_mass = body->inv_mass + other->inv_mass   (other=0 if NULL)
 *      body->pos += normal * correction * body->inv_mass / total_inv_mass
 *      other->pos -= normal * correction * other->inv_mass / total_inv_mass
 *
 * 2. RELATIVE VELOCITY at contact point:
 *      r  = contact_point - body->pos
 *      vel_contact = body->vel + cross(body->angular_vel, r)
 *      (subtract other's vel + cross(other->angular_vel, r2) if other != NULL)
 *
 * 3. NORMAL COMPONENT — skip if bodies already separating:
 *      vn = dot(vel_contact, normal)
 *      if (vn > 0) return;    // positive vn means moving apart — no impulse
 *
 * 4. IMPULSE MAGNITUDE:
 *      rxn      = cross(r, normal)
 *      rot_term = dot(rxn, rxn) * body->inv_inertia
 *                 + (other's equivalent if other != NULL)
 *      j = -(1 + RESTITUTION) * vn / (total_inv_mass + rot_term)
 *      The rot_term captures how much the rotational inertia "absorbs"
 *      the impulse.  For a body with very large inertia, rot_term ≈ 0
 *      and only the linear 1/m terms matter.
 *
 * 5. APPLY LINEAR IMPULSE:
 *      body->vel  += normal * j * body->inv_mass
 *      other->vel -= normal * j * other->inv_mass    (if other != NULL)
 *
 * 6. APPLY ANGULAR IMPULSE:
 *      body->angular_vel  += cross(r,  normal*j) * body->inv_inertia
 *      other->angular_vel -= cross(r2, normal*j) * other->inv_inertia (if other)
 *      (cross(r, J) gives the torque arm; multiplying by inv_inertia gives Δω)
 *
 * 7. Call apply_friction(body, c, other, j) for the tangential component.
 * ══════════════════════════════════════════════════════════════════ */
static void apply_friction(RigidBody *body, Contact *c, RigidBody *other, float j);

static void resolve_collision(RigidBody *body, Contact *c, RigidBody *other) {
    if (!c->collided) return;
    /* TODO: Implement impulse-based collision response
     *
     * Key variables to compute:
     *   float other_inv_mass = other ? other->inv_mass : 0.0f;
     *   float total_inv_mass = body->inv_mass + other_inv_mass;
     *   if (total_inv_mass < EPSILON) return;   // both static, nothing to do
     *
     * Recall helper:
     *   v3_cross(a, b)    — cross product
     *   v3_dot(a, b)      — dot product
     *   v3_scale(v, s)    — scale vector
     *   v3_add / v3_sub   — add / subtract vectors
     */
    (void)body; (void)c; (void)other;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Friction impulse
 *
 * Called from resolve_collision() with the normal impulse magnitude j.
 * Friction opposes the tangential relative motion at the contact point.
 *
 * Steps:
 *   1. Recompute vel_contact (same as in resolve_collision):
 *        r  = c->point - body->pos
 *        vel_contact = body->vel + cross(body->angular_vel, r)
 *        if other: subtract other->vel + cross(other->angular_vel, r2)
 *
 *   2. tangent = vel_contact - normal * dot(vel_contact, normal)
 *      This strips the normal component, leaving only the tangential velocity.
 *      tlen    = length(tangent)
 *      if tlen < EPSILON: no tangential motion, return early
 *      tangent = tangent / tlen     [normalise to get friction direction]
 *
 *   3. jt = -dot(vel_contact, tangent) / total_inv_mass
 *      Clamp to the Coulomb friction cone (can't exceed mu * normal impulse):
 *        float max_jt = FRICTION * fabsf(j);
 *        jt = jt < -max_jt ? -max_jt : (jt > max_jt ? max_jt : jt);
 *      Clamping ensures friction can decelerate but never reverse motion.
 *
 *   4. Apply tangential linear impulse:
 *        body->vel  += tangent * jt * body->inv_mass
 *        other->vel -= tangent * jt * other->inv_mass   (if other)
 *
 *   5. Apply tangential angular impulse:
 *        body->angular_vel  += cross(r,  tangent*jt) * body->inv_inertia
 *        other->angular_vel -= cross(r2, tangent*jt) * other->inv_inertia (if other)
 * ══════════════════════════════════════════════════════════════════ */
static void apply_friction(RigidBody *body, Contact *c, RigidBody *other, float j) {
    /* TODO: Implement friction impulse
     *
     * Remember: r  = c->point - body->pos
     *           r2 = c->point - other->pos   (if other != NULL)
     */
    (void)body; (void)c; (void)other; (void)j;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Integrate a single rigid body (semi-implicit Euler)
 *
 * Order matters — update velocity before position (semi-implicit):
 *   1. vel += GRAVITY * dt               (apply gravitational acceleration)
 *      Equivalent to: b->vel = v3_add(b->vel, v3_scale(GRAVITY, dt));
 *
 *   2. vel *= LINEAR_DAMPING             (energy dissipation each frame)
 *      b->vel = v3_scale(b->vel, LINEAR_DAMPING);
 *      LINEAR_DAMPING < 1.0 slowly bleeds kinetic energy, simulating
 *      air resistance without needing an explicit drag force.
 *
 *   3. angular_vel *= ANGULAR_DAMPING
 *      Same idea for rotational motion.
 *
 *   4. pos += vel * dt                   (update position with NEW velocity)
 *      b->pos = v3_add(b->pos, v3_scale(b->vel, dt));
 *      Using the just-updated vel makes this semi-implicit (symplectic),
 *      which conserves energy better than explicit Euler.
 *
 *   5. Update orientation from angular_vel:
 *        float angle = v3_len(b->angular_vel) * dt;
 *        if (angle > EPSILON) {
 *            vec3 axis = v3_norm(b->angular_vel);
 *            quat dq   = q_from_axis_angle(axis, angle);
 *            b->orientation = q_norm(q_mul(dq, b->orientation));
 *        }
 *      q_norm prevents quaternion length drift due to floating-point errors.
 * ══════════════════════════════════════════════════════════════════ */
static void integrate_body(RigidBody *b, float dt) {
    /* TODO: Semi-implicit Euler integration
     * Steps listed above; use q_from_axis_angle + q_mul + q_norm */
    (void)b; (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Main physics step
 *
 * Called every frame at a fixed timestep (1/60 s).
 * A fixed dt (rather than the variable render dt) makes physics
 * deterministic and independent of frame rate.
 *
 * Steps:
 *   1. Integrate each active body: integrate_body(&bodies[i], dt)
 *      This advances velocities and positions by one timestep.
 *
 *   2. Ground collisions — for each active body:
 *        if SHAPE_SPHERE: c = collide_sphere_ground(&bodies[i])
 *        if SHAPE_BOX:    c = collide_box_ground(&bodies[i])
 *        resolve_collision(&bodies[i], &c, NULL)   // NULL = static ground
 *      NULL for `other` tells resolve_collision the ground has infinite mass
 *      (inv_mass = 0), so only the dynamic body receives velocity changes.
 *
 *   3. Sphere-sphere collisions — for each pair (i < j), both active:
 *        if both SHAPE_SPHERE:
 *          c = collide_sphere_sphere(&bodies[i], &bodies[j])
 *          if (c.collided): resolve_collision(&bodies[i], &c, &bodies[j])
 *      Only sphere-sphere is tested here; sphere-box is an optional extension.
 * ══════════════════════════════════════════════════════════════════ */
static void physics_step(float dt) {
    /* TODO: Integrate, then resolve ground and sphere-sphere collisions */
    (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * SDF RENDERING (phases 3-4 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere_f(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}
static float sdf_box_f(vec3 p, vec3 center, vec3 hs, quat orient) {
    quat inv = (quat){orient.w, -orient.x, -orient.y, -orient.z};
    vec3 local = q_rotate(inv, v3_sub(p, center));
    vec3 d = v3_sub(v3_abs(local), hs);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}
static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    float d;

    d = sdf_plane(p, v3(0,1,0), -GROUND_Y);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    for (int i = 0; i < num_bodies; i++) {
        RigidBody *b = &bodies[i];
        if (!b->active) continue;
        if (b->shape.type == SHAPE_SPHERE)
            d = sdf_sphere_f(p, b->pos, b->shape.radius);
        else
            d = sdf_box_f(p, b->pos, b->shape.half_extents, b->orientation);
        if (d < hit.dist) { hit.dist = d; hit.material_id = 1 + b->color_id; }
    }
    return hit;
}
static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

#define MAX_STEPS 48
#define MAX_DIST  40.0f
#define SURF_DIST 0.003f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;
static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0, ro, 0};
    float t = 0;
    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        SceneHit h = scene_sdf(p);
        if (h.dist < SURF_DIST) { result.hit=1; result.dist=t; result.point=p; result.material_id=h.material_id; return result; }
        t += h.dist;
    }
    result.dist = t; return result;
}
static vec3 calc_normal(vec3 p) {
    const float h = 0.001f;
    return v3_norm(v3(
        scene_dist(v3(p.x+h,p.y,p.z)) - scene_dist(v3(p.x-h,p.y,p.z)),
        scene_dist(v3(p.x,p.y+h,p.z)) - scene_dist(v3(p.x,p.y-h,p.z)),
        scene_dist(v3(p.x,p.y,p.z+h)) - scene_dist(v3(p.x,p.y,p.z-h))
    ));
}

static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10
static const uint8_t MATERIAL_COLORS[][3] = {
    {2,3,2},{5,1,1},{1,3,5},{5,4,0},{5,5,0},{5,0,5},{0,5,5},{2,5,2},{5,2,3},
};
static uint8_t color_216(int r, int g, int b) {
    r=r<0?0:(r>5?5:r); g=g<0?0:(g>5?5:g); b=b<0?0:(b>5?5:b);
    return 16 + 36*r + 6*g + b;
}
typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;
static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 0, 16};
    vec3 light_dir = v3_norm(v3(0.5f, 1.0f, 0.3f));
    vec3 half_vec  = v3_norm(v3_add(light_dir, v3_neg(rd)));
    float ambient  = 0.2f;
    float diffuse  = fmaxf(0.0f, v3_dot(normal, light_dir)) * 0.6f;
    float specular = powf(fmaxf(0.0f, v3_dot(normal, half_vec)), 16.0f) * 0.3f;
    float fog      = expf(-dist * 0.05f);
    float lum      = fminf(1.0f, fmaxf(0.0f, (ambient+diffuse+specular)*fog));
    result.ch = LUMINANCE_RAMP[(int)(lum*(RAMP_LEN-1)+0.5f)];
    int num_mats = sizeof(MATERIAL_COLORS)/sizeof(MATERIAL_COLORS[0]);
    int mi = mat_id < num_mats ? mat_id : 0;
    float cs = 0.3f + lum * 0.7f;
    result.fg = color_216((int)(MATERIAL_COLORS[mi][0]*cs),(int)(MATERIAL_COLORS[mi][1]*cs),(int)(MATERIAL_COLORS[mi][2]*cs));
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * ORBIT CAMERA (phase 4 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    vec3 target; float distance, azimuth, elevation, fov;
    float smooth_az, smooth_el, smooth_dist;
} OrbitCamera;

static vec3 orbit_camera_pos(OrbitCamera *cam) {
    float y   = sinf(cam->smooth_el) * cam->smooth_dist;
    float xzd = cosf(cam->smooth_el) * cam->smooth_dist;
    return v3_add(cam->target, v3(sinf(cam->smooth_az)*xzd, y, cosf(cam->smooth_az)*xzd));
}
static void orbit_camera_get_ray(OrbitCamera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 pos   = orbit_camera_pos(cam);
    vec3 fwd   = v3_norm(v3_sub(cam->target, pos));
    vec3 right = v3_norm(v3_cross(fwd, v3(0,1,0)));
    vec3 up    = v3_cross(right, fwd);
    float hf   = tanf(cam->fov * 0.5f);
    *ro = pos; *rd = v3_norm(v3_add(v3_add(fwd, v3_scale(right, u*hf)), v3_scale(up, v*hf)));
}
static void orbit_camera_update(OrbitCamera *cam, float dt) {
    float s = 1.0f - expf(-10.0f * dt);
    cam->smooth_az   = lerpf(cam->smooth_az,   cam->azimuth,   s);
    cam->smooth_el   = lerpf(cam->smooth_el,   cam->elevation, s);
    cam->smooth_dist = lerpf(cam->smooth_dist, cam->distance,  s);
}
static void render_frame(OrbitCamera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            float u = (2.0f*x/term_w - 1.0f) * aspect;
            float v = -(2.0f*y/term_h - 1.0f);
            vec3 ro, rd;
            orbit_camera_get_ray(cam, u, v, &ro, &rd);
            RayResult hit = ray_march(ro, rd);
            Cell *cell = &screen[y * term_w + x];
            if (hit.hit) {
                vec3 normal = calc_normal(hit.point);
                ShadeResult shade = shade_point(hit.point, rd, normal, hit.material_id, hit.dist);
                cell->ch=shade.ch; cell->fg=shade.fg; cell->bg=shade.bg;
            } else {
                float sky_t = (v+1.0f)*0.5f;
                cell->ch=' '; cell->fg=cell->bg=17+(int)(sky_t*4);
            }
        }
    }
}

static void draw_hud(double fps, int paused) {
    char hud[256];
    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " Phase 5c: Response | FPS: %.0f | Bodies: %d %s"
        "| [Space] add [R] reset [P] pause [WASD] cam [Q] quit ",
        fps, num_bodies, paused ? "[PAUSED] " : "");
    write(STDOUT_FILENO, hud, strlen(hud));
}

static double get_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void sleep_ms(int ms) {
    struct timespec ts = { .tv_sec=ms/1000, .tv_nsec=(ms%1000)*1000000L };
    nanosleep(&ts, NULL);
}

static void reset_scene(void) {
    num_bodies = 0;
    body_init_sphere(&bodies[num_bodies++], v3(-2, 3, 0),  0.5f, 1.0f, 0);
    body_init_sphere(&bodies[num_bodies++], v3( 0, 5, 0),  0.6f, 1.2f, 1);
    body_init_sphere(&bodies[num_bodies++], v3( 2, 4, 0),  0.4f, 0.8f, 2);
    body_init_box   (&bodies[num_bodies++], v3( 0, 2, 2),  v3(0.5f,0.5f,0.5f), 2.0f, 3);
    bodies[num_bodies-1].angular_vel = v3(1, 2, 0.5f);
    bodies[0].vel = v3( 2, 0,  1);
    bodies[1].vel = v3(-1, 0, -1);
}

static void add_random_body(void) {
    if (num_bodies >= MAX_BODIES) return;
    float x = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
    float z = ((float)rand()/RAND_MAX - 0.5f) * 4.0f;
    int color = num_bodies % 8;
    if (rand() % 3 == 0) {
        body_init_box(&bodies[num_bodies++], v3(x, 6, z),
                      v3(0.3f + (rand()%3)*0.1f, 0.3f + (rand()%3)*0.1f, 0.3f + (rand()%3)*0.1f),
                      1.5f, color);
        bodies[num_bodies-1].angular_vel = v3(rand()%3-1, rand()%3-1, rand()%3-1);
    } else {
        body_init_sphere(&bodies[num_bodies++], v3(x, 6, z),
                         0.3f + (rand()%4)*0.1f, 1.0f, color);
    }
}

int main(void) {
    srand((unsigned)time(NULL));
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target     = v3(0,0,0), .distance  = 12.0f,
        .azimuth    = 0.5f,      .elevation = 0.4f,
        .fov        = (float)M_PI / 3.0f,
        .smooth_az  = 0.5f,      .smooth_el = 0.4f, .smooth_dist = 12.0f
    };

    reset_scene();

    double last_time = get_time(), fps_time = last_time;
    int fps_count = 0; double fps_display = 0;
    int running = 1, paused = 0;
    const float physics_dt = 1.0f / 60.0f;
    float physics_accum = 0;

    while (running) {
        double now = get_time();
        float dt = (float)(now - last_time); last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0; get_term_size(); free(screen);
            screen = calloc(term_w * term_h, sizeof(Cell));
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        int key;
        while ((key = read_key()) != KEY_NONE) {
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'p': case 'P': paused = !paused; break;
                case 'r': case 'R': reset_scene(); break;
                case ' ': add_random_body(); break;
                case 'w': case 'W': case KEY_UP:    cam.elevation += 0.1f; break;
                case 's': case 'S': case KEY_DOWN:  cam.elevation -= 0.1f; break;
                case 'a': case 'A': case KEY_LEFT:  cam.azimuth   -= 0.1f; break;
                case 'd': case 'D': case KEY_RIGHT: cam.azimuth   += 0.1f; break;
                case '+': case '=': cam.distance -= 0.5f; break;
                case '-': case '_': cam.distance += 0.5f; break;
            }
        }
        cam.elevation = fmaxf(-(float)M_PI/2+0.1f, fminf((float)M_PI/2-0.1f, cam.elevation));
        cam.distance  = fmaxf(5.0f, fminf(30.0f, cam.distance));

        if (!paused) {
            physics_accum += dt;
            while (physics_accum >= physics_dt) {
                physics_step(physics_dt);
                physics_accum -= physics_dt;
            }
        }

        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0; fps_time = now;
        }
        draw_hud(fps_display, paused);
        sleep_ms(30);
    }

    free(screen); free(out_buf);
    return 0;
}
