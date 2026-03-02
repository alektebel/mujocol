/*
 * phase5_physics.c — Rigid Body Dynamics
 *
 * Rigid body state, forces, semi-implicit Euler integration,
 * collision detection via SDFs, impulse-based collision response.
 *
 * Deliverable: Balls bouncing on a floor, boxes tumbling with gravity.
 *
 * Build: gcc -O3 -o phase5_physics phase5_physics.c -lm
 * Run:   ./phase5_physics
 * Controls: [Space] add ball, [R] reset, [P] pause, [Q] quit
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
 * 3D MATH
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float w, x, y, z; } quat;

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x * s, v.y * s, v.z * s); }
static inline vec3 v3_neg(vec3 v) { return v3(-v.x, -v.y, -v.z); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }
static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    return len < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f / len);
}
static inline vec3 v3_abs(vec3 v) { return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
static inline vec3 v3_max(vec3 a, vec3 b) { return v3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
static inline float v3_max_comp(vec3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

/* Quaternion operations */
static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
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
 * TERMINAL
 * ══════════════════════════════════════════════════════════════════ */

static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;
static volatile sig_atomic_t got_resize = 0;

typedef struct { char ch; uint8_t fg; uint8_t bg; } Cell;
static Cell *screen = NULL;
static char *out_buf = NULL;
static int out_cap = 0, out_len = 0;

static void out_flush(void) {
    if (out_len > 0) { write(STDOUT_FILENO, out_buf, out_len); out_len = 0; }
}

static void out_append(const char *s, int n) {
    while (out_len + n > out_cap) {
        out_cap = out_cap ? out_cap * 2 : 65536;
        out_buf = realloc(out_buf, out_cap);
    }
    memcpy(out_buf + out_len, s, n);
    out_len += n;
}

static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        term_w = ws.ws_col;
        term_h = ws.ws_row;
    }
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
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_mode_enabled = 1;
    atexit(disable_raw_mode);
    signal(SIGWINCH, handle_sigwinch);

    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
}

static void present_screen(void) {
    char seq[64];
    int prev_fg = -1, prev_bg = -1;
    out_append("\033[H", 3);

    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            Cell *c = &screen[y * term_w + x];
            if (c->fg != prev_fg || c->bg != prev_bg) {
                int n = snprintf(seq, sizeof(seq), "\033[38;5;%dm\033[48;5;%dm", c->fg, c->bg);
                out_append(seq, n);
                prev_fg = c->fg;
                prev_bg = c->bg;
            }
            out_append(&c->ch, 1);
        }
    }
    out_flush();
}

/* Input */
#define KEY_NONE  0
#define KEY_ESC   27
#define KEY_UP    1000
#define KEY_DOWN  1001
#define KEY_LEFT  1002
#define KEY_RIGHT 1003

static int read_key(void) {
    struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
    if (poll(&pfd, 1, 0) <= 0) return KEY_NONE;
    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return KEY_NONE;
    if (c == '\033') {
        char seq[3];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return KEY_ESC;
        if (seq[0] == '[') {
            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        return KEY_ESC;
    }
    return (int)(unsigned char)c;
}

/* ══════════════════════════════════════════════════════════════════
 * RIGID BODY
 * ══════════════════════════════════════════════════════════════════ */

typedef enum { SHAPE_SPHERE, SHAPE_BOX } ShapeType;

typedef struct {
    ShapeType type;
    float radius;      /* for sphere */
    vec3 half_extents; /* for box */
} Shape;

typedef struct {
    vec3 pos;
    vec3 vel;
    quat orientation;
    vec3 angular_vel;

    float mass;
    float inv_mass;
    float inertia;      /* simplified: scalar for sphere, approximate for box */
    float inv_inertia;

    Shape shape;
    int color_id;
    int active;
} RigidBody;

#define MAX_BODIES 32
static RigidBody bodies[MAX_BODIES];
static int num_bodies = 0;

static void body_init_sphere(RigidBody *b, vec3 pos, float radius, float mass, int color_id) {
    b->pos = pos;
    b->vel = v3(0, 0, 0);
    b->orientation = q_identity();
    b->angular_vel = v3(0, 0, 0);

    b->mass = mass;
    b->inv_mass = mass > 0 ? 1.0f / mass : 0;
    b->inertia = 0.4f * mass * radius * radius;  /* 2/5 * m * r^2 */
    b->inv_inertia = b->inertia > 0 ? 1.0f / b->inertia : 0;

    b->shape.type = SHAPE_SPHERE;
    b->shape.radius = radius;
    b->color_id = color_id;
    b->active = 1;
}

static void body_init_box(RigidBody *b, vec3 pos, vec3 half_ext, float mass, int color_id) {
    b->pos = pos;
    b->vel = v3(0, 0, 0);
    b->orientation = q_identity();
    b->angular_vel = v3(0, 0, 0);

    b->mass = mass;
    b->inv_mass = mass > 0 ? 1.0f / mass : 0;
    /* Approximate inertia for box */
    float w = half_ext.x * 2, h = half_ext.y * 2, d = half_ext.z * 2;
    b->inertia = mass * (w*w + h*h + d*d) / 12.0f;
    b->inv_inertia = b->inertia > 0 ? 1.0f / b->inertia : 0;

    b->shape.type = SHAPE_BOX;
    b->shape.half_extents = half_ext;
    b->color_id = color_id;
    b->active = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * PHYSICS CONSTANTS
 * ══════════════════════════════════════════════════════════════════ */

static const vec3 GRAVITY = {0, -9.81f, 0};
static const float RESTITUTION = 0.6f;   /* bounciness */
static const float FRICTION = 0.3f;
static const float LINEAR_DAMPING = 0.99f;
static const float ANGULAR_DAMPING = 0.98f;
static const float GROUND_Y = -2.0f;

/* ══════════════════════════════════════════════════════════════════
 * COLLISION DETECTION
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int collided;
    vec3 point;
    vec3 normal;
    float penetration;
} Contact;

/* Sphere vs ground plane (y = GROUND_Y) */
static Contact collide_sphere_ground(RigidBody *b) {
    Contact c = {0};
    if (b->shape.type != SHAPE_SPHERE) return c;

    float dist = b->pos.y - GROUND_Y - b->shape.radius;
    if (dist < 0) {
        c.collided = 1;
        c.normal = v3(0, 1, 0);
        c.penetration = -dist;
        c.point = v3(b->pos.x, GROUND_Y, b->pos.z);
    }
    return c;
}

/* Box vs ground plane (simplified: axis-aligned) */
static Contact collide_box_ground(RigidBody *b) {
    Contact c = {0};
    if (b->shape.type != SHAPE_BOX) return c;

    /* For simplicity, compute AABB after rotation and check against ground */
    vec3 corners[8];
    vec3 he = b->shape.half_extents;
    int idx = 0;
    for (int i = -1; i <= 1; i += 2) {
        for (int j = -1; j <= 1; j += 2) {
            for (int k = -1; k <= 1; k += 2) {
                vec3 local = v3(i * he.x, j * he.y, k * he.z);
                corners[idx++] = v3_add(b->pos, q_rotate(b->orientation, local));
            }
        }
    }

    /* Find lowest corner */
    float min_y = corners[0].y;
    int min_idx = 0;
    for (int i = 1; i < 8; i++) {
        if (corners[i].y < min_y) {
            min_y = corners[i].y;
            min_idx = i;
        }
    }

    float dist = min_y - GROUND_Y;
    if (dist < 0) {
        c.collided = 1;
        c.normal = v3(0, 1, 0);
        c.penetration = -dist;
        c.point = corners[min_idx];
    }
    return c;
}

/* Sphere vs sphere */
static Contact collide_sphere_sphere(RigidBody *a, RigidBody *b) {
    Contact c = {0};
    if (a->shape.type != SHAPE_SPHERE || b->shape.type != SHAPE_SPHERE) return c;

    vec3 diff = v3_sub(b->pos, a->pos);
    float dist = v3_len(diff);
    float min_dist = a->shape.radius + b->shape.radius;

    if (dist < min_dist && dist > EPSILON) {
        c.collided = 1;
        c.normal = v3_scale(diff, 1.0f / dist);
        c.penetration = min_dist - dist;
        c.point = v3_add(a->pos, v3_scale(c.normal, a->shape.radius));
    }
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * COLLISION RESPONSE
 * ══════════════════════════════════════════════════════════════════ */

static void resolve_collision(RigidBody *body, Contact *c, RigidBody *other) {
    if (!c->collided) return;

    /* Position correction (separate overlapping bodies) */
    float correction = c->penetration * 0.8f;
    body->pos = v3_add(body->pos, v3_scale(c->normal, correction * body->inv_mass /
                       (body->inv_mass + (other ? other->inv_mass : 0))));

    /* Calculate relative velocity at contact point */
    vec3 r = v3_sub(c->point, body->pos);
    vec3 vel_at_contact = v3_add(body->vel, v3_cross(body->angular_vel, r));

    if (other) {
        vec3 r_other = v3_sub(c->point, other->pos);
        vec3 vel_other = v3_add(other->vel, v3_cross(other->angular_vel, r_other));
        vel_at_contact = v3_sub(vel_at_contact, vel_other);
    }

    float vn = v3_dot(vel_at_contact, c->normal);

    /* Only resolve if approaching */
    if (vn > 0) return;

    /* Impulse magnitude */
    float j = -(1.0f + RESTITUTION) * vn;
    float inv_mass_sum = body->inv_mass + (other ? other->inv_mass : 0);

    /* Add rotational terms */
    vec3 r_cross_n = v3_cross(r, c->normal);
    float rot_term = v3_dot(r_cross_n, v3_scale(r_cross_n, body->inv_inertia));
    if (other) {
        vec3 r_other = v3_sub(c->point, other->pos);
        vec3 r_cross_n_other = v3_cross(r_other, c->normal);
        rot_term += v3_dot(r_cross_n_other, v3_scale(r_cross_n_other, other->inv_inertia));
    }

    j /= (inv_mass_sum + rot_term);

    /* Apply impulse */
    vec3 impulse = v3_scale(c->normal, j);
    body->vel = v3_add(body->vel, v3_scale(impulse, body->inv_mass));
    body->angular_vel = v3_add(body->angular_vel,
                                v3_scale(v3_cross(r, impulse), body->inv_inertia));

    if (other) {
        other->vel = v3_sub(other->vel, v3_scale(impulse, other->inv_mass));
        vec3 r_other = v3_sub(c->point, other->pos);
        other->angular_vel = v3_sub(other->angular_vel,
                                    v3_scale(v3_cross(r_other, impulse), other->inv_inertia));
    }

    /* Friction impulse */
    vec3 tangent = v3_sub(vel_at_contact, v3_scale(c->normal, vn));
    float tangent_len = v3_len(tangent);
    if (tangent_len > EPSILON) {
        tangent = v3_scale(tangent, 1.0f / tangent_len);
        float jt = -v3_dot(vel_at_contact, tangent) / inv_mass_sum;
        jt = fmaxf(-FRICTION * j, fminf(FRICTION * j, jt));

        vec3 friction_impulse = v3_scale(tangent, jt);
        body->vel = v3_add(body->vel, v3_scale(friction_impulse, body->inv_mass));
    }
}

/* ══════════════════════════════════════════════════════════════════
 * PHYSICS STEP
 * ══════════════════════════════════════════════════════════════════ */

static void physics_step(float dt) {
    /* Integrate forces (semi-implicit Euler) */
    for (int i = 0; i < num_bodies; i++) {
        RigidBody *b = &bodies[i];
        if (!b->active || b->inv_mass == 0) continue;

        /* Apply gravity */
        b->vel = v3_add(b->vel, v3_scale(GRAVITY, dt));

        /* Damping */
        b->vel = v3_scale(b->vel, LINEAR_DAMPING);
        b->angular_vel = v3_scale(b->angular_vel, ANGULAR_DAMPING);

        /* Update position */
        b->pos = v3_add(b->pos, v3_scale(b->vel, dt));

        /* Update orientation */
        float ang_speed = v3_len(b->angular_vel);
        if (ang_speed > EPSILON) {
            vec3 axis = v3_scale(b->angular_vel, 1.0f / ang_speed);
            quat dq = q_from_axis_angle(axis, ang_speed * dt);
            b->orientation = q_norm(q_mul(dq, b->orientation));
        }
    }

    /* Collision detection and response */
    for (int i = 0; i < num_bodies; i++) {
        RigidBody *b = &bodies[i];
        if (!b->active) continue;

        /* Ground collisions */
        Contact c;
        if (b->shape.type == SHAPE_SPHERE) {
            c = collide_sphere_ground(b);
        } else {
            c = collide_box_ground(b);
        }
        resolve_collision(b, &c, NULL);

        /* Body-body collisions */
        for (int j = i + 1; j < num_bodies; j++) {
            RigidBody *other = &bodies[j];
            if (!other->active) continue;

            c = collide_sphere_sphere(b, other);
            if (c.collided) {
                resolve_collision(b, &c, other);
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SDF FOR RENDERING
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}

static float sdf_box(vec3 p, vec3 center, vec3 half_size, quat orient) {
    /* Transform point to box's local space */
    quat inv_orient = (quat){orient.w, -orient.x, -orient.y, -orient.z};
    vec3 local = q_rotate(inv_orient, v3_sub(p, center));
    vec3 d = v3_sub(v3_abs(local), half_size);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}

static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};

    /* Ground plane */
    float d_ground = sdf_plane(p, v3(0, 1, 0), -GROUND_Y);
    if (d_ground < hit.dist) { hit.dist = d_ground; hit.material_id = 0; }

    /* Dynamic bodies */
    for (int i = 0; i < num_bodies; i++) {
        RigidBody *b = &bodies[i];
        if (!b->active) continue;

        float d;
        if (b->shape.type == SHAPE_SPHERE) {
            d = sdf_sphere(p, b->pos, b->shape.radius);
        } else {
            d = sdf_box(p, b->pos, b->shape.half_extents, b->orientation);
        }
        if (d < hit.dist) {
            hit.dist = d;
            hit.material_id = 1 + b->color_id;
        }
    }

    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING + SHADING
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 48
#define MAX_DIST 40.0f
#define SURF_DIST 0.003f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;

static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0, ro, 0};
    float t = 0;
    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        SceneHit h = scene_sdf(p);
        if (h.dist < SURF_DIST) {
            result.hit = 1;
            result.dist = t;
            result.point = p;
            result.material_id = h.material_id;
            return result;
        }
        t += h.dist;
    }
    result.dist = t;
    return result;
}

static vec3 calc_normal(vec3 p) {
    const float h = 0.001f;
    return v3_norm(v3(
        scene_dist(v3(p.x + h, p.y, p.z)) - scene_dist(v3(p.x - h, p.y, p.z)),
        scene_dist(v3(p.x, p.y + h, p.z)) - scene_dist(v3(p.x, p.y - h, p.z)),
        scene_dist(v3(p.x, p.y, p.z + h)) - scene_dist(v3(p.x, p.y, p.z - h))
    ));
}

static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10

static const uint8_t MATERIAL_COLORS[][3] = {
    {2, 3, 2},  /* ground */
    {5, 1, 1},  /* red */
    {1, 3, 5},  /* blue */
    {5, 4, 0},  /* orange */
    {5, 5, 0},  /* yellow */
    {5, 0, 5},  /* magenta */
    {0, 5, 5},  /* cyan */
    {2, 5, 2},  /* green */
    {5, 2, 3},  /* pink */
};

static uint8_t color_216(int r, int g, int b) {
    r = r < 0 ? 0 : (r > 5 ? 5 : r);
    g = g < 0 ? 0 : (g > 5 ? 5 : g);
    b = b < 0 ? 0 : (b > 5 ? 5 : b);
    return 16 + 36 * r + 6 * g + b;
}

typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;

static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 0, 16};
    vec3 light_dir = v3_norm(v3(0.5f, 1.0f, 0.3f));
    vec3 view_dir = v3_neg(rd);
    vec3 half_vec = v3_norm(v3_add(light_dir, view_dir));

    float ambient = 0.2f;
    float diffuse = fmaxf(0.0f, v3_dot(normal, light_dir)) * 0.6f;
    float specular = powf(fmaxf(0.0f, v3_dot(normal, half_vec)), 16.0f) * 0.3f;

    float fog = expf(-dist * 0.05f);
    float lum = (ambient + diffuse + specular) * fog;
    lum = fminf(1.0f, fmaxf(0.0f, lum));

    int ramp_idx = (int)(lum * (RAMP_LEN - 1) + 0.5f);
    result.ch = LUMINANCE_RAMP[ramp_idx];

    int num_mats = sizeof(MATERIAL_COLORS) / sizeof(MATERIAL_COLORS[0]);
    int mi = mat_id < num_mats ? mat_id : 0;
    float color_scale = 0.3f + lum * 0.7f;
    result.fg = color_216((int)(MATERIAL_COLORS[mi][0] * color_scale),
                          (int)(MATERIAL_COLORS[mi][1] * color_scale),
                          (int)(MATERIAL_COLORS[mi][2] * color_scale));
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * CAMERA + RENDER
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    vec3 target;
    float distance, azimuth, elevation, fov;
    float smooth_az, smooth_el, smooth_dist;
} OrbitCamera;

static vec3 orbit_camera_pos(OrbitCamera *cam) {
    float y = sinf(cam->smooth_el) * cam->smooth_dist;
    float xz_dist = cosf(cam->smooth_el) * cam->smooth_dist;
    float x = sinf(cam->smooth_az) * xz_dist;
    float z = cosf(cam->smooth_az) * xz_dist;
    return v3_add(cam->target, v3(x, y, z));
}

static void orbit_camera_get_ray(OrbitCamera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 pos = orbit_camera_pos(cam);
    vec3 forward = v3_norm(v3_sub(cam->target, pos));
    vec3 right = v3_norm(v3_cross(forward, v3(0, 1, 0)));
    vec3 up = v3_cross(right, forward);
    float half_fov = tanf(cam->fov * 0.5f);
    *ro = pos;
    *rd = v3_norm(v3_add(v3_add(forward, v3_scale(right, u * half_fov)), v3_scale(up, v * half_fov)));
}

static void orbit_camera_update(OrbitCamera *cam, float dt) {
    float smoothing = 1.0f - expf(-10.0f * dt);
    cam->smooth_az = lerpf(cam->smooth_az, cam->azimuth, smoothing);
    cam->smooth_el = lerpf(cam->smooth_el, cam->elevation, smoothing);
    cam->smooth_dist = lerpf(cam->smooth_dist, cam->distance, smoothing);
}

static void render_frame(OrbitCamera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            float u = (2.0f * x / term_w - 1.0f) * aspect;
            float v = -(2.0f * y / term_h - 1.0f);
            vec3 ro, rd;
            orbit_camera_get_ray(cam, u, v, &ro, &rd);
            RayResult hit = ray_march(ro, rd);
            Cell *cell = &screen[y * term_w + x];
            if (hit.hit) {
                vec3 normal = calc_normal(hit.point);
                ShadeResult shade = shade_point(hit.point, rd, normal, hit.material_id, hit.dist);
                cell->ch = shade.ch;
                cell->fg = shade.fg;
                cell->bg = shade.bg;
            } else {
                float sky_t = (v + 1.0f) * 0.5f;
                cell->ch = ' ';
                cell->fg = cell->bg = 17 + (int)(sky_t * 4);
            }
        }
    }
}

static void draw_hud(double fps, int paused) {
    char hud[256];
    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " Phase 5: Physics | FPS: %.0f | Bodies: %d %s| [Space] add [R] reset [P] pause [WASD] cam [Q] quit ",
        fps, num_bodies, paused ? "[PAUSED] " : "");
    write(STDOUT_FILENO, hud, strlen(hud));
}

/* ══════════════════════════════════════════════════════════════════
 * TIMING
 * ══════════════════════════════════════════════════════════════════ */

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_ms(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

/* ══════════════════════════════════════════════════════════════════
 * SCENE SETUP
 * ══════════════════════════════════════════════════════════════════ */

static void reset_scene(void) {
    num_bodies = 0;

    /* Initial spheres */
    body_init_sphere(&bodies[num_bodies++], v3(-2, 3, 0), 0.5f, 1.0f, 0);
    body_init_sphere(&bodies[num_bodies++], v3(0, 5, 0), 0.6f, 1.2f, 1);
    body_init_sphere(&bodies[num_bodies++], v3(2, 4, 0), 0.4f, 0.8f, 2);

    /* Initial box */
    body_init_box(&bodies[num_bodies++], v3(0, 2, 2), v3(0.5f, 0.5f, 0.5f), 2.0f, 3);
    bodies[num_bodies - 1].angular_vel = v3(1, 2, 0.5f);

    /* Give them some initial velocity */
    bodies[0].vel = v3(2, 0, 1);
    bodies[1].vel = v3(-1, 0, -1);
}

static void add_random_body(void) {
    if (num_bodies >= MAX_BODIES) return;

    float x = (rand() / (float)RAND_MAX - 0.5f) * 4;
    float z = (rand() / (float)RAND_MAX - 0.5f) * 4;
    int color = num_bodies % 8;

    if (rand() % 3 == 0) {
        body_init_box(&bodies[num_bodies++], v3(x, 6, z),
                      v3(0.3f + rand() % 3 * 0.1f, 0.3f + rand() % 3 * 0.1f, 0.3f + rand() % 3 * 0.1f),
                      1.5f, color);
        bodies[num_bodies - 1].angular_vel = v3(rand() % 3 - 1, rand() % 3 - 1, rand() % 3 - 1);
    } else {
        body_init_sphere(&bodies[num_bodies++], v3(x, 6, z),
                         0.3f + rand() % 4 * 0.1f, 1.0f, color);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    srand((unsigned)time(NULL));
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target = v3(0, 0, 0),
        .distance = 12.0f,
        .azimuth = 0.5f,
        .elevation = 0.4f,
        .fov = M_PI / 3.0f,
        .smooth_az = 0.5f,
        .smooth_el = 0.4f,
        .smooth_dist = 12.0f
    };

    reset_scene();

    double last_time = get_time();
    double fps_time = last_time;
    int fps_count = 0;
    double fps_display = 0;
    int running = 1;
    int paused = 0;

    const float physics_dt = 1.0f / 60.0f;
    float physics_accum = 0;

    while (running) {
        double now = get_time();
        float dt = (float)(now - last_time);
        last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0;
            get_term_size();
            free(screen);
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
                case 'w': case 'W': case KEY_UP:   cam.elevation += 0.1f; break;
                case 's': case 'S': case KEY_DOWN: cam.elevation -= 0.1f; break;
                case 'a': case 'A': case KEY_LEFT: cam.azimuth -= 0.1f; break;
                case 'd': case 'D': case KEY_RIGHT: cam.azimuth += 0.1f; break;
                case '+': case '=': cam.distance -= 0.5f; break;
                case '-': case '_': cam.distance += 0.5f; break;
            }
        }

        cam.elevation = fmaxf(-M_PI/2 + 0.1f, fminf(M_PI/2 - 0.1f, cam.elevation));
        cam.distance = fmaxf(5.0f, fminf(30.0f, cam.distance));

        /* Physics simulation */
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
            fps_count = 0;
            fps_time = now;
        }
        draw_hud(fps_display, paused);

        sleep_ms(30);
    }

    free(screen);
    free(out_buf);
    return 0;
}
