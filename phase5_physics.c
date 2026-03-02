/*
 * phase5_physics.c — Rigid Body Dynamics
 *
 * EXERCISE: Implement rigid body physics: inertia tensors, collision detection
 * (sphere-ground, box-ground, sphere-sphere), and impulse-based collision response.
 *
 * Build: gcc -O2 -o phase5_physics phase5_physics.c -lm
 * Run:   ./phase5_physics
 * Controls: [Space] add body, [R] reset, [P] pause, [WASD] camera, [Q] quit
 *
 * LEARNING GOALS:
 * - Understand rigid body state: position, velocity, orientation (quaternion), angular velocity
 * - Compute inertia tensors for sphere and box shapes
 * - Implement contact detection using geometric primitives
 * - Apply impulse-based collision response (includes rotational effects)
 * - Integrate equations of motion with semi-implicit Euler
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
 * TERMINAL (phase1 — already implemented)
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
 * RIGID BODY DATA STRUCTURES
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

#define MAX_BODIES 32
static RigidBody bodies[MAX_BODIES];
static int num_bodies = 0;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Initialize sphere rigid body
 *
 * Set position, zero vel/angular_vel, identity orientation.
 * For a solid sphere, the moment of inertia is:
 *   I = (2/5) * mass * radius^2
 * Store both inertia and its inverse (1/inertia).
 * If mass == 0 (static body), inv_mass = inv_inertia = 0.
 * ══════════════════════════════════════════════════════════════════ */
static void body_init_sphere(RigidBody *b, vec3 pos, float radius,
                              float mass, int color_id) {
    /* TODO: Initialize rigid body for a sphere
     * - Set position, zero velocity and angular velocity
     * - Set orientation to identity quaternion
     * - Compute inertia: I = 0.4 * mass * radius * radius (2/5 * m * r^2)
     * - Store inv_mass = 1/mass (or 0 if mass==0)
     * - Store inv_inertia = 1/inertia (or 0 if inertia==0)
     * - Set shape type, radius, color, active flag */
    (void)b; (void)pos; (void)radius; (void)mass; (void)color_id;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Initialize box rigid body
 *
 * For a solid box with dimensions w × h × d:
 *   I = mass * (w^2 + h^2 + d^2) / 12
 * where w=2*half_ext.x, h=2*half_ext.y, d=2*half_ext.z
 * ══════════════════════════════════════════════════════════════════ */
static void body_init_box(RigidBody *b, vec3 pos, vec3 half_ext,
                           float mass, int color_id) {
    /* TODO: Initialize rigid body for a box
     * - Same as sphere init but with box shape and box inertia formula */
    (void)b; (void)pos; (void)half_ext; (void)mass; (void)color_id;
}

/* ══════════════════════════════════════════════════════════════════
 * PHYSICS CONSTANTS
 * ══════════════════════════════════════════════════════════════════ */

static const vec3  GRAVITY         = {0, -9.81f, 0};
static const float RESTITUTION     = 0.6f;   /* bounciness 0..1 */
static const float FRICTION        = 0.3f;
static const float LINEAR_DAMPING  = 0.99f;
static const float ANGULAR_DAMPING = 0.98f;
static const float GROUND_Y        = -2.0f;

/* ══════════════════════════════════════════════════════════════════
 * COLLISION DETECTION
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    int   collided;
    vec3  point;      /* contact point in world space */
    vec3  normal;     /* contact normal pointing from surface into body */
    float penetration;
} Contact;

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Sphere vs ground plane collision
 *
 * The ground is at y = GROUND_Y.
 * A sphere at position b->pos with radius r:
 *   dist = b->pos.y - GROUND_Y - radius
 *   if dist < 0: collision!
 *     normal    = (0, 1, 0)          [pointing up into the sphere]
 *     penetration = -dist
 *     contact point = (b->pos.x, GROUND_Y, b->pos.z)
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_sphere_ground(RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect sphere-ground collision */
    (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Box vs ground plane collision
 *
 * Compute all 8 corners of the oriented box:
 *   for each (+/-1, +/-1, +/-1):
 *     local = (i * he.x, j * he.y, k * he.z)
 *     world = b->pos + q_rotate(b->orientation, local)
 *
 * Find the corner with the smallest y.
 * If min_y < GROUND_Y: collision!
 *   penetration = GROUND_Y - min_y
 *   normal = (0, 1, 0)
 *   contact point = that corner
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_box_ground(RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect box-ground collision using corner enumeration */
    (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Sphere vs sphere collision
 *
 * Two spheres a and b collide when:
 *   dist = length(b->pos - a->pos)
 *   min_dist = a->radius + b->radius
 *   if dist < min_dist (and dist > EPSILON to avoid NaN):
 *     normal = (b->pos - a->pos) / dist  [from a toward b]
 *     penetration = min_dist - dist
 *     contact point = a->pos + normal * a->radius
 * ══════════════════════════════════════════════════════════════════ */
static Contact collide_sphere_sphere(RigidBody *a, RigidBody *b) {
    Contact c = {0};
    /* TODO: Detect sphere-sphere collision */
    (void)a; (void)b;
    return c;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Impulse-based collision response
 *
 * Given a contact with body and optionally 'other' (NULL for static ground):
 *
 * 1. POSITION CORRECTION (separate overlapping bodies):
 *    correction = penetration * 0.8
 *    body->pos += normal * correction * inv_mass_body / total_inv_mass
 *    (if other, move other in opposite direction proportionally)
 *
 * 2. RELATIVE VELOCITY at contact point:
 *    r = contact_point - body->pos
 *    vel_at_contact = body->vel + cross(body->angular_vel, r)
 *    (subtract other's velocity if other exists)
 *
 * 3. NORMAL COMPONENT:
 *    vn = dot(vel_at_contact, normal)
 *    if vn > 0: bodies are separating, skip
 *
 * 4. IMPULSE MAGNITUDE:
 *    j = -(1 + RESTITUTION) * vn
 *        / (inv_mass_sum + rotational_terms)
 *    rotational_term = dot(cross(r, n), cross(r, n) * inv_inertia)
 *
 * 5. APPLY LINEAR IMPULSE:
 *    body->vel += normal * j * inv_mass
 *    other->vel -= normal * j * other->inv_mass  (if other)
 *
 * 6. APPLY ANGULAR IMPULSE:
 *    body->angular_vel += cross(r, impulse) * inv_inertia
 *
 * 7. FRICTION IMPULSE (optional but improves realism):
 *    tangent = vel_at_contact - normal * vn
 *    jt = -dot(vel_at_contact, tangent) / inv_mass_sum
 *    clamp jt to [-FRICTION*j, FRICTION*j]
 *    apply tangential impulse similarly
 * ══════════════════════════════════════════════════════════════════ */
static void resolve_collision(RigidBody *body, Contact *c, RigidBody *other) {
    if (!c->collided) return;
    /* TODO: Implement impulse-based collision response */
    (void)body; (void)c; (void)other;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #7: Physics integration step
 *
 * Semi-implicit Euler integration for each active body:
 *   1. Apply gravity:      vel += GRAVITY * dt
 *   2. Apply damping:      vel *= LINEAR_DAMPING
 *                          angular_vel *= ANGULAR_DAMPING
 *   3. Update position:    pos += vel * dt
 *   4. Update orientation: convert angular_vel to angle-axis,
 *                          create delta quaternion, multiply into orientation,
 *                          re-normalize
 *
 * Then detect and resolve all collisions:
 *   - Each body vs ground
 *   - Each pair of spheres
 * ══════════════════════════════════════════════════════════════════ */
static void physics_step(float dt) {
    /* TODO: Integrate forces and velocities, then resolve collisions */
    (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * SDF RENDERING (from phases 3-4 — already implemented)
 * Uses dynamic rigid bodies for scene_sdf
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
    vec3 half_vec = v3_norm(v3_add(light_dir, v3_neg(rd)));
    float ambient = 0.2f;
    float diffuse = fmaxf(0.0f, v3_dot(normal, light_dir)) * 0.6f;
    float specular = powf(fmaxf(0.0f, v3_dot(normal, half_vec)), 16.0f) * 0.3f;
    float fog = expf(-dist * 0.05f);
    float lum = fminf(1.0f, fmaxf(0.0f, (ambient+diffuse+specular)*fog));
    result.ch = LUMINANCE_RAMP[(int)(lum*(RAMP_LEN-1)+0.5f)];
    int num_mats = sizeof(MATERIAL_COLORS)/sizeof(MATERIAL_COLORS[0]);
    int mi = mat_id < num_mats ? mat_id : 0;
    float cs = 0.3f + lum * 0.7f;
    result.fg = color_216((int)(MATERIAL_COLORS[mi][0]*cs),(int)(MATERIAL_COLORS[mi][1]*cs),(int)(MATERIAL_COLORS[mi][2]*cs));
    return result;
}

typedef struct {
    vec3 target; float distance, azimuth, elevation, fov;
    float smooth_az, smooth_el, smooth_dist;
} OrbitCamera;
static vec3 orbit_camera_pos(OrbitCamera *cam) {
    float y = sinf(cam->smooth_el)*cam->smooth_dist;
    float xzd = cosf(cam->smooth_el)*cam->smooth_dist;
    return v3_add(cam->target, v3(sinf(cam->smooth_az)*xzd, y, cosf(cam->smooth_az)*xzd));
}
static void orbit_camera_get_ray(OrbitCamera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 pos = orbit_camera_pos(cam);
    vec3 fwd = v3_norm(v3_sub(cam->target, pos));
    vec3 right = v3_norm(v3_cross(fwd, v3(0,1,0)));
    vec3 up = v3_cross(right, fwd);
    float hf = tanf(cam->fov*0.5f);
    *ro = pos; *rd = v3_norm(v3_add(v3_add(fwd, v3_scale(right, u*hf)), v3_scale(up, v*hf)));
}
static void orbit_camera_update(OrbitCamera *cam, float dt) {
    float s = 1.0f - expf(-10.0f * dt);
    cam->smooth_az = lerpf(cam->smooth_az, cam->azimuth, s);
    cam->smooth_el = lerpf(cam->smooth_el, cam->elevation, s);
    cam->smooth_dist = lerpf(cam->smooth_dist, cam->distance, s);
}
static void render_frame(OrbitCamera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            float u = (2.0f*x/term_w - 1.0f)*aspect;
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
        " Phase 5: Physics | FPS: %.0f | Bodies: %d %s"
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
    body_init_sphere(&bodies[num_bodies++], v3(-2,3,0),  0.5f, 1.0f, 0);
    body_init_sphere(&bodies[num_bodies++], v3(0,5,0),   0.6f, 1.2f, 1);
    body_init_sphere(&bodies[num_bodies++], v3(2,4,0),   0.4f, 0.8f, 2);
    body_init_box(&bodies[num_bodies++],    v3(0,2,2),   v3(0.5f,0.5f,0.5f), 2.0f, 3);
    bodies[num_bodies-1].angular_vel = v3(1,2,0.5f);
    bodies[0].vel = v3(2,0,1);
    bodies[1].vel = v3(-1,0,-1);
}

static void add_random_body(void) {
    if (num_bodies >= MAX_BODIES) return;
    float x = ((float)rand()/RAND_MAX - 0.5f) * 4;
    float z = ((float)rand()/RAND_MAX - 0.5f) * 4;
    int color = num_bodies % 8;
    if (rand() % 3 == 0) {
        body_init_box(&bodies[num_bodies++], v3(x,6,z),
                      v3(0.3f + rand()%3*0.1f, 0.3f + rand()%3*0.1f, 0.3f + rand()%3*0.1f),
                      1.5f, color);
        bodies[num_bodies-1].angular_vel = v3(rand()%3-1, rand()%3-1, rand()%3-1);
    } else {
        body_init_sphere(&bodies[num_bodies++], v3(x,6,z), 0.3f + rand()%4*0.1f, 1.0f, color);
    }
}

int main(void) {
    srand((unsigned)time(NULL));
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target = v3(0,0,0), .distance=12.0f, .azimuth=0.5f, .elevation=0.4f,
        .fov=(float)M_PI/3.0f, .smooth_az=0.5f, .smooth_el=0.4f, .smooth_dist=12.0f
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
        cam.distance = fmaxf(5.0f, fminf(30.0f, cam.distance));

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
