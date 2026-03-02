/*
 * phase9_robot_sdf.c — Quadruped Robot SDF Renderer
 *
 * EXERCISE: Use signed distance fields and forward kinematics to render
 * an animated Unitree GO2-style quadruped robot in the terminal.
 *
 * Build: gcc -O2 -o phase9_robot_sdf phase9_robot_sdf.c -lm
 * Run:   ./phase9_robot_sdf
 * Controls: ← → = yaw,  ↑ ↓ = pitch,  +/- = zoom,  Q = quit
 *
 * LEARNING GOALS:
 * - Implement the capsule SDF: the most important primitive for skeletal bodies
 * - Implement the rounded-box SDF: extends the box formula with a rounding term
 * - Compute forward kinematics: propagate joint angles to joint world positions
 * - Drive joints with a trot gait: two diagonal pairs alternating at 180°
 * - Assemble a complex SDF scene from many primitive SDFs combined with min()
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
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * 3D MATH  (provided — same as previous phases)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;

static inline vec3  v3(float x, float y, float z) { return (vec3){x,y,z}; }
static inline vec3  v3_add(vec3 a, vec3 b)  { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3  v3_sub(vec3 a, vec3 b)  { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3  v3_scale(vec3 v, float s){ return v3(v.x*s, v.y*s, v.z*s); }
static inline float v3_dot(vec3 a, vec3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3  v3_cross(vec3 a, vec3 b){
    return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float v3_len(vec3 v)  { return sqrtf(v3_dot(v,v)); }
static inline vec3  v3_norm(vec3 v) {
    float l = v3_len(v); return l < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f/l);
}
static inline vec3  v3_abs(vec3 v)  { return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
static inline vec3  v3_max(vec3 a, vec3 b){
    return v3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
static inline float v3_max_comp(vec3 v){ return fmaxf(v.x, fmaxf(v.y, v.z)); }

/* ══════════════════════════════════════════════════════════════════
 * TERMINAL FRAMEWORK  (provided)
 * ══════════════════════════════════════════════════════════════════ */

static int  term_w = 80, term_h = 24;
static struct termios orig_termios;
static int  raw_mode_on = 0;

typedef struct { char ch; uint8_t fg; uint8_t bg; } Cell;
static Cell *screen  = NULL;
static char *out_buf = NULL;
static int   out_cap = 0, out_len = 0;

static void out_flush(void) {
    if (out_len > 0) { write(STDOUT_FILENO, out_buf, out_len); out_len = 0; }
}
static void out_append(const char *s, int n) {
    while (out_len + n > out_cap) {
        out_cap = out_cap ? out_cap*2 : 65536;
        out_buf = realloc(out_buf, out_cap);
    }
    memcpy(out_buf + out_len, s, n);
    out_len += n;
}
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        term_w = ws.ws_col; term_h = ws.ws_row;
    }
}
static void disable_raw(void) {
    if (raw_mode_on) {
        write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_on = 0;
    }
}
static void enable_raw(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_mode_on = 1;
    atexit(disable_raw);
    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT|ICRNL|INPCK|ISTRIP|IXON);
    raw.c_oflag &= ~OPOST;
    raw.c_cflag |= CS8;
    raw.c_lflag &= ~(ECHO|ICANON|IEXTEN|ISIG);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
}
static void present(void) {
    char seq[64]; int pf = -1, pb = -1;
    out_append("\033[H", 3);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            Cell *c = &screen[y*term_w + x];
            if (c->fg != pf || c->bg != pb) {
                int n = snprintf(seq, sizeof seq,
                    "\033[38;5;%dm\033[48;5;%dm", c->fg, c->bg);
                out_append(seq, n); pf = c->fg; pb = c->bg;
            }
            out_append(&c->ch, 1);
        }
    }
    out_flush();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement sdf_capsule — the key primitive for robot limbs
 *
 * A capsule is a cylinder with hemispherical end-caps. It is the SDF
 * of the segment between points a and b, inflated by radius r.
 *
 * Algorithm:
 *   pa = p - a             (vector from a to p)
 *   ba = b - a             (segment direction)
 *   h  = clamp(dot(pa,ba) / dot(ba,ba), 0, 1)   (closest point param)
 *   return length(pa - ba*h) - r
 *
 * Hint: use v3_dot, v3_sub, v3_scale, v3_len, fminf, fmaxf
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    /* TODO #1: Implement capsule SDF */
    (void)p; (void)a; (void)b; (void)r;
    return 1e9f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement sdf_rounded_box — a box with rounded edges
 *
 * A rounded box with half-extents 'half' and rounding radius r.
 * p is already relative to the box centre.
 *
 * Algorithm (same as sdf_box but subtract r at the end):
 *   q = abs(p) - half                    (componentwise)
 *   d = length(max(q, 0))               (outside distance to corner)
 *     + min(max_component(q), 0)        (negative when inside)
 *     - r                               (rounding: shrink box by r)
 *
 * Hint: use v3_abs, v3_sub, v3_max, v3_max_comp, v3_len
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_rounded_box(vec3 p, vec3 half, float r) {
    /* TODO #2: Implement rounded-box SDF */
    (void)p; (void)half; (void)r;
    return 1e9f;
}

/* Sphere and plane provided for reference */
static float sdf_sphere(vec3 p, vec3 c, float r) { return v3_len(v3_sub(p,c)) - r; }
static float sdf_plane(vec3 p, float h) { return p.y - h; }

static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a-b), 0.0f) / k;
    return fminf(a, b) - h*h*k*0.25f;
}

/* ══════════════════════════════════════════════════════════════════
 * UNITREE GO2 ROBOT DIMENSIONS  (all in metres)
 * ══════════════════════════════════════════════════════════════════ */

#define BODY_HALF_LEN  0.250f
#define BODY_HALF_H    0.060f
#define BODY_HALF_W    0.110f
#define BODY_HEIGHT    0.320f
#define BODY_ROUND     0.028f

#define HIP_X_OFF      0.200f
#define HIP_Z_OFF      0.110f

#define THIGH_LEN      0.213f
#define SHIN_LEN       0.213f
#define HIP_R          0.033f
#define THIGH_R        0.022f
#define SHIN_R         0.018f
#define FOOT_R         0.028f

#define HIP_NEUTRAL    0.40f
#define KNEE_STAND    -1.40f

#define GAIT_FREQ      1.5f
#define STRIDE_AMP     0.30f
#define KNEE_LIFT      0.45f

/* Material IDs */
#define MAT_GROUND  0
#define MAT_BODY    1
#define MAT_SENSOR  2
#define MAT_HIP     3
#define MAT_THIGH   4
#define MAT_KNEE    5
#define MAT_SHIN    6
#define MAT_FOOT    7

static const uint8_t MAT_COLOR[][3] = {
    {2, 2, 1}, {4, 3, 2}, {0, 3, 5},
    {3, 3, 3}, {3, 3, 3}, {4, 4, 4}, {2, 2, 2}, {1, 1, 1},
};

/* Joint angle state per leg:  0=FL  1=FR  2=RL  3=RR */
static float g_hip[4];
static float g_knee[4];

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement leg_fk — forward kinematics for one leg
 *
 * Leg index convention:
 *   0 = FL (front-left)   hip socket at (+HIP_X_OFF, BODY_HEIGHT, +HIP_Z_OFF)
 *   1 = FR (front-right)  hip socket at (+HIP_X_OFF, BODY_HEIGHT, -HIP_Z_OFF)
 *   2 = RL (rear-left)    hip socket at (-HIP_X_OFF, BODY_HEIGHT, +HIP_Z_OFF)
 *   3 = RR (rear-right)   hip socket at (-HIP_X_OFF, BODY_HEIGHT, -HIP_Z_OFF)
 *
 * FK formula (sagittal plane — motion is only in X and Y):
 *
 *   thigh_dir = ( sin(g_hip[i]),              -cos(g_hip[i]),             0 )
 *   *knee_p   = *hip_j + THIGH_LEN * thigh_dir
 *
 *   shin_angle = g_hip[i] + g_knee[i]
 *   shin_dir   = ( sin(shin_angle),           -cos(shin_angle),           0 )
 *   *foot_p    = *knee_p + SHIN_LEN * shin_dir
 *
 * Steps:
 *   1. Compute hip socket world position from index i (use HIP_X_OFF, HIP_Z_OFF)
 *   2. Compute thigh direction from g_hip[i]
 *   3. Compute knee position = hip + THIGH_LEN * thigh_dir
 *   4. Compute shin direction from (g_hip[i] + g_knee[i])
 *   5. Compute foot position = knee + SHIN_LEN * shin_dir
 * ══════════════════════════════════════════════════════════════════ */
static void leg_fk(int i, vec3 *hip_j, vec3 *knee_p, vec3 *foot_p) {
    /* TODO #3: Compute hip socket, knee, and foot world positions */
    (void)i;
    *hip_j  = v3(0, BODY_HEIGHT, 0);   /* placeholder */
    *knee_p = v3(0, 0.1f, 0);          /* placeholder */
    *foot_p = v3(0, 0.0f, 0);          /* placeholder */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement update_gait — drive joints with a trot gait
 *
 * Trot gait: two diagonal pairs alternate 180° out of phase.
 *   Pair A (FL=0, RR=3): phase = fmod(t * GAIT_FREQ, 1.0)
 *   Pair B (FR=1, RL=2): phase = fmod(t * GAIT_FREQ + 0.5, 1.0)
 *
 * Each leg's gait cycle (phase p in [0, 1)):
 *
 *   Swing phase  p in [0.0, 0.5):  foot in the air, sweeping forward
 *     s = p / 0.5                               (swing progress 0→1)
 *     g_hip[i]  = HIP_NEUTRAL + STRIDE_AMP * (2*s - 1)   (back→front)
 *     g_knee[i] = KNEE_STAND  - KNEE_LIFT * sin(PI * s)  (peak lift at mid)
 *
 *   Stance phase p in [0.5, 1.0):  foot on ground, pushing back
 *     s = (p - 0.5) / 0.5                       (stance progress 0→1)
 *     g_hip[i]  = HIP_NEUTRAL + STRIDE_AMP * (1 - 2*s)   (front→back)
 *     g_knee[i] = KNEE_STAND                              (no knee lift)
 *
 * Steps:
 *   1. Compute gait phase for each of the 4 legs
 *   2. For each leg, check swing or stance, compute g_hip[i] and g_knee[i]
 * ══════════════════════════════════════════════════════════════════ */
static void update_gait(float t) {
    /* TODO #4: Fill g_hip[0..3] and g_knee[0..3] from gait phase */
    for (int i = 0; i < 4; i++) {
        g_hip[i]  = HIP_NEUTRAL;
        g_knee[i] = KNEE_STAND;
    }
    (void)t;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement scene_sdf — assemble the robot from SDF primitives
 *
 * The scene contains:
 *   1. Ground plane (y = 0)  → mat = MAT_GROUND
 *   2. Body (rounded box centred at (0, BODY_HEIGHT, 0))  → MAT_BODY
 *   3. Front sensor (sphere on nose at (BODY_HALF_LEN, BODY_HEIGHT, 0)) → MAT_SENSOR
 *      Blend sensor into body with sdf_smooth_union(db, ds, 0.04f)
 *   4. For each of the 4 legs (call leg_fk to get joint positions):
 *      a. Hip sphere    → MAT_HIP
 *      b. Thigh capsule (hip → knee)   → MAT_THIGH
 *      c. Knee sphere (radius THIGH_R * 1.25)  → MAT_KNEE
 *      d. Shin capsule (knee → foot)   → MAT_SHIN
 *      e. Foot sphere   → MAT_FOOT
 *
 * Pattern: for each shape compute distance d, then:
 *   if (d < h.d) { h.d = d; h.mat = MAT_XXX; }
 *
 * Hint: the body rounded box needs a centred point: bp = p - body_centre
 * ══════════════════════════════════════════════════════════════════ */
typedef struct { float d; int mat; } Hit;

static Hit scene_sdf(vec3 p) {
    Hit h = {1e9f, MAT_GROUND};

    /* TODO #5: Build the full robot scene */
    /* Hint: start with ground, then body, then sensor, then 4 legs */

    /* Placeholder — just a ground plane so it compiles */
    float dg = sdf_plane(p, 0.0f);
    if (dg < h.d) { h.d = dg; h.mat = MAT_GROUND; }

    return h;
}

static float scene_dist(vec3 p) { return scene_sdf(p).d; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING  (provided)
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 120
#define MAX_DIST  12.0f
#define SURF_DIST 0.0008f

typedef struct { int hit; float dist; vec3 point; int mat; } RayHit;

static RayHit ray_march(vec3 ro, vec3 rd) {
    RayHit r = {0, 0.0f, ro, 0};
    float t = 0.0f;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        Hit h = scene_sdf(p);
        if (h.d < SURF_DIST) { r.hit=1; r.dist=t; r.point=p; r.mat=h.mat; return r; }
        t += h.d;
        if (t > MAX_DIST) break;
    }
    return r;
}

static vec3 calc_normal(vec3 p) {
    float e = 0.001f;
    return v3_norm(v3(
        scene_dist(v3_add(p,v3(e,0,0))) - scene_dist(v3_sub(p,v3(e,0,0))),
        scene_dist(v3_add(p,v3(0,e,0))) - scene_dist(v3_sub(p,v3(0,e,0))),
        scene_dist(v3_add(p,v3(0,0,e))) - scene_dist(v3_sub(p,v3(0,0,e)))
    ));
}

static float soft_shadow(vec3 ro, vec3 rd, float maxt) {
    float res = 1.0f, t = 0.02f;
    for (int i = 0; i < 32 && t < maxt; i++) {
        float h = scene_dist(v3_add(ro, v3_scale(rd, t)));
        if (h < 0.001f) return 0.0f;
        res = fminf(res, 8.0f * h / t);
        t += h;
    }
    return res;
}

/* ══════════════════════════════════════════════════════════════════
 * SHADING  (provided)
 * ══════════════════════════════════════════════════════════════════ */

static const char RAMP[] = " .,:;=+*%#@";
#define RAMP_LEN 11

static void shade(RayHit ray, vec3 rd, Cell *out) {
    if (!ray.hit) { out->ch=' '; out->bg=17; out->fg=17; return; }

    vec3 n   = calc_normal(ray.point);
    vec3 lp  = v3_norm(v3(-0.5f, 1.8f, -0.8f));
    vec3 lp2 = v3_norm(v3( 0.8f, 0.5f,  0.6f));
    vec3 vd  = v3_scale(rd, -1.0f);
    float shad = soft_shadow(ray.point, lp, 4.0f);
    float amb  = 0.18f;
    float diff = fmaxf(v3_dot(n, lp),  0.0f) * 0.65f * shad;
    float fill = fmaxf(v3_dot(n, lp2), 0.0f) * 0.15f;
    vec3  hv   = v3_norm(v3_add(lp, vd));
    float spec = powf(fmaxf(v3_dot(n, hv), 0.0f), 24.0f) * 0.35f * shad;
    float lum  = fminf(amb + diff + fill + spec, 1.0f);

    int ci = (int)(lum * (RAMP_LEN-1) + 0.5f);
    if (ci < 0) ci = 0;
    if (ci >= RAMP_LEN) ci = RAMP_LEN-1;
    out->ch = RAMP[ci];

    const uint8_t *mc = MAT_COLOR[ray.mat];
    int lr = (int)(mc[0]*lum+0.5f), lg = (int)(mc[1]*lum+0.5f), lb = (int)(mc[2]*lum+0.5f);
    lr = lr<0?0:lr>5?5:lr; lg = lg<0?0:lg>5?5:lg; lb = lb<0?0:lb>5?5:lb;
    out->fg = (uint8_t)(16 + 36*lr + 6*lg + lb);
    out->bg = 16;
}

/* ══════════════════════════════════════════════════════════════════
 * CAMERA & MAIN LOOP  (provided)
 * ══════════════════════════════════════════════════════════════════ */

static float g_yaw   = 0.8f;
static float g_pitch = 0.35f;
static float g_dist  = 2.2f;
static vec3  g_target = {0.0f, 0.15f, 0.0f};

static void render_frame(void) {
    float cy  = cosf(g_pitch), sy  = sinf(g_pitch);
    float cyw = cosf(g_yaw),   syw = sinf(g_yaw);
    vec3 cam = v3(g_target.x + g_dist*cy*syw,
                  g_target.y + g_dist*sy,
                  g_target.z + g_dist*cy*cyw);
    vec3 fwd   = v3_norm(v3_sub(g_target, cam));
    vec3 right = v3_norm(v3_cross(fwd, v3(0,1,0)));
    vec3 up    = v3_cross(right, fwd);
    float fov    = tanf((float)M_PI / 6.0f);
    float aspect = ((float)term_w / (float)term_h) * 0.5f;

    for (int row = 0; row < term_h; row++)
        for (int col = 0; col < term_w; col++) {
            float u =  (2.0f*(col+0.5f)/term_w  - 1.0f) * fov * aspect;
            float v = -(2.0f*(row+0.5f)/term_h  - 1.0f) * fov;
            vec3 rd = v3_norm(v3_add(fwd, v3_add(v3_scale(right,u), v3_scale(up,v))));
            RayHit hit = ray_march(cam, rd);
            shade(hit, rd, &screen[row*term_w + col]);
        }
}

static void draw_hud(float t) {
    char buf[128];
    int n = snprintf(buf, sizeof buf,
        " GO2 robot | t=%.1fs | yaw=%.2f pitch=%.2f dist=%.2f | Q=quit ",
        t, g_yaw, g_pitch, g_dist);
    for (int i = 0; i < n && i < term_w; i++) {
        screen[i].ch = buf[i]; screen[i].fg = 231; screen[i].bg = 234;
    }
}

static int handle_input(void) {
    char buf[16];
    int n = (int)read(STDIN_FILENO, buf, sizeof buf);
    if (n <= 0) return 1;
    for (int i = 0; i < n; i++) {
        char c = buf[i];
        if (c == 'q' || c == 'Q' || c == 3) return 0;
        if (c == '\033' && i+2 < n && buf[i+1] == '[') {
            char code = buf[i+2]; i += 2;
            if (code == 'A') g_pitch += 0.08f;
            if (code == 'B') g_pitch -= 0.08f;
            if (code == 'C') g_yaw   -= 0.08f;
            if (code == 'D') g_yaw   += 0.08f;
        }
        if (c == '+' || c == '=') g_dist = fmaxf(0.5f, g_dist - 0.15f);
        if (c == '-')             g_dist = fminf(6.0f, g_dist + 0.15f);
    }
    if (g_pitch >  1.4f) g_pitch =  1.4f;
    if (g_pitch < -0.3f) g_pitch = -0.3f;
    return 1;
}

int main(void) {
    enable_raw();
    get_term_size();
    screen  = calloc(term_w * term_h, sizeof *screen);
    out_buf = malloc(65536);
    out_cap = 65536;

    for (int i = 0; i < 4; i++) { g_hip[i] = HIP_NEUTRAL; g_knee[i] = KNEE_STAND; }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (1) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        float t = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec)*1e-9f;

        if (!handle_input()) break;
        get_term_size();
        screen = realloc(screen, term_w * term_h * sizeof *screen);

        update_gait(t);
        render_frame();
        draw_hud(t);
        present();

        struct timespec sl = {0, 50000000L};
        nanosleep(&sl, NULL);
    }

    free(screen); free(out_buf);
    return 0;
}
