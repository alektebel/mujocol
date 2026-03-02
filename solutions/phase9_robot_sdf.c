/*
 * solutions/phase9_robot_sdf.c — Unitree GO2-style Quadruped Robot SDF Renderer
 *
 * Renders an animated GO2-style legged robot using signed distance fields:
 *   - Rounded-box torso, sphere front sensor
 *   - 4 legs: hip sphere → thigh capsule → knee sphere → shin capsule → foot sphere
 *   - Trot gait via forward kinematics (FL+RR vs FR+RL diagonal pairs)
 *   - Phong shading + soft shadows + xterm-256 colour
 *
 * Controls:  ← → = yaw,  ↑ ↓ = pitch,  +/- = zoom,  Q = quit
 * Build: gcc -O2 -o phase9_robot_sdf solutions/phase9_robot_sdf.c -lm
 * Run:   ./phase9_robot_sdf
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
 * 3D MATH
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
 * TERMINAL FRAMEWORK
 * ══════════════════════════════════════════════════════════════════ */

static int  term_w = 80, term_h = 24;
static struct termios orig_termios;
static int  raw_mode_on = 0;

typedef struct { char ch; uint8_t fg; uint8_t bg; } Cell;
static Cell *screen   = NULL;
static char *out_buf  = NULL;
static int   out_cap  = 0, out_len = 0;

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
 * SDF PRIMITIVES
 * ══════════════════════════════════════════════════════════════════ */

/* Sphere centered at c with radius r */
static float sdf_sphere(vec3 p, vec3 c, float r) {
    return v3_len(v3_sub(p, c)) - r;
}

/* Infinite ground plane at y = h */
static float sdf_plane(vec3 p, float h) { return p.y - h; }

/* Capsule: segment from a to b, radius r
 * Find closest point on segment ab to p, then measure distance - r
 */
static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = v3_sub(p, a);
    vec3 ba = v3_sub(b, a);
    float h = fminf(fmaxf(v3_dot(pa, ba) / v3_dot(ba, ba), 0.0f), 1.0f);
    return v3_len(v3_sub(pa, v3_scale(ba, h))) - r;
}

/* Rounded box centred at origin with half-extents 'half' and edge radius r
 * Formula: box SDF (point→nearest surface) minus the rounding radius
 */
static float sdf_rounded_box(vec3 p, vec3 half, float r) {
    vec3 q = v3_sub(v3_abs(p), half);
    return v3_len(v3_max(q, v3(0,0,0))) + fminf(v3_max_comp(q), 0.0f) - r;
}

/* SDF smooth union — blends two shapes over distance k */
static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a-b), 0.0f) / k;
    return fminf(a, b) - h*h*k*0.25f;
}

/* ══════════════════════════════════════════════════════════════════
 * UNITREE GO2 ROBOT DIMENSIONS  (all in metres)
 * ══════════════════════════════════════════════════════════════════ */

#define BODY_HALF_LEN  0.250f   /* body half-length  (X) */
#define BODY_HALF_H    0.060f   /* body half-height  (Y) */
#define BODY_HALF_W    0.110f   /* body half-width   (Z) */
#define BODY_HEIGHT    0.320f   /* body centre above ground */
#define BODY_ROUND     0.028f   /* edge rounding radius */

#define HIP_X_OFF      0.200f   /* front/rear hip X offset from body centre */
#define HIP_Z_OFF      0.110f   /* left/right hip Z offset */

#define THIGH_LEN      0.213f   /* thigh segment length */
#define SHIN_LEN       0.213f   /* shin segment length */
#define HIP_R          0.033f   /* hip sphere radius */
#define THIGH_R        0.022f   /* thigh capsule radius */
#define SHIN_R         0.018f   /* shin capsule radius */
#define FOOT_R         0.028f   /* foot sphere radius */

/* Standing joint angles (radians) */
#define HIP_NEUTRAL    0.40f    /* hip pitch at rest */
#define KNEE_STAND    -1.40f    /* knee angle at rest (negative = folded) */

/* Trot gait parameters */
#define GAIT_FREQ      1.5f     /* trot cycles per second */
#define STRIDE_AMP     0.30f    /* hip pitch amplitude (rad) */
#define KNEE_LIFT      0.45f    /* extra knee bend during swing */

/* ── Material IDs ───────────────────────────────────────────────── */
#define MAT_GROUND  0
#define MAT_BODY    1
#define MAT_SENSOR  2
#define MAT_HIP     3
#define MAT_THIGH   4
#define MAT_KNEE    5
#define MAT_SHIN    6
#define MAT_FOOT    7

/* Material base colours: (R,G,B) in [0,5] → xterm256 = 16+36r+6g+b */
static const uint8_t MAT_COLOR[][3] = {
    {2, 2, 1},   /* 0: ground  — dark olive */
    {4, 3, 2},   /* 1: body    — warm beige (GO2 livery) */
    {0, 3, 5},   /* 2: sensor  — cyan  (LiDAR dome) */
    {3, 3, 3},   /* 3: hip     — mid grey */
    {3, 3, 3},   /* 4: thigh   — mid grey */
    {4, 4, 4},   /* 5: knee    — light grey joint */
    {2, 2, 2},   /* 6: shin    — dark grey */
    {1, 1, 1},   /* 7: foot    — near-black rubber */
};

/* ══════════════════════════════════════════════════════════════════
 * ROBOT STATE  (joint angles, updated each frame)
 * ══════════════════════════════════════════════════════════════════ */

/* Joint angles per leg:  0=FL  1=FR  2=RL  3=RR */
static float g_hip[4];    /* hip pitch  (sagittal plane) */
static float g_knee[4];   /* knee pitch (relative to thigh) */

/*
 * leg_fk — forward kinematics for one leg
 *
 * Leg index convention:
 *   0 = FL (front-left)   hip socket at (+HIP_X_OFF, BODY_HEIGHT, +HIP_Z_OFF)
 *   1 = FR (front-right)  hip socket at (+HIP_X_OFF, BODY_HEIGHT, -HIP_Z_OFF)
 *   2 = RL (rear-left)    hip socket at (-HIP_X_OFF, BODY_HEIGHT, +HIP_Z_OFF)
 *   3 = RR (rear-right)   hip socket at (-HIP_X_OFF, BODY_HEIGHT, -HIP_Z_OFF)
 *
 * All motion is in the sagittal (XY) plane of that leg.
 * Positive hip angle tilts the thigh forward (+X).
 * Negative knee angle folds the shin backward (towards -X).
 *
 *   thigh_dir = ( sin(hip),          -cos(hip),          0 )
 *   knee_pos  = hip_socket + THIGH_LEN * thigh_dir
 *   shin_dir  = ( sin(hip+knee),     -cos(hip+knee),     0 )
 *   foot_pos  = knee_pos  + SHIN_LEN  * shin_dir
 */
static void leg_fk(int i, vec3 *hip_j, vec3 *knee_p, vec3 *foot_p) {
    float hx = (i < 2) ? HIP_X_OFF : -HIP_X_OFF;
    float hz = (i % 2 == 0) ? HIP_Z_OFF : -HIP_Z_OFF;
    *hip_j = v3(hx, BODY_HEIGHT, hz);

    float ha = g_hip[i];
    vec3 thigh_dir = v3(sinf(ha), -cosf(ha), 0.0f);
    *knee_p = v3_add(*hip_j, v3_scale(thigh_dir, THIGH_LEN));

    float ka = ha + g_knee[i];
    vec3 shin_dir = v3(sinf(ka), -cosf(ka), 0.0f);
    *foot_p = v3_add(*knee_p, v3_scale(shin_dir, SHIN_LEN));
}

/*
 * update_gait — compute joint angles from trot gait phase
 *
 * Trot gait: two diagonal pairs alternate.
 *   Pair A (FL + RR): phase  = fmod(t * GAIT_FREQ, 1)
 *   Pair B (FR + RL): phase  = fmod(t * GAIT_FREQ + 0.5, 1)
 *
 * Each leg's gait cycle (phase in [0, 1)):
 *   Swing  [0.0, 0.5):  foot in the air, sweeping forward
 *     s = phase / 0.5                          (progress 0→1)
 *     hip   = HIP_NEUTRAL + STRIDE_AMP*(2s-1)  (back → front)
 *     knee  = KNEE_STAND - KNEE_LIFT*sin(π*s)  (peak lift at mid-swing)
 *
 *   Stance [0.5, 1.0):  foot on the ground, pushing backward
 *     s = (phase - 0.5) / 0.5                  (progress 0→1)
 *     hip   = HIP_NEUTRAL + STRIDE_AMP*(1-2s)  (front → back)
 *     knee  = KNEE_STAND                        (no lift)
 */
static void update_gait(float t) {
    float ph[4];
    ph[0] = fmodf(t * GAIT_FREQ,        1.0f);  /* FL */
    ph[1] = fmodf(t * GAIT_FREQ + 0.5f, 1.0f);  /* FR */
    ph[2] = ph[1];                               /* RL same as FR */
    ph[3] = ph[0];                               /* RR same as FL */

    for (int i = 0; i < 4; i++) {
        float p = ph[i];
        if (p < 0.5f) {
            float s = p / 0.5f;
            g_hip[i]  = HIP_NEUTRAL + STRIDE_AMP * (2.0f*s - 1.0f);
            g_knee[i] = KNEE_STAND  - KNEE_LIFT  * sinf((float)M_PI * s);
        } else {
            float s = (p - 0.5f) / 0.5f;
            g_hip[i]  = HIP_NEUTRAL + STRIDE_AMP * (1.0f - 2.0f*s);
            g_knee[i] = KNEE_STAND;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SCENE SDF
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float d; int mat; } Hit;

static Hit scene_sdf(vec3 p) {
    Hit h = {1e9f, 0};

    /* ── Ground plane ──────────────────────────────────────────── */
    float dg = sdf_plane(p, 0.0f);
    if (dg < h.d) { h.d = dg; h.mat = MAT_GROUND; }

    /* ── Body (rounded box centred at BODY_HEIGHT) ─────────────── */
    vec3 bp = v3_sub(p, v3(0.0f, BODY_HEIGHT, 0.0f));
    float db = sdf_rounded_box(bp, v3(BODY_HALF_LEN, BODY_HALF_H, BODY_HALF_W), BODY_ROUND);
    if (db < h.d) { h.d = db; h.mat = MAT_BODY; }

    /* ── Front LiDAR sensor (sphere on nose) ───────────────────── */
    float ds = sdf_sphere(p, v3(BODY_HALF_LEN + 0.005f, BODY_HEIGHT, 0.0f), 0.055f);
    /* blend sensor into body with smooth union so it looks attached */
    float body_blend = sdf_smooth_union(db, ds, 0.04f);
    if (body_blend < h.d) {
        h.d = body_blend;
        h.mat = (ds < db - 0.01f) ? MAT_SENSOR : MAT_BODY;
    }

    /* ── Four legs ─────────────────────────────────────────────── */
    for (int i = 0; i < 4; i++) {
        vec3 hip_j, knee_p, foot_p;
        leg_fk(i, &hip_j, &knee_p, &foot_p);

        /* Hip sphere */
        float dh = sdf_sphere(p, hip_j, HIP_R);
        if (dh < h.d) { h.d = dh; h.mat = MAT_HIP; }

        /* Thigh capsule */
        float dt = sdf_capsule(p, hip_j, knee_p, THIGH_R);
        if (dt < h.d) { h.d = dt; h.mat = MAT_THIGH; }

        /* Knee joint sphere (slightly fatter for visual joint) */
        float dk = sdf_sphere(p, knee_p, THIGH_R * 1.25f);
        if (dk < h.d) { h.d = dk; h.mat = MAT_KNEE; }

        /* Shin capsule */
        float dsn = sdf_capsule(p, knee_p, foot_p, SHIN_R);
        if (dsn < h.d) { h.d = dsn; h.mat = MAT_SHIN; }

        /* Foot sphere */
        float df = sdf_sphere(p, foot_p, FOOT_R);
        if (df < h.d) { h.d = df; h.mat = MAT_FOOT; }
    }

    return h;
}

static float scene_dist(vec3 p) { return scene_sdf(p).d; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING
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
        if (h.d < SURF_DIST) {
            r.hit = 1; r.dist = t; r.point = p; r.mat = h.mat;
            return r;
        }
        t += h.d;
        if (t > MAX_DIST) break;
    }
    return r;
}

static vec3 calc_normal(vec3 p) {
    float e = 0.001f;
    return v3_norm(v3(
        scene_dist(v3_add(p, v3(e,0,0))) - scene_dist(v3_sub(p, v3(e,0,0))),
        scene_dist(v3_add(p, v3(0,e,0))) - scene_dist(v3_sub(p, v3(0,e,0))),
        scene_dist(v3_add(p, v3(0,0,e))) - scene_dist(v3_sub(p, v3(0,0,e)))
    ));
}

/* Soft shadow: march toward light, return 1=lit, 0=shadowed, values in between */
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
 * SHADING
 * ══════════════════════════════════════════════════════════════════ */

static const char RAMP[] = " .,:;=+*%#@";
#define RAMP_LEN 11

/* Convert normalised luminance [0,1] to (fg, bg, char) for a terminal cell */
static void shade(RayHit ray, vec3 rd, Cell *out) {
    if (!ray.hit) {
        /* Sky gradient */
        out->ch = ' '; out->bg = 17; out->fg = 17;
        return;
    }

    vec3 n   = calc_normal(ray.point);
    vec3 lp  = v3_norm(v3(-0.5f, 1.8f, -0.8f));   /* key light dir */
    vec3 lp2 = v3_norm(v3( 0.8f, 0.5f,  0.6f));   /* fill light dir */
    vec3 vd  = v3_scale(rd, -1.0f);

    float shad = soft_shadow(ray.point, lp, 4.0f);

    float amb  = 0.18f;
    float diff = fmaxf(v3_dot(n, lp),  0.0f) * 0.65f * shad;
    float fill = fmaxf(v3_dot(n, lp2), 0.0f) * 0.15f;
    vec3  hv   = v3_norm(v3_add(lp, vd));
    float spec = powf(fmaxf(v3_dot(n, hv), 0.0f), 24.0f) * 0.35f * shad;

    float lum = fminf(amb + diff + fill + spec, 1.0f);

    /* ASCII character from luminance */
    int ci = (int)(lum * (RAMP_LEN - 1) + 0.5f);
    if (ci < 0) ci = 0;
    if (ci >= RAMP_LEN) ci = RAMP_LEN - 1;
    out->ch = RAMP[ci];

    /* Colour: modulate material base colour by luminance */
    const uint8_t *mc = MAT_COLOR[ray.mat];
    int lr = (int)(mc[0] * lum + 0.5f);
    int lg = (int)(mc[1] * lum + 0.5f);
    int lb = (int)(mc[2] * lum + 0.5f);
    lr = lr < 0 ? 0 : lr > 5 ? 5 : lr;
    lg = lg < 0 ? 0 : lg > 5 ? 5 : lg;
    lb = lb < 0 ? 0 : lb > 5 ? 5 : lb;

    out->fg = (uint8_t)(16 + 36*lr + 6*lg + lb);
    out->bg = 16;   /* black background */
}

/* ══════════════════════════════════════════════════════════════════
 * CAMERA
 * ══════════════════════════════════════════════════════════════════ */

static float g_yaw   = 0.8f;
static float g_pitch = 0.35f;
static float g_dist  = 2.2f;
static vec3  g_target = {0.0f, 0.15f, 0.0f};

static void render_frame(void) {
    /* Camera position (orbit) */
    float cy = cosf(g_pitch), sy = sinf(g_pitch);
    float cyw = cosf(g_yaw),  syw = sinf(g_yaw);
    vec3 cam = v3(
        g_target.x + g_dist * cy * syw,
        g_target.y + g_dist * sy,
        g_target.z + g_dist * cy * cyw
    );

    /* Camera frame */
    vec3 fwd   = v3_norm(v3_sub(g_target, cam));
    vec3 right = v3_norm(v3_cross(fwd, v3(0,1,0)));
    vec3 up    = v3_cross(right, fwd);

    /* Field of view: 60° half-angle */
    float fov = tanf((float)M_PI / 6.0f);
    /* Terminal chars are ~2× taller than wide; halve the aspect ratio */
    float aspect = ((float)term_w / (float)term_h) * 0.5f;

    for (int row = 0; row < term_h; row++) {
        for (int col = 0; col < term_w; col++) {
            float u =  (2.0f * (col + 0.5f) / term_w  - 1.0f) * fov * aspect;
            float v = -(2.0f * (row + 0.5f) / term_h  - 1.0f) * fov;
            vec3 rd = v3_norm(v3_add(fwd, v3_add(v3_scale(right, u), v3_scale(up, v))));
            RayHit hit = ray_march(cam, rd);
            shade(hit, rd, &screen[row * term_w + col]);
        }
    }
}

/* Draw simple HUD overlay */
static void draw_hud(float t) {
    char buf[128];
    int n = snprintf(buf, sizeof buf,
        " GO2 robot | t=%.1fs | yaw=%.2f pitch=%.2f dist=%.2f | Q=quit ",
        t, g_yaw, g_pitch, g_dist);
    for (int i = 0; i < n && i < term_w; i++) {
        screen[i].ch = buf[i];
        screen[i].fg = 231;  /* white */
        screen[i].bg = 234;  /* dark grey */
    }
}

/* ══════════════════════════════════════════════════════════════════
 * INPUT & MAIN LOOP
 * ══════════════════════════════════════════════════════════════════ */

static int handle_input(void) {
    char buf[16];
    int n = (int)read(STDIN_FILENO, buf, sizeof buf);
    if (n <= 0) return 1;

    for (int i = 0; i < n; i++) {
        char c = buf[i];
        if (c == 'q' || c == 'Q' || c == 3) return 0;   /* quit */

        /* Arrow keys: ESC [ A/B/C/D */
        if (c == '\033' && i+2 < n && buf[i+1] == '[') {
            char code = buf[i+2]; i += 2;
            if (code == 'A') g_pitch += 0.08f;  /* up    */
            if (code == 'B') g_pitch -= 0.08f;  /* down  */
            if (code == 'C') g_yaw   -= 0.08f;  /* right */
            if (code == 'D') g_yaw   += 0.08f;  /* left  */
        }
        if (c == '+' || c == '=') g_dist = fmaxf(0.5f, g_dist - 0.15f);
        if (c == '-')             g_dist = fminf(6.0f, g_dist + 0.15f);
    }
    /* Clamp pitch */
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

    /* Initialise joint angles to standing pose */
    for (int i = 0; i < 4; i++) { g_hip[i] = HIP_NEUTRAL; g_knee[i] = KNEE_STAND; }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (1) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        float t = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) * 1e-9f;

        if (!handle_input()) break;
        get_term_size();

        /* Re-allocate screen if terminal was resized */
        screen = realloc(screen, term_w * term_h * sizeof *screen);

        update_gait(t);
        render_frame();
        draw_hud(t);
        present();

        /* ~20 fps */
        struct timespec sl = {0, 50000000L};
        nanosleep(&sl, NULL);
    }

    free(screen);
    free(out_buf);
    return 0;
}
