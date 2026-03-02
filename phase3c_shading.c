/*
 * phase3c_shading.c — Surface Normals and Phong Shading
 *
 * EXERCISE (sub-phase 3c): Implement surface normal estimation via finite
 * differences and Phong shading to create a fully lit SDF scene.
 *
 * Build: gcc -O2 -o phase3c_shading phase3c_shading.c -lm
 * Run:   ./phase3c_shading
 *
 * LEARNING GOALS:
 * - Estimate surface normals as the gradient of the SDF
 * - Apply Phong shading: ambient + diffuse + specular
 * - Map luminance to ASCII characters for terminal rendering
 * - Cast shadow rays to detect shadowing
 * - Apply distance fog for depth cues
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
 * 3D MATH (from phase2 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;

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

/* ══════════════════════════════════════════════════════════════════
 * TERMINAL (from phase1 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;

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
    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
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
                prev_fg = c->fg; prev_bg = c->bg;
            }
            out_append(&c->ch, 1);
        }
    }
    out_flush();
}

/* ══════════════════════════════════════════════════════════════════
 * SDF PRIMITIVES (from phase3a — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

static float sdf_sphere(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}

static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    vec3 q = v3_sub(v3_abs(v3_sub(p, center)), half_size);
    return v3_len(v3_max(q, v3(0,0,0))) + fminf(v3_max_comp(q), 0.0f);
}

static float sdf_torus(vec3 p, vec3 center, float R, float r) {
    vec3 rel = v3_sub(p, center);
    float q   = sqrtf(rel.x*rel.x + rel.z*rel.z) - R;
    return sqrtf(q*q + rel.y*rel.y) - r;
}

static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = v3_sub(p, a), ba = v3_sub(b, a);
    float h = fminf(fmaxf(v3_dot(pa, ba) / v3_dot(ba, ba), 0.0f), 1.0f);
    return v3_len(v3_sub(pa, v3_scale(ba, h))) - r;
}

static float sdf_cylinder(vec3 p, vec3 base, float radius, float height) {
    vec3 rel = v3_sub(p, base);
    float d_radial   = sqrtf(rel.x*rel.x + rel.z*rel.z) - radius;
    float d_vertical = fabsf(rel.y - height * 0.5f) - height * 0.5f;
    return fminf(fmaxf(d_radial, d_vertical), 0.0f) +
           v3_len(v3(fmaxf(d_radial, 0.0f), fmaxf(d_vertical, 0.0f), 0.0f));
}

static inline float sdf_union(float a, float b)     { return fminf(a, b); }
static inline float sdf_intersect(float a, float b) { return fmaxf(a, b); }
static inline float sdf_subtract(float a, float b)  { return fmaxf(a, -b); }
static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

/* ── Scene (already implemented) ────────────────────────────────── */
typedef struct {
    float dist;
    int   material_id;  /* 0=ground, 1=sphere, 2=box, 3=torus */
} SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    float d;

    d = sdf_plane(p, v3(0,1,0), 1.5f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    d = sdf_sphere(p, v3(0, 0, 0), 1.0f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }

    d = sdf_box(p, v3(2.5f, -0.5f, 0), v3(0.7f, 0.7f, 0.7f));
    if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }

    d = sdf_torus(p, v3(-2.5f, -0.8f, 0), 0.6f, 0.25f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

    return hit;
}

/* Convenience wrapper: distance only (used by calc_normal) */
static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING (from phase3b — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 100
#define MAX_DIST  50.0f
#define SURF_DIST 0.001f

typedef struct {
    int   hit;
    float dist;
    vec3  point;
    int   material_id;
} RayResult;

static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0.0f, ro, 0};
    float t = 0.0f;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        SceneHit sh = scene_sdf(p);
        if (sh.dist < SURF_DIST) {
            result.hit = 1;
            result.dist = t;
            result.point = p;
            result.material_id = sh.material_id;
            return result;
        }
        t += sh.dist;
        if (t > MAX_DIST) break;
    }
    return result;
}

/* ── Camera (already implemented) ───────────────────────────────── */
typedef struct {
    vec3  pos;
    vec3  target;
    vec3  up;
    float fov;  /* vertical field-of-view in radians */
} Camera;

static void camera_get_ray(Camera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 forward  = v3_norm(v3_sub(cam->target, cam->pos));
    vec3 right    = v3_norm(v3_cross(forward, cam->up));
    vec3 up       = v3_cross(right, forward);
    float half_fov = tanf(cam->fov * 0.5f);
    *ro = cam->pos;
    *rd = v3_norm(v3_add(v3_add(forward, v3_scale(right, u * half_fov)),
                                          v3_scale(up, v * half_fov)));
}

/* ══════════════════════════════════════════════════════════════════
 * SHADING INFRASTRUCTURE (already provided)
 * ══════════════════════════════════════════════════════════════════ */

static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10

/* Material colours (R,G,B each 0-5 for the 216-colour terminal cube) */
static const uint8_t MATERIAL_COLORS[][3] = {
    {1, 2, 1},  /* 0: ground  — dark green  */
    {5, 1, 1},  /* 1: sphere  — red         */
    {1, 3, 5},  /* 2: box     — blue        */
    {5, 4, 0},  /* 3: torus   — yellow      */
};
#define NUM_MATERIALS 4

/* Convert r,g,b (0-5 each) to a 256-colour terminal palette index */
static uint8_t color_216(int r, int g, int b) {
    r = r < 0 ? 0 : (r > 5 ? 5 : r);
    g = g < 0 ? 0 : (g > 5 ? 5 : g);
    b = b < 0 ? 0 : (b > 5 ? 5 : b);
    return (uint8_t)(16 + 36*r + 6*g + b);
}

/* ShadeResult carries both the renderable cell and the raw luminance
 * so that render_frame_shaded can apply shadow and fog on top. */
typedef struct { char ch; uint8_t fg; uint8_t bg; float lum; } ShadeResult;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Estimate the surface normal at point p
 *
 * The normal = gradient of the SDF evaluated via central differences:
 *
 *   h  = 0.001f
 *   nx = scene_dist(p + (h,0,0)) - scene_dist(p - (h,0,0))
 *   ny = scene_dist(p + (0,h,0)) - scene_dist(p - (0,h,0))
 *   nz = scene_dist(p + (0,0,h)) - scene_dist(p - (0,0,h))
 *   return normalize(nx, ny, nz)
 *
 * Intuition: the SDF increases fastest in the direction perpendicular
 * to the surface, which is exactly the outward normal direction.
 * ══════════════════════════════════════════════════════════════════ */
static vec3 calc_normal(vec3 p) {
    /* TODO: Compute surface normal via central finite differences of scene_dist() */
    (void)p;
    return v3(0, 1, 0);  /* placeholder — returns up vector */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Cast a shadow ray toward the light
 *
 * To check whether point p is in shadow:
 *   1. Offset the origin slightly along light_dir to avoid self-intersection:
 *        shadow_origin = p + light_dir * (SURF_DIST * 10.0f)
 *   2. March a ray from shadow_origin in direction light_dir
 *   3. If the ray hits something within 15 units:
 *        return 0.3f + 0.7f * (hit.dist / 15.0f)
 *      Otherwise (miss):
 *        return 1.0f   [fully lit]
 *
 * The factor maps hit distance → shadow darkness:
 *   close occluder → 0.3 (deep shadow), far occluder → ~1.0 (barely shadowed)
 *
 * Hint: call ray_march(shadow_origin, light_dir) and check result.hit
 * ══════════════════════════════════════════════════════════════════ */
static float cast_shadow(vec3 p, vec3 light_dir) {
    /* TODO: Cast a shadow ray and return a shadow factor in [0.3, 1.0] */
    (void)p; (void)light_dir;
    return 1.0f;  /* placeholder — no shadows */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Apply distance fog
 *
 * Exponential fog attenuates the luminance with distance:
 *   return lum * exp(-dist * 0.05f)
 *
 * Objects further away are dimmer — this creates atmospheric depth.
 * The fog density constant (0.05) can be tuned: larger = thicker fog.
 * ══════════════════════════════════════════════════════════════════ */
static float apply_fog(float lum, float dist) {
    /* TODO: Multiply lum by an exponential falloff based on dist */
    (void)dist;
    return lum;  /* placeholder — no fog applied */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Phong shading
 *
 * Implement the Blinn-Phong lighting model:
 *
 *   light_dir = normalize(0.5, 1.0, 0.3)       [directional light]
 *   view_dir  = normalize(-rd)                  [toward the camera]
 *   half_vec  = normalize(light_dir + view_dir) [half-way vector]
 *
 *   ambient   = 0.15f
 *   diffuse   = max(0, dot(normal, light_dir)) * 0.7f
 *   specular  = pow(max(0, dot(normal, half_vec)), 32) * 0.5f
 *   luminance = clamp(ambient + diffuse + specular, 0, 1)
 *
 * Map to ASCII and colour:
 *   result.lum = luminance
 *   result.ch  = LUMINANCE_RAMP[(int)(luminance * (RAMP_LEN-1) + 0.5f)]
 *   result.fg  = color_216 with material colour scaled by luminance
 *                  mat = MATERIAL_COLORS[mat_id % NUM_MATERIALS]
 *                  r = (int)(mat[0] * luminance + 0.5f)  [etc. for g, b]
 *   result.bg  = 16 (dark background)
 *
 * Note: cast_shadow and apply_fog are intentionally separate — they are
 * applied by render_frame_shaded after this function returns, using result.lum.
 * ══════════════════════════════════════════════════════════════════ */
static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 16, 16, 0.0f};
    /* TODO: Implement Phong shading and fill result.ch, result.fg, result.lum */
    (void)p; (void)rd; (void)normal; (void)mat_id; (void)dist;
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Render one frame with full shading
 *
 * Like render_frame in phase3b, but now with lighting, shadows, and fog:
 *
 * For each pixel (x, y):
 *   1. Compute u, v with aspect correction (same as phase3b)
 *   2. Get ray: camera_get_ray(cam, u, v, &ro, &rd)
 *   3. March:   RayResult hit = ray_march(ro, rd)
 *   4. If hit:
 *        a. normal = calc_normal(hit.point)
 *        b. shade  = shade_point(hit.point, rd, normal,
 *                                hit.material_id, hit.dist)
 *        c. light_dir = v3_norm(v3(0.5f, 1.0f, 0.3f))
 *        d. shadow = cast_shadow(hit.point, light_dir)
 *        e. final_lum = apply_fog(shade.lum * shadow, hit.dist)
 *             clamp final_lum to [0, 1]
 *        f. Re-map char and colour using final_lum:
 *             int idx = (int)(final_lum * (RAMP_LEN-1) + 0.5f)
 *             cell->ch  = LUMINANCE_RAMP[idx]
 *             recompute cell->fg via color_216 with material colour * final_lum
 *        g. cell->bg = 16
 *   5. If miss: sky gradient (same as phase3b)
 * ══════════════════════════════════════════════════════════════════ */
static void render_frame_shaded(Camera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);
    /* TODO: Render scene with full Phong shading, shadows, and fog */
    (void)cam; (void)aspect;
    for (int y = 0; y < term_h; y++)
        for (int x = 0; x < term_w; x++)
            screen[y * term_w + x] = (Cell){' ', 16, 16};  /* placeholder — all black */
}

/* ── Timing ──────────────────────────────────────────────────────── */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Main — animated camera orbit ───────────────────────────────── */
int main(void) {
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    int frame = 0;
    char c = 0;
    do {
        /* Rotate the camera around the scene each frame */
        float angle = frame * 0.4f;
        Camera cam = {
            .pos    = v3(sinf(angle) * 8.0f, 2.5f, cosf(angle) * 8.0f),
            .target = v3(0, 0, 0),
            .up     = v3(0, 1, 0),
            .fov    = (float)M_PI / 3.0f
        };

        double start = get_time();
        render_frame_shaded(&cam);
        double elapsed = get_time() - start;

        present_screen();

        char info[160];
        snprintf(info, sizeof(info),
            "\033[1;1H\033[48;5;16m\033[38;5;226m"
            " MujoCol Phase 3c: Shaded SDF | Frame %d | Render: %.2fs"
            " | [any key] next  [q] quit \033[0m",
            frame, elapsed);
        write(STDOUT_FILENO, info, strlen(info));

        c = 0;
        while (read(STDIN_FILENO, &c, 1) == 0) { /* wait for keypress */ }
        frame++;
    } while (c != 'q');

    free(screen);
    free(out_buf);
    return 0;
}
