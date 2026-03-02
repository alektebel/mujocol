/*
 * phase3_sdf_render.c — SDF Ray Marcher
 *
 * Camera, ray generation, SDF primitives, ray marching, normal estimation,
 * Phong shading, ASCII luminance mapping.
 *
 * Deliverable: Static scene with sphere, box, torus rendered in ASCII with shading.
 *
 * Build: gcc -O2 -o phase3_sdf_render phase3_sdf_render.c -lm
 * Run:   ./phase3_sdf_render
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
 * 3D MATH (from phase2)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float x, y, z, w; } vec4;
typedef struct { float m[16]; } mat4;

#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

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
static inline vec3 v3_reflect(vec3 v, vec3 n) {
    return v3_sub(v, v3_scale(n, 2.0f * v3_dot(v, n)));
}
static inline float v3_max_comp(vec3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
static inline vec3 v3_abs(vec3 v) {
    return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}
static inline vec3 v3_max(vec3 a, vec3 b) {
    return v3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
static inline vec3 v3_min(vec3 a, vec3 b) {
    return v3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

/* ══════════════════════════════════════════════════════════════════
 * TERMINAL
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
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
}

static void present_screen(void) {
    char seq[64];
    int prev_fg = -1, prev_bg = -1;

    out_append("\033[H", 3);  /* home cursor */

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

/* ══════════════════════════════════════════════════════════════════
 * SDF PRIMITIVES
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}

static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    vec3 d = v3_sub(v3_abs(v3_sub(p, center)), half_size);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}

static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

static float sdf_cylinder(vec3 p, vec3 base, float radius, float height) {
    vec3 rel = v3_sub(p, base);
    float d_radial = sqrtf(rel.x * rel.x + rel.z * rel.z) - radius;
    float d_vertical = fabsf(rel.y - height * 0.5f) - height * 0.5f;
    return fminf(fmaxf(d_radial, d_vertical), 0.0f) +
           v3_len(v3(fmaxf(d_radial, 0.0f), fmaxf(d_vertical, 0.0f), 0.0f));
}

static float sdf_torus(vec3 p, vec3 center, float R, float r) {
    vec3 rel = v3_sub(p, center);
    float q = sqrtf(rel.x * rel.x + rel.z * rel.z) - R;
    return sqrtf(q * q + rel.y * rel.y) - r;
}

static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = v3_sub(p, a);
    vec3 ba = v3_sub(b, a);
    float h = fminf(fmaxf(v3_dot(pa, ba) / v3_dot(ba, ba), 0.0f), 1.0f);
    return v3_len(v3_sub(pa, v3_scale(ba, h))) - r;
}

/* ══════════════════════════════════════════════════════════════════
 * SDF COMBINATORS
 * ══════════════════════════════════════════════════════════════════ */

static inline float sdf_union(float a, float b) { return fminf(a, b); }
static inline float sdf_intersect(float a, float b) { return fmaxf(a, b); }
static inline float sdf_subtract(float a, float b) { return fmaxf(a, -b); }

static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

static float sdf_smooth_subtract(float a, float b, float k) {
    float h = fmaxf(k - fabsf(-a - b), 0.0f) / k;
    return fmaxf(a, -b) + h * h * k * 0.25f;
}

/* ══════════════════════════════════════════════════════════════════
 * SCENE DEFINITION
 * ══════════════════════════════════════════════════════════════════ */

/* Object material info returned alongside distance */
typedef struct {
    float dist;
    int material_id;  /* 0=ground, 1=sphere, 2=box, 3=torus, 4=cylinder */
} SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};

    /* Ground plane */
    float d_ground = sdf_plane(p, v3(0, 1, 0), 1.5f);
    if (d_ground < hit.dist) { hit.dist = d_ground; hit.material_id = 0; }

    /* Main sphere */
    float d_sphere = sdf_sphere(p, v3(0, 0, 0), 1.0f);
    if (d_sphere < hit.dist) { hit.dist = d_sphere; hit.material_id = 1; }

    /* Box to the right */
    float d_box = sdf_box(p, v3(2.5f, -0.5f, 0), v3(0.7f, 0.7f, 0.7f));
    if (d_box < hit.dist) { hit.dist = d_box; hit.material_id = 2; }

    /* Torus to the left */
    float d_torus = sdf_torus(p, v3(-2.5f, -0.8f, 0), 0.6f, 0.25f);
    if (d_torus < hit.dist) { hit.dist = d_torus; hit.material_id = 3; }

    /* Cylinder in back */
    float d_cyl = sdf_cylinder(p, v3(0, -1.5f, -2.5f), 0.5f, 2.0f);
    if (d_cyl < hit.dist) { hit.dist = d_cyl; hit.material_id = 4; }

    /* Small spheres in a row */
    for (int i = -2; i <= 2; i++) {
        float d = sdf_sphere(p, v3(i * 1.2f, -1.1f, 2.5f), 0.3f);
        if (d < hit.dist) { hit.dist = d; hit.material_id = 5 + (i + 2); }
    }

    return hit;
}

/* Just the distance for normal calculation */
static float scene_dist(vec3 p) {
    return scene_sdf(p).dist;
}

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 100
#define MAX_DIST 50.0f
#define SURF_DIST 0.001f

typedef struct {
    int hit;
    float dist;
    vec3 point;
    int material_id;
} RayResult;

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

/* ══════════════════════════════════════════════════════════════════
 * NORMAL ESTIMATION
 * ══════════════════════════════════════════════════════════════════ */

static vec3 calc_normal(vec3 p) {
    const float h = 0.001f;
    return v3_norm(v3(
        scene_dist(v3(p.x + h, p.y, p.z)) - scene_dist(v3(p.x - h, p.y, p.z)),
        scene_dist(v3(p.x, p.y + h, p.z)) - scene_dist(v3(p.x, p.y - h, p.z)),
        scene_dist(v3(p.x, p.y, p.z + h)) - scene_dist(v3(p.x, p.y, p.z - h))
    ));
}

/* ══════════════════════════════════════════════════════════════════
 * SHADING
 * ══════════════════════════════════════════════════════════════════ */

/* ASCII luminance ramp from dark to light */
static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10

/* Material colors (RGB 0-5 for 216 color cube) */
static const uint8_t MATERIAL_COLORS[][3] = {
    {1, 2, 1},  /* 0: ground - dark green */
    {5, 1, 1},  /* 1: sphere - red */
    {1, 3, 5},  /* 2: box - blue */
    {5, 4, 0},  /* 3: torus - orange */
    {3, 3, 3},  /* 4: cylinder - gray */
    {5, 5, 0},  /* 5: small sphere 1 - yellow */
    {5, 0, 5},  /* 6: small sphere 2 - magenta */
    {0, 5, 5},  /* 7: small sphere 3 - cyan */
    {5, 2, 0},  /* 8: small sphere 4 - orange */
    {2, 5, 2},  /* 9: small sphere 5 - green */
};

static uint8_t color_216(int r, int g, int b) {
    /* 216 color cube: 16 + 36*r + 6*g + b (r,g,b in 0-5) */
    r = r < 0 ? 0 : (r > 5 ? 5 : r);
    g = g < 0 ? 0 : (g > 5 ? 5 : g);
    b = b < 0 ? 0 : (b > 5 ? 5 : b);
    return 16 + 36 * r + 6 * g + b;
}

typedef struct {
    char ch;
    uint8_t fg;
    uint8_t bg;
} ShadeResult;

static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 0, 0};

    /* Light direction */
    vec3 light_dir = v3_norm(v3(0.5f, 1.0f, 0.3f));

    /* View direction (opposite of ray direction) */
    vec3 view_dir = v3_neg(rd);

    /* Half vector for specular */
    vec3 half_vec = v3_norm(v3_add(light_dir, view_dir));

    /* Phong components */
    float ambient = 0.15f;
    float diffuse = fmaxf(0.0f, v3_dot(normal, light_dir)) * 0.7f;
    float specular = powf(fmaxf(0.0f, v3_dot(normal, half_vec)), 32.0f) * 0.5f;

    /* Shadow ray */
    vec3 shadow_origin = v3_add(p, v3_scale(normal, SURF_DIST * 2));
    RayResult shadow = ray_march(shadow_origin, light_dir);
    if (shadow.hit && shadow.dist < 20.0f) {
        diffuse *= 0.3f;
        specular = 0;
    }

    /* Distance fog */
    float fog = expf(-dist * 0.05f);

    /* Final luminance */
    float lum = (ambient + diffuse + specular) * fog;
    lum = fminf(1.0f, fmaxf(0.0f, lum));

    /* Map to ASCII */
    int ramp_idx = (int)(lum * (RAMP_LEN - 1) + 0.5f);
    result.ch = LUMINANCE_RAMP[ramp_idx];

    /* Get material color and modulate by luminance */
    int num_mats = sizeof(MATERIAL_COLORS) / sizeof(MATERIAL_COLORS[0]);
    int mi = mat_id < num_mats ? mat_id : 0;
    float color_scale = 0.3f + lum * 0.7f;
    int r = (int)(MATERIAL_COLORS[mi][0] * color_scale);
    int g = (int)(MATERIAL_COLORS[mi][1] * color_scale);
    int b = (int)(MATERIAL_COLORS[mi][2] * color_scale);

    result.fg = color_216(r, g, b);
    result.bg = 16;  /* black background */

    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * CAMERA
 * ══════════════════════════════════════════════════════════════════ */

typedef struct {
    vec3 pos;
    vec3 target;
    vec3 up;
    float fov;  /* radians */
} Camera;

static void camera_get_ray(Camera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    /* u, v in [-1, 1] (normalized screen coords) */
    vec3 forward = v3_norm(v3_sub(cam->target, cam->pos));
    vec3 right = v3_norm(v3_cross(forward, cam->up));
    vec3 up = v3_cross(right, forward);

    float half_fov = tanf(cam->fov * 0.5f);
    *ro = cam->pos;
    *rd = v3_norm(v3_add(v3_add(
        forward,
        v3_scale(right, u * half_fov)),
        v3_scale(up, v * half_fov)
    ));
}

/* ══════════════════════════════════════════════════════════════════
 * RENDER
 * ══════════════════════════════════════════════════════════════════ */

static void render_frame(Camera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);  /* chars are ~2:1 tall */

    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            /* Normalized coords [-1, 1] */
            float u = (2.0f * x / term_w - 1.0f) * aspect;
            float v = -(2.0f * y / term_h - 1.0f);  /* flip Y */

            vec3 ro, rd;
            camera_get_ray(cam, u, v, &ro, &rd);

            RayResult hit = ray_march(ro, rd);

            Cell *cell = &screen[y * term_w + x];

            if (hit.hit) {
                vec3 normal = calc_normal(hit.point);
                ShadeResult shade = shade_point(hit.point, rd, normal, hit.material_id, hit.dist);
                cell->ch = shade.ch;
                cell->fg = shade.fg;
                cell->bg = shade.bg;
            } else {
                /* Sky gradient */
                float sky_t = (v + 1.0f) * 0.5f;
                int sky_color = 17 + (int)(sky_t * 4);  /* dark to light blue */
                cell->ch = ' ';
                cell->fg = sky_color;
                cell->bg = sky_color;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════ */

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    Camera cam = {
        .pos = v3(0, 2, 8),
        .target = v3(0, 0, 0),
        .up = v3(0, 1, 0),
        .fov = M_PI / 3.0f  /* 60 degrees */
    };

    printf("Rendering scene (%dx%d)...\n", term_w, term_h);
    fflush(stdout);

    double start = get_time();
    render_frame(&cam);
    double elapsed = get_time() - start;

    present_screen();

    /* Draw title and info */
    char info[128];
    snprintf(info, sizeof(info), "\033[1;1H\033[48;5;16m\033[38;5;226m MujoCol Phase 3: SDF Ray Marcher | Render: %.2fs | Press any key... \033[0m", elapsed);
    write(STDOUT_FILENO, info, strlen(info));

    /* Wait for keypress */
    char c;
    read(STDIN_FILENO, &c, 1);

    free(screen);
    free(out_buf);
    return 0;
}
