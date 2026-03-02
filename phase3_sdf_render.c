/*
 * phase3_sdf_render.c — SDF Ray Marcher
 *
 * EXERCISE: Implement SDF primitives, ray marching, surface normal estimation,
 * and Phong shading to render a 3D scene in the terminal using ASCII characters.
 *
 * Build: gcc -O2 -o phase3_sdf_render phase3_sdf_render.c -lm
 * Run:   ./phase3_sdf_render
 *
 * LEARNING GOALS:
 * - Understand Signed Distance Fields (SDF): a function that returns the shortest
 *   distance from a point to a surface (negative inside, positive outside)
 * - Learn the ray marching algorithm: step along a ray by the SDF distance
 * - Estimate surface normals via finite differences of the SDF gradient
 * - Apply Phong shading (ambient + diffuse + specular) and map to ASCII luminance
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
typedef struct { float x, y, z, w; } vec4;
typedef struct { float m[16]; } mat4;

#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

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
 * TODO #1: Implement SDF for a sphere
 *
 * A sphere SDF returns the distance from point p to the sphere surface.
 * Formula: length(p - center) - radius
 *   - Returns negative when p is INSIDE the sphere
 *   - Returns zero on the surface
 *   - Returns positive when p is OUTSIDE
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_sphere(vec3 p, vec3 center, float radius) {
    /* TODO: Implement sphere SDF */
    (void)p; (void)center; (void)radius;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement SDF for an axis-aligned box
 *
 * Box SDF formula for a box centered at 'center' with half-extents 'half_size':
 *   q = abs(p - center) - half_size       [componentwise]
 *   d = length(max(q, 0)) + min(max(q.x, q.y, q.z), 0)
 *
 * The max(q, 0) gives the outside distance vector.
 * The min(max_component, 0) handles the interior (negative inside).
 * Hint: use v3_abs(), v3_sub(), v3_max(), v3_max_comp()
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    /* TODO: Implement box SDF */
    (void)p; (void)center; (void)half_size;
    return 1e10f;
}

/* Ground plane — already provided */
static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement SDF for a torus
 *
 * A torus centered at 'center' with major radius R and minor radius r:
 *   rel = p - center
 *   q = sqrt(rel.x^2 + rel.z^2) - R   [distance in XZ plane minus major radius]
 *   d = sqrt(q^2 + rel.y^2) - r
 *
 * Think of it as revolving a circle of radius r around the Y axis
 * at distance R from the center.
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_torus(vec3 p, vec3 center, float R, float r) {
    /* TODO: Implement torus SDF */
    (void)p; (void)center; (void)R; (void)r;
    return 1e10f;
}

/* Cylinder — provided for reference */
static float sdf_cylinder(vec3 p, vec3 base, float radius, float height) {
    vec3 rel = v3_sub(p, base);
    float d_radial = sqrtf(rel.x * rel.x + rel.z * rel.z) - radius;
    float d_vertical = fabsf(rel.y - height * 0.5f) - height * 0.5f;
    return fminf(fmaxf(d_radial, d_vertical), 0.0f) +
           v3_len(v3(fmaxf(d_radial, 0.0f), fmaxf(d_vertical, 0.0f), 0.0f));
}

static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = v3_sub(p, a), ba = v3_sub(b, a);
    float h = fminf(fmaxf(v3_dot(pa, ba) / v3_dot(ba, ba), 0.0f), 1.0f);
    return v3_len(v3_sub(pa, v3_scale(ba, h))) - r;
}

/* ── SDF combinators ─────────────────────────────────────────────── */
static inline float sdf_union(float a, float b) { return fminf(a, b); }
static inline float sdf_intersect(float a, float b) { return fmaxf(a, b); }
static inline float sdf_subtract(float a, float b) { return fmaxf(a, -b); }
static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

/* ── Scene definition ────────────────────────────────────────────── */
typedef struct {
    float dist;
    int material_id;  /* 0=ground, 1=sphere, 2=box, 3=torus, 4=cylinder */
} SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};

    float d;
    d = sdf_plane(p, v3(0, 1, 0), 1.5f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    d = sdf_sphere(p, v3(0, 0, 0), 1.0f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }

    d = sdf_box(p, v3(2.5f, -0.5f, 0), v3(0.7f, 0.7f, 0.7f));
    if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }

    d = sdf_torus(p, v3(-2.5f, -0.8f, 0), 0.6f, 0.25f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

    d = sdf_cylinder(p, v3(0, -1.5f, -2.5f), 0.5f, 2.0f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 4; }

    for (int i = -2; i <= 2; i++) {
        d = sdf_sphere(p, v3(i * 1.2f, -1.1f, 2.5f), 0.3f);
        if (d < hit.dist) { hit.dist = d; hit.material_id = 5 + (i + 2); }
    }
    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement ray marching
 *
 * Ray marching (sphere tracing) algorithm:
 *   t = 0                          [distance traveled along ray]
 *   for i in 0..MAX_STEPS:
 *     p = ray_origin + t * ray_dir [current position along ray]
 *     d = scene_sdf(p).dist        [safe step distance]
 *     if d < SURF_DIST:            [we've hit a surface!]
 *       return hit result
 *     t += d                       [march forward by safe amount]
 *     if t > MAX_DIST: break       [gone too far, miss]
 *
 * Key insight: the SDF tells us the MAXIMUM safe step — we can't
 * overshoot a surface if we step by exactly the SDF distance.
 * ══════════════════════════════════════════════════════════════════ */
#define MAX_STEPS 100
#define MAX_DIST  50.0f
#define SURF_DIST 0.001f

typedef struct {
    int hit;
    float dist;
    vec3 point;
    int material_id;
} RayResult;

static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0, ro, 0};
    /* TODO: Implement ray marching loop
     * - Start with t = 0
     * - Each iteration: compute p = ro + rd*t, get scene_sdf(p)
     * - If distance < SURF_DIST: fill result (hit=1, dist=t, point=p, material)
     * - Otherwise: t += distance
     * - Stop if t > MAX_DIST or iterations exhausted */
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement surface normal estimation
 *
 * The surface normal = gradient of the SDF at point p.
 * Compute it using finite differences (central differences):
 *   nx = scene_dist(p + (h,0,0)) - scene_dist(p - (h,0,0))
 *   ny = scene_dist(p + (0,h,0)) - scene_dist(p - (0,h,0))
 *   nz = scene_dist(p + (0,0,h)) - scene_dist(p - (0,0,h))
 *   normal = normalize(nx, ny, nz)
 *
 * Use h = 0.001 (small epsilon for gradient approximation).
 * ══════════════════════════════════════════════════════════════════ */
static vec3 calc_normal(vec3 p) {
    /* TODO: Compute surface normal via central finite differences of scene_dist() */
    (void)p;
    return v3(0, 1, 0);  /* placeholder — returns up vector */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement Phong shading
 *
 * Phong shading formula:
 *   light_dir = normalize(light_pos - hit_point)
 *   view_dir  = normalize(-ray_direction)
 *   half_vec  = normalize(light_dir + view_dir)
 *
 *   ambient  = 0.15
 *   diffuse  = max(0, dot(normal, light_dir)) * 0.7
 *   specular = max(0, dot(normal, half_vec))^32 * 0.5
 *
 *   luminance = clamp(ambient + diffuse + specular, 0, 1)
 *   ascii_char = LUMINANCE_RAMP[luminance * (RAMP_LEN-1)]
 *
 * Color: modulate material RGB by luminance, convert to 256-color cube:
 *   index = 16 + 36*r + 6*g + b   (r,g,b in 0-5)
 * ══════════════════════════════════════════════════════════════════ */
static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10

/* Material colors (R,G,B in range 0-5 for 216-color cube) */
static const uint8_t MATERIAL_COLORS[][3] = {
    {1, 2, 1},  /* 0: ground */
    {5, 1, 1},  /* 1: sphere */
    {1, 3, 5},  /* 2: box */
    {5, 4, 0},  /* 3: torus */
    {3, 3, 3},  /* 4: cylinder */
    {5, 5, 0},  /* 5: small sphere 1 */
    {5, 0, 5},  /* 6: small sphere 2 */
    {0, 5, 5},  /* 7: small sphere 3 */
    {5, 2, 0},  /* 8: small sphere 4 */
    {2, 5, 2},  /* 9: small sphere 5 */
};

static uint8_t color_216(int r, int g, int b) {
    r = r < 0 ? 0 : (r > 5 ? 5 : r);
    g = g < 0 ? 0 : (g > 5 ? 5 : g);
    b = b < 0 ? 0 : (b > 5 ? 5 : b);
    return 16 + 36 * r + 6 * g + b;
}

typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;

static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 16, 16};
    /* TODO: Implement Phong shading
     * 1. Compute light_dir = normalize(0.5, 1.0, 0.3)  [directional light]
     * 2. Compute ambient, diffuse, specular terms
     * 3. (Optional) Cast a shadow ray to detect shadowing
     * 4. Apply distance fog: fog = exp(-dist * 0.05)
     * 5. luminance = clamp((ambient + diffuse + specular) * fog, 0, 1)
     * 6. Map to ASCII: result.ch = LUMINANCE_RAMP[(int)(lum * (RAMP_LEN-1) + 0.5)]
     * 7. Modulate material color by luminance, use color_216() for result.fg
     */
    (void)p; (void)rd; (void)normal; (void)mat_id; (void)dist;
    return result;
}

/* ── Camera ──────────────────────────────────────────────────────── */
typedef struct {
    vec3 pos;
    vec3 target;
    vec3 up;
    float fov;  /* radians */
} Camera;

static void camera_get_ray(Camera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 forward = v3_norm(v3_sub(cam->target, cam->pos));
    vec3 right   = v3_norm(v3_cross(forward, cam->up));
    vec3 up      = v3_cross(right, forward);
    float half_fov = tanf(cam->fov * 0.5f);
    *ro = cam->pos;
    *rd = v3_norm(v3_add(v3_add(forward, v3_scale(right, u * half_fov)),
                                         v3_scale(up, v * half_fov)));
}

/* ── Render ──────────────────────────────────────────────────────── */
static void render_frame(Camera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);

    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            float u = (2.0f * x / term_w - 1.0f) * aspect;
            float v = -(2.0f * y / term_h - 1.0f);

            vec3 ro, rd;
            camera_get_ray(cam, u, v, &ro, &rd);
            RayResult hit = ray_march(ro, rd);

            Cell *cell = &screen[y * term_w + x];
            if (hit.hit) {
                vec3 normal = calc_normal(hit.point);
                ShadeResult shade = shade_point(hit.point, rd, normal,
                                                hit.material_id, hit.dist);
                cell->ch = shade.ch;
                cell->fg = shade.fg;
                cell->bg = shade.bg;
            } else {
                float sky_t = (v + 1.0f) * 0.5f;
                int sky_color = 17 + (int)(sky_t * 4);
                cell->ch = ' ';
                cell->fg = sky_color;
                cell->bg = sky_color;
            }
        }
    }
}

/* ── Timing ──────────────────────────────────────────────────────── */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void) {
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    Camera cam = {
        .pos    = v3(0, 2, 8),
        .target = v3(0, 0, 0),
        .up     = v3(0, 1, 0),
        .fov    = (float)M_PI / 3.0f
    };

    printf("Rendering scene (%dx%d)...\n", term_w, term_h);
    fflush(stdout);

    double start = get_time();
    render_frame(&cam);
    double elapsed = get_time() - start;

    present_screen();

    char info[128];
    snprintf(info, sizeof(info),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " MujoCol Phase 3: SDF Ray Marcher | Render: %.2fs | Press any key... \033[0m",
        elapsed);
    write(STDOUT_FILENO, info, strlen(info));

    char c;
    read(STDIN_FILENO, &c, 1);

    free(screen);
    free(out_buf);
    return 0;
}
