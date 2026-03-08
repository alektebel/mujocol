/*
 * phase3b_ray_march.c — Ray Marching Algorithm
 *
 * EXERCISE (sub-phase 3b): Implement the ray marching (sphere tracing) algorithm
 * to find ray-SDF intersections, and render a static scene.
 *
 * Build: gcc -O2 -o phase3b_ray_march phase3b_ray_march.c -lm
 * Run:   ./phase3b_ray_march
 *
 * LEARNING GOALS:
 * - Understand sphere tracing: step along a ray by the SDF distance
 * - Key insight: SDF gives a guaranteed safe step size (cannot overshoot)
 * - Compute camera rays from field-of-view and pixel coordinates
 * - Combine multiple SDF primitives into a scene using union
 */

/* ══════════════════════════════════════════════════════════════════
 * TECHNIQUE OVERVIEW
 * ══════════════════════════════════════════════════════════════════
 *
 * 1. Sphere tracing
 *    The algorithm: start from ray origin, evaluate d = scene_sdf(pos).
 *    Since d is the guaranteed safe step size (the nearest surface is
 *    at least d units away), advance pos += dir * d.  Repeat until
 *    d < epsilon (hit) or total distance > MAX_DIST (miss).  Never
 *    overshoots a surface.
 *
 * 2. Camera ray generation
 *    For a raster image, each pixel (px, py) maps to a normalized
 *    screen-space coordinate UV in [-aspect/2..aspect/2] × [-0.5..0.5].
 *    The ray direction is:
 *      normalize(right*u + up*v - forward*focal_length)
 *    where the camera's forward/right/up vectors come from the
 *    look-at matrix.
 *
 * 3. Aspect ratio correction
 *    Terminal characters are taller than they are wide (typically ~2:1
 *    ratio).  We correct by scaling the horizontal UV by
 *    `aspect = term_w / term_h / 2.0f` (divide by 2 accounts for
 *    character aspect ratio).
 *
 * 4. 1e10f return / MAX_DIST
 *    The ray marcher has two stopping conditions:
 *    (a) d < HIT_EPS  = surface hit;
 *    (b) total_dist > MAX_DIST = miss (ray escaped the scene).
 *    Returning 1e10f from scene sdf means "nothing here" and the
 *    marcher immediately exceeds MAX_DIST.
 * ══════════════════════════════════════════════════════════════════ */

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

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING CONSTANTS
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 100
#define MAX_DIST  50.0f
#define SURF_DIST 0.001f

/* ── Scene hit result ────────────────────────────────────────────── */
typedef struct {
    float dist;
    int   material_id;  /* 0=ground, 1=sphere, 2=box, 3=torus */
} SceneHit;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Define the scene
 *
 * Build the scene by combining SDF primitives with sdf_union.
 * For each primitive, compute its distance d, then:
 *   if (d < hit.dist) { hit.dist = d; hit.material_id = N; }
 *
 * Add these four objects:
 *   1. Ground plane: sdf_plane(p, v3(0,1,0), 1.5f)          [material 0]
 *   2. Sphere:       sdf_sphere(p, v3(0,0,0), 1.0f)          [material 1]
 *   3. Box:          sdf_box(p, v3(2.5f,-0.5f,0),
 *                            v3(0.7f,0.7f,0.7f))             [material 2]
 *   4. Torus:        sdf_torus(p, v3(-2.5f,-0.8f,0),
 *                              0.6f, 0.25f)                  [material 3]
 *
 * The functions sdf_union, sdf_intersect, sdf_subtract are provided
 * above but you may also use the if-chain pattern shown here.
 * ══════════════════════════════════════════════════════════════════ */
static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    /* TODO: Build the scene — add plane, sphere, box, and torus
     * Build the scene by computing the SDF of each primitive and taking
     * the union (minimum) of all distances.  The plane provides a
     * ground; the sphere, box, and torus demonstrate the three primitive
     * types from phase3a.  Return the minimum:
     *   float d = sdf_plane(...);
     *   d = fminf(d, sdf_sphere(...));
     *   d = fminf(d, sdf_box(...));
     *   d = fminf(d, sdf_torus(...));
     *   return d;
     */
    (void)p;
    return hit;
}

/* ── Ray result ──────────────────────────────────────────────────── */
typedef struct {
    int   hit;
    float dist;
    vec3  point;
    int   material_id;
} RayResult;

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement ray marching (sphere tracing)
 *
 * The key insight: the SDF returns the radius of an empty sphere around
 * the current point — we can safely step that far without hitting anything.
 *
 * Algorithm:
 *   t = 0.0f                              [distance travelled along ray]
 *   for i in 0 .. MAX_STEPS-1:
 *     p  = ro + rd * t                    [current sample point]
 *     sh = scene_sdf(p)                   [safe step distance + material]
 *     if sh.dist < SURF_DIST:             [close enough to surface]
 *       return hit (hit=1, dist=t, point=p, material_id=sh.material_id)
 *     t += sh.dist                        [march forward — cannot overshoot]
 *     if t > MAX_DIST: break              [ray escaped the scene]
 *   return miss (hit=0)
 * ══════════════════════════════════════════════════════════════════ */
static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0.0f, ro, 0};
    /* TODO: Implement the ray marching loop
     * The loop:
     *   float t = 0;
     *   for(int i=0; i<MAX_STEPS; i++) {
     *       float d = scene_dist(v3_add(ro, v3_scale(rd, t)));
     *       if(d < HIT_EPS) return t;
     *       t += d;
     *       if(t > MAX_DIST) break;
     *   }
     *   return MAX_DIST;
     * ro=ray_origin, rd=ray_direction (unit vector).  Advancing by d
     * each step is the sphere-tracing guarantee: d is the minimum
     * distance to any surface in the scene, so we can always safely
     * step that far.
     */
    (void)ro; (void)rd;
    return result;
}

/* ── Camera ──────────────────────────────────────────────────────── */
typedef struct {
    vec3  pos;
    vec3  target;
    vec3  up;
    float fov;  /* vertical field-of-view in radians */
} Camera;

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Compute a camera ray for pixel (u, v)
 *
 * u and v are pre-scaled screen coordinates (u is horizontal,
 * v is vertical, both account for aspect ratio and FOV).
 *
 * Steps:
 *   1. forward  = normalize(cam->target - cam->pos)
 *   2. right    = normalize(cross(forward, cam->up))
 *   3. up_cam   = cross(right, forward)        [re-orthogonalised up]
 *   4. half_fov = tan(cam->fov * 0.5f)
 *   5. *ro = cam->pos
 *   6. *rd = normalize(forward
 *                      + right  * (u * half_fov)
 *                      + up_cam * (v * half_fov))
 *
 * Hint: use v3_norm(), v3_cross(), v3_add(), v3_scale()
 * ══════════════════════════════════════════════════════════════════ */
static void camera_get_ray(Camera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    /* TODO: Compute ray origin and direction from camera parameters
     * Steps:
     * (1) Compute the camera basis:
     *     `forward = normalize(target-eye)`
     *     `right   = normalize(cross(forward, world_up))`
     *     `up      = cross(right, forward)`
     * (2) Compute NDC (Normalized Device Coordinates):
     *     `uv_x = (px / width  - 0.5f) * aspect * 2.0f`
     *     `uv_y = -(py / height - 0.5f)` (negate y because screen y
     *     increases downward but 3D y increases upward)
     * (3) Ray direction:
     *     `normalize(right*uv_x + up*uv_y + forward*focal_len)`
     */
    (void)cam; (void)u; (void)v;
    *ro = cam->pos;
    *rd = v3(0, 0, -1);  /* placeholder — always points toward -Z */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Render one frame
 *
 * For each pixel (x, y):
 *   1. Convert to normalised UV with aspect correction:
 *        aspect = term_w / (term_h * 2.0f)   [terminals are ~2:1 char aspect]
 *        u = (2.0f * x / term_w  - 1.0f) * aspect
 *        v = -(2.0f * y / term_h - 1.0f)
 *
 *   2. Get the ray: camera_get_ray(cam, u, v, &ro, &rd)
 *
 *   3. March: RayResult hit = ray_march(ro, rd)
 *
 *   4. Colour the pixel:
 *      - Hit:  solid white cell  → ch=' ', fg=231, bg=231
 *      - Miss: sky gradient blue → sky_color = 17 + (int)((v+1)*0.5f * 4)
 *                                   ch=' ', fg=sky_color, bg=sky_color
 * ══════════════════════════════════════════════════════════════════ */
static void render_frame(Camera *cam) {
    float aspect = (float)term_w / (term_h * 2.0f);
    /* TODO: Iterate over every pixel, compute UV, get ray, march, colour
     * For each row y, col x: call `camera_ray(x, y)` to get (ro, rd),
     * then `t = ray_march(ro, rd)`.  If t >= MAX_DIST, the pixel is
     * background (draw space character).  Otherwise the ray hit a
     * surface — for now pick a character from the ASCII density ramp
     * based on a simple shading (e.g. based on hit distance).  Use
     * `set_cell(x, y, ch, fg, bg)` to write the pixel.
     */
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

    double start = get_time();
    render_frame(&cam);
    double elapsed = get_time() - start;

    present_screen();

    char info[128];
    snprintf(info, sizeof(info),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " MujoCol Phase 3b: Ray Marcher | Render: %.2fs | Press any key... \033[0m",
        elapsed);
    write(STDOUT_FILENO, info, strlen(info));

    char c;
    while (read(STDIN_FILENO, &c, 1) == 0) { /* wait */ }

    free(screen);
    free(out_buf);
    return 0;
}
