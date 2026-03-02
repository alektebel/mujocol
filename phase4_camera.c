/*
 * phase4_camera.c — Interactive Orbit Camera
 *
 * EXERCISE: Implement an orbit camera system that lets the user rotate and zoom
 * around the scene using keyboard input with smooth interpolation.
 *
 * Build: gcc -O2 -o phase4_camera phase4_camera.c -lm
 * Run:   ./phase4_camera
 * Controls: WASD/Arrows = orbit, +/- = zoom, Q/Esc = quit
 *
 * LEARNING GOALS:
 * - Understand spherical coordinates (azimuth, elevation, distance)
 * - Convert spherical coords to Cartesian camera position
 * - Build a ray from the camera through each screen pixel (perspective projection)
 * - Apply exponential smoothing for smooth camera motion
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
 * 3D MATH + TERMINAL (from phases 1-2 — already implemented)
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
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

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
        term_w = ws.ws_col; term_h = ws.ws_row;
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

/* ── SDF Scene (from phase3 — already implemented) ───────────────── */
static float sdf_sphere(vec3 p, vec3 c, float r) { return v3_len(v3_sub(p, c)) - r; }
static float sdf_box(vec3 p, vec3 c, vec3 hs) {
    vec3 d = v3_sub(v3_abs(v3_sub(p, c)), hs);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}
static float sdf_plane(vec3 p, vec3 n, float o) { return v3_dot(p, n) + o; }
static float sdf_torus(vec3 p, vec3 c, float R, float r) {
    vec3 rel = v3_sub(p, c);
    float q = sqrtf(rel.x*rel.x + rel.z*rel.z) - R;
    return sqrtf(q*q + rel.y*rel.y) - r;
}
static float sdf_cylinder(vec3 p, vec3 base, float radius, float height) {
    vec3 rel = v3_sub(p, base);
    float dr = sqrtf(rel.x*rel.x + rel.z*rel.z) - radius;
    float dv = fabsf(rel.y - height*0.5f) - height*0.5f;
    return fminf(fmaxf(dr, dv), 0.0f) + v3_len(v3(fmaxf(dr,0.0f), fmaxf(dv,0.0f), 0.0f));
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    float d;
    d = sdf_plane(p, v3(0,1,0), 1.5f); if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }
    d = sdf_sphere(p, v3(0,0,0), 1.0f); if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }
    d = sdf_box(p, v3(2.5f,-0.5f,0), v3(0.7f,0.7f,0.7f)); if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }
    d = sdf_torus(p, v3(-2.5f,-0.8f,0), 0.6f, 0.25f); if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }
    d = sdf_cylinder(p, v3(0,-1.5f,-2.5f), 0.5f, 2.0f); if (d < hit.dist) { hit.dist = d; hit.material_id = 4; }
    for (int i = -2; i <= 2; i++) {
        d = sdf_sphere(p, v3(i*1.2f,-1.1f,2.5f), 0.3f);
        if (d < hit.dist) { hit.dist = d; hit.material_id = 5+(i+2); }
    }
    return hit;
}
static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ── Ray marching + shading (from phase3 — already implemented) ───── */
#define MAX_STEPS 64
#define MAX_DIST  50.0f
#define SURF_DIST 0.002f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;

static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0, ro, 0};
    float t = 0;
    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        SceneHit h = scene_sdf(p);
        if (h.dist < SURF_DIST) {
            result.hit = 1; result.dist = t; result.point = p;
            result.material_id = h.material_id; return result;
        }
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
    {1,2,1},{5,1,1},{1,3,5},{5,4,0},{3,3,3},{5,5,0},{5,0,5},{0,5,5},{5,2,0},{2,5,2},
};
static uint8_t color_216(int r, int g, int b) {
    r = r<0?0:(r>5?5:r); g = g<0?0:(g>5?5:g); b = b<0?0:(b>5?5:b);
    return 16 + 36*r + 6*g + b;
}
typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;
static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 0, 0};
    vec3 light_dir = v3_norm(v3(0.5f, 1.0f, 0.3f));
    vec3 half_vec = v3_norm(v3_add(light_dir, v3_neg(rd)));
    float ambient = 0.15f;
    float diffuse = fmaxf(0.0f, v3_dot(normal, light_dir)) * 0.7f;
    float specular = powf(fmaxf(0.0f, v3_dot(normal, half_vec)), 32.0f) * 0.5f;
    vec3 shadow_o = v3_add(p, v3_scale(normal, SURF_DIST*2));
    RayResult shadow = ray_march(shadow_o, light_dir);
    if (shadow.hit && shadow.dist < 15.0f) {
        float sf = 0.3f + 0.7f*(shadow.dist/15.0f);
        diffuse *= sf; specular *= sf;
    }
    float fog = expf(-dist * 0.04f);
    float lum = fminf(1.0f, fmaxf(0.0f, (ambient + diffuse + specular) * fog));
    result.ch = LUMINANCE_RAMP[(int)(lum * (RAMP_LEN-1) + 0.5f)];
    int num_mats = sizeof(MATERIAL_COLORS)/sizeof(MATERIAL_COLORS[0]);
    int mi = mat_id < num_mats ? mat_id : 0;
    float cs = 0.3f + lum * 0.7f;
    result.fg = color_216((int)(MATERIAL_COLORS[mi][0]*cs),
                          (int)(MATERIAL_COLORS[mi][1]*cs),
                          (int)(MATERIAL_COLORS[mi][2]*cs));
    result.bg = 16;
    return result;
}

/* ══════════════════════════════════════════════════════════════════
 * ORBIT CAMERA
 *
 * An orbit camera lives at a point on a sphere centered on 'target'.
 * It is described by:
 *   - azimuth:   horizontal angle around the Y axis (radians)
 *   - elevation: vertical angle above the XZ plane (radians)
 *   - distance:  radius of the sphere (how far from target)
 *
 * The "smooth_*" values are the displayed values, interpolated toward
 * the target values each frame for smooth motion.
 * ══════════════════════════════════════════════════════════════════ */
typedef struct {
    vec3  target;
    float distance;
    float azimuth;    /* horizontal angle, radians */
    float elevation;  /* vertical angle, radians */
    float fov;

    /* Smoothed values (what gets rendered) */
    float smooth_az, smooth_el, smooth_dist;
} OrbitCamera;

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Compute camera world position from spherical coordinates
 *
 * Convert spherical (azimuth, elevation, distance) to Cartesian offset
 * from target, then add to target:
 *
 *   y      = sin(elevation) * distance         [vertical component]
 *   xz_r   = cos(elevation) * distance         [horizontal radius]
 *   x      = sin(azimuth) * xz_r
 *   z      = cos(azimuth) * xz_r
 *   result = target + (x, y, z)
 *
 * Use the smooth_* fields, NOT the raw azimuth/elevation/distance.
 * ══════════════════════════════════════════════════════════════════ */
static vec3 orbit_camera_pos(OrbitCamera *cam) {
    /* TODO: Convert spherical coords to world position */
    (void)cam;
    return v3(0, 5, 10);  /* placeholder */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Generate a ray from the camera through screen pixel (u, v)
 *
 * u, v are normalized screen coords in [-1, 1].
 * Steps:
 *   1. pos     = orbit_camera_pos(cam)
 *   2. forward = normalize(target - pos)
 *   3. right   = normalize(cross(forward, world_up))  world_up = (0,1,0)
 *   4. up      = cross(right, forward)
 *   5. half_fov = tan(fov * 0.5)
 *   6. *ro = pos
 *   7. *rd = normalize(forward + right * u * half_fov + up * v * half_fov)
 * ══════════════════════════════════════════════════════════════════ */
static void orbit_camera_get_ray(OrbitCamera *cam, float u, float v,
                                  vec3 *ro, vec3 *rd) {
    /* TODO: Build ray origin and direction from orbit camera */
    (void)cam; (void)u; (void)v;
    *ro = v3(0, 5, 10);
    *rd = v3_norm(v3(0, -0.5f, -1));  /* placeholder: points roughly at origin */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Update camera smooth interpolation
 *
 * Use exponential smoothing (lerp toward target each frame):
 *   smoothing = 1 - exp(-speed * dt)   [speed ≈ 10]
 *   smooth_az   = lerp(smooth_az, azimuth, smoothing)
 *   smooth_el   = lerp(smooth_el, elevation, smoothing)
 *   smooth_dist = lerp(smooth_dist, distance, smoothing)
 *
 * This creates a "lag" effect where the camera smoothly catches up.
 * ══════════════════════════════════════════════════════════════════ */
static void orbit_camera_update(OrbitCamera *cam, float dt) {
    /* TODO: Smoothly interpolate smooth_* toward target values */
    (void)dt;
    cam->smooth_az   = cam->azimuth;
    cam->smooth_el   = cam->elevation;
    cam->smooth_dist = cam->distance;
}

/* ── Render ──────────────────────────────────────────────────────── */
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
                ShadeResult shade = shade_point(hit.point, rd, normal,
                                                hit.material_id, hit.dist);
                cell->ch = shade.ch; cell->fg = shade.fg; cell->bg = shade.bg;
            } else {
                float sky_t = (v + 1.0f) * 0.5f;
                int sky_c = 17 + (int)(sky_t * 4);
                cell->ch = ' '; cell->fg = sky_c; cell->bg = sky_c;
            }
        }
    }
}

static void draw_hud(OrbitCamera *cam, double fps) {
    char hud[256];
    vec3 pos = orbit_camera_pos(cam);
    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " MujoCol Phase 4 | FPS: %.0f | Cam: (%.1f, %.1f, %.1f)"
        " | [WASD/Arrows] rotate [+/-] zoom [Q] quit ",
        fps, pos.x, pos.y, pos.z);
    write(STDOUT_FILENO, hud, strlen(hud));
}

/* ── Timing ──────────────────────────────────────────────────────── */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void sleep_ms(int ms) {
    struct timespec ts = { .tv_sec = ms/1000, .tv_nsec = (ms%1000)*1000000L };
    nanosleep(&ts, NULL);
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void) {
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target     = v3(0, -0.5f, 0),
        .distance   = 10.0f,
        .azimuth    = 0.3f,
        .elevation  = 0.4f,
        .fov        = (float)M_PI / 3.0f,
        .smooth_az  = 0.3f,
        .smooth_el  = 0.4f,
        .smooth_dist = 10.0f
    };

    double last_time = get_time();
    double fps_time  = last_time;
    int fps_count = 0;
    double fps_display = 0;
    int running = 1;

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
            float move = 0.15f, zoom = 0.5f;
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'w': case 'W': case KEY_UP:    cam.elevation += move; break;
                case 's': case 'S': case KEY_DOWN:  cam.elevation -= move; break;
                case 'a': case 'A': case KEY_LEFT:  cam.azimuth   -= move; break;
                case 'd': case 'D': case KEY_RIGHT: cam.azimuth   += move; break;
                case '+': case '=': cam.distance -= zoom; break;
                case '-': case '_': cam.distance += zoom; break;
            }
        }

        cam.elevation = fmaxf(-(float)M_PI/2 + 0.1f,
                              fminf((float)M_PI/2 - 0.1f, cam.elevation));
        cam.distance  = fmaxf(3.0f, fminf(30.0f, cam.distance));

        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0;
            fps_time = now;
        }
        draw_hud(&cam, fps_display);

        sleep_ms(50);
    }

    free(screen);
    free(out_buf);
    return 0;
}
