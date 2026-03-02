/*
 * phase7_sim.c — Full Simulation Loop
 *
 * Unified simulation combining all modules. Implements classic RL environments:
 * CartPole, Pendulum, Reacher. ASCII HUD shows joint angles, reward, timestep.
 *
 * Deliverable: Interactive CartPole balancing in terminal.
 *
 * Build: gcc -O3 -o phase7_sim phase7_sim.c -lm
 * Run:   ./phase7_sim [cartpole|pendulum|reacher]
 * Controls: [Left/Right] apply force, [R] reset, [P] pause, [WASD] camera, [Q] quit
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
static inline vec3 v3_abs(vec3 v) { return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
static inline vec3 v3_max(vec3 a, vec3 b) { return v3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
static inline float v3_max_comp(vec3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) sum += M4(a, row, k) * M4(b, k, c);
            M4(r, row, c) = sum;
        }
    }
    return r;
}

static mat4 m4_translate(vec3 t) {
    mat4 m = m4_identity();
    M4(m, 0, 3) = t.x; M4(m, 1, 3) = t.y; M4(m, 2, 3) = t.z;
    return m;
}

static mat4 m4_rotate_z(float rad) {
    mat4 m = m4_identity();
    float c = cosf(rad), s = sinf(rad);
    M4(m, 0, 0) = c; M4(m, 0, 1) = -s;
    M4(m, 1, 0) = s; M4(m, 1, 1) = c;
    return m;
}

static mat4 m4_rotate_y(float rad) {
    mat4 m = m4_identity();
    float c = cosf(rad), s = sinf(rad);
    M4(m, 0, 0) = c;  M4(m, 0, 2) = s;
    M4(m, 2, 0) = -s; M4(m, 2, 2) = c;
    return m;
}

static vec3 m4_transform_point(mat4 m, vec3 p) {
    return v3(
        M4(m, 0, 0)*p.x + M4(m, 0, 1)*p.y + M4(m, 0, 2)*p.z + M4(m, 0, 3),
        M4(m, 1, 0)*p.x + M4(m, 1, 1)*p.y + M4(m, 1, 2)*p.z + M4(m, 1, 3),
        M4(m, 2, 0)*p.x + M4(m, 2, 1)*p.y + M4(m, 2, 2)*p.z + M4(m, 2, 3)
    );
}

static mat4 m4_inverse(mat4 m) {
    mat4 inv;
    float *o = inv.m, *i = m.m;
    o[0]  =  i[5]*i[10]*i[15] - i[5]*i[11]*i[14] - i[9]*i[6]*i[15] + i[9]*i[7]*i[14] + i[13]*i[6]*i[11] - i[13]*i[7]*i[10];
    o[4]  = -i[4]*i[10]*i[15] + i[4]*i[11]*i[14] + i[8]*i[6]*i[15] - i[8]*i[7]*i[14] - i[12]*i[6]*i[11] + i[12]*i[7]*i[10];
    o[8]  =  i[4]*i[9]*i[15]  - i[4]*i[11]*i[13] - i[8]*i[5]*i[15] + i[8]*i[7]*i[13] + i[12]*i[5]*i[11] - i[12]*i[7]*i[9];
    o[12] = -i[4]*i[9]*i[14]  + i[4]*i[10]*i[13] + i[8]*i[5]*i[14] - i[8]*i[6]*i[13] - i[12]*i[5]*i[10] + i[12]*i[6]*i[9];
    o[1]  = -i[1]*i[10]*i[15] + i[1]*i[11]*i[14] + i[9]*i[2]*i[15] - i[9]*i[3]*i[14] - i[13]*i[2]*i[11] + i[13]*i[3]*i[10];
    o[5]  =  i[0]*i[10]*i[15] - i[0]*i[11]*i[14] - i[8]*i[2]*i[15] + i[8]*i[3]*i[14] + i[12]*i[2]*i[11] - i[12]*i[3]*i[10];
    o[9]  = -i[0]*i[9]*i[15]  + i[0]*i[11]*i[13] + i[8]*i[1]*i[15] - i[8]*i[3]*i[13] - i[12]*i[1]*i[11] + i[12]*i[3]*i[9];
    o[13] =  i[0]*i[9]*i[14]  - i[0]*i[10]*i[13] - i[8]*i[1]*i[14] + i[8]*i[2]*i[13] + i[12]*i[1]*i[10] - i[12]*i[2]*i[9];
    o[2]  =  i[1]*i[6]*i[15] - i[1]*i[7]*i[14] - i[5]*i[2]*i[15] + i[5]*i[3]*i[14] + i[13]*i[2]*i[7] - i[13]*i[3]*i[6];
    o[6]  = -i[0]*i[6]*i[15] + i[0]*i[7]*i[14] + i[4]*i[2]*i[15] - i[4]*i[3]*i[14] - i[12]*i[2]*i[7] + i[12]*i[3]*i[6];
    o[10] =  i[0]*i[5]*i[15] - i[0]*i[7]*i[13] - i[4]*i[1]*i[15] + i[4]*i[3]*i[13] + i[12]*i[1]*i[7] - i[12]*i[3]*i[5];
    o[14] = -i[0]*i[5]*i[14] + i[0]*i[6]*i[13] + i[4]*i[1]*i[14] - i[4]*i[2]*i[13] - i[12]*i[1]*i[6] + i[12]*i[2]*i[5];
    o[3]  = -i[1]*i[6]*i[11] + i[1]*i[7]*i[10] + i[5]*i[2]*i[11] - i[5]*i[3]*i[10] - i[9]*i[2]*i[7] + i[9]*i[3]*i[6];
    o[7]  =  i[0]*i[6]*i[11] - i[0]*i[7]*i[10] - i[4]*i[2]*i[11] + i[4]*i[3]*i[10] + i[8]*i[2]*i[7] - i[8]*i[3]*i[6];
    o[11] = -i[0]*i[5]*i[11] + i[0]*i[7]*i[9]  + i[4]*i[1]*i[11] - i[4]*i[3]*i[9]  - i[8]*i[1]*i[7] + i[8]*i[3]*i[5];
    o[15] =  i[0]*i[5]*i[10] - i[0]*i[6]*i[9]  - i[4]*i[1]*i[10] + i[4]*i[2]*i[9]  + i[8]*i[1]*i[6] - i[8]*i[2]*i[5];
    float det = i[0]*o[0] + i[1]*o[4] + i[2]*o[8] + i[3]*o[12];
    if (fabsf(det) < EPSILON) return m4_identity();
    float inv_det = 1.0f / det;
    for (int j = 0; j < 16; j++) o[j] *= inv_det;
    return inv;
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

#define KEY_NONE 0
#define KEY_ESC 27
#define KEY_UP 1000
#define KEY_DOWN 1001
#define KEY_LEFT 1002
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
 * ENVIRONMENT INTERFACE
 * ══════════════════════════════════════════════════════════════════ */

typedef enum { ENV_CARTPOLE, ENV_PENDULUM, ENV_REACHER } EnvType;

typedef struct {
    EnvType type;
    int timestep;
    int max_timesteps;
    float total_reward;
    int done;

    /* State variables (meaning depends on env type) */
    float x, x_dot;           /* CartPole: cart position/velocity */
    float theta, theta_dot;   /* CartPole/Pendulum: pole angle/angular velocity */
    float theta2, theta2_dot; /* Reacher: second joint */

    /* Rendering transforms */
    mat4 cart_tf;
    mat4 pole_tf;
    mat4 pole2_tf;  /* Reacher second link */
    mat4 target_tf; /* Reacher target */

    /* Reacher target position */
    float target_x, target_y;
} Environment;

static Environment env;

/* CartPole constants */
#define CARTPOLE_GRAVITY 9.81f
#define CARTPOLE_CART_MASS 1.0f
#define CARTPOLE_POLE_MASS 0.1f
#define CARTPOLE_POLE_LENGTH 1.0f
#define CARTPOLE_FORCE_MAG 10.0f
#define CARTPOLE_DT 0.02f
#define CARTPOLE_X_THRESHOLD 2.4f
#define CARTPOLE_THETA_THRESHOLD (12.0f * M_PI / 180.0f)

/* Pendulum constants */
#define PENDULUM_MAX_SPEED 8.0f
#define PENDULUM_MAX_TORQUE 2.0f
#define PENDULUM_DT 0.05f
#define PENDULUM_G 9.81f
#define PENDULUM_M 1.0f
#define PENDULUM_L 1.0f

/* Reacher constants */
#define REACHER_LINK1_LENGTH 0.8f
#define REACHER_LINK2_LENGTH 0.6f
#define REACHER_DT 0.02f

static float randf(void) {
    return (float)rand() / RAND_MAX;
}

static void env_reset(void) {
    env.timestep = 0;
    env.total_reward = 0;
    env.done = 0;

    switch (env.type) {
        case ENV_CARTPOLE:
            env.max_timesteps = 500;
            env.x = (randf() - 0.5f) * 0.1f;
            env.x_dot = (randf() - 0.5f) * 0.1f;
            env.theta = (randf() - 0.5f) * 0.1f;
            env.theta_dot = (randf() - 0.5f) * 0.1f;
            break;

        case ENV_PENDULUM:
            env.max_timesteps = 200;
            env.theta = M_PI + (randf() - 0.5f) * 0.2f;  /* Start near bottom */
            env.theta_dot = (randf() - 0.5f) * 1.0f;
            break;

        case ENV_REACHER:
            env.max_timesteps = 200;
            env.theta = (randf() - 0.5f) * M_PI;
            env.theta_dot = 0;
            env.theta2 = (randf() - 0.5f) * M_PI;
            env.theta2_dot = 0;
            /* Random target in reachable area */
            float r = 0.5f + randf() * 0.8f;
            float a = randf() * 2.0f * M_PI;
            env.target_x = r * cosf(a);
            env.target_y = r * sinf(a);
            break;
    }
}

static void env_step(float action) {
    if (env.done) return;

    env.timestep++;
    float reward = 0;

    switch (env.type) {
        case ENV_CARTPOLE: {
            float force = action * CARTPOLE_FORCE_MAG;
            float total_mass = CARTPOLE_CART_MASS + CARTPOLE_POLE_MASS;
            float pole_mass_length = CARTPOLE_POLE_MASS * CARTPOLE_POLE_LENGTH;

            float cos_theta = cosf(env.theta);
            float sin_theta = sinf(env.theta);

            float temp = (force + pole_mass_length * env.theta_dot * env.theta_dot * sin_theta) / total_mass;
            float theta_acc = (CARTPOLE_GRAVITY * sin_theta - cos_theta * temp) /
                (CARTPOLE_POLE_LENGTH * (4.0f/3.0f - CARTPOLE_POLE_MASS * cos_theta * cos_theta / total_mass));
            float x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

            /* Semi-implicit Euler */
            env.x_dot += x_acc * CARTPOLE_DT;
            env.x += env.x_dot * CARTPOLE_DT;
            env.theta_dot += theta_acc * CARTPOLE_DT;
            env.theta += env.theta_dot * CARTPOLE_DT;

            /* Reward for staying alive */
            reward = 1.0f;

            /* Termination conditions */
            if (fabsf(env.x) > CARTPOLE_X_THRESHOLD ||
                fabsf(env.theta) > CARTPOLE_THETA_THRESHOLD ||
                env.timestep >= env.max_timesteps) {
                env.done = 1;
            }
            break;
        }

        case ENV_PENDULUM: {
            float torque = fmaxf(-PENDULUM_MAX_TORQUE, fminf(PENDULUM_MAX_TORQUE, action * PENDULUM_MAX_TORQUE));

            float theta_acc = (-3.0f * PENDULUM_G / (2.0f * PENDULUM_L) * sinf(env.theta + M_PI) +
                              3.0f / (PENDULUM_M * PENDULUM_L * PENDULUM_L) * torque);

            env.theta_dot += theta_acc * PENDULUM_DT;
            env.theta_dot = fmaxf(-PENDULUM_MAX_SPEED, fminf(PENDULUM_MAX_SPEED, env.theta_dot));
            env.theta += env.theta_dot * PENDULUM_DT;

            /* Normalize angle to [-pi, pi] */
            while (env.theta > M_PI) env.theta -= 2.0f * M_PI;
            while (env.theta < -M_PI) env.theta += 2.0f * M_PI;

            /* Reward: penalize angle from upright and speed */
            float angle_cost = env.theta * env.theta;
            float speed_cost = 0.1f * env.theta_dot * env.theta_dot;
            float torque_cost = 0.001f * torque * torque;
            reward = -(angle_cost + speed_cost + torque_cost);

            if (env.timestep >= env.max_timesteps) {
                env.done = 1;
            }
            break;
        }

        case ENV_REACHER: {
            /* action is a 2D control mapped from single value or use left/right */
            float torque1 = action * 0.5f;
            float torque2 = action * 0.3f;

            /* Simple dynamics */
            env.theta_dot += torque1 * REACHER_DT;
            env.theta += env.theta_dot * REACHER_DT;
            env.theta_dot *= 0.98f;  /* damping */

            env.theta2_dot += torque2 * REACHER_DT;
            env.theta2 += env.theta2_dot * REACHER_DT;
            env.theta2_dot *= 0.98f;

            /* End effector position */
            float ee_x = REACHER_LINK1_LENGTH * cosf(env.theta) +
                        REACHER_LINK2_LENGTH * cosf(env.theta + env.theta2);
            float ee_y = REACHER_LINK1_LENGTH * sinf(env.theta) +
                        REACHER_LINK2_LENGTH * sinf(env.theta + env.theta2);

            /* Reward: negative distance to target */
            float dx = ee_x - env.target_x;
            float dy = ee_y - env.target_y;
            float dist = sqrtf(dx * dx + dy * dy);
            reward = -dist;

            if (env.timestep >= env.max_timesteps) {
                env.done = 1;
            }
            break;
        }
    }

    env.total_reward += reward;
}

static void env_compute_transforms(void) {
    switch (env.type) {
        case ENV_CARTPOLE: {
            /* Cart */
            env.cart_tf = m4_translate(v3(env.x, 0, 0));
            /* Pole attached to cart top */
            mat4 pole_offset = m4_translate(v3(0, 0.15f, 0));
            mat4 pole_rot = m4_rotate_z(-env.theta);
            env.pole_tf = m4_mul(env.cart_tf, m4_mul(pole_offset, pole_rot));
            break;
        }

        case ENV_PENDULUM: {
            /* Pendulum pivot at origin */
            env.pole_tf = m4_rotate_z(-env.theta + M_PI/2);
            break;
        }

        case ENV_REACHER: {
            /* Link 1 from origin */
            env.pole_tf = m4_rotate_z(env.theta);
            /* Link 2 from end of link 1 */
            mat4 link1_end = m4_translate(v3(REACHER_LINK1_LENGTH, 0, 0));
            mat4 link2_rot = m4_rotate_z(env.theta2);
            env.pole2_tf = m4_mul(env.pole_tf, m4_mul(link1_end, link2_rot));
            /* Target */
            env.target_tf = m4_translate(v3(env.target_x, env.target_y, 0));
            break;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SDF SCENE
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}

static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    vec3 d = v3_sub(v3_abs(v3_sub(p, center)), half_size);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}

static float sdf_cylinder_y(vec3 p, vec3 center, float radius, float height) {
    vec3 rel = v3_sub(p, center);
    float d_radial = sqrtf(rel.x * rel.x + rel.z * rel.z) - radius;
    float d_vertical = fabsf(rel.y) - height * 0.5f;
    return fminf(fmaxf(d_radial, d_vertical), 0.0f) +
           v3_len(v3(fmaxf(d_radial, 0.0f), fmaxf(d_vertical, 0.0f), 0.0f));
}

static float sdf_capsule_x(vec3 p, float length, float radius) {
    float hlen = length * 0.5f;
    p.x = fabsf(p.x) - hlen;
    if (p.x < 0) p.x = 0;
    return v3_len(p) - radius;
}

static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};

    /* Ground */
    float d = sdf_plane(p, v3(0, 1, 0), 0.5f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    switch (env.type) {
        case ENV_CARTPOLE: {
            /* Track */
            d = sdf_box(p, v3(0, -0.45f, 0), v3(3.0f, 0.05f, 0.2f));
            if (d < hit.dist) { hit.dist = d; hit.material_id = 4; }

            /* Cart */
            vec3 cart_p = m4_transform_point(m4_inverse(env.cart_tf), p);
            d = sdf_box(cart_p, v3(0, 0, 0), v3(0.3f, 0.15f, 0.2f));
            if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }

            /* Wheels */
            d = sdf_sphere(cart_p, v3(-0.2f, -0.15f, 0), 0.08f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }
            d = sdf_sphere(cart_p, v3(0.2f, -0.15f, 0), 0.08f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

            /* Pole */
            vec3 pole_p = m4_transform_point(m4_inverse(env.pole_tf), p);
            d = sdf_capsule_x(v3(pole_p.y - CARTPOLE_POLE_LENGTH * 0.5f, pole_p.x, pole_p.z),
                             CARTPOLE_POLE_LENGTH, 0.05f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }

            /* Pole tip */
            d = sdf_sphere(pole_p, v3(0, CARTPOLE_POLE_LENGTH, 0), 0.08f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 5; }
            break;
        }

        case ENV_PENDULUM: {
            /* Pivot */
            d = sdf_sphere(p, v3(0, 0, 0), 0.1f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

            /* Pole */
            vec3 pole_p = m4_transform_point(m4_inverse(env.pole_tf), p);
            d = sdf_capsule_x(v3(pole_p.x - PENDULUM_L * 0.5f, pole_p.y, pole_p.z),
                             PENDULUM_L, 0.06f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }

            /* Mass at end */
            d = sdf_sphere(pole_p, v3(PENDULUM_L, 0, 0), 0.15f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }
            break;
        }

        case ENV_REACHER: {
            /* Base */
            d = sdf_cylinder_y(p, v3(0, 0, 0), 0.15f, 0.1f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

            /* Link 1 */
            vec3 link1_p = m4_transform_point(m4_inverse(env.pole_tf), p);
            d = sdf_capsule_x(v3(link1_p.x - REACHER_LINK1_LENGTH * 0.5f, link1_p.y, link1_p.z),
                             REACHER_LINK1_LENGTH, 0.05f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 1; }

            /* Joint */
            d = sdf_sphere(link1_p, v3(REACHER_LINK1_LENGTH, 0, 0), 0.08f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 3; }

            /* Link 2 */
            vec3 link2_p = m4_transform_point(m4_inverse(env.pole2_tf), p);
            d = sdf_capsule_x(v3(link2_p.x - REACHER_LINK2_LENGTH * 0.5f, link2_p.y, link2_p.z),
                             REACHER_LINK2_LENGTH, 0.04f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 2; }

            /* End effector */
            d = sdf_sphere(link2_p, v3(REACHER_LINK2_LENGTH, 0, 0), 0.06f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 5; }

            /* Target */
            d = sdf_sphere(p, v3(env.target_x, env.target_y, 0), 0.1f);
            if (d < hit.dist) { hit.dist = d; hit.material_id = 6; }
            break;
        }
    }

    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING + SHADING
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 48
#define MAX_DIST 30.0f
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
    {2, 3, 2},  /* 0: ground */
    {5, 1, 1},  /* 1: red (cart/link1) */
    {1, 3, 5},  /* 2: blue (pole/link2) */
    {3, 3, 3},  /* 3: gray (joints) */
    {2, 2, 2},  /* 4: dark gray (track) */
    {5, 5, 0},  /* 5: yellow (tip/ee) */
    {0, 5, 2},  /* 6: green (target) */
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

    float fog = expf(-dist * 0.08f);
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

static void draw_hud(double fps, int paused, float action) {
    const char *env_names[] = {"CartPole", "Pendulum", "Reacher"};
    char hud[512];
    char state[256];

    switch (env.type) {
        case ENV_CARTPOLE:
            snprintf(state, sizeof(state), "x:%.2f v:%.2f th:%.1f° w:%.1f",
                     env.x, env.x_dot, env.theta * 180.0f / M_PI, env.theta_dot);
            break;
        case ENV_PENDULUM:
            snprintf(state, sizeof(state), "th:%.1f° w:%.1f",
                     env.theta * 180.0f / M_PI, env.theta_dot);
            break;
        case ENV_REACHER:
            snprintf(state, sizeof(state), "th1:%.0f° th2:%.0f° tgt:(%.1f,%.1f)",
                     env.theta * 180.0f / M_PI, env.theta2 * 180.0f / M_PI,
                     env.target_x, env.target_y);
            break;
    }

    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " %s | t:%d R:%.1f %s| %s | a:%.1f %s| [</>] act [R]eset [P]ause [Q]uit ",
        env_names[env.type], env.timestep, env.total_reward,
        env.done ? "[DONE] " : "", state, action,
        paused ? "[PAUSED] " : "");
    write(STDOUT_FILENO, hud, strlen(hud));

    /* Second line with FPS */
    char hud2[128];
    snprintf(hud2, sizeof(hud2),
        "\033[2;1H\033[48;5;16m\033[38;5;245m FPS:%.0f | [WASD] cam [+/-] zoom ",
        fps);
    write(STDOUT_FILENO, hud2, strlen(hud2));
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
 * MAIN
 * ══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    /* Select environment */
    env.type = ENV_CARTPOLE;
    if (argc > 1) {
        if (strcmp(argv[1], "pendulum") == 0) env.type = ENV_PENDULUM;
        else if (strcmp(argv[1], "reacher") == 0) env.type = ENV_REACHER;
    }

    env_reset();

    /* Camera settings based on environment */
    vec3 cam_target = v3(0, 0.5f, 0);
    float cam_dist = 5.0f;
    float cam_el = 0.3f;

    if (env.type == ENV_PENDULUM) {
        cam_target = v3(0, 0, 0);
        cam_dist = 4.0f;
    } else if (env.type == ENV_REACHER) {
        cam_target = v3(0, 0, 0);
        cam_dist = 4.0f;
        cam_el = 1.2f;  /* top-down-ish view */
    }

    OrbitCamera cam = {
        .target = cam_target,
        .distance = cam_dist,
        .azimuth = 0.0f,
        .elevation = cam_el,
        .fov = M_PI / 3.0f,
        .smooth_az = 0.0f,
        .smooth_el = cam_el,
        .smooth_dist = cam_dist
    };

    double last_time = get_time();
    double fps_time = last_time;
    int fps_count = 0;
    double fps_display = 0;
    int running = 1;
    int paused = 0;

    float sim_accum = 0;
    float sim_dt = 0.02f;
    float action = 0;
    float action_decay = 0.9f;

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

        /* Input */
        int key;
        while ((key = read_key()) != KEY_NONE) {
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'p': case 'P': paused = !paused; break;
                case 'r': case 'R': env_reset(); break;
                case KEY_LEFT:  action = -1.0f; break;
                case KEY_RIGHT: action = 1.0f; break;
                case 'w': case 'W': case KEY_UP:   cam.elevation += 0.1f; break;
                case 's': case 'S': case KEY_DOWN: cam.elevation -= 0.1f; break;
                case 'a': case 'A': cam.azimuth -= 0.1f; break;
                case 'd': case 'D': cam.azimuth += 0.1f; break;
                case '+': case '=': cam.distance -= 0.3f; break;
                case '-': case '_': cam.distance += 0.3f; break;
            }
        }

        cam.elevation = fmaxf(-M_PI/2 + 0.1f, fminf(M_PI/2 - 0.1f, cam.elevation));
        cam.distance = fmaxf(2.0f, fminf(15.0f, cam.distance));

        /* Simulation */
        if (!paused && !env.done) {
            sim_accum += dt;
            while (sim_accum >= sim_dt) {
                env_step(action);
                sim_accum -= sim_dt;
            }
            action *= action_decay;
        }

        /* Compute transforms for rendering */
        env_compute_transforms();

        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0;
            fps_time = now;
        }
        draw_hud(fps_display, paused, action);

        sleep_ms(30);
    }

    free(screen);
    free(out_buf);
    return 0;
}
