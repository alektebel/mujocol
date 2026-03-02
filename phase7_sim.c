/*
 * phase7_sim.c — Full Simulation Loop
 *
 * EXERCISE: Implement the dynamics for three classic RL environments:
 * CartPole (balance a pole on a cart), Pendulum (swing-up), and
 * Reacher (reach a target with a 2-joint arm).
 *
 * Build: gcc -O2 -o phase7_sim phase7_sim.c -lm
 * Run:   ./phase7_sim [cartpole|pendulum|reacher]
 * Controls: [Left/Right] apply force/torque, [R] reset, [P] pause,
 *           [WASD] camera, [Q] quit
 *
 * LEARNING GOALS:
 * - Derive and implement equations of motion for constrained systems
 * - Understand the CartPole equations (coupled cart-pole dynamics)
 * - Understand the Pendulum swing-up task (torque control)
 * - Understand a 2-link Reacher arm (forward kinematics + reward)
 * - Compute reward functions and termination conditions
 * - Use matrix transforms to build visual representations of env state
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
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static mat4 m4_identity(void) { mat4 m={0}; m.m[0]=m.m[5]=m.m[10]=m.m[15]=1.0f; return m; }
static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r={0};
    for (int c=0; c<4; c++) for (int row=0; row<4; row++) {
        float sum=0; for (int k=0; k<4; k++) sum += M4(a,row,k)*M4(b,k,c);
        M4(r,row,c) = sum;
    }
    return r;
}
static mat4 m4_translate(vec3 t) {
    mat4 m=m4_identity(); M4(m,0,3)=t.x; M4(m,1,3)=t.y; M4(m,2,3)=t.z; return m;
}
static mat4 m4_rotate_z(float rad) {
    mat4 m=m4_identity(); float c=cosf(rad), s=sinf(rad);
    M4(m,0,0)=c; M4(m,0,1)=-s; M4(m,1,0)=s; M4(m,1,1)=c; return m;
}
static mat4 m4_rotate_y(float rad) {
    mat4 m=m4_identity(); float c=cosf(rad), s=sinf(rad);
    M4(m,0,0)=c; M4(m,0,2)=s; M4(m,2,0)=-s; M4(m,2,2)=c; return m;
}
static vec3 m4_transform_point(mat4 m, vec3 p) {
    return v3(M4(m,0,0)*p.x+M4(m,0,1)*p.y+M4(m,0,2)*p.z+M4(m,0,3),
              M4(m,1,0)*p.x+M4(m,1,1)*p.y+M4(m,1,2)*p.z+M4(m,1,3),
              M4(m,2,0)*p.x+M4(m,2,1)*p.y+M4(m,2,2)*p.z+M4(m,2,3));
}
static mat4 m4_inverse(mat4 m) {
    mat4 inv; float *o=inv.m, *i=m.m;
    o[0]= i[5]*i[10]*i[15]-i[5]*i[11]*i[14]-i[9]*i[6]*i[15]+i[9]*i[7]*i[14]+i[13]*i[6]*i[11]-i[13]*i[7]*i[10];
    o[4]=-i[4]*i[10]*i[15]+i[4]*i[11]*i[14]+i[8]*i[6]*i[15]-i[8]*i[7]*i[14]-i[12]*i[6]*i[11]+i[12]*i[7]*i[10];
    o[8]= i[4]*i[9]*i[15] -i[4]*i[11]*i[13]-i[8]*i[5]*i[15]+i[8]*i[7]*i[13]+i[12]*i[5]*i[11]-i[12]*i[7]*i[9];
    o[12]=-i[4]*i[9]*i[14]+i[4]*i[10]*i[13]+i[8]*i[5]*i[14]-i[8]*i[6]*i[13]-i[12]*i[5]*i[10]+i[12]*i[6]*i[9];
    o[1]=-i[1]*i[10]*i[15]+i[1]*i[11]*i[14]+i[9]*i[2]*i[15]-i[9]*i[3]*i[14]-i[13]*i[2]*i[11]+i[13]*i[3]*i[10];
    o[5]= i[0]*i[10]*i[15]-i[0]*i[11]*i[14]-i[8]*i[2]*i[15]+i[8]*i[3]*i[14]+i[12]*i[2]*i[11]-i[12]*i[3]*i[10];
    o[9]=-i[0]*i[9]*i[15] +i[0]*i[11]*i[13]+i[8]*i[1]*i[15]-i[8]*i[3]*i[13]-i[12]*i[1]*i[11]+i[12]*i[3]*i[9];
    o[13]=i[0]*i[9]*i[14] -i[0]*i[10]*i[13]-i[8]*i[1]*i[14]+i[8]*i[2]*i[13]+i[12]*i[1]*i[10]-i[12]*i[2]*i[9];
    o[2]= i[1]*i[6]*i[15]-i[1]*i[7]*i[14]-i[5]*i[2]*i[15]+i[5]*i[3]*i[14]+i[13]*i[2]*i[7]-i[13]*i[3]*i[6];
    o[6]=-i[0]*i[6]*i[15]+i[0]*i[7]*i[14]+i[4]*i[2]*i[15]-i[4]*i[3]*i[14]-i[12]*i[2]*i[7]+i[12]*i[3]*i[6];
    o[10]=i[0]*i[5]*i[15]-i[0]*i[7]*i[13]-i[4]*i[1]*i[15]+i[4]*i[3]*i[13]+i[12]*i[1]*i[7]-i[12]*i[3]*i[5];
    o[14]=-i[0]*i[5]*i[14]+i[0]*i[6]*i[13]+i[4]*i[1]*i[14]-i[4]*i[2]*i[13]-i[12]*i[1]*i[6]+i[12]*i[2]*i[5];
    o[3]=-i[1]*i[6]*i[11]+i[1]*i[7]*i[10]+i[5]*i[2]*i[11]-i[5]*i[3]*i[10]-i[9]*i[2]*i[7]+i[9]*i[3]*i[6];
    o[7]= i[0]*i[6]*i[11]-i[0]*i[7]*i[10]-i[4]*i[2]*i[11]+i[4]*i[3]*i[10]+i[8]*i[2]*i[7]-i[8]*i[3]*i[6];
    o[11]=-i[0]*i[5]*i[11]+i[0]*i[7]*i[9] +i[4]*i[1]*i[11]-i[4]*i[3]*i[9] -i[8]*i[1]*i[7]+i[8]*i[3]*i[5];
    o[15]=i[0]*i[5]*i[10]-i[0]*i[6]*i[9] -i[4]*i[1]*i[10]+i[4]*i[2]*i[9] +i[8]*i[1]*i[6]-i[8]*i[2]*i[5];
    float det=i[0]*o[0]+i[1]*o[4]+i[2]*o[8]+i[3]*o[12];
    if (fabsf(det) < EPSILON) return m4_identity();
    float inv_det=1.0f/det;
    for (int j=0; j<16; j++) o[j] *= inv_det;
    return inv;
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
    while (out_len + n > out_cap) { out_cap=out_cap?out_cap*2:65536; out_buf=realloc(out_buf,out_cap); }
    memcpy(out_buf+out_len, s, n); out_len += n;
}
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO,TIOCGWINSZ,&ws)==0 && ws.ws_col>0) { term_w=ws.ws_col; term_h=ws.ws_row; }
}
static void handle_sigwinch(int sig) { (void)sig; got_resize=1; }
static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios); raw_mode_enabled=0;
    }
}
static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO,&orig_termios); raw_mode_enabled=1; atexit(disable_raw_mode);
    signal(SIGWINCH, handle_sigwinch);
    struct termios raw=orig_termios;
    raw.c_iflag &= ~(BRKINT|ICRNL|INPCK|ISTRIP|IXON); raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8); raw.c_lflag &= ~(ECHO|ICANON|IEXTEN|ISIG);
    raw.c_cc[VMIN]=0; raw.c_cc[VTIME]=0;
    tcsetattr(STDIN_FILENO,TCSAFLUSH,&raw);
    write(STDOUT_FILENO,"\033[?25l\033[2J",10);
}
static void present_screen(void) {
    char seq[64]; int prev_fg=-1, prev_bg=-1;
    out_append("\033[H", 3);
    for (int y=0; y<term_h; y++) {
        for (int x=0; x<term_w; x++) {
            Cell *c = &screen[y*term_w+x];
            if (c->fg!=prev_fg || c->bg!=prev_bg) {
                int n=snprintf(seq,sizeof(seq),"\033[38;5;%dm\033[48;5;%dm",c->fg,c->bg);
                out_append(seq,n); prev_fg=c->fg; prev_bg=c->bg;
            }
            out_append(&c->ch,1);
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
    struct pollfd pfd={.fd=STDIN_FILENO,.events=POLLIN};
    if (poll(&pfd,1,0)<=0) return KEY_NONE;
    char c; if (read(STDIN_FILENO,&c,1)!=1) return KEY_NONE;
    if (c=='\033') {
        char seq[3];
        if (read(STDIN_FILENO,&seq[0],1)!=1) return KEY_ESC;
        if (read(STDIN_FILENO,&seq[1],1)!=1) return KEY_ESC;
        if (seq[0]=='[') { switch(seq[1]) { case 'A': return KEY_UP; case 'B': return KEY_DOWN; case 'C': return KEY_RIGHT; case 'D': return KEY_LEFT; } }
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
    int   timestep;
    int   max_timesteps;
    float total_reward;
    int   done;

    /* State: meaning depends on env type */
    float x, x_dot;           /* CartPole: cart position and velocity */
    float theta, theta_dot;   /* CartPole: pole angle; Pendulum: angle */
    float theta2, theta2_dot; /* Reacher: second joint */

    /* Visual transforms (computed by env_compute_transforms) */
    mat4 cart_tf;
    mat4 pole_tf;
    mat4 pole2_tf;
    mat4 target_tf;

    /* Reacher target */
    float target_x, target_y;
} Environment;

static Environment env;

/* ── Physical constants ──────────────────────────────────────────── */
#define CARTPOLE_GRAVITY    9.81f
#define CARTPOLE_CART_MASS  1.0f
#define CARTPOLE_POLE_MASS  0.1f
#define CARTPOLE_POLE_LENGTH 1.0f
#define CARTPOLE_FORCE_MAG  10.0f
#define CARTPOLE_DT         0.02f
#define CARTPOLE_X_THRESHOLD    2.4f
#define CARTPOLE_THETA_THRESHOLD (12.0f * (float)M_PI / 180.0f)

#define PENDULUM_MAX_SPEED  8.0f
#define PENDULUM_MAX_TORQUE 2.0f
#define PENDULUM_DT         0.05f
#define PENDULUM_G          9.81f
#define PENDULUM_M          1.0f
#define PENDULUM_L          1.0f

#define REACHER_LINK1_LENGTH 0.8f
#define REACHER_LINK2_LENGTH 0.6f
#define REACHER_DT          0.02f

static float randf(void) { return (float)rand() / RAND_MAX; }

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Reset environment to a random initial state
 *
 * CartPole:
 *   - All four state variables (x, x_dot, theta, theta_dot) start near 0
 *   - Typical: uniform random in [-0.05, 0.05]
 *   - max_timesteps = 500
 *
 * Pendulum:
 *   - theta starts near bottom (π) ± small noise
 *   - theta_dot starts with small random velocity
 *   - max_timesteps = 200
 *
 * Reacher:
 *   - theta, theta2: random angles in [-π/2, π/2]
 *   - Random target position within reachable area
 *   - max_timesteps = 200
 * ══════════════════════════════════════════════════════════════════ */
static void env_reset(void) {
    env.timestep = 0;
    env.total_reward = 0;
    env.done = 0;

    switch (env.type) {
        case ENV_CARTPOLE:
            /* TODO: Initialize CartPole state with small random values */
            env.max_timesteps = 500;
            env.x = env.x_dot = env.theta = env.theta_dot = 0;
            break;

        case ENV_PENDULUM:
            /* TODO: Initialize Pendulum starting near the bottom (theta ≈ π) */
            env.max_timesteps = 200;
            env.theta = (float)M_PI;
            env.theta_dot = 0;
            break;

        case ENV_REACHER:
            /* TODO: Initialize Reacher with random joint angles and random target */
            env.max_timesteps = 200;
            env.theta = env.theta2 = 0;
            env.theta_dot = env.theta2_dot = 0;
            env.target_x = 0.8f; env.target_y = 0.0f;
            break;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: CartPole dynamics step
 *
 * The CartPole equations of motion (from Barto, Sutton, Anderson 1983):
 *
 *   total_mass = cart_mass + pole_mass
 *   pole_mass_length = pole_mass * pole_length
 *   cos_theta = cos(theta)
 *   sin_theta = sin(theta)
 *
 *   force = action * CARTPOLE_FORCE_MAG  (action in [-1, 1])
 *
 *   temp = (force + pole_mass_length * theta_dot^2 * sin_theta) / total_mass
 *
 *   theta_acc = (g * sin_theta - cos_theta * temp)
 *             / (pole_length * (4/3 - pole_mass * cos_theta^2 / total_mass))
 *
 *   x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
 *
 * Semi-implicit Euler integration:
 *   x_dot   += x_acc   * dt
 *   x       += x_dot   * dt
 *   theta_dot += theta_acc * dt
 *   theta     += theta_dot * dt
 *
 * Reward: +1 for every timestep the pole stays up.
 *
 * Termination: |x| > 2.4 OR |theta| > 12° OR timestep >= max_timesteps
 * ══════════════════════════════════════════════════════════════════ */
static void cartpole_step(float action) {
    /* TODO: Implement CartPole dynamics */
    (void)action;
    env.timestep++;
    if (env.timestep >= env.max_timesteps) env.done = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Pendulum dynamics step
 *
 * The Pendulum equation (angle 0 = upright, π = hanging down):
 *
 *   torque = clamp(action * MAX_TORQUE, -MAX_TORQUE, MAX_TORQUE)
 *
 *   theta_acc = (-3*g)/(2*L) * sin(theta + π)
 *             + 3/(m*L^2) * torque
 *
 *   theta_dot += theta_acc * dt
 *   theta_dot  = clamp(theta_dot, -MAX_SPEED, MAX_SPEED)
 *   theta     += theta_dot * dt
 *
 *   Normalize theta to [-π, π]:
 *     while theta > π: theta -= 2π
 *     while theta < -π: theta += 2π
 *
 * Reward (negative cost to minimize):
 *   reward = -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
 *
 * No termination (runs until max_timesteps).
 * ══════════════════════════════════════════════════════════════════ */
static void pendulum_step(float action) {
    /* TODO: Implement Pendulum dynamics */
    (void)action;
    env.timestep++;
    if (env.timestep >= env.max_timesteps) env.done = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Reacher dynamics step
 *
 * Simple 2-joint planar arm. Each joint is independently torque-controlled.
 * For this simplified version, map a single action scalar to both joints:
 *   torque1 = action * 0.5
 *   torque2 = action * 0.3
 *
 * Dynamics with damping:
 *   theta_dot  += torque1 * dt
 *   theta      += theta_dot * dt
 *   theta_dot  *= 0.98  (damping)
 *   (same for theta2 with torque2)
 *
 * End effector position (forward kinematics of 2-link arm):
 *   ee_x = L1 * cos(theta) + L2 * cos(theta + theta2)
 *   ee_y = L1 * sin(theta) + L2 * sin(theta + theta2)
 *
 * Reward: negative distance from end effector to target
 *   reward = -sqrt((ee_x - target_x)^2 + (ee_y - target_y)^2)
 *
 * No termination until max_timesteps.
 * ══════════════════════════════════════════════════════════════════ */
static void reacher_step(float action) {
    /* TODO: Implement Reacher dynamics */
    (void)action;
    env.timestep++;
    if (env.timestep >= env.max_timesteps) env.done = 1;
}

static void env_step(float action) {
    if (env.done) return;
    switch (env.type) {
        case ENV_CARTPOLE:  cartpole_step(action);  break;
        case ENV_PENDULUM:  pendulum_step(action);  break;
        case ENV_REACHER:   reacher_step(action);   break;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Compute visual transforms for rendering
 *
 * Map env state variables to 4x4 matrices for each object.
 *
 * CartPole:
 *   cart_tf = m4_translate(x, 0, 0)
 *   pole_tf = cart_tf * m4_translate(0, 0.15, 0) * m4_rotate_z(-theta)
 *
 * Pendulum (pivot at origin):
 *   pole_tf = m4_rotate_z(-theta + π/2)
 *
 * Reacher (2 links from origin):
 *   pole_tf  = m4_rotate_z(theta)
 *   pole2_tf = pole_tf * m4_translate(LINK1_LENGTH, 0, 0) * m4_rotate_z(theta2)
 *   target_tf = m4_translate(target_x, target_y, 0)
 * ══════════════════════════════════════════════════════════════════ */
static void env_compute_transforms(void) {
    /* TODO: Compute env.cart_tf, env.pole_tf, env.pole2_tf, env.target_tf */
    env.cart_tf  = m4_identity();
    env.pole_tf  = m4_identity();
    env.pole2_tf = m4_identity();
    env.target_tf = m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Build SDF scene from environment state
 *
 * Use env.cart_tf, env.pole_tf, etc. to position objects:
 *   - Transform point p to object local space: m4_inverse(obj_tf) * p
 *   - Then evaluate SDF primitive in local space
 *
 * CartPole geometry:
 *   - Track: sdf_box centered at origin
 *   - Cart: sdf_box in cart_tf local space
 *   - Wheels: two sdf_sphere on cart
 *   - Pole: sdf_capsule along Y in pole_tf local space
 *   - Pole tip: sdf_sphere at pole tip
 *
 * Pendulum geometry:
 *   - Pivot: sdf_sphere at origin
 *   - Pole: sdf_capsule in pole_tf local space
 *   - Mass: sdf_sphere at end of pole
 *
 * Reacher geometry:
 *   - Base: sdf_cylinder at origin
 *   - Link 1: sdf_capsule in pole_tf local space
 *   - Joint: sdf_sphere at end of link 1
 *   - Link 2: sdf_capsule in pole2_tf local space
 *   - End effector: sdf_sphere at end of link 2
 *   - Target: sdf_sphere at target position
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere_f(vec3 p, vec3 c, float r) { return v3_len(v3_sub(p,c)) - r; }
static float sdf_box_f(vec3 p, vec3 c, vec3 hs) {
    vec3 d = v3_sub(v3_abs(v3_sub(p,c)), hs);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}
static float sdf_plane(vec3 p, vec3 n, float o) { return v3_dot(p,n) + o; }
static float sdf_capsule_x(vec3 p, float length, float radius) {
    float hlen = length*0.5f; p.x = fabsf(p.x) - hlen;
    if (p.x < 0) p.x = 0; return v3_len(p) - radius;
}
static float sdf_cylinder_y(vec3 p, vec3 center, float radius, float height) {
    vec3 rel = v3_sub(p, center);
    float dr = sqrtf(rel.x*rel.x + rel.z*rel.z) - radius;
    float dv = fabsf(rel.y) - height*0.5f;
    return fminf(fmaxf(dr,dv), 0.0f) + v3_len(v3(fmaxf(dr,0.0f), fmaxf(dv,0.0f), 0.0f));
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    float d;

    d = sdf_plane(p, v3(0,1,0), 0.5f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    switch (env.type) {
        case ENV_CARTPOLE: {
            /* TODO: Add CartPole geometry using env.cart_tf and env.pole_tf
             * Hint for pole: transform p to pole local space with m4_inverse(env.pole_tf)
             *   vec3 pole_p = m4_transform_point(m4_inverse(env.pole_tf), p);
             *   d = sdf_capsule_x(v3(pole_p.y - CARTPOLE_POLE_LENGTH*0.5f, pole_p.x, pole_p.z),
             *                     CARTPOLE_POLE_LENGTH, 0.05f);
             */
            (void)d;
            break;
        }
        case ENV_PENDULUM: {
            /* TODO: Add Pendulum geometry */
            break;
        }
        case ENV_REACHER: {
            /* TODO: Add Reacher geometry */
            break;
        }
    }

    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ── Ray marching + shading (from phases 3-4) ────────────────────── */
#define MAX_STEPS 48
#define MAX_DIST  30.0f
#define SURF_DIST 0.003f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;
static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result={0,0,ro,0}; float t=0;
    for (int i=0; i<MAX_STEPS && t<MAX_DIST; i++) {
        vec3 p=v3_add(ro,v3_scale(rd,t));
        SceneHit h=scene_sdf(p);
        if (h.dist<SURF_DIST) { result.hit=1; result.dist=t; result.point=p; result.material_id=h.material_id; return result; }
        t += h.dist;
    }
    result.dist=t; return result;
}
static vec3 calc_normal(vec3 p) {
    const float h=0.001f;
    return v3_norm(v3(scene_dist(v3(p.x+h,p.y,p.z))-scene_dist(v3(p.x-h,p.y,p.z)),
                      scene_dist(v3(p.x,p.y+h,p.z))-scene_dist(v3(p.x,p.y-h,p.z)),
                      scene_dist(v3(p.x,p.y,p.z+h))-scene_dist(v3(p.x,p.y,p.z-h))));
}
static const char LUMINANCE_RAMP[]=" .:-=+*#%@";
#define RAMP_LEN 10
static const uint8_t MATERIAL_COLORS[][3]={{2,3,2},{5,1,1},{1,3,5},{5,4,0},{5,5,0},{5,0,5},{0,5,5}};
static uint8_t color_216(int r,int g,int b) {
    r=r<0?0:(r>5?5:r); g=g<0?0:(g>5?5:g); b=b<0?0:(b>5?5:b); return 16+36*r+6*g+b;
}
typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;
static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result={' ',0,16};
    vec3 light_dir=v3_norm(v3(0.5f,1.0f,0.3f));
    vec3 half_vec=v3_norm(v3_add(light_dir,v3_neg(rd)));
    float ambient=0.2f, diffuse=fmaxf(0.0f,v3_dot(normal,light_dir))*0.6f;
    float specular=powf(fmaxf(0.0f,v3_dot(normal,half_vec)),16.0f)*0.3f;
    float fog=expf(-dist*0.05f);
    float lum=fminf(1.0f,fmaxf(0.0f,(ambient+diffuse+specular)*fog));
    result.ch=LUMINANCE_RAMP[(int)(lum*(RAMP_LEN-1)+0.5f)];
    int nm=sizeof(MATERIAL_COLORS)/sizeof(MATERIAL_COLORS[0]);
    int mi=mat_id<nm?mat_id:0; float cs=0.3f+lum*0.7f;
    result.fg=color_216((int)(MATERIAL_COLORS[mi][0]*cs),(int)(MATERIAL_COLORS[mi][1]*cs),(int)(MATERIAL_COLORS[mi][2]*cs));
    (void)p; return result;
}

typedef struct {
    vec3 target; float distance, azimuth, elevation, fov;
    float smooth_az, smooth_el, smooth_dist;
} OrbitCamera;
static vec3 orbit_camera_pos(OrbitCamera *cam) {
    float y=sinf(cam->smooth_el)*cam->smooth_dist;
    float xzd=cosf(cam->smooth_el)*cam->smooth_dist;
    return v3_add(cam->target, v3(sinf(cam->smooth_az)*xzd, y, cosf(cam->smooth_az)*xzd));
}
static void orbit_camera_get_ray(OrbitCamera *cam, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 pos=orbit_camera_pos(cam);
    vec3 fwd=v3_norm(v3_sub(cam->target,pos));
    vec3 right=v3_norm(v3_cross(fwd,v3(0,1,0)));
    vec3 up=v3_cross(right,fwd);
    float hf=tanf(cam->fov*0.5f);
    *ro=pos; *rd=v3_norm(v3_add(v3_add(fwd,v3_scale(right,u*hf)),v3_scale(up,v*hf)));
}
static void orbit_camera_update(OrbitCamera *cam, float dt) {
    float s=1.0f-expf(-10.0f*dt);
    cam->smooth_az=lerpf(cam->smooth_az,cam->azimuth,s);
    cam->smooth_el=lerpf(cam->smooth_el,cam->elevation,s);
    cam->smooth_dist=lerpf(cam->smooth_dist,cam->distance,s);
}
static void render_frame(OrbitCamera *cam) {
    float aspect=(float)term_w/(term_h*2.0f);
    for (int y=0; y<term_h; y++) {
        for (int x=0; x<term_w; x++) {
            float u=(2.0f*x/term_w-1.0f)*aspect, v=-(2.0f*y/term_h-1.0f);
            vec3 ro, rd; orbit_camera_get_ray(cam, u, v, &ro, &rd);
            RayResult hit=ray_march(ro,rd);
            Cell *cell=&screen[y*term_w+x];
            if (hit.hit) {
                vec3 n=calc_normal(hit.point);
                ShadeResult shade=shade_point(hit.point,rd,n,hit.material_id,hit.dist);
                cell->ch=shade.ch; cell->fg=shade.fg; cell->bg=shade.bg;
            } else {
                float sky_t=(v+1.0f)*0.5f; int sc=17+(int)(sky_t*4);
                cell->ch=' '; cell->fg=sc; cell->bg=sc;
            }
        }
    }
}

static double get_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void sleep_ms(int ms) {
    struct timespec ts={.tv_sec=ms/1000,.tv_nsec=(ms%1000)*1000000L};
    nanosleep(&ts,NULL);
}

static void draw_hud(double fps, int paused) {
    char hud[256];
    const char *env_names[] = {"cartpole", "pendulum", "reacher"};
    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " Phase 7: %s | t=%d reward=%.1f FPS:%.0f %s"
        " | [L/R] force [R] reset [P] pause [WASD] cam [Q] quit ",
        env_names[env.type], env.timestep, env.total_reward, fps,
        paused ? "[PAUSED]" : "");
    write(STDOUT_FILENO, hud, strlen(hud));
}

int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));

    env.type = ENV_CARTPOLE;
    if (argc > 1) {
        if (strcmp(argv[1], "pendulum") == 0) env.type = ENV_PENDULUM;
        else if (strcmp(argv[1], "reacher") == 0) env.type = ENV_REACHER;
    }

    env_reset();

    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target=v3(0,0,0), .distance=8.0f, .azimuth=0.2f, .elevation=0.3f,
        .fov=(float)M_PI/3.0f, .smooth_az=0.2f, .smooth_el=0.3f, .smooth_dist=8.0f
    };

    double last_time = get_time(), fps_time = last_time;
    int fps_count = 0; double fps_display = 0;
    int running = 1, paused = 0;
    float action = 0;

    while (running) {
        double now = get_time();
        float dt = (float)(now - last_time); last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0; get_term_size(); free(screen);
            screen = calloc(term_w * term_h, sizeof(Cell));
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        action = 0;
        int key;
        while ((key = read_key()) != KEY_NONE) {
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'p': case 'P': paused = !paused; break;
                case 'r': case 'R': env_reset(); break;
                case KEY_LEFT:  action = -1.0f; break;
                case KEY_RIGHT: action =  1.0f; break;
                case 'w': case 'W': cam.elevation += 0.1f; break;
                case 's': case 'S': cam.elevation -= 0.1f; break;
                case 'a': case 'A': cam.azimuth   -= 0.1f; break;
                case 'd': case 'D': cam.azimuth   += 0.1f; break;
                case '+': case '=': cam.distance -= 0.5f; break;
                case '-': case '_': cam.distance += 0.5f; break;
            }
        }

        cam.elevation = fmaxf(-(float)M_PI/2+0.1f, fminf((float)M_PI/2-0.1f, cam.elevation));
        cam.distance = fmaxf(3.0f, fminf(20.0f, cam.distance));

        if (!paused) {
            env_step(action);
            if (env.done) { sleep_ms(500); env_reset(); }
        }

        env_compute_transforms();
        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0; fps_time = now;
        }
        draw_hud(fps_display, paused);

        sleep_ms(20);
    }

    free(screen); free(out_buf);
    return 0;
}
