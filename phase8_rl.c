/*
 * phase8_rl.c — RL Training: CartPole Stick Balancer
 *
 * Uses the physics engine and terminal SDF renderer from phase7 to
 * train a cart-pole balancer with Random Hill Climbing — the simplest
 * RL algorithm that can actually solve CartPole.
 *
 * Algorithm: Random Hill Climbing
 *   Policy:  action = sign(w · obs)   where obs = [x, x_dot, theta, theta_dot]
 *   Step:    sample w' = best_w + noise*N(0,I), keep if score(w') >= best
 *   Restart: randomize weights after RESTART_THRESH consecutive failures
 *
 * Build:  gcc -O2 -o phase8_rl phase8_rl.c -lm
 * Run:    ./phase8_rl
 * Keys:   [T] toggle training  [R] reset all  [WASD/+/-] camera  [Q] quit
 *
 * YOUR TASKS (marked TODO below):
 *   #1 cp_step()        — CartPole dynamics (equations provided)
 *   #2 eval_policy()    — run one episode with a linear sign policy
 *   #3 rl_init()        — initialize RL state with random weights
 *   #4 rl_train_step()  — one hill-climbing update step
 *   #5 vis_compute_tf() — compute cart_tf and pole_tf from vis state
 *   #6 scene_sdf()      — build the CartPole SDF scene
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
#define M4(mat, row, col) ((mat).m[(col)*4+(row)])

static inline vec3 v3(float x, float y, float z) { return (vec3){x,y,z}; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x*s,v.y*s,v.z*s); }
static inline vec3 v3_neg(vec3 v) { return v3(-v.x,-v.y,-v.z); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v,v)); }
static inline vec3 v3_norm(vec3 v) {
    float l=v3_len(v); return l<EPSILON?v3(0,0,0):v3_scale(v,1.0f/l);
}
static inline vec3 v3_abs(vec3 v) { return v3(fabsf(v.x),fabsf(v.y),fabsf(v.z)); }
static inline vec3 v3_max(vec3 a, vec3 b) {
    return v3(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));
}
static inline float v3_max_comp(vec3 v) { return fmaxf(v.x,fmaxf(v.y,v.z)); }
static inline float lerpf(float a, float b, float t) { return a+(b-a)*t; }

static mat4 m4_identity(void) {
    mat4 m={0}; m.m[0]=m.m[5]=m.m[10]=m.m[15]=1.0f; return m;
}
static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r={0};
    for (int c=0;c<4;c++) for (int row=0;row<4;row++) {
        float s=0; for (int k=0;k<4;k++) s+=M4(a,row,k)*M4(b,k,c);
        M4(r,row,c)=s;
    }
    return r;
}
static mat4 m4_translate(vec3 t) {
    mat4 m=m4_identity();
    M4(m,0,3)=t.x; M4(m,1,3)=t.y; M4(m,2,3)=t.z; return m;
}
static mat4 m4_rotate_z(float rad) {
    mat4 m=m4_identity(); float c=cosf(rad),s=sinf(rad);
    M4(m,0,0)=c; M4(m,0,1)=-s; M4(m,1,0)=s; M4(m,1,1)=c; return m;
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
    if (fabsf(det)<EPSILON) return m4_identity();
    float id=1.0f/det;
    for (int j=0;j<16;j++) o[j]*=id;
    return inv;
}

/* ══════════════════════════════════════════════════════════════════
 * TERMINAL (phase1 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static int term_w=80, term_h=24;
static struct termios orig_termios;
static int raw_mode_enabled=0;
static volatile sig_atomic_t got_resize=0;

typedef struct { char ch; uint8_t fg; uint8_t bg; } Cell;
static Cell *screen=NULL;
static char *out_buf=NULL;
static int out_cap=0, out_len=0;

static void out_flush(void) {
    if (out_len>0){write(STDOUT_FILENO,out_buf,out_len);out_len=0;}
}
static void out_append(const char *s, int n) {
    while (out_len+n>out_cap){out_cap=out_cap?out_cap*2:65536;out_buf=realloc(out_buf,out_cap);}
    memcpy(out_buf+out_len,s,n); out_len+=n;
}
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO,TIOCGWINSZ,&ws)==0&&ws.ws_col>0){term_w=ws.ws_col;term_h=ws.ws_row;}
}
static void handle_sigwinch(int sig){(void)sig;got_resize=1;}
static void disable_raw_mode(void) {
    if (raw_mode_enabled){
        write(STDOUT_FILENO,"\033[?25h\033[0m\033[2J\033[H",18);
        tcsetattr(STDIN_FILENO,TCSAFLUSH,&orig_termios);
        raw_mode_enabled=0;
    }
}
static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO,&orig_termios); raw_mode_enabled=1; atexit(disable_raw_mode);
    signal(SIGWINCH,handle_sigwinch);
    struct termios raw=orig_termios;
    raw.c_iflag&=~(BRKINT|ICRNL|INPCK|ISTRIP|IXON); raw.c_oflag&=~(OPOST);
    raw.c_cflag|=(CS8); raw.c_lflag&=~(ECHO|ICANON|IEXTEN|ISIG);
    raw.c_cc[VMIN]=0; raw.c_cc[VTIME]=0;
    tcsetattr(STDIN_FILENO,TCSAFLUSH,&raw);
    write(STDOUT_FILENO,"\033[?25l\033[2J",10);
}
static void present_screen(void) {
    char seq[64]; int prev_fg=-1, prev_bg=-1;
    out_append("\033[H",3);
    for (int y=0;y<term_h;y++) {
        for (int x=0;x<term_w;x++) {
            Cell *c=&screen[y*term_w+x];
            if (c->fg!=prev_fg||c->bg!=prev_bg){
                int n=snprintf(seq,sizeof(seq),"\033[38;5;%dm\033[48;5;%dm",c->fg,c->bg);
                out_append(seq,n); prev_fg=c->fg; prev_bg=c->bg;
            }
            out_append(&c->ch,1);
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
    struct pollfd pfd={.fd=STDIN_FILENO,.events=POLLIN};
    if (poll(&pfd,1,0)<=0) return KEY_NONE;
    char c; if (read(STDIN_FILENO,&c,1)!=1) return KEY_NONE;
    if (c=='\033'){
        char seq[3];
        if (read(STDIN_FILENO,&seq[0],1)!=1) return KEY_ESC;
        if (read(STDIN_FILENO,&seq[1],1)!=1) return KEY_ESC;
        if (seq[0]=='['){switch(seq[1]){
            case 'A': return KEY_UP;  case 'B': return KEY_DOWN;
            case 'C': return KEY_RIGHT; case 'D': return KEY_LEFT;
        }}
        return KEY_ESC;
    }
    return (int)(unsigned char)c;
}

/* ══════════════════════════════════════════════════════════════════
 * CARTPOLE PHYSICS
 * ══════════════════════════════════════════════════════════════════ */

#define CP_GRAVITY    9.81f
#define CP_CART_MASS  1.0f
#define CP_POLE_MASS  0.1f
#define CP_POLE_LEN   1.0f
#define CP_FORCE      10.0f
#define CP_DT         0.02f
#define CP_X_THRESH   2.4f
#define CP_TH_THRESH  (12.0f * (float)M_PI / 180.0f)
#define CP_MAX_STEPS  500

typedef struct {
    float x, x_dot, theta, theta_dot;
    int   timestep;
    int   done;
} CPState;

static float randf(void) { return (float)rand() / (float)RAND_MAX; }

static void cp_reset(CPState *s) {
    /* Small random initial state (uniform [-0.05, 0.05] per variable) */
    s->x         = (randf()-0.5f) * 0.1f;
    s->x_dot     = (randf()-0.5f) * 0.1f;
    s->theta     = (randf()-0.5f) * 0.1f;
    s->theta_dot = (randf()-0.5f) * 0.1f;
    s->timestep  = 0;
    s->done      = 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: CartPole dynamics step
 *
 * Equations of motion (Barto, Sutton & Anderson 1983):
 *
 *   total_mass = CP_CART_MASS + CP_POLE_MASS
 *   ml         = CP_POLE_MASS * CP_POLE_LEN
 *   force      = action * CP_FORCE     (action is -1.0 or +1.0)
 *
 *   tmp = (force + ml * theta_dot^2 * sin(theta)) / total_mass
 *
 *   theta_acc = (CP_GRAVITY * sin(theta) - cos(theta) * tmp)
 *             / (CP_POLE_LEN * (4.0/3.0 - CP_POLE_MASS * cos(theta)^2 / total_mass))
 *
 *   x_acc = tmp - ml * theta_acc * cos(theta) / total_mass
 *
 *   Semi-implicit Euler integration:
 *     x_dot     += x_acc     * CP_DT;   x     += x_dot     * CP_DT
 *     theta_dot += theta_acc * CP_DT;   theta += theta_dot * CP_DT
 *
 *   Increment timestep.
 *   Set done=1 if: |x| > CP_X_THRESH  OR  |theta| > CP_TH_THRESH
 *                  OR  timestep >= CP_MAX_STEPS
 * ══════════════════════════════════════════════════════════════════ */
static void cp_step(CPState *s, float action) {
    /* TODO: implement CartPole dynamics */
    (void)action;
    s->timestep++;
    if (s->timestep >= CP_MAX_STEPS) s->done = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * RL: RANDOM HILL CLIMBING
 *
 * Policy: action = sign(w[0]*x + w[1]*x_dot + w[2]*theta + w[3]*theta_dot)
 *
 * Each training step:
 *   1. Perturb: w' = best_w + noise * gaussian_noise(4)
 *   2. Evaluate score (timesteps survived in one episode)
 *   3. Accept w' if score >= best_score
 *   4. Restart randomly after RESTART_THRESH consecutive failures
 * ══════════════════════════════════════════════════════════════════ */

#define HIST_LEN        80    /* episode score history for learning curve  */
#define N_TRAIN_FRAME   20    /* training episodes per rendered frame      */
#define NOISE_INIT      0.5f  /* initial perturbation scale                */
#define NOISE_MIN       0.05f /* minimum noise (fine-tuning floor)         */
#define RESTART_THRESH  300   /* failures before random restart            */

typedef struct {
    float best_w[4];         /* best policy weights found so far          */
    float best_score;        /* best episode score (timesteps survived)   */
    float hist[HIST_LEN];    /* ring buffer of recent episode scores      */
    int   hist_head;         /* ring buffer write position                */
    int   hist_n;            /* number of valid entries in hist           */
    int   total_ep;          /* total episodes run                        */
    int   no_improve;        /* consecutive episodes without improvement  */
    float noise;             /* current perturbation scale                */
    int   solved;            /* 1 if best_score >= 475 (near-perfect)     */
    int   training;          /* 1 = actively training, 0 = paused         */
} RLState;

static RLState rl;

static float gaussian_rand(void) {
    /* Box-Muller transform: uniform → standard normal */
    float u = randf() + 1e-7f, v = randf();
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * (float)M_PI * v);
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Evaluate a policy
 *
 * Run one full episode using the linear sign policy:
 *   while not done:
 *     q = w[0]*s.x + w[1]*s.x_dot + w[2]*s.theta + w[3]*s.theta_dot
 *     action = (q >= 0) ? 1.0 : -1.0
 *     cp_step(&s, action)
 *
 * Return the number of timesteps survived (s.timestep).
 * ══════════════════════════════════════════════════════════════════ */
static float eval_policy(float *w) {
    /* TODO: run a full episode with the linear sign policy, return timesteps */
    (void)w;
    return 0.0f;
}

static void rl_record(float score) {
    rl.hist[rl.hist_head % HIST_LEN] = score;
    rl.hist_head++;
    if (rl.hist_n < HIST_LEN) rl.hist_n++;
    rl.total_ep++;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Initialize RL state
 *
 * - Set best_w to random values in, e.g., [-1, 1]
 * - Evaluate best_w using eval_policy() to get initial best_score
 * - Reset hist_head=0, hist_n=0; call rl_record(best_score)
 * - Reset no_improve=0, noise=NOISE_INIT
 * - Set solved = (best_score >= 475)
 * - Set training = 1
 * ══════════════════════════════════════════════════════════════════ */
static void rl_init(void) {
    /* TODO: initialize RL state */
    memset(&rl, 0, sizeof(rl));
    rl.training = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: One hill-climbing training step
 *
 * Algorithm:
 *   1. Sample candidate: tw[i] = best_w[i] + noise * gaussian_rand()
 *   2. sc = eval_policy(tw)
 *   3. rl_record(sc)
 *   4. if sc >= best_score:
 *        best_score = sc
 *        best_w = tw
 *        no_improve = 0
 *        if noise > NOISE_MIN: noise *= 0.998  (slowly shrink noise)
 *        if sc >= 475: solved = 1
 *      else:
 *        no_improve++
 *        if no_improve >= RESTART_THRESH:
 *          randomize best_w (e.g., range [-2, 2])
 *          re-evaluate best_score
 *          reset no_improve=0, noise=NOISE_INIT
 * ══════════════════════════════════════════════════════════════════ */
static void rl_train_step(void) {
    /* TODO: implement one hill-climbing step */
}

/* ══════════════════════════════════════════════════════════════════
 * VISUAL ENV — global CartPole rendered each frame
 * ══════════════════════════════════════════════════════════════════ */

static CPState vis;        /* visual CartPole state (rendered each frame) */
static mat4    g_cart_tf;  /* cart transform — set by vis_compute_tf()    */
static mat4    g_pole_tf;  /* pole transform — set by vis_compute_tf()    */
static int     vis_steps;  /* steps survived in current visual episode    */

static void vis_reset(void) { cp_reset(&vis); vis_steps = 0; }

static void vis_tick(void) {
    /* Step visual env using current best policy */
    float q = rl.best_w[0]*vis.x     + rl.best_w[1]*vis.x_dot
            + rl.best_w[2]*vis.theta  + rl.best_w[3]*vis.theta_dot;
    cp_step(&vis, q >= 0.0f ? 1.0f : -1.0f);
    vis_steps++;
    if (vis.done) vis_reset();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Compute visual transforms
 *
 * These drive the SDF renderer — call this before render_frame().
 *
 * cart_tf: translate along x-axis by vis.x
 *   g_cart_tf = m4_translate(v3(vis.x, 0, 0))
 *
 * pole_tf: pivot from top of cart, rotate Z by -vis.theta
 *   In pole-local space the pole runs along +Y.
 *   theta=0 → upright,  theta>0 → tilts right (+X direction)
 *
 *   g_pole_tf = g_cart_tf
 *             * m4_translate(v3(0, 0.15, 0))   (pivot at cart top)
 *             * m4_rotate_z(-vis.theta)
 * ══════════════════════════════════════════════════════════════════ */
static void vis_compute_tf(void) {
    /* TODO: compute g_cart_tf and g_pole_tf */
    g_cart_tf = m4_identity();
    g_pole_tf = m4_identity();
}

/* ══════════════════════════════════════════════════════════════════
 * SDF PRIMITIVES (phases 3-6 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere_f(vec3 p, vec3 c, float r) { return v3_len(v3_sub(p,c))-r; }
static float sdf_box_f(vec3 p, vec3 c, vec3 hs) {
    vec3 d=v3_sub(v3_abs(v3_sub(p,c)),hs);
    return fminf(v3_max_comp(d),0.0f)+v3_len(v3_max(d,v3(0,0,0)));
}
static float sdf_plane(vec3 p, vec3 n, float o) { return v3_dot(p,n)+o; }
static float sdf_capsule_x(vec3 p, float length, float radius) {
    /* Capsule aligned along X, centered at origin */
    float hl=length*0.5f; p.x=fabsf(p.x)-hl;
    if (p.x<0) p.x=0; return v3_len(p)-radius;
}

typedef struct { float dist; int material_id; } SceneHit;

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: SDF Scene — CartPole geometry
 *
 * Material IDs:
 *   0 = ground (dark green)    1 = track rail (gray)
 *   2 = cart body (blue)       3 = wheels (dark gray)
 *   4 = pole (orange)          5 = pole tip (red)
 *
 * Geometry to add (on top of the ground plane which is given):
 *
 *   Track rail: sdf_box_f(p, center=(0,-0.48,0), half_size=(CP_X_THRESH+0.5, 0.04, 0.1))
 *
 *   Cart: transform p to cart local space, then sdf_box_f with half-size (0.3, 0.15, 0.15)
 *     vec3 cp = m4_transform_point(m4_inverse(g_cart_tf), p)
 *
 *   Wheels: four sdf_sphere_f on cp at corners (±0.2, -0.15, ±0.12), radius 0.06
 *
 *   Pole: transform p to pole local space, capsule along Y using the X-axis trick:
 *     vec3 pp = m4_transform_point(m4_inverse(g_pole_tf), p)
 *     d = sdf_capsule_x(v3(pp.y - CP_POLE_LEN*0.5, pp.x, pp.z), CP_POLE_LEN, 0.04)
 *     (swapping Y→X makes the capsule run along Y in pole-local space)
 *
 *   Pole tip: sdf_sphere_f at the top of the pole
 *     vec3 tip = m4_transform_point(g_pole_tf, v3(0, CP_POLE_LEN, 0))
 * ══════════════════════════════════════════════════════════════════ */
static SceneHit scene_sdf(vec3 p) {
    SceneHit h = {1e10f, 0};
    float d;

    /* Ground plane (y = -0.5) */
    d = sdf_plane(p, v3(0,1,0), 0.5f);
    if (d < h.dist) { h.dist=d; h.material_id=0; }

    /* TODO: add track rail, cart, wheels, pole, pole tip */
    (void)d;

    return h;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

/* ══════════════════════════════════════════════════════════════════
 * RAY MARCHING + SHADING (phases 3-4 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_STEPS 48
#define MAX_DIST  30.0f
#define SURF_DIST 0.003f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;

static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult r={0,0,ro,0}; float t=0;
    for (int i=0;i<MAX_STEPS&&t<MAX_DIST;i++){
        vec3 p=v3_add(ro,v3_scale(rd,t));
        SceneHit h=scene_sdf(p);
        if (h.dist<SURF_DIST){r.hit=1;r.dist=t;r.point=p;r.material_id=h.material_id;return r;}
        t+=h.dist;
    }
    r.dist=t; return r;
}

static vec3 calc_normal(vec3 p) {
    const float h=0.001f;
    return v3_norm(v3(
        scene_dist(v3(p.x+h,p.y,p.z))-scene_dist(v3(p.x-h,p.y,p.z)),
        scene_dist(v3(p.x,p.y+h,p.z))-scene_dist(v3(p.x,p.y-h,p.z)),
        scene_dist(v3(p.x,p.y,p.z+h))-scene_dist(v3(p.x,p.y,p.z-h))));
}

static const char LRAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10

/* RGB in 0-5 range for xterm-256 216-color cube */
static const uint8_t MAT_COLORS[][3] = {
    {1,2,1}, /* 0 ground     (dark green)  */
    {3,3,3}, /* 1 track rail (gray)        */
    {0,2,5}, /* 2 cart       (blue)        */
    {2,2,2}, /* 3 wheels     (dark gray)   */
    {5,3,0}, /* 4 pole       (orange)      */
    {5,0,0}, /* 5 pole tip   (red)         */
};

static uint8_t color216(int r, int g, int b) {
    r=r<0?0:(r>5?5:r); g=g<0?0:(g>5?5:g); b=b<0?0:(b>5?5:b);
    return (uint8_t)(16+36*r+6*g+b);
}

typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;

static ShadeResult shade_point(vec3 p, vec3 rd, vec3 n, int mat, float dist) {
    ShadeResult sr={' ',0,16};
    vec3 ld=v3_norm(v3(0.5f,1.0f,0.3f));
    vec3 hv=v3_norm(v3_add(ld,v3_neg(rd)));
    float amb=0.2f, diff=fmaxf(0,v3_dot(n,ld))*0.6f;
    float spec=powf(fmaxf(0,v3_dot(n,hv)),16)*0.3f;
    float fog=expf(-dist*0.05f);
    float lum=fminf(1.0f,fmaxf(0.0f,(amb+diff+spec)*fog));
    sr.ch=LRAMP[(int)(lum*(RAMP_LEN-1)+0.5f)];
    int nm=(int)(sizeof(MAT_COLORS)/sizeof(MAT_COLORS[0]));
    int mi=mat<nm?mat:0; float cs=0.3f+lum*0.7f;
    sr.fg=color216((int)(MAT_COLORS[mi][0]*cs),
                   (int)(MAT_COLORS[mi][1]*cs),
                   (int)(MAT_COLORS[mi][2]*cs));
    (void)p; return sr;
}

/* ── Orbit camera (phase 4 — already implemented) ───────────────── */
typedef struct {
    vec3 target; float distance, azimuth, elevation, fov;
    float saz, sel, sdist;
} OrbitCamera;

static vec3 cam_pos(OrbitCamera *c) {
    float y=sinf(c->sel)*c->sdist, xzd=cosf(c->sel)*c->sdist;
    return v3_add(c->target, v3(sinf(c->saz)*xzd, y, cosf(c->saz)*xzd));
}
static void cam_ray(OrbitCamera *c, float u, float v, vec3 *ro, vec3 *rd) {
    vec3 pos=cam_pos(c);
    vec3 fwd=v3_norm(v3_sub(c->target,pos));
    vec3 right=v3_norm(v3_cross(fwd,v3(0,1,0)));
    vec3 up=v3_cross(right,fwd);
    float hf=tanf(c->fov*0.5f);
    *ro=pos; *rd=v3_norm(v3_add(v3_add(fwd,v3_scale(right,u*hf)),v3_scale(up,v*hf)));
}
static void cam_update(OrbitCamera *c, float dt) {
    float s=1.0f-expf(-10.0f*dt);
    c->saz=lerpf(c->saz,c->azimuth,s);
    c->sel=lerpf(c->sel,c->elevation,s);
    c->sdist=lerpf(c->sdist,c->distance,s);
}
static void render_frame(OrbitCamera *c) {
    float asp=(float)term_w/(term_h*2.0f);
    for (int y=0;y<term_h;y++) {
        for (int x=0;x<term_w;x++) {
            float u=(2.0f*x/term_w-1.0f)*asp, v=-(2.0f*y/term_h-1.0f);
            vec3 ro,rd; cam_ray(c,u,v,&ro,&rd);
            RayResult hit=ray_march(ro,rd);
            Cell *cell=&screen[y*term_w+x];
            if (hit.hit){
                vec3 n=calc_normal(hit.point);
                ShadeResult sr=shade_point(hit.point,rd,n,hit.material_id,hit.dist);
                cell->ch=sr.ch; cell->fg=sr.fg; cell->bg=sr.bg;
            } else {
                float sky_t=(v+1.0f)*0.5f; int sc=17+(int)(sky_t*4);
                cell->ch=' '; cell->fg=(uint8_t)sc; cell->bg=(uint8_t)sc;
            }
        }
    }
}

static double get_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}
static void sleep_ms(int ms) {
    struct timespec ts={.tv_sec=ms/1000,.tv_nsec=(ms%1000)*1000000L};
    nanosleep(&ts,NULL);
}

/* ══════════════════════════════════════════════════════════════════
 * HUD — training stats and learning curve (already implemented)
 *
 * The bottom row shows a live learning curve using Unicode block chars:
 *   ' '=0  ▁=1/8  ▂=2/8  ▃=3/8  ▄=4/8  ▅=5/8  ▆=6/8  ▇=7/8  █=8/8
 * Each column = one episode, height = score / 500.
 * Watch the bars grow taller as rl_train_step() finds better weights!
 * ══════════════════════════════════════════════════════════════════ */
static void draw_hud(double fps) {
    char buf[512]; int n;

    n = snprintf(buf, sizeof(buf),
        "\033[1;1H\033[48;5;16m\033[38;5;%dm"
        " Phase 8 RL | Ep:%-6d | Best:%-3.0f/500 | Noise:%.3f | FPS:%.0f | %s ",
        rl.solved ? 46 : 226,
        rl.total_ep, rl.best_score, rl.noise, fps,
        rl.solved ? "*** SOLVED! ***" : (rl.training ? "Training..." : "Paused"));
    write(STDOUT_FILENO, buf, n);

    n = snprintf(buf, sizeof(buf),
        "\033[%d;1H\033[48;5;16m\033[38;5;245m"
        " Best w=[%+.3f, %+.3f, %+.3f, %+.3f] (x, x_dot, theta, theta_dot)"
        "  vis:%d steps   ",
        term_h-2,
        rl.best_w[0], rl.best_w[1], rl.best_w[2], rl.best_w[3],
        vis_steps);
    write(STDOUT_FILENO, buf, n);

    n = snprintf(buf, sizeof(buf),
        "\033[%d;1H\033[48;5;16m\033[38;5;240m"
        " [T] %s   [R] reset all   [WASD] camera   [+/-] zoom   [Q] quit   ",
        term_h-1,
        rl.training ? "pause training" : "resume training");
    write(STDOUT_FILENO, buf, n);

    n = snprintf(buf, sizeof(buf),
        "\033[%d;1H\033[48;5;16m\033[38;5;%dm[",
        term_h, rl.solved ? 46 : 214);
    write(STDOUT_FILENO, buf, n);

    int avail = term_w - 2;
    int cnt   = rl.hist_n < avail ? rl.hist_n : avail;
    for (int i = 0; i < cnt; i++) {
        int idx = ((rl.hist_head - cnt + i) % HIST_LEN + HIST_LEN) % HIST_LEN;
        float sc = rl.hist[idx];
        int level = (int)(sc / CP_MAX_STEPS * 8.0f + 0.5f);
        if (level == 0) {
            write(STDOUT_FILENO, " ", 1);
        } else {
            unsigned char blk[3] = {0xE2, 0x96, (unsigned char)(0x80 + level)};
            write(STDOUT_FILENO, blk, 3);
        }
    }
    for (int i = cnt; i < avail; i++) write(STDOUT_FILENO, " ", 1);
    write(STDOUT_FILENO, "]", 1);
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN (already implemented — framework wires everything together)
 * ══════════════════════════════════════════════════════════════════ */

int main(void) {
    srand((unsigned)time(NULL));
    rl_init();
    vis_reset();

    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target   = v3(0, 0, 0),
        .distance = 9.0f, .azimuth = 0.3f, .elevation = 0.25f,
        .fov      = (float)M_PI / 3.0f,
        .saz=0.3f, .sel=0.25f, .sdist=9.0f
    };

    double last_time = get_time(), fps_time = last_time;
    int fps_count = 0; double fps_disp = 0;
    int running = 1;

    while (running) {
        double now = get_time();
        float dt = (float)(now - last_time); last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0; get_term_size();
            free(screen); screen = calloc(term_w*term_h, sizeof(Cell));
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        int key;
        while ((key = read_key()) != KEY_NONE) {
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running=0; break;
                case 't': case 'T': rl.training = !rl.training; break;
                case 'r': case 'R': rl_init(); vis_reset(); break;
                case 'w': case 'W': cam.elevation += 0.1f; break;
                case 's': case 'S': cam.elevation -= 0.1f; break;
                case 'a': case 'A': cam.azimuth   -= 0.1f; break;
                case 'd': case 'D': cam.azimuth   += 0.1f; break;
                case '+': case '=': cam.distance  -= 0.5f; break;
                case '-': case '_': cam.distance  += 0.5f; break;
            }
        }
        cam.elevation = fmaxf(-(float)M_PI/2+0.1f, fminf((float)M_PI/2-0.1f, cam.elevation));
        cam.distance  = fmaxf(3.0f, fminf(20.0f, cam.distance));

        /* ── RL training: N fast (invisible) episodes per frame ── */
        if (rl.training)
            for (int i = 0; i < N_TRAIN_FRAME; i++)
                rl_train_step();

        /* ── Step visual env one tick using best found policy ── */
        vis_tick();

        /* ── Render & display ── */
        vis_compute_tf();
        cam_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_disp = fps_count / (now - fps_time);
            fps_count = 0; fps_time = now;
        }
        draw_hud(fps_disp);

        sleep_ms(20);
    }

    free(screen); free(out_buf);
    return 0;
}
