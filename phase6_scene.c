/*
 * phase6_scene.c — Scene Graph + URDF Parser
 *
 * EXERCISE: Build a minimal XML/URDF parser, represent a robot as a
 * kinematic tree, and compute forward kinematics.
 *
 * Build: gcc -O2 -o phase6_scene phase6_scene.c -lm
 * Run:   ./phase6_scene robot.urdf
 * Controls: [1-9] select joint, [+/-] adjust angle, [WASD] camera, [Q] quit
 *
 * LEARNING GOALS:
 * - Parse XML attributes by scanning character-by-character
 * - Build a tree of links and joints from URDF data
 * - Compute forward kinematics: propagate transforms from root to leaves
 * - Render articulated robot geometry using per-link world transforms
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
#include <ctype.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * 3D MATH (phases 1-2 — already implemented)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float w, x, y, z; } quat;
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

static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    axis = v3_norm(axis); float half = rad*0.5f, s = sinf(half);
    return (quat){cosf(half), axis.x*s, axis.y*s, axis.z*s};
}
static inline quat q_mul(quat a, quat b) {
    return (quat){a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z, a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
                  a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x, a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};
}
static inline quat q_norm(quat q) {
    float len = sqrtf(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f/len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}
static vec3 q_rotate(quat q, vec3 v) {
    vec3 qv = v3(q.x, q.y, q.z);
    vec3 uv = v3_cross(qv, v), uuv = v3_cross(qv, uv);
    return v3_add(v, v3_scale(v3_add(v3_scale(uv, q.w), uuv), 2.0f));
}

static inline mat4 m4_identity(void) {
    mat4 m = {0}; m.m[0]=m.m[5]=m.m[10]=m.m[15]=1.0f; return m;
}
static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) sum += M4(a,row,k)*M4(b,k,c);
            M4(r,row,c) = sum;
        }
    return r;
}
static mat4 m4_translate(vec3 t) {
    mat4 m = m4_identity();
    M4(m,0,3)=t.x; M4(m,1,3)=t.y; M4(m,2,3)=t.z; return m;
}
static mat4 m4_rotate_axis(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float c=cosf(rad), s=sinf(rad), t=1.0f-c;
    float x=axis.x, y=axis.y, z=axis.z;
    mat4 m = m4_identity();
    M4(m,0,0)=t*x*x+c;   M4(m,0,1)=t*x*y-s*z; M4(m,0,2)=t*x*z+s*y;
    M4(m,1,0)=t*x*y+s*z; M4(m,1,1)=t*y*y+c;   M4(m,1,2)=t*y*z-s*x;
    M4(m,2,0)=t*x*z-s*y; M4(m,2,1)=t*y*z+s*x; M4(m,2,2)=t*z*z+c;
    return m;
}
static mat4 m4_from_quat(quat q) {
    q = q_norm(q);
    float xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
    float xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
    float wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
    mat4 m = m4_identity();
    M4(m,0,0)=1-2*(yy+zz); M4(m,0,1)=2*(xy-wz);   M4(m,0,2)=2*(xz+wy);
    M4(m,1,0)=2*(xy+wz);   M4(m,1,1)=1-2*(xx+zz); M4(m,1,2)=2*(yz-wx);
    M4(m,2,0)=2*(xz-wy);   M4(m,2,1)=2*(yz+wx);   M4(m,2,2)=1-2*(xx+yy);
    return m;
}
static vec3 m4_transform_point(mat4 m, vec3 p) {
    return v3(M4(m,0,0)*p.x+M4(m,0,1)*p.y+M4(m,0,2)*p.z+M4(m,0,3),
              M4(m,1,0)*p.x+M4(m,1,1)*p.y+M4(m,1,2)*p.z+M4(m,1,3),
              M4(m,2,0)*p.x+M4(m,2,1)*p.y+M4(m,2,2)*p.z+M4(m,2,3));
}
static mat4 m4_inverse(mat4 m) {
    mat4 inv; float *o = inv.m, *i = m.m;
    o[0]  =  i[5]*i[10]*i[15]-i[5]*i[11]*i[14]-i[9]*i[6]*i[15]+i[9]*i[7]*i[14]+i[13]*i[6]*i[11]-i[13]*i[7]*i[10];
    o[4]  = -i[4]*i[10]*i[15]+i[4]*i[11]*i[14]+i[8]*i[6]*i[15]-i[8]*i[7]*i[14]-i[12]*i[6]*i[11]+i[12]*i[7]*i[10];
    o[8]  =  i[4]*i[9]*i[15] -i[4]*i[11]*i[13]-i[8]*i[5]*i[15]+i[8]*i[7]*i[13]+i[12]*i[5]*i[11]-i[12]*i[7]*i[9];
    o[12] = -i[4]*i[9]*i[14] +i[4]*i[10]*i[13]+i[8]*i[5]*i[14]-i[8]*i[6]*i[13]-i[12]*i[5]*i[10]+i[12]*i[6]*i[9];
    o[1]  = -i[1]*i[10]*i[15]+i[1]*i[11]*i[14]+i[9]*i[2]*i[15]-i[9]*i[3]*i[14]-i[13]*i[2]*i[11]+i[13]*i[3]*i[10];
    o[5]  =  i[0]*i[10]*i[15]-i[0]*i[11]*i[14]-i[8]*i[2]*i[15]+i[8]*i[3]*i[14]+i[12]*i[2]*i[11]-i[12]*i[3]*i[10];
    o[9]  = -i[0]*i[9]*i[15] +i[0]*i[11]*i[13]+i[8]*i[1]*i[15]-i[8]*i[3]*i[13]-i[12]*i[1]*i[11]+i[12]*i[3]*i[9];
    o[13] =  i[0]*i[9]*i[14] -i[0]*i[10]*i[13]-i[8]*i[1]*i[14]+i[8]*i[2]*i[13]+i[12]*i[1]*i[10]-i[12]*i[2]*i[9];
    o[2]  =  i[1]*i[6]*i[15]-i[1]*i[7]*i[14]-i[5]*i[2]*i[15]+i[5]*i[3]*i[14]+i[13]*i[2]*i[7]-i[13]*i[3]*i[6];
    o[6]  = -i[0]*i[6]*i[15]+i[0]*i[7]*i[14]+i[4]*i[2]*i[15]-i[4]*i[3]*i[14]-i[12]*i[2]*i[7]+i[12]*i[3]*i[6];
    o[10] =  i[0]*i[5]*i[15]-i[0]*i[7]*i[13]-i[4]*i[1]*i[15]+i[4]*i[3]*i[13]+i[12]*i[1]*i[7]-i[12]*i[3]*i[5];
    o[14] = -i[0]*i[5]*i[14]+i[0]*i[6]*i[13]+i[4]*i[1]*i[14]-i[4]*i[2]*i[13]-i[12]*i[1]*i[6]+i[12]*i[2]*i[5];
    o[3]  = -i[1]*i[6]*i[11]+i[1]*i[7]*i[10]+i[5]*i[2]*i[11]-i[5]*i[3]*i[10]-i[9]*i[2]*i[7]+i[9]*i[3]*i[6];
    o[7]  =  i[0]*i[6]*i[11]-i[0]*i[7]*i[10]-i[4]*i[2]*i[11]+i[4]*i[3]*i[10]+i[8]*i[2]*i[7]-i[8]*i[3]*i[6];
    o[11] = -i[0]*i[5]*i[11]+i[0]*i[7]*i[9] +i[4]*i[1]*i[11]-i[4]*i[3]*i[9] -i[8]*i[1]*i[7]+i[8]*i[3]*i[5];
    o[15] =  i[0]*i[5]*i[10]-i[0]*i[6]*i[9] -i[4]*i[1]*i[10]+i[4]*i[2]*i[9] +i[8]*i[1]*i[6]-i[8]*i[2]*i[5];
    float det = i[0]*o[0]+i[1]*o[4]+i[2]*o[8]+i[3]*o[12];
    if (fabsf(det) < EPSILON) return m4_identity();
    float inv_det = 1.0f / det;
    for (int j = 0; j < 16; j++) o[j] *= inv_det;
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
    while (out_len + n > out_cap) { out_cap = out_cap ? out_cap*2 : 65536; out_buf = realloc(out_buf, out_cap); }
    memcpy(out_buf + out_len, s, n); out_len += n;
}
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) { term_w = ws.ws_col; term_h = ws.ws_row; }
}
static void handle_sigwinch(int sig) { (void)sig; got_resize = 1; }
static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios); raw_mode_enabled = 0;
    }
}
static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios); raw_mode_enabled = 1; atexit(disable_raw_mode);
    signal(SIGWINCH, handle_sigwinch);
    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT|ICRNL|INPCK|ISTRIP|IXON); raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8); raw.c_lflag &= ~(ECHO|ICANON|IEXTEN|ISIG);
    raw.c_cc[VMIN]=0; raw.c_cc[VTIME]=0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
}
static void present_screen(void) {
    char seq[64]; int prev_fg=-1, prev_bg=-1;
    out_append("\033[H", 3);
    for (int y = 0; y < term_h; y++) {
        for (int x = 0; x < term_w; x++) {
            Cell *c = &screen[y*term_w+x];
            if (c->fg != prev_fg || c->bg != prev_bg) {
                int n = snprintf(seq, sizeof(seq), "\033[38;5;%dm\033[48;5;%dm", c->fg, c->bg);
                out_append(seq, n); prev_fg=c->fg; prev_bg=c->bg;
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
    struct pollfd pfd = {.fd=STDIN_FILENO,.events=POLLIN};
    if (poll(&pfd,1,0) <= 0) return KEY_NONE;
    char c; if (read(STDIN_FILENO,&c,1) != 1) return KEY_NONE;
    if (c == '\033') {
        char seq[3];
        if (read(STDIN_FILENO,&seq[0],1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO,&seq[1],1) != 1) return KEY_ESC;
        if (seq[0]=='[') { switch(seq[1]) { case 'A': return KEY_UP; case 'B': return KEY_DOWN; case 'C': return KEY_RIGHT; case 'D': return KEY_LEFT; } }
        return KEY_ESC;
    }
    return (int)(unsigned char)c;
}

/* ══════════════════════════════════════════════════════════════════
 * URDF DATA STRUCTURES
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_NAME  64
#define MAX_LINKS  32
#define MAX_JOINTS 32

typedef enum { GEOM_BOX, GEOM_CYLINDER, GEOM_SPHERE } GeomType;

typedef struct {
    GeomType type;
    vec3  size;       /* box: half extents */
    float radius;     /* cylinder/sphere */
    float length;     /* cylinder */
    vec3  origin_xyz;
    vec3  origin_rpy;
    int   color_id;
} Geometry;

typedef struct {
    char  name[MAX_NAME];
    Geometry visual;
    float mass;
    int   parent_joint;   /* index of joint connecting to parent, -1 for root */
    mat4  world_transform; /* computed by forward kinematics */
} Link;

typedef enum { JOINT_FIXED, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_CONTINUOUS } JointType;

typedef struct {
    char  name[MAX_NAME];
    JointType type;
    int   parent_link;
    int   child_link;
    vec3  origin_xyz;
    vec3  origin_rpy;
    vec3  axis;
    float lower_limit;
    float upper_limit;
    float position;   /* current joint position (angle for revolute) */
} Joint;

typedef struct {
    char  name[MAX_NAME];
    Link  links[MAX_LINKS];
    int   num_links;
    Joint joints[MAX_JOINTS];
    int   num_joints;
    int   root_link;
} Robot;

static Robot robot;

/* ══════════════════════════════════════════════════════════════════
 * SIMPLE XML PARSER
 * ══════════════════════════════════════════════════════════════════ */

static char *xml_content = NULL;
static char *xml_ptr     = NULL;

static void skip_whitespace(void) {
    while (*xml_ptr && isspace((unsigned char)*xml_ptr)) xml_ptr++;
}

static void skip_comment(void) {
    if (strncmp(xml_ptr, "<!--", 4) == 0) {
        char *end = strstr(xml_ptr, "-->");
        if (end) xml_ptr = end + 3;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement XML tag name parser
 *
 * Parse the next XML tag name from xml_ptr.
 * Skip whitespace, skip comments, expect '<', then read the tag name.
 * Handle:
 *   - Closing tags: if next char after '<' is '/', skip it
 *   - XML declaration: if next char is '?', skip to '>' and recurse
 *   - Self-closing tags handled by parse_attribute (it detects '/')
 *
 * Return 1 if a tag name was parsed, 0 otherwise.
 * Write tag name to out[]. Return length > 0 on success.
 *
 * Hint: a tag name ends at whitespace, '>', or '/'.
 * ══════════════════════════════════════════════════════════════════ */
static int parse_tag_name(char *out, int max_len) {
    /* TODO: Parse next XML tag name from xml_ptr */
    (void)out; (void)max_len;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement XML attribute parser
 *
 * Parse one "name=value" attribute from the current tag.
 * Skip whitespace. If next char is '>', '/', or '\0': return 0.
 *
 * Steps:
 *   1. Read attribute name (until '=' or whitespace)
 *   2. Skip whitespace, expect '='
 *   3. Skip whitespace, expect '"' or '\''
 *   4. Read attribute value until matching quote
 *   5. Advance xml_ptr past closing quote
 *
 * Return 1 if an attribute was parsed, 0 if no more attributes.
 * ══════════════════════════════════════════════════════════════════ */
static int parse_attribute(char *name, int name_max, char *value, int val_max) {
    /* TODO: Parse one XML attribute (name="value") */
    (void)name; (void)name_max; (void)value; (void)val_max;
    return 0;
}

static void skip_to_tag_end(void) {
    while (*xml_ptr && *xml_ptr != '>') xml_ptr++;
    if (*xml_ptr == '>') xml_ptr++;
}
static void skip_to_closing_tag(const char *tag) {
    char closing[128];
    snprintf(closing, sizeof(closing), "</%s>", tag);
    char *end = strstr(xml_ptr, closing);
    if (end) xml_ptr = end + strlen(closing);
}
static vec3 parse_vec3(const char *s) {
    vec3 v = {0,0,0}; sscanf(s, "%f %f %f", &v.x, &v.y, &v.z); return v;
}
static float parse_float(const char *s) {
    float f = 0; sscanf(s, "%f", &f); return f;
}

/* ══════════════════════════════════════════════════════════════════
 * URDF PARSER
 * Helpers are provided; implement the main parsing loops.
 * ══════════════════════════════════════════════════════════════════ */

static int find_link_by_name(const char *name) {
    for (int i = 0; i < robot.num_links; i++)
        if (strcmp(robot.links[i].name, name) == 0) return i;
    return -1;
}

static void parse_geometry(Geometry *geom) {
    char tag[64], attr_name[64], attr_value[256];
    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "/geometry") == 0) break;
        if (strcmp(tag, "box") == 0) {
            geom->type = GEOM_BOX;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value)))
                if (strcmp(attr_name, "size") == 0) geom->size = v3_scale(parse_vec3(attr_value), 0.5f);
        } else if (strcmp(tag, "cylinder") == 0) {
            geom->type = GEOM_CYLINDER;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "radius") == 0) geom->radius = parse_float(attr_value);
                if (strcmp(attr_name, "length") == 0) geom->length = parse_float(attr_value);
            }
        } else if (strcmp(tag, "sphere") == 0) {
            geom->type = GEOM_SPHERE;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value)))
                if (strcmp(attr_name, "radius") == 0) geom->radius = parse_float(attr_value);
        }
        skip_to_tag_end();
    }
}

static void parse_origin(vec3 *xyz, vec3 *rpy) {
    char attr_name[64], attr_value[256];
    while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
        if (strcmp(attr_name, "xyz") == 0) *xyz = parse_vec3(attr_value);
        if (strcmp(attr_name, "rpy") == 0) *rpy = parse_vec3(attr_value);
    }
    skip_to_tag_end();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Parse a <link> element from URDF
 *
 * A link element looks like:
 *   <link name="link1">
 *     <visual>
 *       <origin xyz="0 0 0" rpy="0 0 0"/>
 *       <geometry><cylinder radius="0.05" length="0.3"/></geometry>
 *       <material name="blue"/>
 *     </visual>
 *     <inertial>
 *       <mass value="1.0"/>
 *     </inertial>
 *   </link>
 *
 * Steps:
 *   1. Read name attribute from <link name="...">
 *   2. Add a new Link to robot.links[], copy the name
 *   3. Parse child tags until </link>:
 *      - <visual>: parse_visual() sets link->visual
 *      - <inertial>: look for <mass value="..."> inside
 * ══════════════════════════════════════════════════════════════════ */
static void parse_link(void) {
    /* TODO: Parse a <link> XML element and add to robot.links[] */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Parse a <joint> element from URDF
 *
 * A joint element looks like:
 *   <joint name="joint1" type="revolute">
 *     <parent link="base_link"/>
 *     <child link="link1"/>
 *     <origin xyz="0 0 0.1" rpy="0 0 0"/>
 *     <axis xyz="0 0 1"/>
 *     <limit lower="-1.57" upper="1.57"/>
 *   </joint>
 *
 * Steps:
 *   1. Read name and type attributes
 *   2. Add a new Joint to robot.joints[]
 *   3. Parse child tags until </joint>:
 *      - <parent link="...">: find parent link index
 *      - <child link="...">: find child link index
 *      - <origin>: parse_origin() for xyz and rpy
 *      - <axis>: read xyz attribute
 *      - <limit>: read lower and upper attributes
 *   4. Set parent_joint on the child link
 * ══════════════════════════════════════════════════════════════════ */
static void parse_joint(void) {
    /* TODO: Parse a <joint> XML element and add to robot.joints[] */
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Load URDF file
 *
 * Steps:
 *   1. Open the file, read its entire content into xml_content
 *   2. Set xml_ptr = xml_content
 *   3. Loop calling parse_tag_name():
 *      - If tag == "link":  call parse_link()
 *      - If tag == "joint": call parse_joint()
 *      - Otherwise: skip_to_tag_end()
 *   4. Find the root link (the one with no parent joint)
 * ══════════════════════════════════════════════════════════════════ */
static int load_urdf(const char *filename) {
    /* TODO: Open file, read content, parse all links and joints */
    (void)filename;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Compute forward kinematics
 *
 * Traverse the kinematic tree from root to leaves.
 * For each link, its world_transform = parent_world_transform * joint_transform
 *
 * For a revolute joint connecting parent→child:
 *   origin_tf   = m4_translate(joint.origin_xyz) * rpy_to_mat(joint.origin_rpy)
 *   joint_rot   = m4_rotate_axis(joint.axis, joint.position)
 *   joint_tf    = origin_tf * joint_rot
 *   child.world = parent.world * joint_tf
 *
 * For the root link: world_transform = identity
 *
 * Strategy: traverse joints in order (assumes tree is listed root-first).
 * For each joint: child_link.world = parent_link.world * joint_local_transform
 *
 * RPY to matrix: Rz(yaw) * Ry(pitch) * Rx(roll)
 * ══════════════════════════════════════════════════════════════════ */
static void compute_fk(void) {
    /* TODO: Compute world_transform for all links using forward kinematics */

    /* Set root link to identity */
    if (robot.root_link >= 0 && robot.root_link < robot.num_links)
        robot.links[robot.root_link].world_transform = m4_identity();

    /* Traverse joints and compute child transforms */
    for (int i = 0; i < robot.num_joints; i++) {
        (void)i;  /* TODO: compute child link world_transform */
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SDF RENDERING using link world transforms
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float dist; int material_id; } SceneHit;

static float sdf_sphere_f(vec3 p, vec3 c, float r) { return v3_len(v3_sub(p,c)) - r; }
static float sdf_box_f(vec3 p, vec3 c, vec3 hs) {
    vec3 d = v3_sub(v3_abs(v3_sub(p,c)), hs);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}
static float sdf_cylinder_f(vec3 p, vec3 base, float radius, float height) {
    vec3 rel = v3_sub(p, base);
    float dr = sqrtf(rel.x*rel.x + rel.z*rel.z) - radius;
    float dv = fabsf(rel.y - height*0.5f) - height*0.5f;
    return fminf(fmaxf(dr,dv), 0.0f) + v3_len(v3(fmaxf(dr,0.0f), fmaxf(dv,0.0f), 0.0f));
}
static float sdf_plane(vec3 p, vec3 n, float o) { return v3_dot(p,n) + o; }

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};
    float d;

    d = sdf_plane(p, v3(0,1,0), 0.5f);
    if (d < hit.dist) { hit.dist = d; hit.material_id = 0; }

    /* Render each link using its world_transform */
    for (int i = 0; i < robot.num_links; i++) {
        Link *link = &robot.links[i];
        Geometry *geom = &link->visual;

        /* Transform point to link's local space */
        mat4 inv = m4_inverse(link->world_transform);
        vec3 lp = m4_transform_point(inv, p);

        /* Apply visual origin offset */
        vec3 origin_p = v3_sub(lp, geom->origin_xyz);

        switch (geom->type) {
            case GEOM_SPHERE:
                d = sdf_sphere_f(origin_p, v3(0,0,0), geom->radius);
                break;
            case GEOM_BOX:
                d = sdf_box_f(origin_p, v3(0,0,0), geom->size);
                break;
            case GEOM_CYLINDER:
                d = sdf_cylinder_f(origin_p, v3(0,-geom->length*0.5f,0),
                                   geom->radius, geom->length);
                break;
            default:
                d = 1e10f;
        }

        if (d < hit.dist) { hit.dist = d; hit.material_id = 1 + (geom->color_id % 8); }
    }

    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

#define MAX_STEPS 64
#define MAX_DIST  40.0f
#define SURF_DIST 0.002f

typedef struct { int hit; float dist; vec3 point; int material_id; } RayResult;
static RayResult ray_march(vec3 ro, vec3 rd) {
    RayResult result = {0, 0, ro, 0}; float t = 0;
    for (int i = 0; i < MAX_STEPS && t < MAX_DIST; i++) {
        vec3 p = v3_add(ro, v3_scale(rd, t));
        SceneHit h = scene_sdf(p);
        if (h.dist < SURF_DIST) { result.hit=1; result.dist=t; result.point=p; result.material_id=h.material_id; return result; }
        t += h.dist;
    }
    result.dist = t; return result;
}
static vec3 calc_normal(vec3 p) {
    const float h = 0.001f;
    return v3_norm(v3(scene_dist(v3(p.x+h,p.y,p.z))-scene_dist(v3(p.x-h,p.y,p.z)),
                      scene_dist(v3(p.x,p.y+h,p.z))-scene_dist(v3(p.x,p.y-h,p.z)),
                      scene_dist(v3(p.x,p.y,p.z+h))-scene_dist(v3(p.x,p.y,p.z-h))));
}
static const char LUMINANCE_RAMP[] = " .:-=+*#%@";
#define RAMP_LEN 10
static const uint8_t MATERIAL_COLORS[][3] = {
    {2,3,2},{5,1,1},{1,3,5},{5,4,0},{5,5,0},{5,0,5},{0,5,5},{2,5,2},{5,2,3},
};
static uint8_t color_216(int r, int g, int b) {
    r=r<0?0:(r>5?5:r); g=g<0?0:(g>5?5:g); b=b<0?0:(b>5?5:b);
    return 16+36*r+6*g+b;
}
typedef struct { char ch; uint8_t fg; uint8_t bg; } ShadeResult;
static ShadeResult shade_point(vec3 p, vec3 rd, vec3 normal, int mat_id, float dist) {
    ShadeResult result = {' ', 0, 16};
    vec3 light_dir = v3_norm(v3(0.5f,1.0f,0.3f));
    vec3 half_vec = v3_norm(v3_add(light_dir, v3_neg(rd)));
    float ambient=0.2f, diffuse=fmaxf(0.0f,v3_dot(normal,light_dir))*0.6f;
    float specular=powf(fmaxf(0.0f,v3_dot(normal,half_vec)),16.0f)*0.3f;
    float fog=expf(-dist*0.05f);
    float lum=fminf(1.0f,fmaxf(0.0f,(ambient+diffuse+specular)*fog));
    result.ch=LUMINANCE_RAMP[(int)(lum*(RAMP_LEN-1)+0.5f)];
    int nm=sizeof(MATERIAL_COLORS)/sizeof(MATERIAL_COLORS[0]);
    int mi=mat_id<nm?mat_id:0; float cs=0.3f+lum*0.7f;
    result.fg=color_216((int)(MATERIAL_COLORS[mi][0]*cs),(int)(MATERIAL_COLORS[mi][1]*cs),(int)(MATERIAL_COLORS[mi][2]*cs));
    return result;
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

int main(int argc, char *argv[]) {
    const char *urdf_file = argc > 1 ? argv[1] : "robot.urdf";

    memset(&robot, 0, sizeof(robot));
    robot.root_link = -1;
    for (int i = 0; i < MAX_LINKS; i++)
        robot.links[i].parent_joint = -1;

    if (!load_urdf(urdf_file)) {
        fprintf(stderr, "Error: could not load '%s'\n", urdf_file);
        return 1;
    }

    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    OrbitCamera cam = {
        .target=v3(0,0.5f,0), .distance=5.0f, .azimuth=0.5f, .elevation=0.5f,
        .fov=(float)M_PI/3.0f, .smooth_az=0.5f, .smooth_el=0.5f, .smooth_dist=5.0f
    };

    int selected_joint = 0;
    double last_time = get_time(), fps_time = last_time;
    int fps_count = 0; double fps_display = 0;
    int running = 1;

    compute_fk();

    while (running) {
        double now = get_time();
        float dt = (float)(now - last_time); last_time = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0; get_term_size(); free(screen);
            screen = calloc(term_w * term_h, sizeof(Cell));
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        int key; int fk_dirty = 0;
        while ((key = read_key()) != KEY_NONE) {
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'w': case 'W': cam.elevation += 0.1f; break;
                case 's': case 'S': cam.elevation -= 0.1f; break;
                case 'a': case 'A': cam.azimuth   -= 0.1f; break;
                case 'd': case 'D': cam.azimuth   += 0.1f; break;
                case '+': case '=': cam.distance -= 0.3f; break;
                case '-': case '_': cam.distance += 0.3f; break;
                default:
                    if (key >= '1' && key <= '9') {
                        selected_joint = key - '1';
                    } else if ((key == KEY_UP || key == KEY_RIGHT) && selected_joint < robot.num_joints) {
                        robot.joints[selected_joint].position += 0.1f;
                        fk_dirty = 1;
                    } else if ((key == KEY_DOWN || key == KEY_LEFT) && selected_joint < robot.num_joints) {
                        robot.joints[selected_joint].position -= 0.1f;
                        fk_dirty = 1;
                    }
            }
        }

        if (fk_dirty) compute_fk();

        cam.elevation = fmaxf(-(float)M_PI/2+0.1f, fminf((float)M_PI/2-0.1f, cam.elevation));
        cam.distance = fmaxf(2.0f, fminf(20.0f, cam.distance));

        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0; fps_time = now;
        }

        char hud[256];
        snprintf(hud, sizeof(hud),
            "\033[1;1H\033[48;5;16m\033[38;5;226m"
            " Phase 6: %s | FPS:%.0f | Joint[%d]: %.2f rad"
            " | [1-9]sel [+/-]adjust [WASD]cam [Q]quit ",
            robot.name, fps_display, selected_joint,
            selected_joint < robot.num_joints ? robot.joints[selected_joint].position : 0.0f);
        write(STDOUT_FILENO, hud, strlen(hud));

        sleep_ms(50);
    }

    free(screen); free(out_buf); free(xml_content);
    return 0;
}
