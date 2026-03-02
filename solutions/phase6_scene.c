/*
 * phase6_scene.c — Scene Graph + URDF Parser
 *
 * Simple XML parser, URDF subset parsing (link, joint, visual, collision, inertial),
 * kinematic tree, forward kinematics, articulated body rendering.
 *
 * Deliverable: Load a simple robot URDF (e.g., 2-link arm) and render it.
 *
 * Build: gcc -O3 -o phase6_scene phase6_scene.c -lm
 * Run:   ./phase6_scene [robot.urdf]
 * Controls: [1-9] select joint, [+/-] adjust angle, [WASD] camera, [Q] quit
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
 * 3D MATH
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float w, x, y, z; } quat;
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

/* Quaternion */
static inline quat q_identity(void) { return (quat){1, 0, 0, 0}; }
static inline quat q_from_axis_angle(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float half = rad * 0.5f;
    float s = sinf(half);
    return (quat){cosf(half), axis.x * s, axis.y * s, axis.z * s};
}
static inline quat q_mul(quat a, quat b) {
    return (quat){
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}
static inline float q_len(quat q) { return sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); }
static inline quat q_norm(quat q) {
    float len = q_len(q);
    if (len < EPSILON) return q_identity();
    float inv = 1.0f / len;
    return (quat){q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}
static vec3 q_rotate(quat q, vec3 v) {
    vec3 qv = v3(q.x, q.y, q.z);
    vec3 uv = v3_cross(qv, v);
    vec3 uuv = v3_cross(qv, uv);
    return v3_add(v, v3_scale(v3_add(v3_scale(uv, q.w), uuv), 2.0f));
}

/* Matrix operations */
static inline mat4 m4_identity(void) {
    mat4 m = {0};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0;
            for (int k = 0; k < 4; k++) {
                sum += M4(a, row, k) * M4(b, k, c);
            }
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

static mat4 m4_rotate_axis(vec3 axis, float rad) {
    axis = v3_norm(axis);
    float c = cosf(rad), s = sinf(rad), t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    mat4 m = m4_identity();
    M4(m, 0, 0) = t*x*x + c;   M4(m, 0, 1) = t*x*y - s*z; M4(m, 0, 2) = t*x*z + s*y;
    M4(m, 1, 0) = t*x*y + s*z; M4(m, 1, 1) = t*y*y + c;   M4(m, 1, 2) = t*y*z - s*x;
    M4(m, 2, 0) = t*x*z - s*y; M4(m, 2, 1) = t*y*z + s*x; M4(m, 2, 2) = t*z*z + c;
    return m;
}

static mat4 m4_from_quat(quat q) {
    q = q_norm(q);
    float xx = q.x*q.x, yy = q.y*q.y, zz = q.z*q.z;
    float xy = q.x*q.y, xz = q.x*q.z, yz = q.y*q.z;
    float wx = q.w*q.x, wy = q.w*q.y, wz = q.w*q.z;
    mat4 m = m4_identity();
    M4(m, 0, 0) = 1 - 2*(yy + zz); M4(m, 0, 1) = 2*(xy - wz);     M4(m, 0, 2) = 2*(xz + wy);
    M4(m, 1, 0) = 2*(xy + wz);     M4(m, 1, 1) = 1 - 2*(xx + zz); M4(m, 1, 2) = 2*(yz - wx);
    M4(m, 2, 0) = 2*(xz - wy);     M4(m, 2, 1) = 2*(yz + wx);     M4(m, 2, 2) = 1 - 2*(xx + yy);
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
 * URDF DATA STRUCTURES
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_NAME 64
#define MAX_LINKS 32
#define MAX_JOINTS 32

typedef enum { GEOM_BOX, GEOM_CYLINDER, GEOM_SPHERE } GeomType;

typedef struct {
    GeomType type;
    vec3 size;       /* box: half extents */
    float radius;    /* cylinder/sphere */
    float length;    /* cylinder */
    vec3 origin_xyz;
    vec3 origin_rpy;
    int color_id;
} Geometry;

typedef struct {
    char name[MAX_NAME];
    Geometry visual;
    float mass;
    int parent_joint;  /* index of joint connecting to parent, -1 for root */
    mat4 world_transform;
} Link;

typedef enum { JOINT_FIXED, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_CONTINUOUS } JointType;

typedef struct {
    char name[MAX_NAME];
    JointType type;
    int parent_link;   /* index */
    int child_link;    /* index */
    vec3 origin_xyz;
    vec3 origin_rpy;
    vec3 axis;
    float lower_limit;
    float upper_limit;
    float position;    /* current joint position */
} Joint;

typedef struct {
    char name[MAX_NAME];
    Link links[MAX_LINKS];
    int num_links;
    Joint joints[MAX_JOINTS];
    int num_joints;
    int root_link;
} Robot;

static Robot robot;

/* ══════════════════════════════════════════════════════════════════
 * SIMPLE XML PARSER
 * ══════════════════════════════════════════════════════════════════ */

static char *xml_content = NULL;
static char *xml_ptr = NULL;

static void skip_whitespace(void) {
    while (*xml_ptr && isspace(*xml_ptr)) xml_ptr++;
}

static void skip_comment(void) {
    if (strncmp(xml_ptr, "<!--", 4) == 0) {
        char *end = strstr(xml_ptr, "-->");
        if (end) xml_ptr = end + 3;
    }
}

static int parse_tag_name(char *out, int max_len) {
    skip_whitespace();
    skip_comment();
    if (*xml_ptr != '<') return 0;
    xml_ptr++;
    if (*xml_ptr == '/') { xml_ptr++; }  /* closing tag */
    if (*xml_ptr == '?') { /* skip XML declaration */
        while (*xml_ptr && *xml_ptr != '>') xml_ptr++;
        if (*xml_ptr == '>') xml_ptr++;
        return parse_tag_name(out, max_len);
    }
    int i = 0;
    while (*xml_ptr && !isspace(*xml_ptr) && *xml_ptr != '>' && *xml_ptr != '/' && i < max_len - 1) {
        out[i++] = *xml_ptr++;
    }
    out[i] = '\0';
    return i > 0;
}

static int parse_attribute(char *name, int name_max, char *value, int value_max) {
    skip_whitespace();
    if (*xml_ptr == '>' || *xml_ptr == '/' || *xml_ptr == '\0') return 0;

    int i = 0;
    while (*xml_ptr && *xml_ptr != '=' && !isspace(*xml_ptr) && i < name_max - 1) {
        name[i++] = *xml_ptr++;
    }
    name[i] = '\0';
    if (i == 0) return 0;

    skip_whitespace();
    if (*xml_ptr != '=') return 0;
    xml_ptr++;
    skip_whitespace();

    char quote = *xml_ptr;
    if (quote != '"' && quote != '\'') return 0;
    xml_ptr++;

    i = 0;
    while (*xml_ptr && *xml_ptr != quote && i < value_max - 1) {
        value[i++] = *xml_ptr++;
    }
    value[i] = '\0';
    if (*xml_ptr == quote) xml_ptr++;

    return 1;
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
    vec3 v = {0, 0, 0};
    sscanf(s, "%f %f %f", &v.x, &v.y, &v.z);
    return v;
}

static float parse_float(const char *s) {
    float f = 0;
    sscanf(s, "%f", &f);
    return f;
}

/* ══════════════════════════════════════════════════════════════════
 * URDF PARSER
 * ══════════════════════════════════════════════════════════════════ */

static int find_link_by_name(const char *name) {
    for (int i = 0; i < robot.num_links; i++) {
        if (strcmp(robot.links[i].name, name) == 0) return i;
    }
    return -1;
}

static void parse_geometry(Geometry *geom) {
    char tag[64], attr_name[64], attr_value[256];

    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "/geometry") == 0) break;
        if (strcmp(tag, "box") == 0) {
            geom->type = GEOM_BOX;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "size") == 0) {
                    vec3 size = parse_vec3(attr_value);
                    geom->size = v3_scale(size, 0.5f);  /* convert to half extents */
                }
            }
        } else if (strcmp(tag, "cylinder") == 0) {
            geom->type = GEOM_CYLINDER;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "radius") == 0) geom->radius = parse_float(attr_value);
                if (strcmp(attr_name, "length") == 0) geom->length = parse_float(attr_value);
            }
        } else if (strcmp(tag, "sphere") == 0) {
            geom->type = GEOM_SPHERE;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "radius") == 0) geom->radius = parse_float(attr_value);
            }
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

static void parse_visual(Link *link) {
    char tag[64], attr_name[64], attr_value[256];

    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "/visual") == 0) break;
        if (strcmp(tag, "origin") == 0) {
            parse_origin(&link->visual.origin_xyz, &link->visual.origin_rpy);
        } else if (strcmp(tag, "geometry") == 0) {
            skip_to_tag_end();
            parse_geometry(&link->visual);
        } else if (strcmp(tag, "material") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                /* Simple color assignment based on material name */
                if (strcmp(attr_name, "name") == 0) {
                    if (strstr(attr_value, "red")) link->visual.color_id = 1;
                    else if (strstr(attr_value, "blue")) link->visual.color_id = 2;
                    else if (strstr(attr_value, "green")) link->visual.color_id = 3;
                    else if (strstr(attr_value, "yellow")) link->visual.color_id = 4;
                    else if (strstr(attr_value, "orange")) link->visual.color_id = 5;
                    else link->visual.color_id = 6;
                }
            }
            skip_to_closing_tag("material");
        } else {
            skip_to_tag_end();
        }
    }
}

static void parse_link(void) {
    if (robot.num_links >= MAX_LINKS) return;
    Link *link = &robot.links[robot.num_links];
    memset(link, 0, sizeof(Link));
    link->parent_joint = -1;
    link->visual.color_id = robot.num_links % 7;

    char attr_name[64], attr_value[256], tag[64];
    while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
        if (strcmp(attr_name, "name") == 0) {
            strncpy(link->name, attr_value, MAX_NAME - 1);
        }
    }
    skip_to_tag_end();

    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "/link") == 0) break;
        if (strcmp(tag, "visual") == 0) {
            skip_to_tag_end();
            parse_visual(link);
        } else if (strcmp(tag, "inertial") == 0) {
            skip_to_closing_tag("inertial");
        } else if (strcmp(tag, "collision") == 0) {
            skip_to_closing_tag("collision");
        } else {
            skip_to_tag_end();
        }
    }

    robot.num_links++;
}

static void parse_joint(void) {
    if (robot.num_joints >= MAX_JOINTS) return;
    Joint *joint = &robot.joints[robot.num_joints];
    memset(joint, 0, sizeof(Joint));
    joint->axis = v3(0, 0, 1);  /* default axis */
    joint->lower_limit = -M_PI;
    joint->upper_limit = M_PI;
    joint->parent_link = -1;
    joint->child_link = -1;

    char attr_name[64], attr_value[256], tag[64];
    while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
        if (strcmp(attr_name, "name") == 0) {
            strncpy(joint->name, attr_value, MAX_NAME - 1);
        } else if (strcmp(attr_name, "type") == 0) {
            if (strcmp(attr_value, "revolute") == 0) joint->type = JOINT_REVOLUTE;
            else if (strcmp(attr_value, "continuous") == 0) joint->type = JOINT_CONTINUOUS;
            else if (strcmp(attr_value, "prismatic") == 0) joint->type = JOINT_PRISMATIC;
            else joint->type = JOINT_FIXED;
        }
    }
    skip_to_tag_end();

    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "/joint") == 0) break;
        if (strcmp(tag, "origin") == 0) {
            parse_origin(&joint->origin_xyz, &joint->origin_rpy);
        } else if (strcmp(tag, "parent") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "link") == 0) {
                    joint->parent_link = find_link_by_name(attr_value);
                }
            }
            skip_to_tag_end();
        } else if (strcmp(tag, "child") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "link") == 0) {
                    joint->child_link = find_link_by_name(attr_value);
                }
            }
            skip_to_tag_end();
        } else if (strcmp(tag, "axis") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "xyz") == 0) joint->axis = parse_vec3(attr_value);
            }
            skip_to_tag_end();
        } else if (strcmp(tag, "limit") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "lower") == 0) joint->lower_limit = parse_float(attr_value);
                if (strcmp(attr_name, "upper") == 0) joint->upper_limit = parse_float(attr_value);
            }
            skip_to_tag_end();
        } else {
            skip_to_tag_end();
        }
    }

    /* Link the child link to this joint */
    if (joint->child_link >= 0) {
        robot.links[joint->child_link].parent_joint = robot.num_joints;
    }

    robot.num_joints++;
}

static int parse_urdf(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    xml_content = malloc(size + 1);
    fread(xml_content, 1, size, f);
    xml_content[size] = '\0';
    fclose(f);

    xml_ptr = xml_content;
    memset(&robot, 0, sizeof(robot));

    char tag[64], attr_name[64], attr_value[256];

    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "robot") == 0) {
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
                if (strcmp(attr_name, "name") == 0) {
                    strncpy(robot.name, attr_value, MAX_NAME - 1);
                }
            }
            skip_to_tag_end();
        } else if (strcmp(tag, "link") == 0) {
            parse_link();
        } else if (strcmp(tag, "joint") == 0) {
            parse_joint();
        } else {
            skip_to_tag_end();
        }
    }

    /* Find root link (link with no parent joint) */
    robot.root_link = 0;
    for (int i = 0; i < robot.num_links; i++) {
        if (robot.links[i].parent_joint < 0) {
            robot.root_link = i;
            break;
        }
    }

    free(xml_content);
    xml_content = NULL;
    return 1;
}

/* ══════════════════════════════════════════════════════════════════
 * FORWARD KINEMATICS
 * ══════════════════════════════════════════════════════════════════ */

static mat4 make_transform(vec3 xyz, vec3 rpy) {
    /* RPY: roll (X), pitch (Y), yaw (Z) */
    mat4 t = m4_translate(xyz);
    mat4 rz = m4_rotate_axis(v3(0, 0, 1), rpy.z);
    mat4 ry = m4_rotate_axis(v3(0, 1, 0), rpy.y);
    mat4 rx = m4_rotate_axis(v3(1, 0, 0), rpy.x);
    return m4_mul(t, m4_mul(rz, m4_mul(ry, rx)));
}

static void compute_forward_kinematics(void) {
    /* Initialize all transforms */
    for (int i = 0; i < robot.num_links; i++) {
        robot.links[i].world_transform = m4_identity();
    }

    /* Compute transforms recursively from root */
    int computed[MAX_LINKS] = {0};
    computed[robot.root_link] = 1;

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int j = 0; j < robot.num_joints; j++) {
            Joint *joint = &robot.joints[j];
            if (joint->parent_link < 0 || joint->child_link < 0) continue;
            if (!computed[joint->parent_link] || computed[joint->child_link]) continue;

            /* Compute joint transform */
            mat4 joint_origin = make_transform(joint->origin_xyz, joint->origin_rpy);
            mat4 joint_motion = m4_identity();

            if (joint->type == JOINT_REVOLUTE || joint->type == JOINT_CONTINUOUS) {
                joint_motion = m4_rotate_axis(joint->axis, joint->position);
            } else if (joint->type == JOINT_PRISMATIC) {
                joint_motion = m4_translate(v3_scale(joint->axis, joint->position));
            }

            mat4 parent_tf = robot.links[joint->parent_link].world_transform;
            robot.links[joint->child_link].world_transform =
                m4_mul(parent_tf, m4_mul(joint_origin, joint_motion));

            computed[joint->child_link] = 1;
            changed = 1;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * DEFAULT ROBOT (if no URDF provided)
 * ══════════════════════════════════════════════════════════════════ */

static void create_default_robot(void) {
    memset(&robot, 0, sizeof(robot));
    strcpy(robot.name, "SimpleArm");

    /* Base link */
    Link *base = &robot.links[robot.num_links++];
    strcpy(base->name, "base_link");
    base->visual.type = GEOM_BOX;
    base->visual.size = v3(0.3f, 0.1f, 0.3f);
    base->visual.color_id = 0;
    base->parent_joint = -1;

    /* Link 1 */
    Link *link1 = &robot.links[robot.num_links++];
    strcpy(link1->name, "link1");
    link1->visual.type = GEOM_CYLINDER;
    link1->visual.radius = 0.08f;
    link1->visual.length = 1.0f;
    link1->visual.origin_xyz = v3(0, 0.5f, 0);
    link1->visual.color_id = 1;

    /* Link 2 */
    Link *link2 = &robot.links[robot.num_links++];
    strcpy(link2->name, "link2");
    link2->visual.type = GEOM_CYLINDER;
    link2->visual.radius = 0.06f;
    link2->visual.length = 0.8f;
    link2->visual.origin_xyz = v3(0, 0.4f, 0);
    link2->visual.color_id = 2;

    /* End effector */
    Link *ee = &robot.links[robot.num_links++];
    strcpy(ee->name, "end_effector");
    ee->visual.type = GEOM_SPHERE;
    ee->visual.radius = 0.1f;
    ee->visual.color_id = 3;

    /* Joint 1: base to link1 (revolute around Y) */
    Joint *j1 = &robot.joints[robot.num_joints++];
    strcpy(j1->name, "joint1");
    j1->type = JOINT_REVOLUTE;
    j1->parent_link = 0;
    j1->child_link = 1;
    j1->origin_xyz = v3(0, 0.1f, 0);
    j1->axis = v3(0, 1, 0);
    j1->lower_limit = -M_PI;
    j1->upper_limit = M_PI;
    robot.links[1].parent_joint = 0;

    /* Joint 2: link1 to link2 (revolute around Z) */
    Joint *j2 = &robot.joints[robot.num_joints++];
    strcpy(j2->name, "joint2");
    j2->type = JOINT_REVOLUTE;
    j2->parent_link = 1;
    j2->child_link = 2;
    j2->origin_xyz = v3(0, 1.0f, 0);
    j2->axis = v3(0, 0, 1);
    j2->lower_limit = -M_PI * 0.75f;
    j2->upper_limit = M_PI * 0.75f;
    robot.links[2].parent_joint = 1;

    /* Joint 3: link2 to end_effector (revolute around Z) */
    Joint *j3 = &robot.joints[robot.num_joints++];
    strcpy(j3->name, "joint3");
    j3->type = JOINT_REVOLUTE;
    j3->parent_link = 2;
    j3->child_link = 3;
    j3->origin_xyz = v3(0, 0.8f, 0);
    j3->axis = v3(0, 0, 1);
    j3->lower_limit = -M_PI * 0.5f;
    j3->upper_limit = M_PI * 0.5f;
    robot.links[3].parent_joint = 2;

    robot.root_link = 0;
}

/* ══════════════════════════════════════════════════════════════════
 * SDF + RENDERING
 * ══════════════════════════════════════════════════════════════════ */

static float sdf_sphere(vec3 p, vec3 center, float radius) {
    return v3_len(v3_sub(p, center)) - radius;
}

static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    vec3 d = v3_sub(v3_abs(v3_sub(p, center)), half_size);
    return fminf(v3_max_comp(d), 0.0f) + v3_len(v3_max(d, v3(0,0,0)));
}

static float sdf_cylinder(vec3 p, vec3 center, float radius, float height) {
    vec3 rel = v3_sub(p, center);
    float d_radial = sqrtf(rel.x * rel.x + rel.z * rel.z) - radius;
    float d_vertical = fabsf(rel.y) - height * 0.5f;
    return fminf(fmaxf(d_radial, d_vertical), 0.0f) +
           v3_len(v3(fmaxf(d_radial, 0.0f), fmaxf(d_vertical, 0.0f), 0.0f));
}

static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

typedef struct { float dist; int material_id; } SceneHit;

static SceneHit scene_sdf(vec3 p) {
    SceneHit hit = {1e10f, 0};

    /* Ground plane */
    float d_ground = sdf_plane(p, v3(0, 1, 0), 0.0f);
    if (d_ground < hit.dist) { hit.dist = d_ground; hit.material_id = 7; }

    /* Robot links */
    for (int i = 0; i < robot.num_links; i++) {
        Link *link = &robot.links[i];
        Geometry *g = &link->visual;

        /* Transform point to link local space */
        mat4 inv_tf = m4_inverse(link->world_transform);

        /* Also account for visual origin */
        mat4 vis_tf = make_transform(g->origin_xyz, g->origin_rpy);
        mat4 full_inv = m4_mul(m4_inverse(vis_tf), inv_tf);

        vec3 local_p = m4_transform_point(full_inv, p);

        float d;
        switch (g->type) {
            case GEOM_SPHERE:
                d = sdf_sphere(local_p, v3(0,0,0), g->radius);
                break;
            case GEOM_BOX:
                d = sdf_box(local_p, v3(0,0,0), g->size);
                break;
            case GEOM_CYLINDER:
                d = sdf_cylinder(local_p, v3(0,0,0), g->radius, g->length);
                break;
            default:
                d = 1e10f;
        }

        if (d < hit.dist) {
            hit.dist = d;
            hit.material_id = g->color_id;
        }
    }

    return hit;
}

static float scene_dist(vec3 p) { return scene_sdf(p).dist; }

#define MAX_STEPS 48
#define MAX_DIST 40.0f
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
    {3, 3, 3},  /* 0: gray (base) */
    {5, 1, 1},  /* 1: red */
    {1, 3, 5},  /* 2: blue */
    {5, 5, 0},  /* 3: yellow */
    {5, 0, 5},  /* 4: magenta */
    {5, 3, 0},  /* 5: orange */
    {0, 5, 5},  /* 6: cyan */
    {1, 2, 1},  /* 7: ground (dark green) */
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

    float fog = expf(-dist * 0.05f);
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

static void draw_hud(int selected_joint, double fps) {
    char hud[512];
    char joints_str[256] = "";
    int len = 0;
    for (int i = 0; i < robot.num_joints && i < 9; i++) {
        Joint *j = &robot.joints[i];
        if (j->type == JOINT_FIXED) continue;
        len += snprintf(joints_str + len, sizeof(joints_str) - len,
            "%s[%d]%.1f ", i == selected_joint ? ">" : "", i+1, j->position * 180.0f / M_PI);
    }
    snprintf(hud, sizeof(hud),
        "\033[1;1H\033[48;5;16m\033[38;5;226m"
        " Phase 6: URDF | %s | FPS:%.0f | Joints: %s| [1-9] sel [+/-] move [WASD] cam [Q] quit ",
        robot.name, fps, joints_str);
    write(STDOUT_FILENO, hud, strlen(hud));
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
    enable_raw_mode();
    get_term_size();
    screen = calloc(term_w * term_h, sizeof(Cell));

    /* Load URDF or create default robot */
    if (argc > 1) {
        if (!parse_urdf(argv[1])) {
            fprintf(stderr, "Failed to load URDF: %s\n", argv[1]);
            create_default_robot();
        }
    } else {
        create_default_robot();
    }

    compute_forward_kinematics();

    OrbitCamera cam = {
        .target = v3(0, 1.0f, 0),
        .distance = 6.0f,
        .azimuth = 0.5f,
        .elevation = 0.3f,
        .fov = M_PI / 3.0f,
        .smooth_az = 0.5f,
        .smooth_el = 0.3f,
        .smooth_dist = 6.0f
    };

    int selected_joint = 0;
    double last_time = get_time();
    double fps_time = last_time;
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
            switch (key) {
                case 'q': case 'Q': case KEY_ESC: running = 0; break;
                case 'w': case 'W': case KEY_UP:   cam.elevation += 0.1f; break;
                case 's': case 'S': case KEY_DOWN: cam.elevation -= 0.1f; break;
                case 'a': case 'A': case KEY_LEFT: cam.azimuth -= 0.1f; break;
                case 'd': case 'D': case KEY_RIGHT: cam.azimuth += 0.1f; break;
                case '=': case '+': {
                    if (selected_joint < robot.num_joints) {
                        Joint *j = &robot.joints[selected_joint];
                        j->position += 0.1f;
                        if (j->type == JOINT_REVOLUTE)
                            j->position = fminf(j->position, j->upper_limit);
                        compute_forward_kinematics();
                    }
                    break;
                }
                case '-': case '_': {
                    if (selected_joint < robot.num_joints) {
                        Joint *j = &robot.joints[selected_joint];
                        j->position -= 0.1f;
                        if (j->type == JOINT_REVOLUTE)
                            j->position = fmaxf(j->position, j->lower_limit);
                        compute_forward_kinematics();
                    }
                    break;
                }
                default:
                    if (key >= '1' && key <= '9') {
                        int idx = key - '1';
                        if (idx < robot.num_joints) selected_joint = idx;
                    }
                    break;
            }
        }

        cam.elevation = fmaxf(-M_PI/2 + 0.1f, fminf(M_PI/2 - 0.1f, cam.elevation));
        cam.distance = fmaxf(3.0f, fminf(20.0f, cam.distance));

        orbit_camera_update(&cam, dt);
        render_frame(&cam);
        present_screen();

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0;
            fps_time = now;
        }
        draw_hud(selected_joint, fps_display);

        sleep_ms(50);
    }

    free(screen);
    free(out_buf);
    return 0;
}
