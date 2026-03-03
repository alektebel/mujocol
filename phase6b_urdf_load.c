/*
 * phase6b_urdf_load.c — URDF Robot Structure Loading
 *
 * EXERCISE (sub-phase 6b): Build the kinematic tree data structures
 * (Link and Joint arrays) by parsing a URDF file.
 *
 * Build: gcc -O2 -o phase6b_urdf_load phase6b_urdf_load.c -lm
 * Run:   ./phase6b_urdf_load robot.urdf
 *
 * LEARNING GOALS:
 * - Build Link and Joint data structures from parsed XML
 * - Understand URDF structure: robot → links + joints
 * - Find the root link (the one with no parent joint)
 * - Verify tree structure by printing the kinematic chain
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-5f

/* ══════════════════════════════════════════════════════════════════
 * 3D MATH — vec3 and mat4 (provided, implemented)
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;
typedef struct { float w, x, y, z; } quat;
typedef struct { float m[16]; } mat4;

#define M4(mat, row, col) ((mat).m[(col) * 4 + (row)])

static inline vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }
static inline vec3 v3_add(vec3 a, vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3 v3_sub(vec3 a, vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3 v3_scale(vec3 v, float s) { return v3(v.x*s, v.y*s, v.z*s); }
static inline float v3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float v3_len(vec3 v) { return sqrtf(v3_dot(v, v)); }
static inline vec3 v3_norm(vec3 v) {
    float len = v3_len(v);
    return len < EPSILON ? v3(0,0,0) : v3_scale(v, 1.0f/len);
}
static inline vec3 v3_abs(vec3 v) { return v3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
static inline vec3 v3_max(vec3 a, vec3 b) {
    return v3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
static inline float v3_max_comp(vec3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }
static inline float lerpf(float a, float b, float t) { return a + (b-a)*t; }

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
 * URDF DATA STRUCTURES (provided)
 * ══════════════════════════════════════════════════════════════════ */

#define MAX_NAME  64
#define MAX_LINKS  32
#define MAX_JOINTS 32

typedef enum { GEOM_BOX, GEOM_CYLINDER, GEOM_SPHERE } GeomType;

typedef struct {
    GeomType type;
    vec3  size;         /* box: half extents (pre-scaled by 0.5) */
    float radius;       /* cylinder / sphere */
    float length;       /* cylinder */
    vec3  origin_xyz;
    vec3  origin_rpy;
    int   color_id;
} Geometry;

typedef struct {
    char  name[MAX_NAME];
    Geometry visual;
    float mass;
    int   parent_joint;   /* index of joint whose child is this link; -1 for root */
    mat4  world_transform; /* filled in by forward kinematics */
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
    float position;       /* current joint angle / displacement */
} Joint;

typedef struct {
    char  name[MAX_NAME];
    Link  links[MAX_LINKS];
    int   num_links;
    Joint joints[MAX_JOINTS];
    int   num_joints;
    int   root_link;      /* index of link with no parent joint */
} Robot;

static Robot robot;

/* ══════════════════════════════════════════════════════════════════
 * XML PARSER — completed (provided, implemented)
 * The TODOs from phase6a are solved here so you can focus on URDF loading.
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

/* Parse the next XML tag name into out[].
 * Closing tags (</foo>) appear as "foo" — callers must track nesting depth. */
static int parse_tag_name(char *out, int max_len) {
    for (;;) {
        skip_whitespace();
        skip_comment();
        if (*xml_ptr == '\0') return 0;
        if (*xml_ptr == '<') break;
        xml_ptr++;   /* skip text-node characters */
    }
    xml_ptr++;  /* consume '<' */
    /* Processing instruction: <?xml ... ?> */
    if (*xml_ptr == '?') {
        while (*xml_ptr && *xml_ptr != '>') xml_ptr++;
        if (*xml_ptr == '>') xml_ptr++;
        return parse_tag_name(out, max_len);
    }
    /* Closing tag: skip the '/' — caller sees bare name */
    if (*xml_ptr == '/') xml_ptr++;
    int len = 0;
    while (*xml_ptr && !isspace((unsigned char)*xml_ptr)
           && *xml_ptr != '>' && *xml_ptr != '/') {
        if (len < max_len - 1) out[len++] = *xml_ptr;
        xml_ptr++;
    }
    out[len] = '\0';
    return len > 0;
}

/* Parse one name="value" attribute.  Returns 1 on success, 0 when exhausted. */
static int parse_attribute(char *name, int name_max, char *value, int val_max) {
    skip_whitespace();
    if (*xml_ptr == '>' || *xml_ptr == '/' || *xml_ptr == '\0') return 0;
    /* Read attribute name */
    int len = 0;
    while (*xml_ptr && *xml_ptr != '=' && !isspace((unsigned char)*xml_ptr)
           && *xml_ptr != '>' && *xml_ptr != '\0') {
        if (len < name_max - 1) name[len++] = *xml_ptr;
        xml_ptr++;
    }
    name[len] = '\0';
    skip_whitespace();
    if (*xml_ptr != '=') return 0;
    xml_ptr++; /* consume '=' */
    skip_whitespace();
    if (*xml_ptr != '"' && *xml_ptr != '\'') return 0;
    char quote = *xml_ptr++;
    len = 0;
    while (*xml_ptr && *xml_ptr != quote) {
        if (len < val_max - 1) value[len++] = *xml_ptr;
        xml_ptr++;
    }
    value[len] = '\0';
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
 * URDF PARSER HELPERS (provided, implemented)
 * ══════════════════════════════════════════════════════════════════ */

static int find_link_by_name(const char *name) {
    for (int i = 0; i < robot.num_links; i++)
        if (strcmp(robot.links[i].name, name) == 0) return i;
    return -1;
}

/* Parse <geometry> … </geometry> and fill in *geom. */
static void parse_geometry(Geometry *geom) {
    char tag[64], attr_name[64], attr_value[256];
    while (parse_tag_name(tag, sizeof(tag))) {
        if (strcmp(tag, "geometry") == 0) break;
        if (strcmp(tag, "box") == 0) {
            geom->type = GEOM_BOX;
            while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value)))
                if (strcmp(attr_name, "size") == 0)
                    geom->size = v3_scale(parse_vec3(attr_value), 0.5f);
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

/* Parse <origin xyz="…" rpy="…"/> attributes and advance past '>'. */
static void parse_origin(vec3 *xyz, vec3 *rpy) {
    char attr_name[64], attr_value[256];
    while (parse_attribute(attr_name, sizeof(attr_name), attr_value, sizeof(attr_value))) {
        if (strcmp(attr_name, "xyz") == 0) *xyz = parse_vec3(attr_value);
        if (strcmp(attr_name, "rpy") == 0) *rpy = parse_vec3(attr_value);
    }
    skip_to_tag_end();
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: parse_link — parse a <link> element and add it to robot
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
 * Algorithm:
 *   1. Read the "name" attribute from the current tag (the <link ...> opening).
 *      Hint: call parse_attribute() in a loop, looking for attr_name=="name".
 *   2. Bounds-check: if robot.num_links >= MAX_LINKS, skip and return.
 *   3. Add a new Link at robot.links[robot.num_links]:
 *        strncpy(link->name, name_value, MAX_NAME - 1);
 *        link->parent_joint = -1;           (will be set by parse_joint)
 *        link->world_transform = m4_identity();
 *        robot.num_links++;
 *   4. Parse child tags in a loop until the closing </link> is found.
 *      Use parse_tag_name() to read each child tag name.
 *      - "link"     → break  (closing tag — same name as opening)
 *      - "visual"   → parse until </visual>:
 *                       "origin"   → parse_origin(&link->visual.origin_xyz,
 *                                                  &link->visual.origin_rpy)
 *                       "geometry" → parse_geometry(&link->visual)
 *                       "material" → read "name" attribute; map to color_id
 *                                    (use: color_id = name[0] % 8 or similar)
 *                       "visual"   → break  (closing </visual>)
 *                       else       → skip_to_tag_end()
 *      - "inertial" → parse until </inertial>:
 *                       "mass"     → read "value" attribute → link->mass
 *                       "inertial" → break
 *                       else       → skip_to_tag_end()
 *      - else       → skip_to_tag_end()
 *
 * Hint: Robot takes a Robot* parameter so you can access robot.links[].
 * ══════════════════════════════════════════════════════════════════ */
static void parse_link(Robot *r) {
    /* TODO: Parse a <link> XML element and add to r->links[] */
    (void)r;
    (void)parse_geometry; (void)parse_origin; (void)find_link_by_name;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: parse_joint — parse a <joint> element and add it to robot
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
 * Algorithm:
 *   1. Read "name" and "type" attributes from the opening <joint ...> tag.
 *      Map type string → JointType:
 *        "revolute"   → JOINT_REVOLUTE
 *        "fixed"      → JOINT_FIXED
 *        "prismatic"  → JOINT_PRISMATIC
 *        "continuous" → JOINT_CONTINUOUS
 *        default      → JOINT_FIXED
 *   2. Bounds-check: if r->num_joints >= MAX_JOINTS, skip and return.
 *   3. Add a new Joint at r->joints[r->num_joints]:
 *        joint->axis = v3(0,0,1);   (default)
 *        joint->lower_limit = -(float)M_PI;
 *        joint->upper_limit =  (float)M_PI;
 *        r->num_joints++;
 *   4. Parse child tags in a loop until closing </joint>:
 *      - "joint"  → break  (closing tag)
 *      - "parent" → read "link" attribute → joint->parent_link =
 *                     find_link_by_name(value)
 *      - "child"  → read "link" attribute → joint->child_link =
 *                     find_link_by_name(value);
 *                   also set r->links[child_idx].parent_joint = joint_index
 *      - "origin" → parse_origin(&joint->origin_xyz, &joint->origin_rpy)
 *      - "axis"   → read "xyz" attribute → joint->axis = parse_vec3(value)
 *      - "limit"  → read "lower"/"upper" attributes
 *      - else     → skip_to_tag_end()
 * ══════════════════════════════════════════════════════════════════ */
static void parse_joint(Robot *r) {
    /* TODO: Parse a <joint> XML element and add to r->joints[] */
    (void)r;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: load_urdf — parse the entire URDF file into *r
 *
 * Algorithm:
 *   1. memset(r, 0, sizeof(*r));
 *      r->root_link = -1;
 *      for (int i = 0; i < MAX_LINKS; i++) r->links[i].parent_joint = -1;
 *   2. Load the file into xml_content using load_file() below.
 *      On failure print "Error: cannot open '<filename>'\n" and return 0.
 *   3. Set xml_ptr = xml_content.
 *   4. Loop calling parse_tag_name(tag, ...):
 *      - "robot"  → read "name" attribute → strncpy(r->name, ...)
 *                   then skip_to_tag_end()
 *      - "link"   → parse_link(r)
 *      - "joint"  → parse_joint(r)
 *      - else     → skip_to_tag_end()
 *   5. Find the root link: the first link with parent_joint == -1.
 *      Store its index in r->root_link.
 *   6. Return 1 on success.
 *
 * Helper (provided, implemented):
 * ══════════════════════════════════════════════════════════════════ */
static long load_file(const char *filename, char **out) {
    FILE *f = fopen(filename, "r");
    if (!f) { perror(filename); return -1; }
    fseek(f, 0, SEEK_END); long size = ftell(f); rewind(f);
    char *buf = malloc(size + 1);
    if (!buf) { fclose(f); return -1; }
    size_t n = fread(buf, 1, size, f);
    buf[n] = '\0';
    fclose(f);
    *out = buf;
    return (long)n;
}

static int load_urdf(Robot *r, const char *filename) {
    /* TODO: Clear robot, load file, parse all tags, find root link */
    (void)r; (void)filename;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: print_robot_tree — print the kinematic chain as ASCII art
 *
 * Walk the kinematic tree starting at the root link, following joints
 * depth-first.  For each level of depth, indent with two spaces.
 *
 * Expected output:
 *   Robot: my_robot (5 links, 4 joints)
 *     [0] base_link  (sphere r=0.05)
 *       joint0 (revolute) → [1] link1  (cylinder r=0.02 l=0.30)
 *         joint1 (revolute) → [2] link2  (box 0.10×0.10×0.10)
 *         ...
 *
 * Algorithm (recursive helper print_link(r, link_idx, depth)):
 *   1. Print "  " * depth then link info:
 *        "[<idx>] <name>  (<geom type and size>)"
 *      Geometry formatting:
 *        GEOM_SPHERE   → "sphere r=%.2f"
 *        GEOM_CYLINDER → "cylinder r=%.2f l=%.2f"
 *        GEOM_BOX      → "box %.2f×%.2f×%.2f"  (sizes are half-extents×2)
 *   2. For each joint j where j.parent_link == link_idx:
 *        Print "  " * (depth+1) then:
 *          "<joint_name> (<joint_type>) → "
 *        Call print_link(r, j.child_link, depth + 2).
 *
 * Joint type strings: "fixed" / "revolute" / "prismatic" / "continuous"
 *
 * print_robot_tree() prints the header and calls print_link(r, r->root_link, 1).
 * ══════════════════════════════════════════════════════════════════ */
static void print_robot_tree(Robot *r) {
    /* TODO: Print ASCII tree of the kinematic chain */
    (void)r;
    (void)m4_mul; (void)m4_translate; (void)m4_rotate_axis;
    (void)m4_transform_point; (void)m4_inverse; (void)lerpf;
    (void)v3_abs; (void)v3_max; (void)v3_max_comp;
    (void)v3_add; (void)v3_sub; (void)v3_dot;
}

int main(int argc, char *argv[]) {
    const char *filename = argc > 1 ? argv[1] : "robot.urdf";
    if (!load_urdf(&robot, filename)) {
        fprintf(stderr, "Error: could not load '%s'\n", filename);
        return 1;
    }
    print_robot_tree(&robot);
    free(xml_content);
    return 0;
}
