/*
 * phase3a_sdf_prims.c — SDF Primitive Functions
 *
 * EXERCISE (sub-phase 3a): Implement Signed Distance Field primitives
 * and test them by evaluating distances at known points.
 *
 * Build: gcc -O2 -o phase3a_sdf_prims phase3a_sdf_prims.c -lm
 * Run:   ./phase3a_sdf_prims
 *
 * LEARNING GOALS:
 * - Understand what a Signed Distance Field (SDF) is:
 *   a function f(p) that returns the shortest signed distance from
 *   point p to a surface (negative inside, zero on surface, positive outside)
 * - Implement SDF for sphere, box, torus, and capsule
 * - Understand SDF boolean operations: union, intersection, subtraction
 * - Test your SDFs at known points to verify correctness
 */

/* ══════════════════════════════════════════════════════════════════
 * TECHNIQUE OVERVIEW
 * ══════════════════════════════════════════════════════════════════
 *
 * 1. What is an SDF (Signed Distance Field)?
 *    A function f(p) that, for any 3D point p, returns the shortest
 *    signed distance to a surface.  Negative = inside the object,
 *    zero = on the surface, positive = outside.  The "signed" part is
 *    what makes them powerful: you always know if you're inside or
 *    outside.
 *
 * 2. Why SDFs are useful
 *    (a) Ray marching: you can safely step a ray forward by f(p) each
 *        iteration — you can never overshoot the surface.
 *    (b) Boolean operations: union is just min(), intersection is
 *        max(), subtraction is max(a,-b).
 *    (c) Smooth blending: smin() lets two objects blend into each
 *        other.
 *    These are impossible with triangle meshes.
 *
 * 3. The box SDF trick
 *    The formula `length(max(q,0)) + min(max_component(q),0)`
 *    elegantly handles all three cases:
 *    (a) Outside all faces: q has positive components, max(q,0)=q,
 *        length gives the corner distance.
 *    (b) On a face: one component 0, length is edge distance.
 *    (c) Inside: all q negative, max(q,0)=zero vector so length=0,
 *        min(max_comp,0) gives the negative interior distance (the
 *        penetration depth of the nearest face).
 *
 * 4. SDF combinators
 *    Already provided in the file:
 *      sdf_union(a,b)     = min(a,b)  — take the closer surface
 *      sdf_intersection(a,b) = max(a,b)  — keep only overlap
 *      sdf_subtract(a,b)  = max(a,-b) — cut b out of a
 *    Note that sdf_subtract negates b to flip its inside/outside.
 *
 * 5. 1e10f sentinel
 *    The TODO stubs return 1e10f (10 billion), which acts as "no
 *    intersection" (infinitely far away).  The ray marcher stops when
 *    a hit is detected (SDF < small epsilon), so returning a huge
 *    value means "miss".
 * ══════════════════════════════════════════════════════════════════ */

#include <stdio.h>
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
static inline float v3_min_comp(vec3 v) { return fminf(v.x, fminf(v.y, v.z)); }

/* ══════════════════════════════════════════════════════════════════
 * Ground plane — provided for reference
 *
 * A plane is defined by a unit normal and a scalar offset.
 * The SDF is simply the dot product of point with the normal plus offset.
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_plane(vec3 p, vec3 normal, float offset) {
    return v3_dot(p, normal) + offset;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement SDF for a sphere
 *
 * A sphere SDF returns the distance from point p to the sphere surface.
 * Formula: length(p - center) - radius
 *   - Returns negative when p is INSIDE the sphere
 *   - Returns zero on the surface
 *   - Returns positive when p is OUTSIDE
 *
 * Hint: use v3_sub() and v3_len()
 *
 * Test: sdf_sphere(v3(1,0,0), v3(0,0,0), 1.0f) should return ~0.0
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_sphere(vec3 p, vec3 center, float radius) {
    /* TODO: Implement sphere SDF
     * The sphere is the simplest SDF: distance to center minus radius.
     * In code: `return v3_len(v3_sub(p, center)) - radius;`
     * Think of it as: if you draw a circle of radius `r` around the
     * center, any point on that circle is at distance 0 from the
     * sphere surface.
     */
    (void)p; (void)center; (void)radius;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement SDF for an axis-aligned box
 *
 * Box SDF formula for a box centered at 'center' with half-extents 'half_size':
 *   q = abs(p - center) - half_size       [componentwise]
 *   d = length(max(q, 0)) + min(max_component(q), 0)
 *
 * The length(max(q,0)) computes the exterior distance (distance to nearest
 * corner/edge/face when outside). The min(max_component(q), 0) adds the
 * signed interior distance (the most negative component when inside).
 *
 * Hint: use v3_abs(), v3_sub(), v3_max(), v3_max_comp(), v3_len()
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_box(vec3 p, vec3 center, vec3 half_size) {
    /* TODO: Implement box SDF
     * Step by step:
     * (1) `q = v3_abs(v3_sub(p, center))` — fold space into the
     *     positive octant (exploit box symmetry).  Then
     *     `q = v3_sub(q, half_size)` — shift so q.x=0 at the face,
     *     positive outside, negative inside.
     * (2) `v3_max(q, v3(0,0,0))` keeps only the outside (positive)
     *     parts — for corners/edges this is the 3D distance vector to
     *     the nearest corner.
     * (3) `v3_max_comp(q)` is the largest component — if inside the
     *     box (all q negative), this is the least-negative, i.e.
     *     distance to nearest face from inside (negative).
     * Combine:
     *   `v3_len(v3_max(q,0)) + fminf(v3_max_comp(q), 0.0f)`
     */
    (void)p; (void)center; (void)half_size;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement SDF for a torus
 *
 * A torus centered at 'center' with major radius R and minor radius r:
 *   rel = p - center
 *   q   = sqrt(rel.x^2 + rel.z^2) - R   [ring radius in XZ plane]
 *   d   = sqrt(q^2 + rel.y^2) - r
 *
 * Think of it as revolving a circle of radius r around the Y axis
 * at distance R from the center. The torus lies in the XZ plane.
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_torus(vec3 p, vec3 center, float R, float r) {
    /* TODO: Implement torus SDF
     * `q = sqrtf(rel.x*rel.x + rel.z*rel.z) - R` computes how far the
     * point is from the torus ring axis (a circle of radius R in the
     * XZ plane).  The result q is now a 2D "distance from the ring"
     * scalar.  Then `sqrtf(q*q + rel.y*rel.y) - r` is just a 2D
     * circle SDF in the (q, rel.y) plane — finding the distance to
     * the tube of radius r.
     */
    (void)p; (void)center; (void)R; (void)r;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement SDF for a capsule
 *
 * A capsule is a sphere swept along a line segment from point a to point b:
 *   pa = p - a
 *   ba = b - a
 *   h  = clamp(dot(pa, ba) / dot(ba, ba), 0, 1)   [closest point param]
 *   d  = length(pa - ba * h) - r
 *
 * h=0 clamps to endpoint a, h=1 clamps to endpoint b.
 * This gives the signed distance to the swept sphere.
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    /* TODO: Implement capsule SDF
     * The key insight: project the query point p onto the line segment
     * a→b.  `h = clamp(dot(pa,ba)/dot(ba,ba), 0, 1)` gives the
     * parameter of the closest point.  When h=0, the closest point is
     * a; when h=1 it's b; in between it's on the segment.  The
     * distance from p to that closest point minus r gives the SDF.
     * `dot(ba,ba)` = `|ba|²` normalizes the parameter to [0,1] range.
     */
    (void)p; (void)a; (void)b; (void)r;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement SDF for a finite cylinder
 *
 * A finite cylinder with base centre at 'base', given 'radius' and 'height'
 * (the cylinder spans from base.y to base.y + height):
 *   rel        = p - base
 *   d_radial   = sqrt(rel.x^2 + rel.z^2) - radius
 *   d_vertical = |rel.y - height/2| - height/2
 *
 * Combine like the box SDF:
 *   d = length(max(d_radial, 0), max(d_vertical, 0))
 *       + min(max(d_radial, d_vertical), 0)
 *
 * The second term is negative inside and zero outside (interior handling).
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_cylinder(vec3 p, vec3 base, float radius, float height) {
    /* TODO: Implement cylinder SDF
     * We decompose into 2D:
     * (1) `d_radial = sqrt(rel.x²+rel.z²) - radius` is the signed
     *     distance from the infinite cylinder wall.
     * (2) `d_vertical = fabsf(rel.y - height/2.0f) - height/2.0f` is
     *     the signed distance from the top/bottom cap planes (we center
     *     rel.y at height/2).
     * Then combine like the box SDF:
     *   `d = length(max2(v2(d_radial,d_vertical), 0))
     *        + min(max(d_radial,d_vertical), 0)`
     * When both d_radial<0 and d_vertical<0 (inside), the combined
     * distance is `max(d_radial,d_vertical)` (distance to nearest
     * boundary).
     */
    (void)p; (void)base; (void)radius; (void)height;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #6: Implement approximate SDF for an ellipsoid
 *
 * An ellipsoid centered at 'center' with semi-axes 'radii' (rx, ry, rz):
 *   q      = p - center
 *   scaled = (q.x/radii.x,  q.y/radii.y,  q.z/radii.z)
 *   d      = (length(scaled) - 1.0) * min(radii.x, radii.y, radii.z)
 *
 * Dividing by radii "squashes" space so the ellipsoid becomes a unit sphere,
 * then we scale back out by the smallest radius for an approximate world-space
 * distance. Returns 0 on the surface, negative inside, positive outside.
 *
 * Note: This is an approximation — the exact ellipsoid SDF has no closed form.
 *
 * Hint: use v3_min_comp(), v3_len()
 * ══════════════════════════════════════════════════════════════════ */
static float sdf_ellipsoid(vec3 p, vec3 center, vec3 radii) {
    /* TODO: Implement approximate ellipsoid SDF
     * `q = v3_sub(p,center)`.  Scale:
     * `v3(q.x/radii.x, q.y/radii.y, q.z/radii.z)` transforms space
     * so the ellipsoid becomes the unit sphere.  `v3_len(scaled)-1.0f`
     * is the unit-sphere SDF in scaled space.  Multiplying by
     * `v3_min_comp(radii)` converts back to an approximate world-space
     * distance.  This underestimates the true distance (so it's
     * conservative — you won't overshoot), making it safe for ray
     * marching.
     */
    (void)p; (void)center; (void)radii;
    return 1e10f;
}

/* ══════════════════════════════════════════════════════════════════
 * SDF combinators — provided
 *
 * These combine multiple SDF primitives into complex shapes.
 *   union      — closest surface (logical OR)
 *   intersect  — only where both shapes overlap (logical AND)
 *   subtract   — cut shape b out of shape a (logical NOT b)
 *   smooth_union — blended/melted union with radius k
 *
 * The union combinator min(a,b) picks the closer surface — this is
 * how scenes are built: union all primitives together and you get a
 * single SDF for the whole scene.  The smooth-min variant (smin)
 * blends surfaces within distance k, creating organic merging of
 * shapes.
 * ══════════════════════════════════════════════════════════════════ */
static inline float sdf_union(float a, float b)     { return fminf(a, b); }
static inline float sdf_intersect(float a, float b) { return fmaxf(a, b); }
static inline float sdf_subtract(float a, float b)  { return fmaxf(a, -b); }
static float sdf_smooth_union(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

/* ══════════════════════════════════════════════════════════════════
 * Unit test helpers
 * ══════════════════════════════════════════════════════════════════ */

#define TOLERANCE 1e-4f

static int g_pass = 0, g_fail = 0;

static void check(float val, float expected, const char *name) {
    if (fabsf(val - expected) < TOLERANCE) {
        printf("  PASS: %-40s got %+.6f\n", name, val);
        g_pass++;
    } else {
        printf("  FAIL: %-40s got %+.6f, expected %+.6f\n", name, val, expected);
        g_fail++;
    }
}

static void check_pos(float val, const char *name) {
    if (val > 0.0f) {
        printf("  PASS: %-40s got %+.6f (positive)\n", name, val);
        g_pass++;
    } else {
        printf("  FAIL: %-40s got %+.6f (expected positive)\n", name, val);
        g_fail++;
    }
}

static void check_neg(float val, const char *name) {
    if (val < 0.0f) {
        printf("  PASS: %-40s got %+.6f (negative)\n", name, val);
        g_pass++;
    } else {
        printf("  FAIL: %-40s got %+.6f (expected negative)\n", name, val);
        g_fail++;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Main — unit tests
 * ══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("=== Phase 3a: SDF Primitive Tests ===\n\n");

    /* ── Plane (provided — should always pass) ───────────────────── */
    printf("Plane (normal=(0,1,0), offset=0):\n");
    check_pos(sdf_plane(v3(0,  1, 0), v3(0,1,0), 0.0f), "above plane (0, 1,0)");
    check    (sdf_plane(v3(0,  0, 0), v3(0,1,0), 0.0f), 0.0f, "on plane    (0, 0,0)");
    check_neg(sdf_plane(v3(0, -1, 0), v3(0,1,0), 0.0f), "below plane (0,-1,0)");
    printf("\n");

    /* ── Sphere ──────────────────────────────────────────────────── */
    printf("Sphere (center=origin, radius=1.0):\n");
    check_pos(sdf_sphere(v3(2, 0, 0), v3(0,0,0), 1.0f), "exterior (2,0,0)");
    check    (sdf_sphere(v3(1, 0, 0), v3(0,0,0), 1.0f),  0.0f, "surface  (1,0,0)");
    check    (sdf_sphere(v3(0, 0, 0), v3(0,0,0), 1.0f), -1.0f, "interior (0,0,0)");
    printf("\n");

    /* ── Box ─────────────────────────────────────────────────────── */
    printf("Box (center=origin, half_size=(1,1,1)):\n");
    check_pos(sdf_box(v3(2,    0, 0), v3(0,0,0), v3(1,1,1)), "exterior (2,0,0)");
    check    (sdf_box(v3(1,    0, 0), v3(0,0,0), v3(1,1,1)),  0.0f, "surface  (1,0,0)");
    check    (sdf_box(v3(0,    0, 0), v3(0,0,0), v3(1,1,1)), -1.0f, "interior (0,0,0)");
    printf("\n");

    /* ── Torus ───────────────────────────────────────────────────── */
    printf("Torus (center=origin, R=1.0, r=0.25):\n");
    check_pos(sdf_torus(v3(2.5f,  0, 0), v3(0,0,0), 1.0f, 0.25f), "exterior (2.5,0,0)");
    check    (sdf_torus(v3(1.25f, 0, 0), v3(0,0,0), 1.0f, 0.25f),  0.0f, "surface  (1.25,0,0)");
    check_neg(sdf_torus(v3(1.0f,  0, 0), v3(0,0,0), 1.0f, 0.25f), "interior (1,0,0)");
    printf("\n");

    /* ── Capsule ─────────────────────────────────────────────────── */
    printf("Capsule (a=(0,-1,0), b=(0,1,0), r=0.5):\n");
    check_pos(sdf_capsule(v3(1,    0, 0), v3(0,-1,0), v3(0,1,0), 0.5f), "exterior (1,0,0)");
    check    (sdf_capsule(v3(0.5f, 0, 0), v3(0,-1,0), v3(0,1,0), 0.5f),  0.0f, "surface  (0.5,0,0)");
    check_neg(sdf_capsule(v3(0.2f, 0, 0), v3(0,-1,0), v3(0,1,0), 0.5f), "interior (0.2,0,0)");
    printf("\n");

    /* ── Cylinder ────────────────────────────────────────────────── */
    /* Cylinder spans y=0..2 (base at origin, height=2, so centre at y=1) */
    printf("Cylinder (base=origin, radius=1.0, height=2.0):\n");
    check_pos(sdf_cylinder(v3(2, 1, 0), v3(0,0,0), 1.0f, 2.0f), "exterior (2,1,0)");
    check    (sdf_cylinder(v3(1, 1, 0), v3(0,0,0), 1.0f, 2.0f),  0.0f, "surface  (1,1,0)");
    check_neg(sdf_cylinder(v3(0, 1, 0), v3(0,0,0), 1.0f, 2.0f), "interior (0,1,0)");
    printf("\n");

    /* ── Ellipsoid ───────────────────────────────────────────────── */
    /* Ellipsoid with rx=2, ry=1, rz=1.5 — surface point along X at (2,0,0) */
    printf("Ellipsoid (center=origin, radii=(2,1,1.5)):\n");
    check_pos(sdf_ellipsoid(v3(3, 0, 0), v3(0,0,0), v3(2,1,1.5f)), "exterior (3,0,0)");
    check    (sdf_ellipsoid(v3(2, 0, 0), v3(0,0,0), v3(2,1,1.5f)),  0.0f, "surface  (2,0,0)");
    check_neg(sdf_ellipsoid(v3(1, 0, 0), v3(0,0,0), v3(2,1,1.5f)), "interior (1,0,0)");
    printf("\n");

    /* ── Combinators ─────────────────────────────────────────────── */
    /* Two overlapping spheres centred at ±1 on X, radius 1.2 — origin is inside both */
    printf("Combinators (two spheres at x=±1, r=1.2, tested at origin):\n");
    {
        float sa = sdf_sphere(v3(0,0,0), v3(-1,0,0), 1.2f);
        float sb = sdf_sphere(v3(0,0,0), v3( 1,0,0), 1.2f);
        check_neg(sdf_union(sa, sb),      "union     (inside both → negative)");
        check_neg(sdf_intersect(sa, sb),  "intersect (inside both → negative)");
        check_pos(sdf_subtract(sb, sa),   "subtract  (cut sa from sb → positive at origin)");
        /* Smooth union should be <= regular union */
        float su = sdf_smooth_union(sa, sb, 0.5f);
        if (su <= sdf_union(sa, sb) + TOLERANCE)
            printf("  PASS: %-40s smooth_union(%.4f) <= union(%.4f)\n",
                   "smooth_union ≤ union", su, sdf_union(sa, sb));
        else
            printf("  FAIL: %-40s smooth_union(%.4f) > union(%.4f)\n",
                   "smooth_union ≤ union", su, sdf_union(sa, sb));
    }
    printf("\n");

    printf("=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
