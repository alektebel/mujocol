/*
 * phase6a_xml_parse.c — Minimal XML/URDF Parser
 *
 * EXERCISE (sub-phase 6a): Implement a character-by-character XML parser
 * that can extract tag names and attribute key-value pairs from URDF files.
 *
 * Build: gcc -O2 -o phase6a_xml_parse phase6a_xml_parse.c -lm
 * Run:   ./phase6a_xml_parse robot.urdf
 *
 * LEARNING GOALS:
 * - Parse XML by scanning characters manually (no external libraries)
 * - Extract attribute name=value pairs from XML tags
 * - Handle XML whitespace, comments, and self-closing tags
 * - Test your parser by printing all link names and joint names from a URDF
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

/* ══════════════════════════════════════════════════════════════════
 * MINIMAL MATH HELPERS
 * Used by parse_vec3 / parse_float (provided, implemented).
 * ══════════════════════════════════════════════════════════════════ */

typedef struct { float x, y, z; } vec3;

static vec3 v3(float x, float y, float z) { return (vec3){x, y, z}; }

/* ══════════════════════════════════════════════════════════════════
 * XML PARSER STATE
 * ══════════════════════════════════════════════════════════════════ */

static char *xml_content = NULL;
static char *xml_ptr     = NULL;

/* ── Provided helpers (implemented) ────────────────────────────── */

/* Skip ASCII whitespace at the current position. */
static void skip_whitespace(void) {
    while (*xml_ptr && isspace((unsigned char)*xml_ptr)) xml_ptr++;
}

/* If the current position starts an XML comment (<!--), skip past -->. */
static void skip_comment(void) {
    if (strncmp(xml_ptr, "<!--", 4) == 0) {
        char *end = strstr(xml_ptr, "-->");
        if (end) xml_ptr = end + 3;
    }
}

/* Advance past the closing '>' of the current tag. */
static void skip_to_tag_end(void) {
    while (*xml_ptr && *xml_ptr != '>') xml_ptr++;
    if (*xml_ptr == '>') xml_ptr++;
}

/* Jump xml_ptr past the next occurrence of </tag>. */
static void skip_to_closing_tag(const char *tag) {
    char closing[128];
    snprintf(closing, sizeof(closing), "</%s>", tag);
    char *end = strstr(xml_ptr, closing);
    if (end) xml_ptr = end + strlen(closing);
}

/* Parse a space-separated vec3, e.g. "1.0 0.0 -0.5". */
static vec3 parse_vec3(const char *s) {
    vec3 v = {0, 0, 0};
    sscanf(s, "%f %f %f", &v.x, &v.y, &v.z);
    (void)v3; /* suppress unused-function warning */
    return v;
}

/* Parse a single float from a string. */
static float parse_float(const char *s) {
    float f = 0;
    sscanf(s, "%f", &f);
    return f;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: parse_tag_name — scan the next XML tag name from xml_ptr
 *
 * Algorithm:
 *   1. Loop: skip_whitespace(), then skip_comment().
 *      Repeat while *xml_ptr != '<' (stay in this loop skipping text nodes).
 *      Break out when *xml_ptr == '<' or *xml_ptr == '\0'.
 *   2. If *xml_ptr is '\0' (end of input), return 0.
 *   3. Advance past '<'.
 *   4. If next char is '?': skip to '>' (XML processing instruction) and
 *      loop back to step 1.
 *   5. If next char is '/': it is a closing tag.
 *      CHOICE: skip the '/' so out[] receives just the bare tag name.
 *      The caller distinguishes closing tags by tracking nesting depth.
 *      (Alternative: keep the '/' prefix in out[] and document that choice.)
 *   6. Read characters into out[] while the char is not whitespace, '>', '/':
 *        out[len++] = *xml_ptr++
 *      Stop when len == max_len - 1.
 *   7. Null-terminate out[len] and return 1 if len > 0, else 0.
 *
 * Return 1 on success, 0 if no tag was found.
 * ══════════════════════════════════════════════════════════════════ */
static int parse_tag_name(char *out, int max_len) {
    /* TODO: Parse next XML tag name from xml_ptr */
    (void)out; (void)max_len;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: parse_attribute — parse one name="value" pair
 *
 * The parser is positioned somewhere inside a tag (after the tag name).
 *
 * Algorithm:
 *   1. skip_whitespace().
 *   2. If *xml_ptr is '>', '/', or '\0': no more attributes — return 0.
 *   3. Read attribute name into name[] until '=', whitespace, or end.
 *      Null-terminate.
 *   4. skip_whitespace(); expect '=' — if not found return 0; advance past it.
 *   5. skip_whitespace(); expect '"' or '\''.
 *      Save the quote character, then advance past it.
 *   6. Read into value[] until the matching closing quote (or end of string).
 *      Null-terminate.
 *   7. Advance past the closing quote.  Return 1.
 *
 * Return 1 if an attribute was successfully parsed, 0 otherwise.
 * ══════════════════════════════════════════════════════════════════ */
static int parse_attribute(char *name, int name_max, char *value, int val_max) {
    /* TODO: Parse one XML attribute (name="value") */
    (void)name; (void)name_max; (void)value; (void)val_max;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: load_file — read an entire file into a heap buffer
 *
 * Algorithm:
 *   1. fopen(filename, "r") — print error and return -1 on failure.
 *   2. fseek(f, 0, SEEK_END); size = ftell(f); rewind(f).
 *   3. buf = malloc(size + 1) — return -1 on allocation failure.
 *   4. fread(buf, 1, size, f); buf[size] = '\0'; fclose(f).
 *   5. Store the pointer in *out and return size.
 *
 * Return the file size (>= 0) on success, -1 on error.
 * ══════════════════════════════════════════════════════════════════ */
static long load_file(const char *filename, char **out) {
    /* TODO: Open file, allocate buffer, read contents, null-terminate */
    (void)filename; (void)out;
    return -1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: xml_print_structure — demonstrate the parser
 *
 * Algorithm:
 *   1. Call load_file(filename, &xml_content).
 *      On failure print "Error: cannot open '<filename>'\n" and return.
 *   2. Set xml_ptr = xml_content.
 *   3. Declare int link_count = 0, joint_count = 0.
 *   4. Loop while parse_tag_name(tag, sizeof(tag)) succeeds:
 *      a. printf("<%s>", tag).
 *      b. Loop parse_attribute(aname, ..., aval, ...):
 *           printf("  %s=\"%s\"", aname, aval)
 *         Track link/joint counts:
 *           if strcmp(tag,"link")==0 && strcmp(aname,"name")==0 → link_count++
 *           if strcmp(tag,"joint")==0 && strcmp(aname,"name")==0 → joint_count++
 *      c. printf("\n").
 *      d. Call skip_to_tag_end() to advance past the '>'.
 *   5. After the loop:
 *      printf("Total: %d links, %d joints\n", link_count, joint_count).
 *
 * Note: closing tags (e.g. </link>) will also be printed; that is fine for
 * this diagnostic tool.
 *
 * Expected output (excerpt):
 *   <robot>  name="my_robot"
 *   <link>   name="base_link"
 *   ...
 *   Total: 5 links, 4 joints
 * ══════════════════════════════════════════════════════════════════ */
static void xml_print_structure(const char *filename) {
    /* TODO: Load file and print all tags with their attributes */
    (void)filename;
    (void)parse_vec3; (void)parse_float;            /* suppress unused warnings */
    (void)skip_to_closing_tag; (void)skip_comment;  /* provided but not used here */
}

int main(int argc, char *argv[]) {
    const char *filename = argc > 1 ? argv[1] : "robot.urdf";
    xml_print_structure(filename);
    free(xml_content);
    return 0;
}
