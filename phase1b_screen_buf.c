/*
 * phase1b_screen_buf.c — Double-Buffered Screen Rendering
 *
 * EXERCISE (sub-phase 1b): Implement a double-buffered terminal renderer
 * using Cell structs and ANSI 256-color escape sequences.
 *
 * Build: gcc -O2 -o phase1b_screen_buf phase1b_screen_buf.c -lm
 * Run:   ./phase1b_screen_buf
 * Quit:  Press any key
 *
 * LEARNING GOALS:
 * - Represent screen state as an array of Cell structs
 * - Implement double buffering: back_buf → present → front_buf
 * - Only emit ANSI sequences for cells that changed (differential update)
 * - Use \033[row;colH to position cursor (1-indexed)
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
#include <math.h>
#include <signal.h>

/* ── Terminal dimensions ────────────────────────────────────────── */
static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;

/* ── Screen buffer cell ─────────────────────────────────────────── */
typedef struct {
    char    ch;     /* character to display        */
    uint8_t fg;     /* foreground 256-color index  */
    uint8_t bg;     /* background 256-color index  */
} Cell;

/* ── Double buffer ──────────────────────────────────────────────── */
static Cell *front_buf = NULL;
static Cell *back_buf  = NULL;
static int   buf_w = 0, buf_h = 0;

/* ── Output buffer for batched writes ───────────────────────────── */
static char *out_buf = NULL;
static int   out_cap = 0, out_len = 0;

static void out_flush(void) {
    if (out_len > 0) {
        write(STDOUT_FILENO, out_buf, out_len);
        out_len = 0;
    }
}

static void out_append(const char *s, int n) {
    while (out_len + n > out_cap) {
        out_cap = out_cap ? out_cap * 2 : 65536;
        out_buf = realloc(out_buf, out_cap);
    }
    memcpy(out_buf + out_len, s, n);
    out_len += n;
}

/* ── Terminal size ──────────────────────────────────────────────── */
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        term_w = ws.ws_col;
        term_h = ws.ws_row;
    }
}

/* ── Raw mode (fully provided for this sub-phase) ───────────────── */
static void disable_raw_mode(void) {
    if (!raw_mode_enabled) return;
    write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    raw_mode_enabled = 0;
}

static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);
    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |=  (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
    raw_mode_enabled = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement alloc_buffers(int w, int h)
 *
 * Should:
 *   - free() any existing front_buf and back_buf
 *   - Store w and h in buf_w, buf_h
 *   - malloc both buffers: n = w * h cells
 *   - Zero-initialize back_buf with calloc or memset
 *   - Initialize front_buf to impossible values to force a full first
 *     redraw: set every cell's fg = 255, bg = 255, ch = 0
 * ══════════════════════════════════════════════════════════════════ */
static void alloc_buffers(int w, int h) {
    /* TODO: Allocate and initialize front/back cell buffers */
    (void)w; (void)h;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement clear_back_buf(char ch, uint8_t fg, uint8_t bg)
 *
 * Fill every cell in back_buf (buf_w * buf_h cells) with the given
 * character, foreground color, and background color.
 * ══════════════════════════════════════════════════════════════════ */
static void clear_back_buf(char ch, uint8_t fg, uint8_t bg) {
    /* TODO: Fill back_buf uniformly */
    (void)ch; (void)fg; (void)bg;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement set_cell(int x, int y, char ch, uint8_t fg, uint8_t bg)
 *
 * Should:
 *   - Return immediately if x or y is out of bounds
 *   - Write ch, fg, bg into back_buf[y * buf_w + x]
 * ══════════════════════════════════════════════════════════════════ */
static void set_cell(int x, int y, char ch, uint8_t fg, uint8_t bg) {
    /* TODO: Bounds-check and write cell into back_buf */
    (void)x; (void)y; (void)ch; (void)fg; (void)bg;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement present()
 *
 * This is the heart of the double-buffer system.  For every cell i:
 *   - Compute row = i / buf_w, col = i % buf_w  (0-indexed)
 *   - If back_buf[i] equals front_buf[i], skip it (no change)
 *   - Otherwise:
 *       1. Move cursor: snprintf "\033[<row+1>;<col+1>H" and out_append()
 *       2. Set foreground if it changed from prev_fg:
 *            "\033[38;5;<fg>m"
 *       3. Set background if it changed from prev_bg:
 *            "\033[48;5;<bg>m"
 *       4. Append the single character with out_append()
 *       5. Copy back_buf[i] → front_buf[i]
 *       6. Update prev_fg / prev_bg
 *   - After the loop, call out_flush()
 *
 * Use local variables prev_fg and prev_bg (initialise to 255) to
 * avoid emitting redundant color-change sequences.
 * ══════════════════════════════════════════════════════════════════ */
static void present(void) {
    /* TODO: Render changed cells to the terminal */
    (void)front_buf;
    (void)back_buf;
    out_flush();
}

/* ── Draw a string into back buffer (provided) ──────────────────── */
static void draw_str(int x, int y, const char *s, uint8_t fg, uint8_t bg) {
    while (*s) {
        set_cell(x++, y, *s++, fg, bg);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement draw_box(int x, int y, int w, int h,
 *                             uint8_t fg, uint8_t bg)
 *
 * Draw a rectangular border using ASCII box-drawing characters:
 *   - Corners : '+'
 *   - Top/bottom edges : '-'
 *   - Left/right edges : '|'
 *
 * Only draw the border cells; leave the interior untouched.
 * All characters use the provided fg and bg color indices.
 * ══════════════════════════════════════════════════════════════════ */
static void draw_box(int x, int y, int w, int h, uint8_t fg, uint8_t bg) {
    /* TODO: Draw box border into back_buf */
    (void)x; (void)y; (void)w; (void)h; (void)fg; (void)bg;
}

/* ── Main ───────────────────────────────────────────────────────── */
int main(void) {
    enable_raw_mode();
    get_term_size();
    alloc_buffers(term_w, term_h);

    /* Background fill */
    clear_back_buf(' ', 7, 232);

    /* Outer frame */
    draw_box(0, 0, buf_w, buf_h, 245, 232);

    /* A few nested colored boxes */
    draw_box(2,  1, 30, 10, 196, 232);
    draw_box(4,  2, 26,  8, 208, 232);
    draw_box(6,  3, 22,  6, 226, 232);

    draw_box(buf_w - 32, 1, 30, 10, 51, 232);
    draw_box(buf_w - 30, 2, 26,  8, 87, 232);
    draw_box(buf_w - 28, 3, 22,  6, 123, 232);

    /* Labels */
    draw_str(2, 0,         " phase1b: double-buffered screen ", 46, 232);
    draw_str(2, buf_h - 1, " press any key to quit ",          250, 232);

    draw_str(4,  2, "box A (red)",    196, 232);
    draw_str(buf_w - 30, 2, "box B (cyan)", 51, 232);

    /* Center info panel */
    int px = buf_w / 2 - 16, py = buf_h / 2 - 2;
    draw_box(px, py, 32, 5, 220, 232);
    draw_str(px + 2, py + 1, "Double-Buffer Renderer", 220, 232);
    draw_str(px + 2, py + 2, "Only diffs are redrawn!", 250, 232);

    present();

    /* Wait for any keypress */
    char c;
    read(STDIN_FILENO, &c, 1);

    free(front_buf);
    free(back_buf);
    free(out_buf);
    return 0;
}
