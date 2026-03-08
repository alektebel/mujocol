/*
 * phase1_term.c — Terminal Framework
 *
 * EXERCISE: Implement raw terminal mode, double-buffered screen rendering,
 * ANSI 256-color output, frame timing, and non-blocking keyboard input.
 *
 * Build: gcc -O2 -o phase1_term phase1_term.c -lm
 * Run:   ./phase1_term
 * Quit:  Press 'q' or Escape
 *
 * LEARNING GOALS:
 * - Understand termios for raw terminal mode
 * - Learn ANSI escape sequences for cursor/color control
 * - Implement double buffering for flicker-free rendering
 * - Use poll() for non-blocking input
 * - Use clock_gettime() for precise timing
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

/* ── Terminal dimensions ────────────────────────────────────────── */
static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;

/* ── Screen buffer cell ─────────────────────────────────────────── */
typedef struct {
    char ch;        /* character to display */
    uint8_t fg;     /* foreground 256-color index */
    uint8_t bg;     /* background 256-color index */
} Cell;

/* ── Double buffer ──────────────────────────────────────────────── */
static Cell *front_buf = NULL;
static Cell *back_buf  = NULL;
static int buf_w = 0, buf_h = 0;

/* ── Output buffer for batched writes ───────────────────────────── */
static char *out_buf = NULL;
static int out_cap = 0, out_len = 0;

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

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement raw terminal mode
 *
 * disable_raw_mode() should:
 *   - Show cursor: \033[?25h
 *   - Reset attributes: \033[0m
 *   - Clear screen: \033[2J
 *   - Move cursor home: \033[H
 *   - Restore original termios settings with tcsetattr()
 *
 * enable_raw_mode() should:
 *   - Save original termios with tcgetattr()
 *   - Register atexit(disable_raw_mode)
 *   - Modify termios flags to disable:
 *     - c_iflag: BRKINT, ICRNL, INPCK, ISTRIP, IXON
 *     - c_oflag: OPOST
 *     - c_lflag: ECHO, ICANON, IEXTEN, ISIG
 *   - Enable c_cflag: CS8
 *   - Set c_cc[VMIN]=0, c_cc[VTIME]=0 for non-blocking
 *   - Apply with tcsetattr()
 *   - Hide cursor: \033[?25l
 *   - Clear screen: \033[2J
 * ══════════════════════════════════════════════════════════════════ */
static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
	write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = 0;
        (void)orig_termios; /* remove when implemented */
    }
}

static void enable_raw_mode(void) {
    /* TODO: Implement - save termios, set raw mode, hide cursor */
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_mode_enabled = 1;
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);                                 raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* hide cursor, clear screen */
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);

    raw_mode_enabled = 1;
    atexit(disable_raw_mode);
}

/* ── Signal handler for clean exit ──────────────────────────────── */
static volatile sig_atomic_t got_resize = 0;

static void handle_sigwinch(int sig) {
    (void)sig;
    got_resize = 1;
}

/* ── Buffer management ──────────────────────────────────────────── */
static void alloc_buffers(int w, int h) {
    free(front_buf);
    free(back_buf);
    buf_w = w;
    buf_h = h;
    int n = w * h;
    front_buf = calloc(n, sizeof(Cell));
    back_buf  = calloc(n, sizeof(Cell));
    /* initialize front buffer to impossible values to force full redraw */
    for (int i = 0; i < n; i++) {
        front_buf[i].ch = 0;
        front_buf[i].fg = 255;
        front_buf[i].bg = 255;
    }
}

static void clear_back_buf(char ch, uint8_t fg, uint8_t bg) {
    int n = buf_w * buf_h;
    for (int i = 0; i < n; i++) {
        back_buf[i].ch = ch;
        back_buf[i].fg = fg;
        back_buf[i].bg = bg;
    }
}

static void set_cell(int x, int y, char ch, uint8_t fg, uint8_t bg) {
    if (x < 0 || x >= buf_w || y < 0 || y >= buf_h) return;
    Cell *c = &back_buf[y * buf_w + x];
    c->ch = ch;
    c->fg = fg;
    c->bg = bg;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement present() - render back buffer to terminal
 *
 * This is the core of double buffering. For each cell:
 *   - Compare back_buf[i] with front_buf[i]
 *   - If different, output ANSI sequences to update that cell
 *   - Use \033[row;colH to move cursor (1-indexed!)
 *   - Use \033[38;5;Nm for foreground color (N = 0-255)
 *   - Use \033[48;5;Nm for background color
 *   - Track prev_fg/prev_bg to avoid redundant color changes
 *   - Copy back_buf to front_buf after drawing
 *   - Call out_flush() at the end
 * ══════════════════════════════════════════════════════════════════ */
static void present(void) {
    /* TODO: Implement double-buffer rendering with ANSI escape codes */
    (void)front_buf;
    (void)back_buf;
    out_flush();
}

/* ── Draw a string into back buffer ─────────────────────────────── */
static void draw_str(int x, int y, const char *s, uint8_t fg, uint8_t bg) {
    while (*s) {
        set_cell(x++, y, *s++, fg, bg);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement non-blocking key read
 *
 * Use poll() with timeout=0 to check if input is available
 * If available, read one byte
 * Handle escape sequences for arrow keys:
 *   \033[A = UP, \033[B = DOWN, \033[C = RIGHT, \033[D = LEFT
 * Return KEY_NONE if no input available
 * ══════════════════════════════════════════════════════════════════ */
#define KEY_NONE  0
#define KEY_ESC   27
#define KEY_UP    1000
#define KEY_DOWN  1001
#define KEY_LEFT  1002
#define KEY_RIGHT 1003

static int read_key(void) {
    /* TODO: Implement non-blocking keyboard input with arrow key support */
    return KEY_NONE;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement timing functions
 *
 * get_time(): Use clock_gettime(CLOCK_MONOTONIC, &ts) and convert to double
 * sleep_ms(): Use nanosleep() with ts.tv_sec and ts.tv_nsec
 * ══════════════════════════════════════════════════════════════════ */
static double get_time(void) {
    /* TODO: Return current time in seconds (use CLOCK_MONOTONIC) */
    return 0.0;
}

static void sleep_ms(int ms) {
    /* TODO: Sleep for ms milliseconds using nanosleep() */
    (void)ms;
}

/* ── Bouncing ball entities ─────────────────────────────────────── */
#define MAX_BALLS 8

typedef struct {
    float x, y;
    float vx, vy;
    uint8_t color;
    char ch;
} Ball;

static Ball balls[MAX_BALLS];
static int num_balls = 5;

static void init_balls(void) {
    const char chars[] = "@O*o.#$&";
    for (int i = 0; i < num_balls; i++) {
        balls[i].x  = 5 + (rand() % (buf_w - 10));
        balls[i].y  = 2 + (rand() % (buf_h - 5));
        balls[i].vx = 10.0f + (rand() % 30);
        balls[i].vy = 5.0f + (rand() % 20);
        if (rand() & 1) balls[i].vx = -balls[i].vx;
        if (rand() & 1) balls[i].vy = -balls[i].vy;
        balls[i].color = 196 + (i * 30) % 60;
        balls[i].ch = chars[i % 8];
    }
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement ball physics update
 *
 * For each ball:
 *   - Update position: x += vx * dt, y += vy * dt
 *   - Bounce off walls: if x < 0 or x >= buf_w, reverse vx
 *   - Same for y with buf_h (leave room for border)
 * ══════════════════════════════════════════════════════════════════ */
static void update_balls(float dt) {
    /* TODO: Update ball positions and handle wall collisions */
    (void)dt;
}

static void draw_balls(void) {
    for (int i = 0; i < num_balls; i++) {
        Ball *b = &balls[i];
        int ix = (int)(b->x + 0.5f);
        int iy = (int)(b->y + 0.5f);
        set_cell(ix, iy, b->ch, b->color, 0);
    }
}

/* ── Draw border ────────────────────────────────────────────────── */
static void draw_border(void) {
    for (int x = 0; x < buf_w; x++) {
        set_cell(x, 0, '-', 245, 0);
        set_cell(x, buf_h - 1, '-', 245, 0);
    }
    for (int y = 0; y < buf_h; y++) {
        set_cell(0, y, '|', 245, 0);
        set_cell(buf_w - 1, y, '|', 245, 0);
    }
    set_cell(0, 0, '+', 245, 0);
    set_cell(buf_w - 1, 0, '+', 245, 0);
    set_cell(0, buf_h - 1, '+', 245, 0);
    set_cell(buf_w - 1, buf_h - 1, '+', 245, 0);
}

/* ── Main ───────────────────────────────────────────────────────── */
int main(void) {
    srand((unsigned)time(NULL));
    signal(SIGWINCH, handle_sigwinch);
    enable_raw_mode();
    get_term_size();
    alloc_buffers(term_w, term_h);
    init_balls();

    double t0 = get_time();
    double fps_time = t0;
    int fps_count = 0;
    double fps_display = 0.0;
    int running = 1;
    int paused = 0;

    char title[] = "MujoCol - Phase 1: Terminal Framework";
    char help[]  = "[q]uit  [p]ause  [+/-]balls  [arrows]nudge";

    while (running) {
        double now = get_time();
        float dt = (float)(now - t0);
        t0 = now;
        if (dt > 0.1f) dt = 0.1f;

        if (got_resize) {
            got_resize = 0;
            get_term_size();
            alloc_buffers(term_w, term_h);
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        int key = read_key();
        switch (key) {
            case 'q': case KEY_ESC: running = 0; break;
            case 'p': paused = !paused; break;
            case '+': case '=':
                if (num_balls < MAX_BALLS) num_balls++;
                break;
            case '-': case '_':
                if (num_balls > 1) num_balls--;
                break;
        }

        if (!paused) {
            update_balls(dt);
        }

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0;
            fps_time = now;
        }

        clear_back_buf(' ', 7, 0);
        draw_border();
        draw_balls();
        draw_str(2, 0, title, 46, 0);
        draw_str(2, buf_h - 1, help, 250, 0);

        char info[128];
        snprintf(info, sizeof(info), " FPS: %.0f | Balls: %d %s",
                 fps_display, num_balls, paused ? "| PAUSED" : "");
        draw_str(buf_w - (int)strlen(info) - 2, 0, info, 226, 0);

        present();
        sleep_ms(16);
    }

    free(front_buf);
    free(back_buf);
    free(out_buf);
    return 0;
}
