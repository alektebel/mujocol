/*
 * phase1_term.c — Terminal Framework
 *
 * Raw terminal mode, double-buffered screen, ANSI 256-color output,
 * frame timing, non-blocking keyboard input.
 *
 * Deliverable: Bouncing colored text on screen, responds to keypresses, shows FPS.
 *
 * Build: gcc -O2 -o phase1_term phase1_term.c -lm
 * Run:   ./phase1_term
 * Quit:  Press 'q' or Escape
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

static void out_str(const char *s) {
    out_append(s, strlen(s));
}

/* ── Terminal size ──────────────────────────────────────────────── */
static void get_term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        term_w = ws.ws_col;
        term_h = ws.ws_row;
    }
}

/* ── Raw mode ───────────────────────────────────────────────────── */
static void disable_raw_mode(void) {
    if (raw_mode_enabled) {
        /* show cursor, reset attributes, clear screen */
        write(STDOUT_FILENO, "\033[?25h\033[0m\033[2J\033[H", 18);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        raw_mode_enabled = 0;
    }
}

static void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw_mode_enabled = 1;
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    /* hide cursor, clear screen */
    write(STDOUT_FILENO, "\033[?25l\033[2J", 10);
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

/* ── Render back buffer to terminal (diff against front) ────────── */
static void present(void) {
    char seq[64];
    int prev_fg = -1, prev_bg = -1;
    int need_move = 1;

    for (int y = 0; y < buf_h; y++) {
        for (int x = 0; x < buf_w; x++) {
            int idx = y * buf_w + x;
            Cell *b = &back_buf[idx];
            Cell *f = &front_buf[idx];

            if (b->ch == f->ch && b->fg == f->fg && b->bg == f->bg) {
                need_move = 1;
                continue;
            }

            if (need_move) {
                int n = snprintf(seq, sizeof(seq), "\033[%d;%dH", y + 1, x + 1);
                out_append(seq, n);
                need_move = 0;
            }

            if (b->fg != prev_fg || b->bg != prev_bg) {
                int n = snprintf(seq, sizeof(seq), "\033[38;5;%dm\033[48;5;%dm",
                                 b->fg, b->bg);
                out_append(seq, n);
                prev_fg = b->fg;
                prev_bg = b->bg;
            }

            out_append(&b->ch, 1);
            *f = *b;
        }
        need_move = 1;
    }

    out_flush();
}

/* ── Draw a string into back buffer ─────────────────────────────── */
static void draw_str(int x, int y, const char *s, uint8_t fg, uint8_t bg) {
    while (*s) {
        set_cell(x++, y, *s++, fg, bg);
    }
}

/* ── Non-blocking key read ──────────────────────────────────────── */
#define KEY_NONE  0
#define KEY_ESC   27
#define KEY_UP    1000
#define KEY_DOWN  1001
#define KEY_LEFT  1002
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

/* ── Timing ─────────────────────────────────────────────────────── */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sleep_ms(int ms) {
    struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
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
        balls[i].color = 196 + (i * 30) % 60;  /* various reds/greens/blues */
        balls[i].ch = chars[i % 8];
    }
}

static void update_balls(float dt) {
    for (int i = 0; i < num_balls; i++) {
        Ball *b = &balls[i];
        b->x += b->vx * dt;
        b->y += b->vy * dt;

        if (b->x < 0)       { b->x = 0;        b->vx = -b->vx; }
        if (b->x >= buf_w)  { b->x = buf_w - 1; b->vx = -b->vx; }
        if (b->y < 1)       { b->y = 1;          b->vy = -b->vy; }
        if (b->y >= buf_h - 1) { b->y = buf_h - 2; b->vy = -b->vy; }
    }
}

static void draw_balls(void) {
    for (int i = 0; i < num_balls; i++) {
        Ball *b = &balls[i];
        int ix = (int)(b->x + 0.5f);
        int iy = (int)(b->y + 0.5f);
        /* draw a small trail */
        int trail_len = 3;
        for (int t = trail_len; t >= 0; t--) {
            float frac = 1.0f - (float)t / (trail_len + 1);
            int tx = ix - (int)(b->vx * 0.01f * t);
            int ty = iy - (int)(b->vy * 0.01f * t);
            uint8_t c = b->color;
            if (t > 0) c = 240 + (int)(frac * 10); /* gray trail */
            set_cell(tx, ty, t == 0 ? b->ch : '.', c, 0);
        }
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

/* ── Draw sine wave ─────────────────────────────────────────────── */
static void draw_wave(double t) {
    for (int x = 1; x < buf_w - 1; x++) {
        float fx = (float)x / buf_w * 4.0f * M_PI;
        int y = (int)(buf_h / 2 + sin(fx + t * 2.0) * (buf_h / 4 - 2));
        if (y > 0 && y < buf_h - 1) {
            /* color based on position along wave */
            uint8_t c = 21 + (x * 6 / buf_w); /* blues */
            set_cell(x, y, '~', c, 0);
        }
    }
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
    int last_key = 0;

    char title[] = "MujoCol - Phase 1: Terminal Framework";
    char help[]  = "[q]uit  [p]ause  [+/-]balls  [arrows]nudge";

    while (running) {
        double now = get_time();
        float dt = (float)(now - t0);
        t0 = now;
        if (dt > 0.1f) dt = 0.1f;  /* clamp large deltas */

        /* handle resize */
        if (got_resize) {
            got_resize = 0;
            get_term_size();
            alloc_buffers(term_w, term_h);
            write(STDOUT_FILENO, "\033[2J", 4);
        }

        /* input */
        int key = read_key();
        if (key != KEY_NONE) last_key = key;
        switch (key) {
            case 'q': case KEY_ESC: running = 0; break;
            case 'p': paused = !paused; break;
            case '+': case '=':
                if (num_balls < MAX_BALLS) {
                    balls[num_balls].x = buf_w / 2;
                    balls[num_balls].y = buf_h / 2;
                    balls[num_balls].vx = 15 + rand() % 20;
                    balls[num_balls].vy = 10 + rand() % 15;
                    balls[num_balls].color = 196 + (num_balls * 30) % 60;
                    balls[num_balls].ch = "@O*o.#$&"[num_balls % 8];
                    num_balls++;
                }
                break;
            case '-': case '_':
                if (num_balls > 1) num_balls--;
                break;
            case KEY_UP:
                for (int i = 0; i < num_balls; i++) balls[i].vy -= 5;
                break;
            case KEY_DOWN:
                for (int i = 0; i < num_balls; i++) balls[i].vy += 5;
                break;
            case KEY_LEFT:
                for (int i = 0; i < num_balls; i++) balls[i].vx -= 5;
                break;
            case KEY_RIGHT:
                for (int i = 0; i < num_balls; i++) balls[i].vx += 5;
                break;
        }

        /* update */
        if (!paused) {
            update_balls(dt);
        }

        /* FPS calculation */
        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count = 0;
            fps_time = now;
        }

        /* draw */
        clear_back_buf(' ', 7, 0);
        draw_border();
        draw_wave(now);
        draw_balls();

        /* HUD */
        draw_str(2, 0, title, 46, 0);
        draw_str(2, buf_h - 1, help, 250, 0);

        char info[128];
        snprintf(info, sizeof(info), " FPS: %.0f | Balls: %d | Key: %d %s",
                 fps_display, num_balls, last_key, paused ? "| PAUSED" : "");
        draw_str(buf_w - (int)strlen(info) - 2, 0, info, 226, 0);

        /* present */
        present();
        sleep_ms(16);  /* ~60 FPS target */
    }

    /* cleanup */
    free(front_buf);
    free(back_buf);
    free(out_buf);
    return 0;
}
