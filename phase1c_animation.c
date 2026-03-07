/*
 * phase1c_animation.c — Animation Loop with Input and Timing
 *
 * EXERCISE (sub-phase 1c): Implement non-blocking keyboard input,
 * precise frame timing, and a bouncing-ball animation using the
 * double-buffered renderer from phase1b.
 *
 * Build: gcc -O2 -o phase1c_animation phase1c_animation.c -lm
 * Run:   ./phase1c_animation
 * Quit:  Press 'q' or Escape
 *
 * LEARNING GOALS:
 * - Use poll() for non-blocking keyboard input
 * - Parse ANSI escape sequences for arrow keys
 * - Use clock_gettime(CLOCK_MONOTONIC) for precise timing
 * - Use nanosleep() for frame rate control
 * - Animate objects using dt-based physics
 *
 * ── TECHNIQUE OVERVIEW ──────────────────────────────────────────────
 *
 * 1. poll() FOR NON-BLOCKING I/O
 *    We need to check for keyboard input without halting the animation.
 *    read() with VMIN=0 does return immediately when no data is present,
 *    but poll() is the cleaner, more portable POSIX approach: it lets
 *    you query multiple file descriptors at once and express intent
 *    explicitly.  Key fields of struct pollfd:
 *      fd      — which file descriptor to watch (STDIN_FILENO here)
 *      events  — what we want to know: POLLIN means "data is readable"
 *      revents — filled in by poll() with what actually happened
 *    Passing timeout=0 makes poll() return instantly (pure polling, no
 *    blocking), which is exactly what a real-time game loop needs.
 *
 * 2. CLOCK_MONOTONIC
 *    Two clock sources are available in POSIX:
 *      CLOCK_REALTIME  — wall-clock time; can jump forward or backward
 *                        when NTP adjusts the system clock.
 *      CLOCK_MONOTONIC — guaranteed to never go backwards; ticks
 *                        continuously from some arbitrary epoch.
 *    For measuring frame intervals (dt) we must use CLOCK_MONOTONIC.
 *    If CLOCK_REALTIME jumped back by 1 second we would get a negative
 *    dt and the physics would run in reverse.
 *
 * 3. nanosleep()
 *    POSIX function to sleep for a precise duration.  The argument is a
 *    struct timespec with two fields:
 *      tv_sec  — whole seconds
 *      tv_nsec — remaining nanoseconds (must be in range 0-999999999)
 *    We sleep at the end of each frame to cap CPU usage; the animation
 *    loop targets ~60 fps (16 ms per frame).
 *
 * 4. dt-BASED PHYSICS
 *    Rather than moving a ball by a fixed number of pixels per frame, we
 *    multiply velocity by dt (elapsed time in seconds):
 *      x += vx * dt
 *    This makes motion frame-rate independent: a ball with vx = 30
 *    cells/sec travels exactly 30 cells per second whether the loop
 *    runs at 30 fps (dt ≈ 0.033) or 120 fps (dt ≈ 0.008).
 *
 * 5. volatile sig_atomic_t
 *    Signal handlers run asynchronously — they can interrupt any
 *    instruction.  A plain int shared between the main loop and a
 *    signal handler could be read mid-write on some architectures.
 *    sig_atomic_t is an integer type whose reads and writes are
 *    guaranteed atomic with respect to signal delivery.  The volatile
 *    qualifier tells the compiler to never cache the value in a
 *    register: every read must go to memory, ensuring the main loop
 *    sees updates written by the signal handler immediately.
 *
 * 6. SIGWINCH
 *    The kernel sends SIGWINCH ("window changed") to a process whenever
 *    the terminal window is resized.  We install a handler that sets
 *    got_resize = 1, then check that flag at the top of each frame loop
 *    iteration.  When set, we re-query the terminal dimensions with
 *    get_term_size() (ioctl/TIOCGWINSZ, explained in phase1a) and
 *    reallocate the cell buffers to match the new size.
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
#include <poll.h>      /* struct pollfd, poll() — non-blocking I/O readiness */
#include <math.h>
#include <signal.h>

/* ── Terminal dimensions ────────────────────────────────────────── */
static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;

/* ── Screen buffer cell ─────────────────────────────────────────── */
typedef struct {
    char    ch;
    uint8_t fg;
    uint8_t bg;
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

/* ── Raw mode (fully provided) ──────────────────────────────────── */
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

/* ── Signal handler for terminal resize ─────────────────────────── */
static volatile sig_atomic_t got_resize = 0;
static void handle_sigwinch(int sig) { (void)sig; got_resize = 1; }

/* ── Buffer management (fully provided) ─────────────────────────── */
static void alloc_buffers(int w, int h) {
    free(front_buf);
    free(back_buf);
    buf_w = w; buf_h = h;
    int n = w * h;
    front_buf = calloc(n, sizeof(Cell));
    back_buf  = calloc(n, sizeof(Cell));
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
    c->ch = ch; c->fg = fg; c->bg = bg;
}

/* ── Present (fully provided) ───────────────────────────────────── */
static void present(void) {
    int n = buf_w * buf_h;
    uint8_t prev_fg = 255, prev_bg = 255;
    char esc[64];
    for (int i = 0; i < n; i++) {
        Cell *b = &back_buf[i];
        Cell *f = &front_buf[i];
        if (b->ch == f->ch && b->fg == f->fg && b->bg == f->bg) continue;
        int row = i / buf_w, col = i % buf_w;
        int len = snprintf(esc, sizeof(esc), "\033[%d;%dH", row + 1, col + 1);
        out_append(esc, len);
        if (b->fg != prev_fg) {
            len = snprintf(esc, sizeof(esc), "\033[38;5;%dm", b->fg);
            out_append(esc, len);
            prev_fg = b->fg;
        }
        if (b->bg != prev_bg) {
            len = snprintf(esc, sizeof(esc), "\033[48;5;%dm", b->bg);
            out_append(esc, len);
            prev_bg = b->bg;
        }
        out_append(&b->ch, 1);
        *f = *b;
    }
    out_flush();
}

/* ── Draw helpers (fully provided) ──────────────────────────────── */
static void draw_str(int x, int y, const char *s, uint8_t fg, uint8_t bg) {
    while (*s) set_cell(x++, y, *s++, fg, bg);
}

static void draw_border(void) {
    for (int x = 0; x < buf_w; x++) {
        set_cell(x, 0,          '-', 245, 0);
        set_cell(x, buf_h - 1,  '-', 245, 0);
    }
    for (int y = 0; y < buf_h; y++) {
        set_cell(0,          y, '|', 245, 0);
        set_cell(buf_w - 1,  y, '|', 245, 0);
    }
    set_cell(0,          0,         '+', 245, 0);
    set_cell(buf_w - 1,  0,         '+', 245, 0);
    set_cell(0,          buf_h - 1, '+', 245, 0);
    set_cell(buf_w - 1,  buf_h - 1, '+', 245, 0);
}

/* ── Key constants ──────────────────────────────────────────────── */
#define KEY_NONE  0
#define KEY_ESC   27
#define KEY_UP    1000
#define KEY_DOWN  1001
#define KEY_LEFT  1002
#define KEY_RIGHT 1003

/* ══════════════════════════════════════════════════════════════════
 * TODO #1: Implement read_key()
 *
 * Use poll() to check whether stdin has data ready (timeout = 0 ms).
 * If no data, return KEY_NONE immediately.
 * If data, read one byte:
 *   - If it is not '\033', return it as-is.
 *   - If it is '\033', try to read two more bytes (also with poll so
 *     we don't block).  If you get '[' followed by:
 *       'A' → return KEY_UP
 *       'B' → return KEY_DOWN
 *       'C' → return KEY_RIGHT
 *       'D' → return KEY_LEFT
 *   - If the sequence is unrecognized, return KEY_ESC.
 *
 * Hint: struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
 *       poll(&pfd, 1, 0);
 *
 * struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 }  — field by field:
 *   fd      = STDIN_FILENO : watch standard input
 *   events  = POLLIN       : we want to know when data can be read
 *   revents = 0            : output field, initialised to 0 by us
 *
 * poll(&pfd, 1, 0) arguments:
 *   &pfd   — pointer to the array of pollfd structs
 *   1      — number of structs in the array (we only watch one fd)
 *   0      — timeout in milliseconds; 0 = return immediately
 *
 * ARROW KEY SEQUENCES:
 *   Terminals encode arrow keys as multi-byte ANSI escape sequences:
 *     Up    → ESC '[' 'A'   (3 bytes)
 *     Down  → ESC '[' 'B'
 *     Right → ESC '[' 'C'
 *     Left  → ESC '[' 'D'
 *   We first read one byte and check whether it is ESC (0x1B / '\033').
 *   If so, we poll() once more to see if there is a continuation; if
 *   yes, we read the '[' and the letter.  Using poll() for the follow-up
 *   read prevents blocking if the user pressed a bare ESC key with no
 *   subsequent bytes.
 * ══════════════════════════════════════════════════════════════════ */
static int read_key(void) {
    /* TODO: Non-blocking keyboard input with arrow-key parsing */
    return KEY_NONE;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement get_time()
 *
 * Return the current wall-clock time as a double (seconds) using
 * clock_gettime(CLOCK_MONOTONIC, &ts).
 * Formula: ts.tv_sec + ts.tv_nsec * 1e-9
 *
 * clock_gettime(CLOCK_MONOTONIC, &ts) fills the struct timespec ts:
 *   tv_sec  — whole seconds since the monotonic clock's epoch
 *   tv_nsec — additional nanoseconds (0 … 999999999)
 * Combining them: ts.tv_sec + ts.tv_nsec * 1e-9 gives a single double
 * in seconds with nanosecond precision.  We use CLOCK_MONOTONIC (not
 * CLOCK_REALTIME) so the value never jumps — see the TECHNIQUE OVERVIEW.
 * ══════════════════════════════════════════════════════════════════ */
static double get_time(void) {
    /* TODO: Return monotonic time in seconds */
    return 0.0;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement sleep_ms(int ms)
 *
 * Sleep for the requested number of milliseconds using nanosleep().
 * Assumes ms >= 0.
 * struct timespec ts = { ms / 1000, (ms % 1000) * 1000000L };
 * nanosleep(&ts, NULL);
 *
 * BREAKING DOWN THE TIMESPEC INITIALIZER:
 *   ms / 1000          — integer division gives the whole-second part.
 *                        e.g. 2500 ms → tv_sec = 2
 *   (ms % 1000) * 1000000L — the leftover milliseconds converted to
 *                        nanoseconds (1 ms = 1,000,000 ns).
 *                        e.g. 2500 ms → (2500 % 1000) * 1000000L
 *                             = 500 * 1000000L = 500000000 ns
 * The L suffix on 1000000L promotes the literal to long before the
 * multiplication, preventing integer overflow on platforms where int is
 * 16 bits (where 999 * 1000000 = 999000000 > 32767 = INT_MAX).
 * ══════════════════════════════════════════════════════════════════ */
static void sleep_ms(int ms) {
    /* TODO: Sleep using nanosleep() */
    (void)ms;
}

/* ── Bouncing ball entities ─────────────────────────────────────── */
#define MAX_BALLS 8

typedef struct {
    float   x, y;
    float   vx, vy;
    uint8_t color;
    char    ch;
} Ball;

static Ball balls[MAX_BALLS];
static int  num_balls = 5;

/* ══════════════════════════════════════════════════════════════════
 * TODO #4: Implement update_balls(float dt)
 *
 * For every active ball (index 0 .. num_balls-1):
 *   1. Advance position:  x += vx * dt;  y += vy * dt
 *   2. Bounce off the left/right walls:
 *        if x < 1         → x = 1;         vx = fabs(vx);
 *        if x >= buf_w-1  → x = buf_w-2;   vx = -fabs(vx);
 *   3. Bounce off the top/bottom walls:
 *        if y < 1         → y = 1;         vy = fabs(vy);
 *        if y >= buf_h-1  → y = buf_h-2;   vy = -fabs(vy);
 *
 * Using fabs() for the bounce ensures the velocity is always directed
 * away from the wall even if multiple frames are skipped.
 *
 * WHY fabs() INSTEAD OF JUST NEGATING?
 *   A naive bounce: vx = -vx.  This can fail when a ball penetrates the
 *   wall by more than one frame's worth of movement (e.g. after a
 *   resize or a hiccup in frame timing): the ball oscillates inside the
 *   wall, toggling direction each frame and never escaping.
 *   Using fabs() is a one-sided clamp: after hitting the left wall we
 *   set vx = +|vx|, which is always positive (pointing right) regardless
 *   of what vx was before.  The ball is guaranteed to move away from the
 *   wall on the very next frame.
 * ══════════════════════════════════════════════════════════════════ */
static void update_balls(float dt) {
    /* TODO: Advance positions and bounce off borders */
    (void)dt;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #5: Implement init_balls()
 *
 * Seed the balls array with random starting positions, velocities,
 * colors, and display characters for all MAX_BALLS entries.
 *
 * Suggested ranges (adjust freely):
 *   x        : 5 … buf_w - 10
 *   y        : 2 … buf_h - 5
 *   |vx|     : 10 … 40  (randomly negate half of them)
 *   |vy|     : 5  … 25  (randomly negate half of them)
 *   color    : 196 + (i * 30) % 60   (bright spectrum)
 *   ch       : pick from "@O*o.#$&"
 *
 * THE rand()%(range)+base PATTERN:
 *   rand() returns a pseudo-random int in [0, RAND_MAX].
 *   rand() % range maps it to [0, range-1].
 *   Adding base shifts it to [base, base+range-1].
 *   Example: rand() % 30 + 10 gives a value in [10, 39].
 *   To randomly negate a velocity: if (rand() % 2) vx = -vx;
 *
 * ARRAY POPULATION:
 *   Loop i from 0 to MAX_BALLS-1 and fill balls[i].  All MAX_BALLS
 *   entries are initialised even though only num_balls are active;
 *   this allows the user to enable more balls at runtime ('+' key)
 *   without reinitialising them on the fly.
 * ══════════════════════════════════════════════════════════════════ */
static void init_balls(void) {
    /* TODO: Randomize ball starting state */
}

/* ── Draw all active balls (provided) ───────────────────────────── */
static void draw_balls(void) {
    for (int i = 0; i < num_balls; i++) {
        Ball *b = &balls[i];
        int ix = (int)(b->x + 0.5f);
        int iy = (int)(b->y + 0.5f);
        set_cell(ix, iy, b->ch, b->color, 0);
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

    double t0       = get_time();
    double fps_time = t0;
    int    fps_count = 0;
    double fps_display = 0.0;
    int    running = 1;
    int    paused  = 0;

    const char *title = "MujoCol - Phase 1c: Animation";
    const char *help  = "[q]uit  [p]ause  [+/-]balls";

    while (running) {
        double now = get_time();
        float  dt  = (float)(now - t0);
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

        if (!paused) update_balls(dt);

        fps_count++;
        if (now - fps_time >= 0.5) {
            fps_display = fps_count / (now - fps_time);
            fps_count   = 0;
            fps_time    = now;
        }

        clear_back_buf(' ', 7, 0);
        draw_border();
        draw_balls();
        draw_str(2, 0, title, 46, 0);
        draw_str(2, buf_h - 1, help, 250, 0);

        char info[128];
        snprintf(info, sizeof(info), " FPS: %.0f | Balls: %d%s",
                 fps_display, num_balls, paused ? " | PAUSED" : "");
        draw_str(buf_w - (int)strlen(info) - 2, 0, info, 226, 0);

        present();
        sleep_ms(16);
    }

    free(front_buf);
    free(back_buf);
    free(out_buf);
    return 0;
}
