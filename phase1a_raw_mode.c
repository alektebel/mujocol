/*
 * phase1a_raw_mode.c — Terminal Raw Mode Basics
 *
 * EXERCISE (sub-phase 1a): Implement raw terminal mode and basic
 * ANSI color output using write() and escape sequences.
 *
 * Build: gcc -O2 -o phase1a_raw_mode phase1a_raw_mode.c -lm
 * Run:   ./phase1a_raw_mode
 * Quit:  Press any key
 *
 * LEARNING GOALS:
 * - Understand termios structures and tcgetattr/tcsetattr
 * - Learn ANSI escape sequences for color and cursor control
 * - Use write() for unbuffered terminal output
 * - Register atexit() for clean-up on exit
 */

#define _POSIX_C_SOURCE 199309L
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
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
 * TODO #1: Implement disable_raw_mode()
 *
 * Should:
 *   - Skip if raw_mode_enabled == 0
 *   - Write "\033[?25h\033[0m\033[2J\033[H" to show cursor, reset colors, clear
 *   - Restore original termios with tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios)
 *   - Set raw_mode_enabled = 0
 * ══════════════════════════════════════════════════════════════════ */
static void disable_raw_mode(void) {
    /* TODO: Restore terminal state */
    (void)orig_termios;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement enable_raw_mode()
 *
 * Should:
 *   - Save original termios with tcgetattr(STDIN_FILENO, &orig_termios)
 *   - Register atexit(disable_raw_mode) for cleanup
 *   - Modify flags:
 *       c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
 *       c_oflag &= ~(OPOST)
 *       c_cflag |= (CS8)
 *       c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG)
 *   - Set c_cc[VMIN]=0, c_cc[VTIME]=0 for non-blocking
 *   - Apply with tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
 *   - Hide cursor: write(STDOUT_FILENO, "\033[?25l\033[2J", 10)
 *   - Set raw_mode_enabled = 1
 * ══════════════════════════════════════════════════════════════════ */
static void enable_raw_mode(void) {
    /* TODO: Enter raw terminal mode */
    raw_mode_enabled = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement write_colored_text()
 *
 * Move the cursor to position (x, y) and write text with ANSI 256-color
 * foreground and background codes.
 *
 * Steps:
 *   1. Build an escape sequence with snprintf:
 *        "\033[<row>;<col>H\033[38;5;<fg>m\033[48;5;<bg>m"
 *      where row = y+1, col = x+1  (terminal coords are 1-indexed)
 *   2. Append that sequence with out_append()
 *   3. Append the text itself with out_append()
 *   4. Append the reset sequence "\033[0m" with out_append()
 *   5. Call out_flush() to emit everything
 * ══════════════════════════════════════════════════════════════════ */
static void write_colored_text(int x, int y, const char *text, int fg, int bg) {
    /* TODO: Position cursor and write ANSI 256-color text */
    (void)x; (void)y; (void)text; (void)fg; (void)bg;
    out_flush();
}

/* ── Main ───────────────────────────────────────────────────────── */
int main(void) {
    enable_raw_mode();
    get_term_size();

    /* Draw a colorful character grid across the terminal */
    const char glyphs[] = "#@%&*O+.";
    int num_glyphs = (int)(sizeof(glyphs) - 1);
    for (int row = 0; row < term_h - 2; row++) {
        for (int col = 0; col < term_w; col++) {
            int fg = 196 + ((row * 7 + col * 3) % 60);  /* bright spectrum  */
            int bg = (col % 2 == 0) ? 232 : 234;         /* subtle dark grid */
            char glyph[2] = { glyphs[(row + col) % num_glyphs], '\0' };
            write_colored_text(col, row, glyph, fg, bg);
        }
    }

    /* Centered title banner */
    const char *title = "[ phase1a: raw mode & ANSI color — press any key ]";
    int title_len = (int)strlen(title);
    int tx = (term_w - title_len) / 2;
    int ty = term_h / 2;
    write_colored_text(tx, ty, title, 226, 0);

    /* Wait for a single keypress then exit */
    char c;
    read(STDIN_FILENO, &c, 1);

    free(out_buf);
    return 0;
}
