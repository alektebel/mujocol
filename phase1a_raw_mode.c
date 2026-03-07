/*
 * phase1a_raw_mode.c — Terminal Raw Mode Basics
 *
 * EXERCISE (sub-phase 1a): Implement raw terminal mode and basic
 * ANSI color output using write() and escape sequences.
 *
 * Build: gcc -O2 -o phase1a_raw_mode phase1a_raw_mode.c -lm
 *        # aarch64 cross-compile:
 *        # aarch64-linux-gnu-gcc -O2 -o phase1a_raw_mode phase1a_raw_mode.c -lm
 * Run:   ./phase1a_raw_mode
 * Quit:  Press any key
 *
 * ══════════════════════════════════════════════════════════════════
 * TECHNIQUE OVERVIEW — what you will learn in this sub-phase
 * ══════════════════════════════════════════════════════════════════
 *  1. Raw terminal mode  — how to switch the kernel's tty layer from
 *     "cooked" (line-buffered, with echo) to "raw" (every keystroke
 *     is immediately readable, no echo, no special-key processing).
 *  2. termios API        — the POSIX struct/functions that control the
 *     tty driver: tcgetattr / tcsetattr, struct termios flags.
 *  3. ANSI escape sequences — byte sequences embedded in output that
 *     instruct the terminal emulator to move the cursor, set colors,
 *     clear the screen, hide/show the cursor, etc.
 *  4. write() vs printf() — why we use the low-level unbuffered write()
 *     syscall instead of printf() for real-time terminal control.
 *  5. atexit() / cleanup  — how to register a function that runs
 *     automatically when the process exits, to restore the terminal.
 *  6. ioctl / TIOCGWINSZ  — how to query the terminal's current size
 *     (columns × rows) at runtime.
 *  7. Output batching      — accumulating output in a heap buffer and
 *     flushing in a single write() to avoid flickering.
 *
 * LEARNING GOALS:
 * - Understand termios structures and tcgetattr/tcsetattr
 * - Learn ANSI escape sequences for color and cursor control
 * - Use write() for unbuffered terminal output
 * - Register atexit() for clean-up on exit
 */

/* ── Feature-test macros ─────────────────────────────────────────────
 * _POSIX_C_SOURCE 199309L  — request POSIX.1b (real-time extensions).
 *   This is required on Linux to expose nanosleep(), clock_gettime()
 *   and the full termios interface from <termios.h>.  The number
 *   199309L is the year+month the standard was published (1993-09).
 *
 * _DEFAULT_SOURCE           — on glibc, this unlocks BSD/misc
 *   extensions that are not strictly POSIX (e.g. SIGWINCH, strsep).
 *   It replaces the older _BSD_SOURCE and _SVID_SOURCE.
 * ──────────────────────────────────────────────────────────────── */
#define _POSIX_C_SOURCE 199309L
#define _DEFAULT_SOURCE

/* ── Standard headers ────────────────────────────────────────────────
 * <stdio.h>     — snprintf, printf (used for building escape strings)
 * <stdlib.h>    — malloc/realloc/free, atexit, exit
 * <string.h>    — memcpy, strlen
 * <unistd.h>    — write(), read(), STDOUT_FILENO, STDIN_FILENO
 *                 STDOUT_FILENO == 1, STDIN_FILENO == 0 (POSIX constants)
 * <termios.h>   — struct termios, tcgetattr, tcsetattr, VMIN, VTIME …
 * <sys/ioctl.h> — ioctl(), TIOCGWINSZ, struct winsize
 * <time.h>      — time() (used only for srand seed in later phases)
 * <math.h>      — math functions (-lm flag needed at link time)
 * <signal.h>    — signal(), SIGWINCH (used in later sub-phases)
 * ──────────────────────────────────────────────────────────────── */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <time.h>
#include <math.h>
#include <signal.h>

/* ── Terminal dimensions ─────────────────────────────────────────────
 * term_w / term_h — current width and height of the terminal window in
 *   character cells.  Defaults (80×24) match the historical VT100
 *   terminal and are used if ioctl fails.
 *
 * orig_termios — a snapshot of the terminal settings as they were
 *   before we touched them.  We must restore this on exit so that
 *   the shell the user returns to still works correctly.
 *
 * raw_mode_enabled — a guard flag.  disable_raw_mode() checks it so
 *   it is safe to call multiple times (e.g. both from atexit AND
 *   from an explicit call) without double-restoring termios.
 * ──────────────────────────────────────────────────────────────── */
static int term_w = 80, term_h = 24;
static struct termios orig_termios;
static int raw_mode_enabled = 0;

/* ── Output buffer for batched writes ───────────────────────────────
 * Writing one character at a time to the terminal with write() causes
 * the kernel to wake the terminal emulator for EVERY byte, leading to
 * visible flickering.  Instead we collect everything into a growable
 * heap buffer (out_buf) and emit it all at once with a single write()
 * in out_flush().
 *
 * out_buf  — pointer to the allocated buffer (NULL until first append)
 * out_cap  — allocated size in bytes (grows exponentially: ×2 each time)
 * out_len  — number of valid bytes currently in the buffer
 * ──────────────────────────────────────────────────────────────── */
static char *out_buf = NULL;
static int out_cap = 0, out_len = 0;

/* out_flush — send everything accumulated in out_buf to stdout.
 *   write(fd, buf, n) — POSIX syscall: write n bytes from buf to fd.
 *   STDOUT_FILENO     — file descriptor 1 (standard output).
 *   After writing we reset out_len to 0 to reuse the buffer. */
static void out_flush(void) {
    if (out_len > 0) {
        write(STDOUT_FILENO, out_buf, out_len);
        out_len = 0;
    }
}

/* out_append — copy n bytes from s into the output buffer.
 *   We double out_cap whenever the buffer is too small (amortised O(1)).
 *   Starting capacity of 65536 bytes avoids many early reallocations. */
static void out_append(const char *s, int n) {
    while (out_len + n > out_cap) {
        out_cap = out_cap ? out_cap * 2 : 65536;
        out_buf = realloc(out_buf, out_cap);
    }
    memcpy(out_buf + out_len, s, n);
    out_len += n;
}

/* ── Terminal size ───────────────────────────────────────────────────
 * ioctl(fd, request, arg) — "input/output control": a generic kernel
 *   interface for device-specific operations that don't fit the
 *   read/write model.
 *
 * TIOCGWINSZ  — "Teletype Input/Output Control Get WINdow SiZe"
 *   Fills a struct winsize with:
 *     ws_col  — number of columns (characters wide)
 *     ws_row  — number of rows (characters tall)
 *     ws_xpixel, ws_ypixel — pixel dimensions (often 0)
 *
 * We guard with ws.ws_col > 0 to avoid replacing our defaults with
 * zeroes if the ioctl call succeeds but returns an empty struct.
 * ──────────────────────────────────────────────────────────────── */
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
 * This function is called either explicitly or automatically via
 * atexit() when the program exits.  It must:
 *
 *   1. Guard: if raw_mode_enabled == 0, return immediately.
 *      (Prevents double-restore if called more than once.)
 *
 *   2. Write the cleanup escape sequence to stdout:
 *        "\033[?25h\033[0m\033[2J\033[H"
 *
 *      Breaking this string apart character by character:
 *        \033        — the ESC byte (octal 033 = decimal 27 = hex 1B).
 *                      Every ANSI/VT100 control sequence starts with
 *                      this byte followed by '[' (together called CSI,
 *                      Control Sequence Introducer).
 *        [?25h       — DEC private mode set.  '?' introduces a DEC
 *                      private parameter.  '25' is the cursor-visible
 *                      mode.  'h' means "high" = enable/turn on.
 *                      ⇒ SHOW the cursor (undo our earlier hide).
 *        \033[0m     — SGR (Select Graphic Rendition) parameter 0.
 *                      '0' resets ALL attributes (color, bold, etc.)
 *                      back to the terminal default.  'm' ends SGR.
 *                      ⇒ RESET all colors and text styles.
 *        \033[2J     — Erase Display.  '2' = erase the entire screen
 *                      (0=from cursor, 1=to cursor, 2=whole screen).
 *                      'J' is the ED (Erase in Display) command.
 *                      ⇒ CLEAR the screen.
 *        \033[H      — Cursor Position.  No row/col arguments means
 *                      row=1, col=1 (top-left corner).  'H' is the
 *                      CUP (Cursor Position) command.
 *                      ⇒ MOVE cursor to top-left.
 *
 *      The string has exactly 18 bytes — that is the third argument.
 *      You can count: \033=1, [=1, ?=1, 2=1, 5=1, h=1 (6 for ?25h)
 *        + \033=1, [=1, 0=1, m=1 (4 for 0m)
 *        + \033=1, [=1, 2=1, J=1 (4 for 2J)
 *        + \033=1, [=1, H=1      (3 for H)  → total = 18.
 *
 *   3. Restore original termios settings:
 *        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios)
 *        — STDIN_FILENO: the tty attributes live on the *input* side
 *          (file descriptor 0), even though we read AND write to the
 *          same physical terminal device.
 *        — TCSAFLUSH: apply the change *after* draining the output
 *          queue and discarding any pending unread input.  This is
 *          the safest option when restoring on exit.
 *        — &orig_termios: pointer to the saved settings struct.
 *
 *   4. Set raw_mode_enabled = 0.
 * ══════════════════════════════════════════════════════════════════ */
static void disable_raw_mode(void) {
    /* TODO: Restore terminal state */
    (void)orig_termios;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #2: Implement enable_raw_mode()
 *
 * "Raw mode" (also called "non-canonical mode") bypasses the kernel's
 * line-discipline layer so that:
 *   - Every key reaches your program immediately (no waiting for Enter).
 *   - Characters are NOT echoed back to the screen automatically.
 *   - Special key combinations (Ctrl-C, Ctrl-Z, Ctrl-S …) do NOT
 *     send signals or pause the process; they are just bytes.
 *
 * Steps:
 *
 *   1. Save original settings:
 *        tcgetattr(STDIN_FILENO, &orig_termios)
 *        Reads the current termios struct into orig_termios so we can
 *        restore it later.
 *
 *   2. Register cleanup:
 *        atexit(disable_raw_mode)
 *        atexit() registers a function to be called automatically
 *        when the process exits (via return from main, or exit()).
 *        This guarantees the terminal is always restored even if the
 *        program crashes or exits early.
 *
 *   3. Copy: struct termios raw = orig_termios;
 *        Work on a copy so orig_termios stays pristine.
 *
 *   4. Modify input flags (raw.c_iflag):
 *        c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
 *
 *        These are bit-flags.  The &= ~(...) idiom CLEARS the listed
 *        bits while leaving all others unchanged.
 *
 *        BRKINT  — on a BREAK signal, generate SIGINT.  We disable
 *                  this so a break condition is just delivered as a
 *                  NUL byte (or ignored).
 *        ICRNL   — translate carriage-return (CR, '\r') to newline
 *                  ('\n') on input.  We disable it so Ctrl-M and Enter
 *                  are distinguishable.
 *        INPCK   — enable input parity checking.  Irrelevant for
 *                  modern virtual terminals but traditionally disabled.
 *        ISTRIP  — strip the 8th bit of each input byte (forces
 *                  7-bit ASCII).  We need full 8-bit for UTF-8 /
 *                  extended keys.
 *        IXON    — enable XON/XOFF software flow control (Ctrl-S
 *                  pauses, Ctrl-Q resumes).  We disable it so these
 *                  keys reach the program as normal bytes.
 *
 *   5. Modify output flags (raw.c_oflag):
 *        c_oflag &= ~(OPOST)
 *        OPOST  — "output post-processing": translates '\n' to '\r\n',
 *                 expands tabs, etc.  We disable it because our ANSI
 *                 sequences already position the cursor explicitly and
 *                 we don't want the kernel mangling our output.
 *
 *   6. Modify control flags (raw.c_cflag):
 *        c_cflag |= (CS8)
 *        CS8    — character size: 8 bits per byte (most common).
 *                 The |= sets these bits without touching others.
 *
 *   7. Modify local flags (raw.c_lflag):
 *        c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG)
 *        ECHO    — echo input characters back to the screen.  We
 *                  handle all drawing ourselves, so we turn this off.
 *        ICANON  — "canonical mode": buffer input until newline.
 *                  Disabling gives us each byte as it arrives.
 *        IEXTEN  — extended processing (e.g. Ctrl-V literal-next).
 *                  Disabled so these bytes pass through unmodified.
 *        ISIG    — generate signals for Ctrl-C (SIGINT), Ctrl-Z
 *                  (SIGTSTP).  Disabled so the game loop receives
 *                  these as plain bytes and can handle them itself.
 *
 *   8. Set control characters for non-blocking read:
 *        raw.c_cc[VMIN]  = 0   — minimum bytes before read() returns.
 *                                0 means return immediately even with
 *                                no data (polling mode).
 *        raw.c_cc[VTIME] = 0   — timeout in tenths of a second.
 *                                0 means no timeout.
 *        Together these make read() non-blocking: it returns 0 if no
 *        key has been pressed, or 1 if one byte is available.
 *
 *   9. Apply the new settings:
 *        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)
 *        TCSAFLUSH: flush (discard) any unread input before applying.
 *
 *  10. Hide the cursor (avoids flicker during drawing):
 *        write(STDOUT_FILENO, "\033[?25l\033[2J", 10)
 *        \033[?25l — DEC private mode '25' low (l = disable).
 *                    ⇒ HIDE the cursor.
 *        \033[2J   — erase entire screen (same as in disable_raw_mode).
 *        10 bytes total: count the characters to verify.
 *
 *  11. Set raw_mode_enabled = 1.
 * ══════════════════════════════════════════════════════════════════ */
static void enable_raw_mode(void) {
    /* TODO: Enter raw terminal mode */
    raw_mode_enabled = 1;
}

/* ══════════════════════════════════════════════════════════════════
 * TODO #3: Implement write_colored_text()
 *
 * Move the cursor to pixel (x, y) — measured in character cells from
 * the top-left corner — and draw text in a chosen 256-color palette
 * foreground and background.
 *
 * Steps:
 *   1. Build the cursor-move + color-set escape string with snprintf:
 *
 *        "\033[<row>;<col>H\033[38;5;<fg>m\033[48;5;<bg>m"
 *
 *      Breakdown of each sequence:
 *        \033[<row>;<col>H
 *          — CUP: Cursor Position.  row = y+1, col = x+1 because
 *            ANSI terminal coordinates are 1-indexed (row 1 = top).
 *            Two numbers separated by ';', terminated by 'H'.
 *
 *        \033[38;5;<fg>m
 *          — SGR parameter 38 = "set foreground color using extended
 *            palette".  Sub-parameter 5 = "use 256-color index".
 *            <fg> is the palette index (0–255).  'm' ends SGR.
 *            Indices 0-15: standard colors; 16-231: 6×6×6 RGB cube;
 *            232-255: grayscale ramp.
 *
 *        \033[48;5;<bg>m
 *          — Same as above but SGR 48 = "set background color".
 *
 *   2. Append that sequence with out_append(esc, len).
 *   3. Append the text itself with out_append(text, strlen(text)).
 *   4. Append the reset sequence "\033[0m" with out_append().
 *      This resets colors so the NEXT draw call starts from a clean
 *      state and doesn't inherit stale colors.
 *   5. Call out_flush() to emit everything in one write().
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

    /* Draw a colorful character grid across the terminal.
     * Color indices 196–255 are the bright end of the 256-color ramp:
     *   196–231 → 6×6×6 RGB cube (high-saturation reds, greens, blues)
     *   232–255 → grayscale (232 ≈ very dark, 255 = white)
     * We offset by row/col to create a diagonal color pattern. */
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
    write_colored_text(tx, ty, title, 226, 0);  /* 226 = bright yellow */

    /* Wait for a single keypress then exit.
     * read(fd, buf, 1) blocks until one byte is available.
     * Because raw mode has VMIN=0, VTIME=0, this would normally
     * return immediately with 0 bytes; in practice after enable_raw_mode
     * the TODO is a stub so it may not set those flags — for the
     * purpose of this exercise a blocking read is fine. */
    char c;
    read(STDIN_FILENO, &c, 1);

    free(out_buf);
    return 0;
}
