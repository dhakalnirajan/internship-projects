/* k3_tui.c — ANSI terminal UI implementation */
#include "k3_tui.h"
#include "k3_model.h"
#include "k3_tokenizer.h"
#include "k3_sampler.h"
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>
#include <signal.h>
#include <sys/ioctl.h>

/* ---------- ANSI helpers ---------- */
#define ESC "\033["

static void buf_append(Tui *t, const char *s) {
    size_t len = strlen(s);
    if (t->buf_len + len >= t->buf_cap) {
        t->buf_cap *= 2;
        t->buf = realloc(t->buf, t->buf_cap);
    }
    memcpy(t->buf + t->buf_len, s, len);
    t->buf_len += len;
}

static void buf_appendf(Tui *t, const char *fmt, ...) {
    char tmp[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    buf_append(t, tmp);
}

/* ---------- Terminal ---------- */
static struct termios orig_termios;
static volatile int got_sigwinch = 0;

static void on_sigwinch(int sig) { (void)sig; got_sigwinch = 1; }

static void tty_raw(void) {
    struct termios raw;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw = orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

static void tty_restore(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

Tui *tui_init(void) {
    Tui *t = calloc(1, sizeof(Tui));
    t->buf_cap = 65536;
    t->buf = malloc(t->buf_cap);
    t->fg = -1; t->bg = -1;
    signal(SIGWINCH, on_sigwinch);
    tty_raw();
    tui_get_size(t);
    buf_append(t, ESC "?25l"); /* hide cursor */
    buf_append(t, ESC "2J");   /* clear screen */
    tui_refresh(t);
    return t;
}

void tui_shutdown(Tui *t) {
    if (!t) return;
    buf_append(t, ESC "?25h"); /* show cursor */
    buf_append(t, ESC "0m");
    buf_append(t, ESC "2J" ESC "H");
    tui_refresh(t);
    tty_restore();
    free(t->buf);
    free(t);
}

void tui_get_size(Tui *t) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        t->w = ws.ws_col;
        t->h = ws.ws_row;
    } else {
        t->w = 80; t->h = 24;
    }
}

void tui_clear(Tui *t) {
    buf_append(t, ESC "2J" ESC "H");
    t->fg = -1; t->bg = -1; t->bold = 0;
}

void tui_goto(Tui *t, int x, int y) {
    buf_appendf(t, ESC "%d;%dH", y + 1, x + 1);
}

void tui_fg(Tui *t, int color) {
    if (color == t->fg) return;
    t->fg = color;
    if (color >= 0) buf_appendf(t, ESC "38;5;%dm", color);
    else buf_append(t, ESC "39m");
}

void tui_bg(Tui *t, int color) {
    if (color == t->bg) return;
    t->bg = color;
    if (color >= 0) buf_appendf(t, ESC "48;5;%dm", color);
    else buf_append(t, ESC "49m");
}

void tui_bold(Tui *t, int on) {
    if (on == t->bold) return;
    t->bold = on;
    buf_append(t, on ? ESC "1m" : ESC "22m");
}

void tui_reset(Tui *t) {
    buf_append(t, ESC "0m");
    t->fg = -1; t->bg = -1; t->bold = 0;
}

void tui_puts(Tui *t, int x, int y, const char *s) {
    tui_goto(t, x, y);
    buf_append(t, s);
}

void tui_putc(Tui *t, int x, int y, char c) {
    tui_goto(t, x, y);
    char tmp[2] = {c, 0};
    buf_append(t, tmp);
}

void tui_hline(Tui *t, int x, int y, int len, char c) {
    tui_goto(t, x, y);
    for (int i = 0; i < len; i++) {
        char tmp[2] = {c, 0};
        buf_append(t, tmp);
    }
}

void tui_vline(Tui *t, int x, int y, int len, char c) {
    for (int i = 0; i < len; i++) tui_putc(t, x, y + i, c);
}

void tui_box(Tui *t, int x, int y, int w, int h) {
    tui_hline(t, x, y, w, '─');
    tui_hline(t, x, y + h - 1, w, '─');
    tui_vline(t, x, y + 1, h - 2, '│');
    tui_vline(t, x + w - 1, y + 1, h - 2, '│');
    tui_putc(t, x, y, '┌');
    tui_putc(t, x + w - 1, y, '┐');
    tui_putc(t, x, y + h - 1, '└');
    tui_putc(t, x + w - 1, y + h - 1, '┘');
}

void tui_fill(Tui *t, int x, int y, int w, int h, int bg) {
    tui_bg(t, bg);
    for (int row = 0; row < h; row++) {
        tui_goto(t, x, y + row);
        for (int col = 0; col < w; col++) buf_append(t, " ");
    }
    tui_bg(t, -1);
}

void tui_refresh(Tui *t) {
    if (t->buf_len > 0) {
        write(STDOUT_FILENO, t->buf, t->buf_len);
        t->buf_len = 0;
    }
}

/* ---------- Input ---------- */
Event tui_poll_event(Tui *t, int timeout_ms) {
    Event ev = {EV_NONE, 0, 0, 0};
    if (got_sigwinch) {
        got_sigwinch = 0;
        tui_get_size(t);
        ev.type = EV_RESIZE;
        return ev;
    }
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) <= 0)
        return ev;

    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return ev;

    if (c == 3)  { ev.type = EV_KEY; ev.key = KEY_CTRL_C; ev.ctrl = 1; return ev; }
    if (c == 12) { ev.type = EV_KEY; ev.key = KEY_CTRL_L; ev.ctrl = 1; return ev; }
    if (c == 14) { ev.type = EV_KEY; ev.key = KEY_CTRL_N; ev.ctrl = 1; return ev; }
    if (c == 19) { ev.type = EV_KEY; ev.key = KEY_CTRL_S; ev.ctrl = 1; return ev; }

    if (c == 27) { /* Escape sequence */
        char seq[4] = {0};
        if (read(STDIN_FILENO, &seq[0], 1) != 1) { ev.type = EV_KEY; ev.key = KEY_ESC; return ev; }
        if (read(STDIN_FILENO, &seq[1], 1) != 1) { ev.type = EV_KEY; ev.key = KEY_ESC; return ev; }
        if (seq[0] == '[') {
            switch (seq[1]) {
                case 'A': ev.key = KEY_UP; break;
                case 'B': ev.key = KEY_DOWN; break;
                case 'C': ev.key = KEY_RIGHT; break;
                case 'D': ev.key = KEY_LEFT; break;
                case 'H': ev.key = KEY_HOME; break;
                case 'F': ev.key = KEY_END; break;
                case '3': read(STDIN_FILENO, &seq[2], 1); ev.key = KEY_DELETE; break;
                case '5': read(STDIN_FILENO, &seq[2], 1); ev.key = KEY_PAGEUP; break;
                case '6': read(STDIN_FILENO, &seq[2], 1); ev.key = KEY_PAGEDOWN; break;
                default: ev.key = KEY_ESC; break;
            }
            ev.type = EV_KEY;
            return ev;
        }
        ev.type = EV_KEY; ev.key = KEY_ESC; return ev;
    }

    ev.type = EV_KEY;
    ev.key = (unsigned char)c;
    return ev;
}

/* ---------- App State ---------- */
AppState *app_state_new(void) {
    AppState *s = calloc(1, sizeof(AppState));
    s->n_sessions = 1;
    strcpy(s->sessions[0].name, "Chat 1");
    s->active_session = 0;
    s->focus = 2; /* input focus */
    s->sampler.temperature = 0.7f;
    s->sampler.top_k = 40;
    s->sampler.top_p = 0.9f;
    s->sampler.max_new_tokens = 512;
    s->sampler.repetition_penalty = 1.0f;
    s->sampler.do_sample = 1;
    k3_rng_init(&s->rng, (uint64_t)time(NULL));
    strcpy(s->status, "Ready. Ctrl+N=new  Ctrl+C=quit");
    return s;
}

void app_state_free(AppState *s) {
    if (!s) return;
    for (int i = 0; i < s->n_sessions; i++) {
        for (int j = 0; j < s->sessions[i].n_msgs; j++)
            free(s->sessions[i].msgs[j].text);
    }
    free(s);
}

void app_add_message(AppState *s, int role, const char *text) {
    Session *sess = &s->sessions[s->active_session];
    if (sess->n_msgs >= MAX_MSGS) return;
    Message *m = &sess->msgs[sess->n_msgs++];
    m->role = role;
    m->len = (int)strlen(text);
    m->text = malloc(m->len + 1);
    strcpy(m->text, text);
    s->chat_scroll = 999999; /* auto-scroll to bottom */
}

void app_new_session(AppState *s) {
    if (s->n_sessions >= MAX_SESSIONS) return;
    int n = s->n_sessions++;
    snprintf(s->sessions[n].name, sizeof(s->sessions[n].name), "Chat %d", n + 1);
    s->active_session = n;
    s->chat_scroll = 0;
    s->input_len = 0;
    s->input[0] = '\0';
    s->input_cursor = 0;
}

/* ---------- Rendering ---------- */
#define C_FG       235
#define C_DIM      244
#define C_BORDER   180
#define C_ACCENT   130
#define C_TITLE    94
#define C_BG       230
#define C_HIGHLIGHT 187
#define C_USER     235
#define C_ASST     130
#define C_SYS      244

static void draw_header(Tui *t, AppState *app) {
    (void)app;
    int w = t->w;
    tui_fill(t, 0, 0, w, 1, C_BG);
    tui_fg(t, C_TITLE);
    tui_bold(t, 1);
    tui_puts(t, 2, 0, " K3-Edu Inference ");
    tui_bold(t, 0);
    tui_fg(t, C_DIM);
    char info[128];
    snprintf(info, sizeof(info), "│ temp=%.2f top_k=%d top_p=%.2f ",
             app->sampler.temperature, app->sampler.top_k, app->sampler.top_p);
    tui_puts(t, w - (int)strlen(info) - 2, 0, info);
    tui_reset(t);
    tui_fg(t, C_BORDER);
    tui_hline(t, 0, 1, w, '─');
    tui_reset(t);
}

static void draw_sidebar(Tui *t, AppState *app, int x, int y, int w, int h) {
    tui_fill(t, x, y, w, h, C_BG);
    tui_fg(t, C_BORDER);
    tui_vline(t, x + w - 1, y, h, '│');
    tui_reset(t);

    tui_fg(t, C_TITLE);
    tui_bold(t, 1);
    tui_puts(t, x + 1, y, " Sessions ");
    tui_bold(t, 0);
    tui_reset(t);

    for (int i = app->sess_scroll; i < app->n_sessions && (i - app->sess_scroll) < h - 2; i++) {
        int row = y + 1 + (i - app->sess_scroll);
        int is_active = (i == app->active_session);
        if (is_active) {
            tui_bg(t, C_HIGHLIGHT);
            tui_fg(t, C_TITLE);
            tui_bold(t, 1);
        } else {
            tui_fg(t, C_FG);
        }
        char line[128];
        snprintf(line, sizeof(line), " %s", app->sessions[i].name);
        if ((int)strlen(line) > w - 2) line[w - 2] = '\0';
        tui_puts(t, x, row, line);
        for (int j = (int)strlen(line); j < w - 1; j++) tui_putc(t, x + j, row, ' ');
        tui_reset(t);
        tui_bg(t, -1);
    }
}

static void draw_chat(Tui *t, AppState *app, int x, int y, int w, int h) {
    tui_fill(t, x, y, w, h, C_BG);
    Session *sess = &app->sessions[app->active_session];

    /* Calculate total lines */
    int total_lines = 0;
    for (int i = 0; i < sess->n_msgs; i++) {
        total_lines += 2; /* role header + content lines */
        int text_w = w - 4;
        if (text_w < 10) text_w = 10;
        total_lines += (sess->msgs[i].len + text_w - 1) / text_w;
    }

    int max_scroll = total_lines > h ? total_lines - h : 0;
    if (app->chat_scroll > max_scroll) app->chat_scroll = max_scroll;
    if (app->chat_scroll < 0) app->chat_scroll = 0;

    int skip = app->chat_scroll;
    int row = y;
    for (int i = 0; i < sess->n_msgs && row < y + h; i++) {
        Message *m = &sess->msgs[i];
        const char *role_str = (m->role == 0) ? " you " : (m->role == 1) ? " assistant " : " system ";
        int role_color = (m->role == 0) ? C_USER : (m->role == 1) ? C_ASST : C_SYS;

        /* Role header */
        if (skip > 0) { skip--; }
        else if (row < y + h) {
            tui_fg(t, role_color);
            tui_bold(t, 1);
            tui_puts(t, x + 1, row, role_str);
            tui_bold(t, 0);
            tui_reset(t);
            row++;
        }

        /* Text wrapped */
        int text_w = w - 4;
        if (text_w < 10) text_w = 10;
        const char *p = m->text;
        while (*p && row < y + h) {
            int line_len = 0;
            while (p[line_len] && line_len < text_w && p[line_len] != '\n') line_len++;
            if (skip > 0) { skip--; }
            else {
                tui_fg(t, C_FG);
                char buf[512];
                int n = line_len;
                if (n >= sizeof(buf)) n = sizeof(buf) - 1;
                memcpy(buf, p, n);
                buf[n] = '\0';
                tui_puts(t, x + 2, row, buf);
                tui_reset(t);
                row++;
            }
            p += line_len;
            if (*p == '\n') p++;
            if (*p == ' ') p++;
        }
    }

    /* Scrollbar indicator */
    if (max_scroll > 0) {
        int bar_h = h * h / (h + max_scroll);
        if (bar_h < 1) bar_h = 1;
        int bar_y = y + (app->chat_scroll * (h - bar_h) / max_scroll);
        tui_fg(t, C_BORDER);
        for (int i = 0; i < bar_h && bar_y + i < y + h; i++)
            tui_putc(t, x + w - 1, bar_y + i, '┃');
        tui_reset(t);
    }
}

static void draw_input(Tui *t, AppState *app, int x, int y, int w) {
    tui_fg(t, C_BORDER);
    tui_hline(t, x, y, w, '─');
    tui_reset(t);

    tui_fg(t, C_ACCENT);
    tui_puts(t, x + 1, y + 1, "> ");
    tui_reset(t);

    int prompt_w = w - 4;
    if (prompt_w < 1) prompt_w = 1;
    int display_start = 0;
    if (app->input_cursor > prompt_w - 1)
        display_start = app->input_cursor - prompt_w + 1;

    char buf[INPUT_MAX];
    int n = app->input_len - display_start;
    if (n > prompt_w) n = prompt_w;
    if (n < 0) n = 0;
    memcpy(buf, app->input + display_start, n);
    buf[n] = '\0';
    tui_fg(t, C_FG);
    tui_puts(t, x + 3, y + 1, buf);
    tui_reset(t);

    /* Cursor */
    int cx = x + 3 + (app->input_cursor - display_start);
    tui_goto(t, cx, y + 1);
    buf_append(t, ESC "?25h");
}

static void draw_footer(Tui *t, AppState *app, int x, int y, int w) {
    tui_fill(t, x, y, w, 1, C_BG);
    tui_fg(t, C_DIM);
    tui_puts(t, x + 1, y, app->status);
    tui_reset(t);
}

void app_render(Tui *tui, AppState *app) {
    if (got_sigwinch) {
        got_sigwinch = 0;
        tui_get_size(tui);
    }
    tui_clear(tui);
    int w = tui->w, h = tui->h;
    if (h < 8 || w < 40) {
        tui_puts(tui, 0, 0, "Terminal too small");
        tui_refresh(tui);
        return;
    }

    int sidebar_w = 18;
    if (sidebar_w > w / 4) sidebar_w = w / 4;
    int content_x = sidebar_w;
    int content_w = w - sidebar_w;
    int header_h = 2;
    int footer_h = 1;
    int input_h = 2;
    int chat_h = h - header_h - input_h - footer_h;

    draw_header(tui, app);
    draw_sidebar(tui, app, 0, header_h, sidebar_w, chat_h);
    draw_chat(tui, app, content_x, header_h, content_w, chat_h);
    draw_input(tui, app, 0, h - input_h - footer_h, w);
    draw_footer(tui, app, 0, h - 1, w);

    tui_refresh(tui);
}

/* ---------- Event Handling ---------- */
static void input_insert(AppState *s, char c) {
    if (s->input_len >= INPUT_MAX - 1) return;
    memmove(s->input + s->input_cursor + 1, s->input + s->input_cursor,
            s->input_len - s->input_cursor);
    s->input[s->input_cursor] = c;
    s->input_len++;
    s->input[s->input_len] = '\0';
    s->input_cursor++;
}

static void input_delete(AppState *s) {
    if (s->input_cursor < s->input_len) {
        memmove(s->input + s->input_cursor, s->input + s->input_cursor + 1,
                s->input_len - s->input_cursor);
        s->input_len--;
        s->input[s->input_len] = '\0';
    }
}

static void input_backspace(AppState *s) {
    if (s->input_cursor > 0) {
        s->input_cursor--;
        input_delete(s);
    }
}

void app_handle_event(Tui *tui, AppState *app, Event *ev) {
    (void)tui;
    if (ev->type == EV_RESIZE) return;
    if (ev->type != EV_KEY) return;

    if (ev->key == KEY_CTRL_C) {
        app->focus = -1; /* signal quit */
        return;
    }
    if (ev->key == KEY_CTRL_N) {
        app_new_session(app);
        snprintf(app->status, sizeof(app->status), "New session created");
        return;
    }

    if (app->focus == 2) { /* Input focus */
        if (ev->key == KEY_ENTER) {
            if (app->input_len > 0) {
                app->input[app->input_len] = '\0';
                app_add_message(app, 0, app->input);
                app->input_len = 0;
                app->input[0] = '\0';
                app->input_cursor = 0;
                app->generating = 1;
                snprintf(app->status, sizeof(app->status), "Generating...");
            }
        } else if (ev->key == KEY_BACKSPACE) {
            input_backspace(app);
        } else if (ev->key == KEY_DELETE) {
            input_delete(app);
        } else if (ev->key == KEY_LEFT) {
            if (app->input_cursor > 0) app->input_cursor--;
        } else if (ev->key == KEY_RIGHT) {
            if (app->input_cursor < app->input_len) app->input_cursor++;
        } else if (ev->key == KEY_HOME) {
            app->input_cursor = 0;
        } else if (ev->key == KEY_END) {
            app->input_cursor = app->input_len;
        } else if (ev->key == KEY_TAB) {
            app->focus = 0;
        } else if (ev->key >= 32 && ev->key < 127) {
            input_insert(app, (char)ev->key);
        }
    } else if (app->focus == 0) { /* Sidebar focus */
        if (ev->key == KEY_UP) {
            if (app->active_session > 0) app->active_session--;
            if (app->active_session < app->sess_scroll) app->sess_scroll = app->active_session;
        } else if (ev->key == KEY_DOWN) {
            if (app->active_session < app->n_sessions - 1) app->active_session++;
        } else if (ev->key == KEY_TAB || ev->key == KEY_ENTER) {
            app->focus = 2;
        } else if (ev->key == 'd' && app->n_sessions > 1) {
            /* Delete session */
            Session *sess = &app->sessions[app->active_session];
            for (int i = 0; i < sess->n_msgs; i++) free(sess->msgs[i].text);
            memmove(&app->sessions[app->active_session],
                    &app->sessions[app->active_session + 1],
                    (app->n_sessions - app->active_session - 1) * sizeof(Session));
            app->n_sessions--;
            if (app->active_session >= app->n_sessions) app->active_session = app->n_sessions - 1;
            snprintf(app->status, sizeof(app->status), "Session deleted");
        }
    }
}

/* ---------- Generation ---------- */
static int format_prompt(AppState *app, int *ids, int max_len) {
    Tokenizer *tok = (Tokenizer*)app->tokenizer;
    Session *sess = &app->sessions[app->active_session];
    char prompt[8192] = {0};
    int p = 0;
    const char *sys = "You are a helpful assistant.";
    p += snprintf(prompt + p, sizeof(prompt) - p, "<|im_start|>system\n%s<|im_end|>\n", sys);
    for (int i = 0; i < sess->n_msgs; i++) {
        const char *role = (sess->msgs[i].role == 0) ? "user" : "assistant";
        p += snprintf(prompt + p, sizeof(prompt) - p, "<|im_start|>%s\n%s<|im_end|>\n",
                      role, sess->msgs[i].text);
    }
    p += snprintf(prompt + p, sizeof(prompt) - p, "<|im_start|>assistant\n");
    return tokenizer_encode(tok, prompt, ids, max_len);
}

void app_generate_step(AppState *app) {
    if (!app->generating) return;
    TransformerModel *model = (TransformerModel*)app->model;
    Tokenizer *tok = (Tokenizer*)app->tokenizer;
    KVCache *kv = (KVCache*)app->kv_cache;
    if (!model || !tok || !kv) {
        app_add_message(app, 1, "[Model not loaded — this is a demo build]");
        app->generating = 0;
        snprintf(app->status, sizeof(app->status), "Ready");
        return;
    }

    static int prompt_ids[MAX_SEQ_LEN];
    static int prev_ids[MAX_SEQ_LEN];
    static int n_prev = 0;
    static int n_prompt = 0;
    static int step = 0;

    if (step == 0) {
        kv_cache_clear(kv);
        n_prompt = format_prompt(app, prompt_ids, MAX_SEQ_LEN);
        n_prev = n_prompt;
        memcpy(prev_ids, prompt_ids, n_prompt * sizeof(int));
        step = 0;
    }

    float logits[MAX_VOCAB];
    model_forward(model, prev_ids, n_prev, logits, kv);

    int next_tok;
    sample_logits(logits, model->vocab_size, &app->sampler,
                  prev_ids, n_prev, &app->rng, &next_tok);

    if (next_tok == TOK_EOS || step >= app->sampler.max_new_tokens) {
        app->generating = 0;
        step = 0;
        n_prev = 0;
        snprintf(app->status, sizeof(app->status), "Ready");
        return;
    }

    /* Append token to last assistant message */
    Session *sess = &app->sessions[app->active_session];
    if (sess->n_msgs == 0 || sess->msgs[sess->n_msgs - 1].role != 1) {
        app_add_message(app, 1, "");
    }
    Message *m = &sess->msgs[sess->n_msgs - 1];
    const char *tok_str = tokenizer_get_token(tok, next_tok);
    int tlen = (int)strlen(tok_str);
    if (tlen > 0 && tok_str[0] != '<') {
        m->text = realloc(m->text, m->len + tlen + 2);
        if (m->len > 0 && m->text[m->len - 1] != ' ') {
            m->text[m->len] = ' ';
            m->len++;
        }
        memcpy(m->text + m->len, tok_str, tlen);
        m->len += tlen;
        m->text[m->len] = '\0';
    }

    prev_ids[0] = next_tok;
    n_prev = 1;
    step++;
}
