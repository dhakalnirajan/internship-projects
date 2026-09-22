/* k3_tui.h — Terminal UI (ANSI, no ncurses) */
#ifndef K3_TUI_H
#define K3_TUI_H
#include "k3_common.h"

/* ---------- Events ---------- */
typedef enum {
    EV_NONE,
    EV_KEY,
    EV_RESIZE,
    EV_QUIT
} EvType;

typedef struct {
    EvType type;
    int key;
    int ctrl;
    int alt;
} Event;

#define KEY_ENTER     '\r'
#define KEY_BACKSPACE 127
#define KEY_TAB       '\t'
#define KEY_ESC       27
#define KEY_UP        1001
#define KEY_DOWN      1002
#define KEY_LEFT      1003
#define KEY_RIGHT     1004
#define KEY_HOME      1005
#define KEY_END       1006
#define KEY_PAGEUP    1007
#define KEY_PAGEDOWN  1008
#define KEY_DELETE    1009
#define KEY_CTRL_C    3
#define KEY_CTRL_N    14
#define KEY_CTRL_S    19
#define KEY_CTRL_L    12

/* ---------- TUI ---------- */
typedef struct {
    int w, h;
    char *buf;
    size_t buf_cap;
    size_t buf_len;
    int fg, bg;
    int bold;
    int dirty;
} Tui;

Tui *tui_init(void);
void tui_shutdown(Tui *t);
void tui_clear(Tui *t);
void tui_goto(Tui *t, int x, int y);
void tui_fg(Tui *t, int color);
void tui_bg(Tui *t, int color);
void tui_bold(Tui *t, int on);
void tui_reset(Tui *t);
void tui_puts(Tui *t, int x, int y, const char *s);
void tui_putc(Tui *t, int x, int y, char c);
void tui_hline(Tui *t, int x, int y, int len, char c);
void tui_vline(Tui *t, int x, int y, int len, char c);
void tui_box(Tui *t, int x, int y, int w, int h);
void tui_fill(Tui *t, int x, int y, int w, int h, int bg);
void tui_refresh(Tui *t);
Event tui_poll_event(Tui *t, int timeout_ms);
void tui_get_size(Tui *t);

/* ---------- App State ---------- */
#define MAX_MSGS     1024
#define MAX_SESSIONS 64
#define INPUT_MAX    4096

typedef struct {
    char *text;
    int role;   /* 0=user, 1=assistant, 2=system */
    int len;
} Message;

typedef struct {
    Message msgs[MAX_MSGS];
    int n_msgs;
    char name[64];
} Session;

typedef struct {
    Session sessions[MAX_SESSIONS];
    int n_sessions;
    int active_session;
    int sess_scroll;

    int chat_scroll;
    int focus; /* 0=sidebar, 1=chat, 2=input */
    int generating;
    char status[256];

    char input[INPUT_MAX];
    int input_len;
    int input_cursor;

    void *model;
    void *tokenizer;
    void *kv_cache;
    SamplerConfig sampler;
    K3Rng rng;
} AppState;

AppState *app_state_new(void);
void app_state_free(AppState *s);
void app_add_message(AppState *s, int role, const char *text);
void app_new_session(AppState *s);
void app_render(Tui *tui, AppState *app);
void app_handle_event(Tui *tui, AppState *app, Event *ev);
void app_generate_step(AppState *app);

#endif
