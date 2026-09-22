/* k3_main.c — Entry point */
#include "k3_common.h"
#include "k3_model.h"
#include "k3_tokenizer.h"
#include "k3_sampler.h"
#include "k3_tui.h"
#include <signal.h>

static volatile int running = 1;

static void on_int(int sig) { (void)sig; running = 0; }

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -m, --model <path>     Model checkpoint file\n");
    printf("  -t, --tokenizer <path> Tokenizer vocab file\n");
    printf("  --temp <float>         Sampling temperature (default: 0.7)\n");
    printf("  --top-k <int>          Top-k filtering (default: 40)\n");
    printf("  --top-p <float>        Top-p nucleus filtering (default: 0.9)\n");
    printf("  --max-tokens <int>     Max tokens to generate (default: 512)\n");
    printf("  -p, --prompt <text>    One-shot prompt (non-TUI mode)\n");
    printf("  -h, --help             Show this help\n");
}

static void run_oneshot(const char *model_path, const char *tok_path,
                        SamplerConfig *cfg, const char *prompt) {
    TransformerModel *model = model_load(model_path);
    Tokenizer *tok = tokenizer_load(tok_path);
    KVCache *kv = kv_cache_create(model->n_layers, model->n_heads,
                                  MAX_SEQ_LEN, HEAD_DIM);
    K3Rng rng;
    k3_rng_init(&rng, (uint64_t)time(NULL));

    int ids[MAX_SEQ_LEN];
    int n = tokenizer_encode(tok, prompt, ids, MAX_SEQ_LEN);
    if (n == 0) { printf("Error: empty prompt\n"); return; }

    printf("Prompt: %s\n", prompt);
    printf("---\n");

    int prev[MAX_SEQ_LEN];
    int n_prev = n;
    memcpy(prev, ids, n * sizeof(int));

    for (int step = 0; step < cfg->max_new_tokens; step++) {
        float logits[MAX_VOCAB];
        model_forward(model, prev, n_prev, logits, kv);
        int next_tok;
        sample_logits(logits, model->vocab_size, cfg, prev, n_prev, &rng, &next_tok);
        if (next_tok == TOK_EOS) break;
        const char *tok_str = tokenizer_get_token(tok, next_tok);
        if (tok_str[0] != '<') printf("%s ", tok_str);
        fflush(stdout);
        prev[0] = next_tok;
        n_prev = 1;
    }
    printf("\n");

    kv_cache_free(kv);
    tokenizer_free(tok);
    model_free(model);
}

static void run_tui(const char *model_path, const char *tok_path,
                    SamplerConfig *cfg) {
    TransformerModel *model = model_load(model_path);
    Tokenizer *tok = tokenizer_load(tok_path);
    KVCache *kv = kv_cache_create(model->n_layers, model->n_heads,
                                  MAX_SEQ_LEN, HEAD_DIM);

    AppState *app = app_state_new();
    app->model = model;
    app->tokenizer = tok;
    app->kv_cache = kv;
    app->sampler = *cfg;

    app_add_message(app, 2, "Welcome to K3-Edu Inference.\n"
        "Type a message and press Enter.\n"
        "Ctrl+N = new chat  |  Ctrl+C = quit  |  Tab = switch focus");

    Tui *tui = tui_init();
    signal(SIGINT, on_int);

    while (running && app->focus >= 0) {
        app_render(tui, app);
        Event ev = tui_poll_event(tui, 50);
        if (ev.type == EV_KEY && ev.key == KEY_CTRL_C) break;
        if (ev.type != EV_NONE) app_handle_event(tui, app, &ev);
        if (app->generating) app_generate_step(app);
    }

    tui_shutdown(tui);
    app_state_free(app);
    kv_cache_free(kv);
    tokenizer_free(tok);
    model_free(model);
}

int main(int argc, char **argv) {
    const char *model_path = "k3_model.bin";
    const char *tok_path = "tokenizer.bin";
    const char *prompt = NULL;
    SamplerConfig cfg = {0.7f, 40, 0.9f, 512, 1.0f, 1};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokenizer") == 0) && i + 1 < argc) {
            tok_path = argv[++i];
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            cfg.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            cfg.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            cfg.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            cfg.max_new_tokens = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    if (prompt) {
        run_oneshot(model_path, tok_path, &cfg, prompt);
    } else {
        run_tui(model_path, tok_path, &cfg);
    }
    return 0;
}
