/**
 * train_main.c - Unified training entry point
 * 
 * Usage:
 *   ./train_base --dataset <dir> --tokenizer <file> --config <file> --checkpoint-dir <dir>
 */

#include "train.h"
#include "model.h"
#include "tokenizer.h"
#include "config.h"
#include "metrics.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ------------------------------------------------------------------
 * Live-dashboard metrics: run id + emitter shared with the trainer.
 * The Python server (scripts/monitor_server.py) tails this file and
 * renders web/dashboard.html.
 * ------------------------------------------------------------------ */
static MetricsEmitter* g_metrics = NULL;

static void make_run_id(char* buf, size_t buflen) {
    time_t t = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    snprintf(buf, buflen, "runs/run_%04d%02d%02d_%02d%02d%02d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}

int main(int argc, char** argv) {
    config_init();

    char* dataset_dir = "datasets/train";
    char* val_dir = "datasets/val";
    char* tokenizer_path = "tokenizer.bin";
    char* config_path = "config.json";
    char* checkpoint_dir = "checkpoints";
    int phase = 1; /* 1=base, 2=instruction */
    char* base_model = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) dataset_dir = argv[++i];
        else if (strcmp(argv[i], "--val-dataset") == 0 && i + 1 < argc) val_dir = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) config_path = argv[++i];
        else if (strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) checkpoint_dir = argv[++i];
        else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) phase = atoi(argv[++i]);
        else if (strcmp(argv[i], "--base-model") == 0 && i + 1 < argc) base_model = argv[++i];
    }

    /* Load config */
    if (config_load(config_path) == 0) {
        log_info("Configuration loaded from %s", config_path);
    }
    config_auto_detect_hardware();
    config_print();

    /* Load tokenizer */
    printf("Loading tokenizer from %s...\n", tokenizer_path);
    Tokenizer* tok = tokenizer_load(tokenizer_path);
    if (!tok) {
        printf("Tokenizer not found. Training new tokenizer...\n");
        tok = tokenizer_create(TOKENIZER_VOCAB_SIZE);
        tokenizer_train(tok, dataset_dir, TOKENIZER_VOCAB_SIZE);
        tokenizer_save(tok, tokenizer_path);
    }
    g_config.vocab_size = tok->vocab_size;

    /* Create or load model */
    TransformerModel* model;
    if (base_model) {
        printf("Loading base model from %s...\n", base_model);
        model = model_load(base_model, tok);
    } else {
        printf("Creating new model...\n");
        model = model_create(&g_config, tok);
        model_init_weights(model);
    }

    printf("Model: %zu parameters\n", model->n_params);

    /* Create datasets */
    Dataset* train_ds = dataset_create(dataset_dir, g_config.max_seq_len, g_config.batch_size);
    Dataset* val_ds = dataset_create(val_dir, g_config.max_seq_len, g_config.batch_size);

    if (!train_ds) {
        printf("Error: Failed to load training dataset\n");
        return 1;
    }

    /* Initialize training */
    TrainState* state = train_init(model, tok, &g_config);

    /* Live dashboard: open metrics emitter and describe the run. */
    char run_id[128];
    make_run_id(run_id, sizeof(run_id));
    g_metrics = metrics_open("runs", run_id);
    if (g_metrics) {
        /* serialize the essential config for the dashboard meta line */
        char cfg_json[512];
        snprintf(cfg_json, sizeof(cfg_json),
                 "{\"d_model\":%d,\"n_layers\":%d,\"n_heads\":%d,"
                 "\"head_dim\":%d,\"d_ffn\":%d,\"max_seq_len\":%d,"
                 "\"vocab_size\":%d,\"batch_size\":%d,\"peak_lr\":%.6g,"
                 "\"total_steps\":%d,\"warmup_steps\":%d}",
                 g_config.d_model, g_config.n_layers, g_config.n_heads,
                 g_config.head_dim, g_config.d_ffn, g_config.max_seq_len,
                 g_config.vocab_size, g_config.batch_size, g_config.peak_lr,
                 g_config.total_steps, g_config.warmup_steps);
        metrics_emit_meta(g_metrics, "K3-Edu-200M", model->n_params, cfg_json);
        printf("Live dashboard: python scripts/monitor_server.py  (run: %s)\n", run_id);
    }

    /* Run training phase with live metrics emission.
     *
     * The training loop lives here so every step can push a metrics line to
     * the dashboard (loss, lr, throughput, cost, ETA). If train_base/train_inst
     * are implemented elsewhere (e.g. cuda/train_gpu.cu), call them instead and
     * move the metrics_emit_step block into that loop. */
    printf("\n=== Phase %d: %s ===\n", phase,
           phase == 1 ? "Base Model Training" : "Instruction Fine-Tuning");
    {
        const int total_steps = g_config.total_steps;
        const double t_start = (double)clock() / CLOCKS_PER_SEC;
        double tokens_total = 0.0, samples_total = 0.0;
        const int log_every = 10;
        const float gpu_hourly_usd = 2.14f;   /* adjust for your hardware */
        int input_ids[64 * 256];              /* batch*seq window (bounded demo) */
        int target_ids[64 * 256];
        const int bs = g_config.batch_size > 64 ? 64 : g_config.batch_size;
        const int sl = g_config.max_seq_len > 256 ? 256 : g_config.max_seq_len;
        float last_val = 0.0f;

        for (long step = 1; step <= total_steps; step++) {
            int got = dataset_next_batch(train_ds, input_ids, target_ids, bs, sl);
            if (got <= 0) { dataset_reset(train_ds); got = dataset_next_batch(train_ds, input_ids, target_ids, bs, sl); }

            float loss = train_step(state, input_ids, target_ids, bs, sl);
            tokens_total += (double)got * sl;
            samples_total += got;

            const double elapsed = (double)clock() / CLOCKS_PER_SEC - t_start;
            const double tps = tokens_total / (elapsed > 0 ? elapsed : 1.0);
            const double eta = elapsed * (total_steps - step);

            if (g_metrics && (step % log_every == 0 || step == 1 || step == total_steps)) {
                /* data-streaming snapshot: which file is being consumed now */
                MetricsData md;
                memset(&md, 0, sizeof(md));
                md.dataset_dir = train_ds->dataset_dir ? train_ds->dataset_dir : dataset_dir;
                md.dataset_files = train_ds->n_files;
                md.dataset_tokens = 0.0;
                md.val_files = val_ds ? val_ds->n_files : 0;
                md.file_pos = train_ds->current_file;
                md.file_name = (train_ds->file_paths && train_ds->current_file < train_ds->n_files)
                                   ? train_ds->file_paths[train_ds->current_file] : NULL;
                md.file_progress = 0.0f;

                /* hardware snapshot: fill from your environment where available;
                 * untracked fields stay 0/NULL and are omitted from the JSON. */
                MetricsHardware hw;
                memset(&hw, 0, sizeof(hw));
#ifdef USE_CUDA
                hw.device = "CUDA GPU";
                hw.cuda_version = 12.0f;
#else
                hw.device = "CPU";
#endif
                hw.cpu_threads = 8;

                metrics_emit_step(g_metrics,
                    step, phase == 1 ? "base" : "inst",
                    loss, last_val,
                    state->optimizer ? state->optimizer->lr : g_config.peak_lr,
                    0.0f,
                    tokens_total, tps,
                    samples_total, train_ds->epoch,
                    bs, sl,
                    &md,
                    elapsed, eta, (float)((double)step / total_steps),
                    &hw,
                    metrics_cost_usd(elapsed, gpu_hourly_usd), gpu_hourly_usd,
                    NULL);
            }
        }
    }

    /* Save final model */
    char final_path[256];
    snprintf(final_path, sizeof(final_path), "%s/final_%s.bin", 
             checkpoint_dir, phase == 1 ? "base" : "inst");
    model_save(model, final_path);
    printf("Final model saved to %s\n", final_path);

    if (g_metrics) {
        metrics_emit_event(g_metrics, "finished", final_path);
        metrics_close(g_metrics);
        g_metrics = NULL;
    }

    /* Cleanup */
    train_free(state);
    dataset_free(train_ds);
    dataset_free(val_ds);
    model_free(model);
    tokenizer_free(tok);

    return 0;
}
