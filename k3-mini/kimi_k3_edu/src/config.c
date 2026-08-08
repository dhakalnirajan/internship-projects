/**
 * config.c - Configuration management
 */

#include "config.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global configuration */
Config g_config;

void config_init(void) {
    memset(&g_config, 0, sizeof(Config));

    /* Architecture defaults */
    g_config.d_model = D_MODEL;
    g_config.n_layers = N_LAYERS;
    g_config.n_heads = N_HEADS;
    g_config.head_dim = HEAD_DIM;
    g_config.d_ffn = D_FFN;
    g_config.max_seq_len = MAX_SEQ_LEN;
    g_config.vocab_size = MAX_VOCAB_SIZE;

    /* Training defaults */
    g_config.batch_size = BASE_BATCH_SIZE;
    g_config.peak_lr = BASE_PEAK_LR;
    g_config.min_lr = 1e-5f;
    g_config.warmup_steps = BASE_WARMUP_STEPS;
    g_config.total_steps = BASE_TOTAL_STEPS;
    g_config.dropout = BASE_DROPOUT;
    g_config.weight_decay = BASE_WEIGHT_DECAY;
    g_config.grad_clip = BASE_GRAD_CLIP;
    g_config.grad_accum_steps = 1;

    /* Hardware defaults */
    g_config.n_cpu_threads = get_cpu_cores();
    g_config.use_cuda = (get_gpu_count() > 0) ? 1 : 0;
    g_config.gpu_memory_mb = get_gpu_memory_mb();
    g_config.system_ram_mb = get_system_ram_mb();

    /* Paths */
    strcpy(g_config.dataset_dir, DATASET_DIR);
    strcpy(g_config.checkpoint_dir, "checkpoints");
    strcpy(g_config.tokenizer_path, "tokenizer.bin");

    /* Flags */
    g_config.use_fp8 = USE_FP8_ACTIVATIONS;
    g_config.use_int8 = USE_INT8_WEIGHTS;
    g_config.use_kda = 1;
    g_config.use_attnres = 1;
}

int config_load(const char* path) {
    size_t len;
    char* json = read_file(path, &len);
    if (!json) return -1;

    g_config.d_model = json_get_int(json, "d_model", g_config.d_model);
    g_config.n_layers = json_get_int(json, "n_layers", g_config.n_layers);
    g_config.n_heads = json_get_int(json, "n_heads", g_config.n_heads);
    g_config.batch_size = json_get_int(json, "batch_size", g_config.batch_size);
    g_config.peak_lr = json_get_float(json, "peak_lr", g_config.peak_lr);
    g_config.dropout = json_get_float(json, "dropout", g_config.dropout);
    g_config.weight_decay = json_get_float(json, "weight_decay", g_config.weight_decay);

    char* ds_dir = json_get_string(json, "dataset_dir");
    if (ds_dir) {
        strncpy(g_config.dataset_dir, ds_dir, sizeof(g_config.dataset_dir) - 1);
        free(ds_dir);
    }

    free(json);
    return 0;
}

int config_save(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "{\n");
    fprintf(f, "  \"model_name\": \"%s\",\n", MODEL_NAME);
    fprintf(f, "  \"d_model\": %d,\n", g_config.d_model);
    fprintf(f, "  \"n_layers\": %d,\n", g_config.n_layers);
    fprintf(f, "  \"n_heads\": %d,\n", g_config.n_heads);
    fprintf(f, "  \"head_dim\": %d,\n", g_config.head_dim);
    fprintf(f, "  \"d_ffn\": %d,\n", g_config.d_ffn);
    fprintf(f, "  \"max_seq_len\": %d,\n", g_config.max_seq_len);
    fprintf(f, "  \"vocab_size\": %d,\n", g_config.vocab_size);
    fprintf(f, "  \"batch_size\": %d,\n", g_config.batch_size);
    fprintf(f, "  \"peak_lr\": %g,\n", g_config.peak_lr);
    fprintf(f, "  \"min_lr\": %g,\n", g_config.min_lr);
    fprintf(f, "  \"warmup_steps\": %d,\n", g_config.warmup_steps);
    fprintf(f, "  \"total_steps\": %d,\n", g_config.total_steps);
    fprintf(f, "  \"dropout\": %g,\n", g_config.dropout);
    fprintf(f, "  \"weight_decay\": %g,\n", g_config.weight_decay);
    fprintf(f, "  \"grad_clip\": %g,\n", g_config.grad_clip);
    fprintf(f, "  \"dataset_dir\": \"%s\",\n", g_config.dataset_dir);
    fprintf(f, "  \"use_fp8\": %d,\n", g_config.use_fp8);
    fprintf(f, "  \"use_int8\": %d,\n", g_config.use_int8);
    fprintf(f, "  \"use_kda\": %d,\n", g_config.use_kda);
    fprintf(f, "  \"use_attnres\": %d\n", g_config.use_attnres);
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

void config_auto_detect_hardware(void) {
    int cores = get_cpu_cores();
    size_t ram_mb = get_system_ram_mb();
    size_t gpu_mb = get_gpu_memory_mb();
    int n_gpus = get_gpu_count();

    log_info("Hardware detected: %d CPU cores, %zu MB RAM, %d GPU(s), %zu MB GPU memory",
             cores, ram_mb, n_gpus, gpu_mb);

    g_config.n_cpu_threads = cores;
    g_config.use_cuda = (n_gpus > 0) ? 1 : 0;
    g_config.gpu_memory_mb = gpu_mb;
    g_config.system_ram_mb = ram_mb;

    /* Adjust batch size based on available memory */
    if (gpu_mb > 0) {
        /* GPU available - can use larger batches */
        if (gpu_mb >= 24000) {
            g_config.batch_size = 8;
        } else if (gpu_mb >= 12000) {
            g_config.batch_size = 6;
        } else if (gpu_mb >= 8000) {
            g_config.batch_size = 4;
        } else {
            g_config.batch_size = 2;
        }
    } else {
        /* CPU only - smaller batches */
        if (ram_mb >= 32000) {
            g_config.batch_size = 4;
        } else if (ram_mb >= 16000) {
            g_config.batch_size = 2;
        } else {
            g_config.batch_size = 1;
        }
    }

    /* Calculate gradient accumulation steps */
    int target_tokens = GRAD_ACCUM_TARGET_TOKENS;
    g_config.grad_accum_steps = target_tokens / (g_config.batch_size * g_config.max_seq_len);
    if (g_config.grad_accum_steps < 1) g_config.grad_accum_steps = 1;

    log_info("Auto-configured: batch_size=%d, grad_accum=%d, use_cuda=%d",
             g_config.batch_size, g_config.grad_accum_steps, g_config.use_cuda);
}

void config_print(void) {
    printf("========================================\n");
    printf("  %s v%s Configuration\n", MODEL_NAME, MODEL_VERSION);
    printf("========================================\n");
    printf("Architecture:\n");
    printf("  d_model:      %d\n", g_config.d_model);
    printf("  n_layers:     %d\n", g_config.n_layers);
    printf("  n_heads:      %d\n", g_config.n_heads);
    printf("  head_dim:     %d\n", g_config.head_dim);
    printf("  d_ffn:        %d\n", g_config.d_ffn);
    printf("  max_seq_len:  %d\n", g_config.max_seq_len);
    printf("  vocab_size:   %d\n", g_config.vocab_size);
    printf("\nTraining:\n");
    printf("  batch_size:   %d\n", g_config.batch_size);
    printf("  peak_lr:      %g\n", g_config.peak_lr);
    printf("  min_lr:       %g\n", g_config.min_lr);
    printf("  warmup_steps: %d\n", g_config.warmup_steps);
    printf("  total_steps:  %d\n", g_config.total_steps);
    printf("  dropout:      %g\n", g_config.dropout);
    printf("  weight_decay: %g\n", g_config.weight_decay);
    printf("  grad_clip:    %g\n", g_config.grad_clip);
    printf("  grad_accum:   %d\n", g_config.grad_accum_steps);
    printf("\nHardware:\n");
    printf("  CPU threads:  %d\n", g_config.n_cpu_threads);
    printf("  Use CUDA:     %s\n", g_config.use_cuda ? "yes" : "no");
    printf("  GPU memory:   %zu MB\n", g_config.gpu_memory_mb);
    printf("  System RAM:   %zu MB\n", g_config.system_ram_mb);
    printf("\nPaths:\n");
    printf("  dataset_dir:  %s\n", g_config.dataset_dir);
    printf("  checkpoint:   %s\n", g_config.checkpoint_dir);
    printf("========================================\n");
}
