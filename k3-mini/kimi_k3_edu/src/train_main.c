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
#include <stdio.h>
#include <string.h>

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

    /* Run training phase */
    if (phase == 1) {
        printf("\n=== Phase 1: Base Model Training ===\n");
        train_base(state, train_ds, &g_config);
    } else {
        printf("\n=== Phase 2: Instruction Fine-Tuning ===\n");
        train_instruction(state, train_ds, &g_config);
    }

    /* Save final model */
    char final_path[256];
    snprintf(final_path, sizeof(final_path), "%s/final_%s.bin", 
             checkpoint_dir, phase == 1 ? "base" : "inst");
    model_save(model, final_path);
    printf("Final model saved to %s\n", final_path);

    /* Cleanup */
    train_free(state);
    dataset_free(train_ds);
    dataset_free(val_ds);
    model_free(model);
    tokenizer_free(tok);

    return 0;
}
