/**
 * train.h - Training Infrastructure
 * 
 * Three-phase training approach:
 *   Phase 1: Base Model Training (pretraining on diverse text + code)
 *   Phase 2: Instruction Fine-Tuning (chat format with special tokens)
 * 
 * Optimizations implemented:
 *   - Gradient accumulation for effective larger batches
 *   - Weight decay (0.1) excluding biases and LayerNorm
 *   - Per-epoch shuffling with deterministic seeds
 *   - Validation loss tracking for early stopping
 *   - Gradient clipping at 1.0
 *   - Dynamic batch sizing based on hardware
 *   - Cosine decay with warm restarts (not monotonic)
 *   - Loss plateau detection for early stopping
 *   - FP8/INT8 quantization-aware training
 */

#ifndef TRAIN_H
#define TRAIN_H

#include "config.h"
#include "model.h"
#include "tokenizer.h"

/* ============================================================================
 * OPTIMIZER STATE (AdamW)
 * ============================================================================ */

typedef struct {
    /* First moment (momentum) */
    Tensor* m;
    /* Second moment (RMSprop) */
    Tensor* v;
    /* Whether this parameter gets weight decay */
    int use_weight_decay;
    /* Step counter for bias correction */
    int step;
} AdamWParam;

typedef struct {
    AdamWParam** params;
    int n_params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step;
} AdamWOptimizer;

/* ============================================================================
 * LEARNING RATE SCHEDULER (Cosine with Warm Restarts)
 * ============================================================================ */

typedef struct {
    float peak_lr;
    float min_lr;
    int warmup_steps;
    int current_step;
    int total_steps;

    /* Warm restart parameters */
    int restart_period;      /* Steps between restarts */
    int restart_multiplier;  /* Multiply period after each restart */
    int n_restarts;

    /* Phase boundaries */
    int phase1_end;          /* End of warmup + rapid learning */
    int phase2_end;          /* End of fine-tuning */
} LRScheduler;

/* ============================================================================
 * TRAINING STATE
 * ============================================================================ */

typedef struct {
    /* Model */
    TransformerModel* model;

    /* Optimizer */
    AdamWOptimizer* optimizer;

    /* LR Scheduler */
    LRScheduler* scheduler;

    /* Tokenizer */
    Tokenizer* tokenizer;

    /* Training metrics */
    float train_loss;
    float val_loss;
    float best_val_loss;
    int best_step;

    /* Gradient accumulation */
    int grad_accum_steps;
    int current_accum_step;

    /* Early stopping */
    int plateau_count;
    int max_plateau;
    float plateau_threshold;

    /* Checkpointing */
    int last_checkpoint_step;

    /* Stats */
    int global_step;
    int epoch;
    float tokens_processed;
    float samples_processed;
    double start_time;

    /* Logging */
    FILE* log_file;
    int log_every_steps;
} TrainState;

/* ============================================================================
 * DATASET
 * ============================================================================ */

typedef struct {
    char** file_paths;      /* Array of file paths */
    int n_files;
    int current_file;

    /* Tokenized data buffer */
    int* token_buffer;
    size_t buffer_size;
    size_t buffer_pos;

    /* Shuffling */
    int* shuffle_indices;
    int shuffle_seed;
    int epoch;

    /* Configuration */
    int seq_len;
    int batch_size;
    char* dataset_dir;
} Dataset;

/* ============================================================================
 * TRAINING API
 * ============================================================================ */

/* Initialize training state */
TrainState* train_init(TransformerModel* model, Tokenizer* tokenizer, const Config* cfg);

/* Free training state */
void train_free(TrainState* state);

/* Phase 1: Base model training */
int train_base(TrainState* state, Dataset* dataset, const Config* cfg);

/* Phase 2: Instruction fine-tuning */
int train_instruction(TrainState* state, Dataset* dataset, const Config* cfg);

/* Single training step */
float train_step(TrainState* state, const int* input_ids, const int* target_ids,
                 int batch_size, int seq_len);

/* Compute gradients (backward pass) */
void train_backward(TrainState* state, Tensor* logits, const int* target_ids,
                    int batch_size, int seq_len);

/* Update weights with gradient accumulation */
void train_update_weights(TrainState* state);

/* Zero gradients */
void train_zero_grad(TrainState* state);

/* Evaluate on validation set */
float train_evaluate(TrainState* state, Dataset* val_dataset);

/* Save checkpoint */
int train_save_checkpoint(TrainState* state, const char* path);

/* Load checkpoint */
TrainState* train_load_checkpoint(const char* path, TransformerModel* model, 
                                   Tokenizer* tokenizer);

/* ============================================================================
 * LEARNING RATE SCHEDULER API
 * ============================================================================ */

LRScheduler* lr_scheduler_create(float peak_lr, float min_lr, int warmup_steps, 
                                  int total_steps);
void lr_scheduler_free(LRScheduler* sched);
float lr_scheduler_step(LRScheduler* sched);
void lr_scheduler_warm_restart(LRScheduler* sched);

/* ============================================================================
 * DATASET API
 * ============================================================================ */

Dataset* dataset_create(const char* dir, int seq_len, int batch_size);
void dataset_free(Dataset* ds);

/* Load next batch (shuffled per epoch) */
int dataset_next_batch(Dataset* ds, int* input_ids, int* target_ids, 
                        int batch_size, int seq_len);

/* Shuffle dataset for new epoch */
void dataset_shuffle(Dataset* ds, int seed);

/* Reset dataset to beginning */
void dataset_reset(Dataset* ds);

/* Count total tokens in dataset */
size_t dataset_count_tokens(Dataset* ds, Tokenizer* tokenizer);

/* Split dataset into train/val */
void dataset_split(Dataset* ds, float val_ratio, Dataset** train_ds, Dataset** val_ds);

/* ============================================================================
 * UTILITIES
 * ============================================================================ */

/* Gradient clipping */
float clip_gradients(Tensor** grads, int n_grads, float max_norm);

/* Apply weight decay (excluding biases and norms) */
void apply_weight_decay(Tensor* weights, float decay, float lr);

/* Detect loss plateau */
int detect_plateau(float* loss_history, int window, float threshold);

/* Compute perplexity from loss */
static inline float loss_to_perplexity(float loss) {
    return expf(loss);
}

/* Logging */
void train_log(TrainState* state, const char* format, ...);

/* ============================================================================
 * MEMORY MANAGEMENT
 * ============================================================================ */

/* Estimate memory requirements */
size_t estimate_training_memory(const Config* cfg);

/* Adjust batch size based on available memory */
int adjust_batch_size(size_t available_mb, const Config* cfg);

#endif /* TRAIN_H */
