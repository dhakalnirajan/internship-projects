/**
 * config.h - Hyperparameters and configuration for the K3-Edu model
 *
 * Inspired by MoonshotAI's Kimi K3 architecture:
 * - Kimi Delta Attention (KDA) for efficient linear attention
 * - Attention Residuals (AttnRes) for selective depth-wise aggregation
 * - MXFP8/INT8 quantization-aware training
 *
 * Model: ~50M parameters dense Transformer (non-MoE for education)
 *
 * References:
 * - Kimi Linear paper (arXiv:2510.26692)
 * - Attention Residuals paper (arXiv:2603.15031)
 * - Kimi K3 Tech Blog (kimi.com/blog/kimi-k3)
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * MODEL ARCHITECTURE PARAMETERS
 * ============================================================================ */

#define MODEL_NAME "K3-Edu-50M"
#define MODEL_VERSION "1.0.0"

/* Embedding dimension */
#define D_MODEL 384

/* Number of transformer layers */
#define N_LAYERS 8

/* Number of attention heads */
#define N_HEADS 6

/* Attention head dimension (D_MODEL / N_HEADS) */
#define HEAD_DIM 64

/* Feed-forward network dimension (4 * D_MODEL) */
#define D_FFN 1536

/* Maximum context window length */
#define MAX_SEQ_LEN 8192

/* Vocabulary size (will be set at runtime from tokenizer) */
#define MAX_VOCAB_SIZE 100000

/* KDA (KIMI DELTA ATTENTION) PARAMETERS */

/* Number of KDA layers per block (KDA:MLA ratio is 3:1 in K3) */
#define KDA_PER_BLOCK 3

/* Number of full attention (MLA-style) layers per block */
#define MLA_PER_BLOCK 1

/* Block size for KDA chunkwise computation */
#define KDA_CHUNK_SIZE 128

/* Channel-wise gating dimension for KDA */
#define KDA_GATE_DIM HEAD_DIM

/* ATTENTION RESIDUALS (AttnRes) PARAMETERS */

/* Number of blocks for Block AttnRes */
#define ATTNRES_N_BLOCKS 4

/* Layers per AttnRes block */
#define ATTNRES_LAYERS_PER_BLOCK (N_LAYERS / ATTNRES_N_BLOCKS)

/* TRAINING HYPERPARAMETERS */

/* Phase 1: Base Model Training */
#define BASE_BATCH_SIZE 4 /* Adjustable 4-8 */
#define BASE_PEAK_LR 3e-4f
#define BASE_WARMUP_STEPS 2000
#define BASE_TOTAL_STEPS 150000
#define BASE_WARMUP_END_STEP 2000
#define BASE_RAPID_END_STEP 50000
#define BASE_DROPOUT 0.1f
#define BASE_WEIGHT_DECAY 0.1f
#define BASE_GRAD_CLIP 1.0f
#define BASE_MAX_EPOCHS 100

/* Phase 2: Instruction Fine-Tuning */
#define INST_BATCH_SIZE 2
#define INST_PEAK_LR 1e-5f
#define INST_MIN_LR 1e-6f
#define INST_TOTAL_STEPS 30000 /* ~20% of base training */
#define INST_WARMUP_STEPS 500
#define INST_DROPOUT 0.05f
#define INST_WEIGHT_DECAY 0.01f

/* OPTIMIZER PARAMETERS (AdamW-style) */

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.95f
#define ADAM_EPS 1e-8f

/* QUANTIZATION PARAMETERS */

/* Use FP8 (E4M3) for activations, INT8 for weights during inference */
#define USE_FP8_ACTIVATIONS 1
#define USE_INT8_WEIGHTS 1

/* FP8 quantization scale */
#define FP8_SCALE 448.0f

/* DATASET PARAMETERS */

#define DATASET_DIR "datasets"
#define MAX_FILE_SIZE_MB 50
#define SHUFFLE_BUFFER_SIZE 10000
#define VALIDATION_SPLIT 0.05f

/* HARDWARE ADAPTATION */

/* Auto-detect and scale */
#define MIN_BATCH_SIZE 1
#define MAX_BATCH_SIZE 32
#define GRAD_ACCUM_TARGET_TOKENS 32768 /* Effective batch target */

/* CHECKPOINTING */

#define CHECKPOINT_EVERY_STEPS 5000
#define KEEP_N_CHECKPOINTS 3

/* INFERENCE PARAMETERS */

#define DEFAULT_TEMPERATURE 0.7f
#define DEFAULT_TOP_K 40
#define DEFAULT_TOP_P 0.9f
#define DEFAULT_MAX_NEW_TOKENS 512
#define KV_CACHE_ENABLED 1

/* SPECIAL TOKENS (Instruction-tuning format) */

#define SPECIAL_IM_START "<|im_start|>"
#define SPECIAL_IM_END "<|im_end|>"
#define SPECIAL_SYSTEM "<|system|>"
#define SPECIAL_USER "<|user|>"
#define SPECIAL_ASSISTANT "<|assistant|>"
#define SPECIAL_PAD "<|pad|>"
#define SPECIAL_UNK "<|unk|>"
#define SPECIAL_BOS "<|bos|>"
#define SPECIAL_EOS "<|eos|>"
#define SPECIAL_MASK "<|mask|>"

/* TOKENIZER PARAMETERS */

#define TOKENIZER_VOCAB_SIZE 50000 /* Initial BPE vocab size */
#define TOKENIZER_MIN_FREQ 2
#define TOKENIZER_MAX_MERGES 48000
#define TOKENIZER_PRETOKENIZER "gpt2" /* GPT-2 style pretokenization */

/* TYPE DEFINITIONS */

typedef float f32;
typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;

/* Quantized weight type */
typedef struct
{
    i8 qval;
    f32 scale;
} QuantizedWeight;

/* FP8 (E4M3) type - stored as uint8, interpreted at runtime */
typedef u8 fp8_t;

/* CONFIGURATION STRUCTURE (runtime-modifiable) */

typedef struct
{
    /* Architecture */
    int d_model;
    int n_layers;
    int n_heads;
    int head_dim;
    int d_ffn;
    int max_seq_len;
    int vocab_size;

    /* Training */
    int batch_size;
    float peak_lr;
    float min_lr;
    int warmup_steps;
    int total_steps;
    float dropout;
    float weight_decay;
    float grad_clip;
    int grad_accum_steps;

    /* Hardware */
    int n_cpu_threads;
    int use_cuda;
    size_t gpu_memory_mb;
    size_t system_ram_mb;

    /* Paths */
    char dataset_dir[256];
    char checkpoint_dir[256];
    char tokenizer_path[256];

    /* Flags */
    int use_fp8;
    int use_int8;
    int use_kda;
    int use_attnres;
} Config;

/* Global configuration instance */
extern Config g_config;

/* Initialize config with defaults */
void config_init(void);

/* Load config from JSON file */
int config_load(const char *path);

/* Save config to JSON file */
int config_save(const char *path);

/* Auto-detect hardware and adjust config */
void config_auto_detect_hardware(void);

/* Print current configuration */
void config_print(void);

#endif /* CONFIG_H */
