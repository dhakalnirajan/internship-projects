/**
 * model.h - Transformer Model Architecture
 * 
 * Inspired by MoonshotAI's Kimi K3:
 * - Kimi Delta Attention (KDA) for efficient linear attention
 * - Attention Residuals (AttnRes) for selective depth-wise aggregation
 * - RMSNorm instead of LayerNorm (standard in modern LLMs)
 * - SwiGLU activation in FFN (Gated Linear Unit with Swish)
 * - RoPE (Rotary Position Embedding) for positional encoding
 * - INT8 weights with FP8 activations for efficient training/inference
 * 
 * Model: ~200M parameters
 *   - Embedding: vocab_size * 768
 *   - 12 layers * (attention + FFN)
 *   - Output projection: 768 * vocab_size
 */

#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "tokenizer.h"
#include <stddef.h>

/* ============================================================================
 * TENSOR STRUCTURE
 * ============================================================================ */

typedef struct {
    f32* data;          /* Raw float data */
    int* shape;         /* Dimensions */
    int ndim;           /* Number of dimensions */
    size_t size;        /* Total elements */
    int owns_data;      /* Whether this tensor owns the data pointer */
} Tensor;

/* Create a tensor with given shape */
Tensor* tensor_create(int ndim, const int* shape);

/* Create tensor from existing data (no copy) */
Tensor* tensor_from_data(f32* data, int ndim, const int* shape);

/* Free tensor (frees data if owns_data) */
void tensor_free(Tensor* t);

/* Get element at indices */
static inline f32 tensor_get(Tensor* t, ...) {
    /* Varargs implementation in .c file */
    return 0.0f;
}

/* Set element at indices */
static inline void tensor_set(Tensor* t, f32 val, ...) {
    /* Varargs implementation in .c file */
}

/* Reshape tensor (no copy) */
Tensor* tensor_reshape(Tensor* t, int ndim, const int* shape);

/* Print tensor info */
void tensor_print(const Tensor* t, const char* name);

/* ============================================================================
 * QUANTIZED TENSOR (INT8 weights)
 * ============================================================================ */

typedef struct {
    i8* qdata;          /* Quantized int8 data */
    f32* scales;        /* Per-channel or per-block scales */
    int* shape;
    int ndim;
    size_t size;
    int block_size;     /* Quantization block size */
} QTensor;

QTensor* qtensor_create(int ndim, const int* shape, int block_size);
void qtensor_free(QTensor* qt);
void qtensor_quantize(QTensor* qt, const Tensor* src);
void qtensor_dequantize(Tensor* dst, const QTensor* qt);

/* ============================================================================
 * RMS NORM
 * ============================================================================ */

typedef struct {
    Tensor* weight;     /* Learnable gain parameter [d_model] */
    float eps;
} RMSNorm;

RMSNorm* rmsnorm_create(int dim);
void rmsnorm_free(RMSNorm* norm);
void rmsnorm_forward(Tensor* out, const Tensor* in, const RMSNorm* norm, int batch, int seq_len);

/* ============================================================================
 * ROTARY POSITION EMBEDDING (RoPE)
 * ============================================================================ */

typedef struct {
    float* sin_cache;   /* Precomputed sin values [max_seq_len][head_dim/2] */
    float* cos_cache;   /* Precomputed cos values */
    int max_seq_len;
    int head_dim;
    float theta;        /* Base frequency (typically 10000.0) */
} RoPE;

RoPE* rope_create(int max_seq_len, int head_dim, float theta);
void rope_free(RoPE* rope);
void rope_apply(Tensor* q, Tensor* k, const RoPE* rope, int seq_len);

/* ============================================================================
 * KIMI DELTA ATTENTION (KDA)
 * ============================================================================ */

/**
 * KDA implements linear attention with channel-wise gating.
 * 
 * Recurrence: S_t = (I - beta_t * k_t * k_t^T) * Diag(alpha_t) * S_{t-1} + beta_t * k_t * v_t^T
 * 
 * Where:
 *   - alpha_t: per-channel forget gate (vector of size head_dim)
 *   - beta_t: learning rate-like update strength (scalar per head)
 *   - S_t: memory state matrix [head_dim x head_dim]
 * 
 * For education, we use a simplified chunkwise parallel implementation.
 */

typedef struct {
    /* Projections: input -> Q, K, V */
    Tensor* w_q;        /* [d_model, d_model] */
    Tensor* w_k;        /* [d_model, d_model] */
    Tensor* w_v;        /* [d_model, d_model] */
    Tensor* w_o;        /* [d_model, d_model] output projection */

    /* Biases (optional, often omitted in modern LLMs) */
    Tensor* b_q;
    Tensor* b_k;
    Tensor* b_v;
    Tensor* b_o;

    /* KDA-specific: channel-wise gating parameters */
    Tensor* w_gate;     /* [d_model, head_dim] for computing alpha_t */
    Tensor* w_beta;     /* [d_model, 1] for computing beta_t */

    /* Pre-norm */
    RMSNorm* norm;

    /* Number of heads */
    int n_heads;
    int head_dim;
    int d_model;
} KDAAttention;

KDAAttention* kda_attention_create(int d_model, int n_heads);
void kda_attention_free(KDAAttention* attn);
void kda_attention_forward(Tensor* out, const Tensor* in, KDAAttention* attn, 
                           const RoPE* rope, int batch, int seq_len, int training);

/* ============================================================================
 * MULTI-HEAD LATENT ATTENTION (MLA) - Full attention for hybrid architecture
 * ============================================================================ */

typedef struct {
    /* Compressed KV projection (MLA style) */
    Tensor* w_dkv;      /* Down-projection for KV [d_model, kv_lora_rank] */
    Tensor* w_uq;       /* Up-projection for Q [d_model, n_heads * qk_rope_head_dim] */
    Tensor* w_uk;       /* Up-projection for K [kv_lora_rank, n_heads * qk_nope_head_dim] */
    Tensor* w_uv;       /* Up-projection for V [kv_lora_rank, n_heads * v_head_dim] */
    Tensor* w_o;        /* Output projection [n_heads * v_head_dim, d_model] */

    /* RoPE-specific projections */
    Tensor* w_q_rope;   /* [d_model, n_heads * qk_rope_head_dim] */
    Tensor* w_k_rope;   /* [d_model, qk_rope_head_dim] */

    /* Dimensions */
    int n_heads;
    int head_dim;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int v_head_dim;
    int kv_lora_rank;
    int d_model;

    RMSNorm* norm;
} MLAAttention;

MLAAttention* mla_attention_create(int d_model, int n_heads);
void mla_attention_free(MLAAttention* attn);
void mla_attention_forward(Tensor* out, const Tensor* in, MLAAttention* attn,
                           const RoPE* rope, int batch, int seq_len, int training);

/* ============================================================================
 * FEED-FORWARD NETWORK (SwiGLU)
 * ============================================================================ */

typedef struct {
    /* SwiGLU: gate_proj(x) * silu(up_proj(x)) -> down_proj */
    Tensor* w_gate;     /* [d_model, d_ffn] */
    Tensor* w_up;       /* [d_model, d_ffn] */
    Tensor* w_down;     /* [d_ffn, d_model] */

    Tensor* b_gate;
    Tensor* b_up;
    Tensor* b_down;

    RMSNorm* norm;
    int d_model;
    int d_ffn;
} FFN;

FFN* ffn_create(int d_model, int d_ffn);
void ffn_free(FFN* ffn);
void ffn_forward(Tensor* out, const Tensor* in, FFN* ffn, int batch, int seq_len);

/* ============================================================================
 * ATTENTION RESIDUALS (AttnRes)
 * ============================================================================ */

/**
 * AttnRes replaces fixed residual accumulation with softmax attention over depth.
 * 
 * For a 12-layer model with 4 blocks (Block AttnRes):
 *   - Each block has 3 layers
 *   - Layers within a block use standard residuals
 *   - Between blocks, each layer attends to all previous block outputs
 * 
 * h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
 * where alpha = softmax(w_l^T * RMSNorm(v_i))
 */

typedef struct {
    /* Pseudo-query vectors: one per layer [n_layers, d_model] */
    Tensor* w_queries;  /* Learnable, initialized to zero */

    /* Block summaries for Block AttnRes */
    Tensor* block_summaries;  /* [n_blocks, d_model] */

    /* Number of blocks and layers */
    int n_blocks;
    int n_layers;
    int d_model;
    int layers_per_block;
} AttnRes;

AttnRes* attnres_create(int n_layers, int d_model, int n_blocks);
void attnres_free(AttnRes* ar);
void attnres_forward(Tensor* out, Tensor** layer_outputs, int layer_idx, 
                     AttnRes* ar, int batch, int seq_len);

/* ============================================================================
 * TRANSFORMER LAYER
 * ============================================================================ */

typedef enum {
    LAYER_KDA,
    LAYER_MLA
} LayerType;

typedef struct {
    LayerType type;
    union {
        KDAAttention* kda;
        MLAAttention* mla;
    } attn;
    FFN* ffn;
    int layer_idx;
} TransformerLayer;

TransformerLayer* transformer_layer_create(LayerType type, int d_model, int n_heads, int layer_idx);
void transformer_layer_free(TransformerLayer* layer);
void transformer_layer_forward(Tensor* out, const Tensor* in, TransformerLayer* layer,
                               const RoPE* rope, int batch, int seq_len, int training);

/* ============================================================================
 * FULL MODEL
 * ============================================================================ */

typedef struct {
    /* Token embedding */
    Tensor* token_embedding;    /* [vocab_size, d_model] */

    /* Transformer layers */
    TransformerLayer** layers;
    int n_layers;

    /* Final RMSNorm */
    RMSNorm* final_norm;

    /* Output projection (shared with embedding or separate) */
    Tensor* output_proj;        /* [d_model, vocab_size] */
    int tie_weights;            /* Whether output_proj shares with token_embedding */

    /* RoPE */
    RoPE* rope;

    /* AttnRes */
    AttnRes* attnres;

    /* Tokenizer */
    Tokenizer* tokenizer;

    /* Dimensions */
    int d_model;
    int n_heads;
    int vocab_size;
    int max_seq_len;

    /* Dropout rate (0 for inference) */
    float dropout_rate;

    /* Parameter count */
    size_t n_params;
    size_t n_active_params;  /* For MoE this would differ; dense = same */
} TransformerModel;

/* Create model with given configuration */
TransformerModel* model_create(const Config* cfg, Tokenizer* tokenizer);

/* Free model and all components */
void model_free(TransformerModel* model);

/* Forward pass: returns logits [batch, seq_len, vocab_size] */
Tensor* model_forward(TransformerModel* model, const int* token_ids, 
                      int batch, int seq_len, int training);

/* Compute loss for a batch (cross-entropy) */
float model_compute_loss(TransformerModel* model, const int* token_ids, 
                         const int* target_ids, int batch, int seq_len);

/* Count parameters */
size_t model_count_params(const TransformerModel* model);

/* Save model weights */
int model_save(TransformerModel* model, const char* path);

/* Load model weights */
TransformerModel* model_load(const char* path, Tokenizer* tokenizer);

/* Initialize weights (Xavier/He init) */
void model_init_weights(TransformerModel* model);

/* Apply dropout mask */
void model_apply_dropout(Tensor* x, float rate, unsigned int* seed);

/* ============================================================================
 * KV CACHE (for efficient inference)
 * ============================================================================ */

typedef struct {
    Tensor* k_cache;    /* [batch, n_heads, max_seq_len, head_dim] */
    Tensor* v_cache;    /* [batch, n_heads, max_seq_len, head_dim] */
    int cache_len;      /* Current cached sequence length */
    int max_cache_len;
    int batch_size;
    int n_heads;
    int head_dim;
} KVCache;

KVCache* kv_cache_create(int batch_size, int n_heads, int max_seq_len, int head_dim);
void kv_cache_free(KVCache* cache);
void kv_cache_append(KVCache* cache, const Tensor* k, const Tensor* v, int seq_len);
void kv_cache_clear(KVCache* cache);

/* ============================================================================
 * INFERENCE STATE
 * ============================================================================ */

typedef struct {
    TransformerModel* model;
    KVCache* kv_cache;
    int current_len;
    int max_new_tokens;
    float temperature;
    int top_k;
    float top_p;
} InferenceState;

InferenceState* inference_state_create(TransformerModel* model);
void inference_state_free(InferenceState* state);
void inference_state_reset(InferenceState* state);

#endif /* MODEL_H */
