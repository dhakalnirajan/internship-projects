/* k3_model.h — Transformer architecture */
#ifndef K3_MODEL_H
#define K3_MODEL_H
#include "k3_tensor.h"

#define D_MODEL     768
#define N_LAYERS    12
#define N_HEADS     12
#define HEAD_DIM    64
#define D_FFN       3072
#define MAX_SEQ_LEN 8192
#define MAX_VOCAB   100000

typedef struct {
    float *weight;
    int    dim;
    float  eps;
} RMSNorm;

typedef struct {
    float *sin_cache;
    float *cos_cache;
    int    max_seq_len;
    int    head_dim;
} RoPE;

typedef struct {
    float *w_q, *w_k, *w_v, *w_o;
    float *w_gate, *w_beta;
    RMSNorm norm;
    int n_heads, head_dim, d_model;
} KDAAttention;

typedef struct {
    float *w_dkv, *w_uq, *w_uk, *w_uv, *w_o;
    float *w_q_rope, *w_k_rope;
    int n_heads, head_dim, kv_rank, d_model;
    RMSNorm norm;
} MLAAttention;

typedef struct {
    float *w_gate, *w_up, *w_down;
    RMSNorm norm;
    int d_model, d_ffn;
} FFN;

typedef enum { LAYER_KDA, LAYER_MLA } LayerType;

typedef struct {
    LayerType type;
    union {
        KDAAttention kda;
        MLAAttention mla;
    } attn;
    FFN ffn;
} TransformerLayer;

typedef struct {
    float *k_cache;
    float *v_cache;
    int cache_len;
    int max_seq_len;
    int n_layers, n_heads, head_dim;
} KVCache;

typedef struct {
    float *token_embedding;
    TransformerLayer layers[N_LAYERS];
    float *output_proj;
    RMSNorm final_norm;
    RoPE rope;
    int vocab_size, d_model, n_layers, n_heads;
    int tie_weights;
} TransformerModel;

TransformerModel *model_load(const char *path);
void model_free(TransformerModel *m);
void model_forward(TransformerModel *m, const int *tokens, int n_tokens,
                   float *logits_out, KVCache *kv);
KVCache *kv_cache_create(int n_layers, int n_heads, int max_seq, int head_dim);
void kv_cache_free(KVCache *c);
void kv_cache_clear(KVCache *c);
void model_init_random(TransformerModel *m);

#endif
