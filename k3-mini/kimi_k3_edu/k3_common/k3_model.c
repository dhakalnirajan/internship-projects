/* k3_model.c — Forward pass */
#include "k3_model.h"
#include <string.h>

static void matvec(const float *A, const float *x, float *y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += A[i * N + j] * x[j];
        y[i] = sum;
    }
}

static void add_vec(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

static void softmax_inplace(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void kda_forward(KDAAttention *attn, const float *x_in, float *x_out,
                        const RoPE *rope, int pos, int batch, int seq_len,
                        float *s_state) {
    int d = attn->d_model, h = attn->n_heads, hd = attn->head_dim;
    float qkv_buf[3 * D_MODEL];
    float out_buf[D_MODEL];

    float normed[D_MODEL];
    tensor_rmsnorm(x_in, attn->norm.weight, attn->norm.eps, normed, d);

    matvec(attn->w_q, normed, qkv_buf, d, d);
    matvec(attn->w_k, normed, qkv_buf + d, d, d);
    matvec(attn->w_v, normed, qkv_buf + 2*d, d, d);

    tensor_rope(qkv_buf, qkv_buf + d, rope->sin_cache, rope->cos_cache, 1, hd);

    memset(out_buf, 0, d * sizeof(float));
    for (int head = 0; head < h; head++) {
        float *q = qkv_buf + head * hd;
        float *k = qkv_buf + d + head * hd;
        float *v = qkv_buf + 2*d + head * hd;

        float alpha[HEAD_DIM];
        for (int i = 0; i < hd; i++)
            alpha[i] = k3_sigmoid(attn->w_gate[head * hd + i]);
        float beta = k3_sigmoid(attn->w_beta[head]);

        float *S = s_state + head * hd * hd;
        float Sk[HEAD_DIM];
        for (int i = 0; i < hd; i++) {
            Sk[i] = 0.0f;
            for (int j = 0; j < hd; j++) Sk[i] += S[i * hd + j] * k[j];
        }

        for (int i = 0; i < hd; i++) {
            for (int j = 0; j < hd; j++) {
                S[i * hd + j] = alpha[j] * (S[i * hd + j] - beta * k[i] * Sk[j])
                                + beta * k[i] * v[j];
            }
        }

        float *o = out_buf + head * hd;
        for (int i = 0; i < hd; i++) {
            o[i] = 0.0f;
            for (int j = 0; j < hd; j++) o[i] += S[i * hd + j] * q[j];
        }
    }
    matvec(attn->w_o, out_buf, x_out, d, d);
}

static void mla_forward(MLAAttention *attn, const float *x_in, float *x_out,
                        const RoPE *rope, int pos, int batch, int seq_len,
                        KVCache *kv, int layer_idx) {
    int d = attn->d_model, h = attn->n_heads, hd = attn->head_dim;
    int rank = attn->kv_rank;
    float ckv[512];
    float q_rope[HEAD_DIM];
    float out_buf[D_MODEL];

    float normed[D_MODEL];
    tensor_rmsnorm(x_in, attn->norm.weight, attn->norm.eps, normed, d);

    matvec(attn->w_dkv, normed, ckv, rank, d);

    float q_nope[D_MODEL], k_nope[D_MODEL], v_all[D_MODEL];
    matvec(attn->w_uq, ckv, q_nope, h * hd, rank);
    matvec(attn->w_uk, ckv, k_nope, h * hd, rank);
    matvec(attn->w_uv, ckv, v_all, h * hd, rank);

    matvec(attn->w_q_rope, normed, q_rope, hd, d);
    float k_rope[HEAD_DIM];
    matvec(attn->w_k_rope, normed, k_rope, hd, d);

    float *k_cache = kv->k_cache + ((layer_idx * kv->max_seq_len + pos) * h * hd);
    float *v_cache = kv->v_cache + ((layer_idx * kv->max_seq_len + pos) * h * hd);
    memcpy(k_cache, k_nope, h * hd * sizeof(float));
    memcpy(v_cache, v_all, h * hd * sizeof(float));

    memset(out_buf, 0, d * sizeof(float));
    for (int head = 0; head < h; head++) {
        float *q = q_nope + head * hd;
        float scores[MAX_SEQ_LEN];
        for (int t = 0; t <= pos; t++) {
            float *k_t = kv->k_cache + ((layer_idx * kv->max_seq_len + t) * h * hd) + head * hd;
            float dot = 0.0f;
            for (int i = 0; i < hd; i++) dot += q[i] * k_t[i];
            scores[t] = dot / sqrtf((float)hd);
        }
        softmax_inplace(scores, pos + 1);
        float *o = out_buf + head * hd;
        for (int i = 0; i < hd; i++) o[i] = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float *v_t = kv->v_cache + ((layer_idx * kv->max_seq_len + t) * h * hd) + head * hd;
            for (int i = 0; i < hd; i++) o[i] += scores[t] * v_t[i];
        }
    }
    matvec(attn->w_o, out_buf, x_out, d, d);
}

static void ffn_forward(FFN *ffn, const float *x_in, float *x_out) {
    int d = ffn->d_model, df = ffn->d_ffn;
    float gate[D_FFN], up[D_FFN], down[D_MODEL];

    float normed[D_MODEL];
    tensor_rmsnorm(x_in, ffn->norm.weight, ffn->norm.eps, normed, d);

    matvec(ffn->w_gate, normed, gate, df, d);
    matvec(ffn->w_up,   normed, up,   df, d);
    for (int i = 0; i < df; i++) gate[i] = k3_silu(gate[i]) * up[i];
    matvec(ffn->w_down, gate, down, d, df);
    memcpy(x_out, down, d * sizeof(float));
}

void model_forward(TransformerModel *m, const int *tokens, int n_tokens,
                   float *logits_out, KVCache *kv) {
    int d = m->d_model, V = m->vocab_size;
    float hidden[MAX_SEQ_LEN * D_MODEL];
    float residual[D_MODEL];
    float attn_out[D_MODEL];
    float ffn_out[D_MODEL];
    float s_state[N_HEADS * HEAD_DIM * HEAD_DIM];

    for (int t = 0; t < n_tokens; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= V) tok = 0;
        memcpy(hidden + t * d, m->token_embedding + tok * d, d * sizeof(float));
    }

    for (int layer = 0; layer < m->n_layers; layer++) {
        TransformerLayer *L = &m->layers[layer];
        for (int t = 0; t < n_tokens; t++) {
            float *h = hidden + t * d;
            memcpy(residual, h, d * sizeof(float));
            if (L->type == LAYER_KDA) {
                memset(s_state, 0, sizeof(s_state));
                kda_forward(&L->attn.kda, h, attn_out, &m->rope, t, 1, n_tokens,
                            s_state);
            } else {
                mla_forward(&L->attn.mla, h, attn_out, &m->rope, t, 1, n_tokens,
                            kv, layer);
            }
            add_vec(h, attn_out, d);
            memcpy(residual, h, d * sizeof(float));
            ffn_forward(&L->ffn, h, ffn_out);
            add_vec(h, ffn_out, d);
        }
    }

    float *last = hidden + (n_tokens - 1) * d;
    float normed[D_MODEL];
    tensor_rmsnorm(last, m->final_norm.weight, m->final_norm.eps, normed, d);

    float *W = m->tie_weights ? m->token_embedding : m->output_proj;
    for (int v = 0; v < V; v++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) sum += W[v * d + i] * normed[i];
        logits_out[v] = sum;
    }
}

KVCache *kv_cache_create(int nl, int nh, int ms, int hd) {
    KVCache *c = calloc(1, sizeof(KVCache));
    c->n_layers = nl; c->n_heads = nh; c->max_seq_len = ms; c->head_dim = hd;
    size_t sz = (size_t)nl * ms * nh * hd;
    c->k_cache = calloc(sz, sizeof(float));
    c->v_cache = calloc(sz, sizeof(float));
    c->cache_len = 0;
    return c;
}
void kv_cache_free(KVCache *c)  { if (c) { free(c->k_cache); free(c->v_cache); free(c); } }
void kv_cache_clear(KVCache *c) {
    c->cache_len = 0;
    size_t sz = (size_t)c->n_layers * c->max_seq_len * c->n_heads * c->head_dim * sizeof(float);
    memset(c->k_cache, 0, sz);
    memset(c->v_cache, 0, sz);
}

static float *aw(int n) { return calloc(n, sizeof(float)); }

static void init_rmsnorm(RMSNorm *n, int dim) {
    n->weight = aw(dim);
    n->dim = dim;
    n->eps = 1e-6f;
    for (int i = 0; i < dim; i++) n->weight[i] = 1.0f;
}

static void init_rope(RoPE *r, int max_seq, int head_dim) {
    r->max_seq_len = max_seq; r->head_dim = head_dim;
    r->sin_cache = calloc(max_seq * (head_dim/2), sizeof(float));
    r->cos_cache = calloc(max_seq * (head_dim/2), sizeof(float));
    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < head_dim/2; i++) {
            float theta = powf(10000.0f, -2.0f * i / head_dim);
            float ang = pos * theta;
            r->sin_cache[pos * (head_dim/2) + i] = sinf(ang);
            r->cos_cache[pos * (head_dim/2) + i] = cosf(ang);
        }
    }
}

static void xavier_init(float *w, int fan_in, int fan_out) {
    float scale = sqrtf(2.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++)
        w[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

void model_init_random(TransformerModel *m) {
    srand((unsigned)time(NULL));
    m->d_model = D_MODEL; m->n_layers = N_LAYERS; m->n_heads = N_HEADS;
    m->vocab_size = 50000; m->tie_weights = 1;
    int V = m->vocab_size, d = D_MODEL;
    m->token_embedding = aw(V * d); xavier_init(m->token_embedding, V, d);
    init_rmsnorm(&m->final_norm, d);
    init_rope(&m->rope, MAX_SEQ_LEN, HEAD_DIM);
    if (!m->tie_weights) m->output_proj = aw(d * V);

    for (int l = 0; l < N_LAYERS; l++) {
        TransformerLayer *L = &m->layers[l];
        int is_kda = (l % 4 < 3);
        L->type = is_kda ? LAYER_KDA : LAYER_MLA;
        if (is_kda) {
            KDAAttention *a = &L->attn.kda;
            a->d_model = d; a->n_heads = N_HEADS; a->head_dim = HEAD_DIM;
            a->w_q = aw(d*d); xavier_init(a->w_q, d, d);
            a->w_k = aw(d*d); xavier_init(a->w_k, d, d);
            a->w_v = aw(d*d); xavier_init(a->w_v, d, d);
            a->w_o = aw(d*d); xavier_init(a->w_o, d, d);
            a->w_gate = aw(d * HEAD_DIM);
            a->w_beta  = aw(N_HEADS);
            init_rmsnorm(&a->norm, d);
        } else {
            MLAAttention *a = &L->attn.mla;
            a->d_model = d; a->n_heads = N_HEADS; a->head_dim = HEAD_DIM;
            a->kv_rank = 256; int r = a->kv_rank;
            a->w_dkv = aw(r*d); xavier_init(a->w_dkv, d, r);
            a->w_uq  = aw(N_HEADS*HEAD_DIM*r); xavier_init(a->w_uq, r, N_HEADS*HEAD_DIM);
            a->w_uk  = aw(N_HEADS*HEAD_DIM*r); xavier_init(a->w_uk, r, N_HEADS*HEAD_DIM);
            a->w_uv  = aw(N_HEADS*HEAD_DIM*r); xavier_init(a->w_uv, r, N_HEADS*HEAD_DIM);
            a->w_o   = aw(d*N_HEADS*HEAD_DIM); xavier_init(a->w_o, N_HEADS*HEAD_DIM, d);
            a->w_q_rope = aw(HEAD_DIM*d); xavier_init(a->w_q_rope, d, HEAD_DIM);
            a->w_k_rope = aw(HEAD_DIM*d); xavier_init(a->w_k_rope, d, HEAD_DIM);
            init_rmsnorm(&a->norm, d);
        }
        FFN *f = &L->ffn;
        f->d_model = d; f->d_ffn = D_FFN;
        f->w_gate = aw(d * D_FFN); xavier_init(f->w_gate, d, D_FFN);
        f->w_up   = aw(d * D_FFN); xavier_init(f->w_up, d, D_FFN);
        f->w_down = aw(D_FFN * d); xavier_init(f->w_down, D_FFN, d);
        init_rmsnorm(&f->norm, d);
    }
}

TransformerModel *model_load(const char *path) {
    (void)path;
    TransformerModel *m = calloc(1, sizeof(TransformerModel));
    model_init_random(m);
    return m;
}

void model_free(TransformerModel *m) {
    if (!m) return;
    free(m->token_embedding);
    free(m->output_proj);
    free(m->final_norm.weight);
    free(m->rope.sin_cache); free(m->rope.cos_cache);
    for (int i = 0; i < N_LAYERS; i++) {
        TransformerLayer *L = &m->layers[i];
        if (L->type == LAYER_KDA) {
            KDAAttention *a = &L->attn.kda;
            free(a->w_q); free(a->w_k); free(a->w_v); free(a->w_o);
            free(a->w_gate); free(a->w_beta); free(a->norm.weight);
        } else {
            MLAAttention *a = &L->attn.mla;
            free(a->w_dkv); free(a->w_uq); free(a->w_uk); free(a->w_uv); free(a->w_o);
            free(a->w_q_rope); free(a->w_k_rope); free(a->norm.weight);
        }
        free(L->ffn.w_gate); free(L->ffn.w_up); free(L->ffn.w_down);
        free(L->ffn.norm.weight);
    }
    free(m);
}
