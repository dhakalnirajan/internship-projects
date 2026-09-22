/* k3_sampler.c */
#include "k3_sampler.h"
#include <string.h>

typedef struct { float val; int idx; } SortPair;

static int cmp_desc(const void *a, const void *b) {
    float d = ((SortPair*)b)->val - ((SortPair*)a)->val;
    return (d > 0) - (d < 0);
}

void sample_logits(float *logits, int vocab_size, const SamplerConfig *cfg,
                   const int *prev_tokens, int n_prev, K3Rng *rng, int *out_token) {
    /* Temperature */
    if (cfg->temperature > 0.0f && cfg->temperature != 1.0f) {
        float inv_t = 1.0f / cfg->temperature;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_t;
    }

    /* Repetition penalty */
    if (cfg->repetition_penalty != 1.0f && n_prev > 0) {
        for (int i = 0; i < n_prev; i++) {
            int tok = prev_tokens[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits[tok] > 0) logits[tok] /= cfg->repetition_penalty;
                else logits[tok] *= cfg->repetition_penalty;
            }
        }
    }

    /* Top-k */
    int k = cfg->top_k > 0 ? k3_min(cfg->top_k, vocab_size) : vocab_size;

    /* Build sorted index */
    static SortPair pairs[MAX_VOCAB];
    for (int i = 0; i < vocab_size; i++) { pairs[i].val = logits[i]; pairs[i].idx = i; }
    qsort(pairs, vocab_size, sizeof(SortPair), cmp_desc);

    /* Top-p (nucleus) */
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += expf(pairs[i].val);
    float cum = 0.0f;
    int cutoff = k;
    for (int i = 0; i < k; i++) {
        cum += expf(pairs[i].val) / sum;
        if (cum >= cfg->top_p) { cutoff = i + 1; break; }
    }

    /* Mask out everything beyond cutoff */
    float max_val = pairs[0].val;
    float local_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        pairs[i].val = expf(pairs[i].val - max_val);
        local_sum += pairs[i].val;
    }
    for (int i = 0; i < cutoff; i++) pairs[i].val /= local_sum;

    /* Sample */
    if (!cfg->do_sample || cfg->temperature <= 0.0f) {
        *out_token = pairs[0].idx;
        return;
    }
    float r = k3_rng_float(rng);
    float acc = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        acc += pairs[i].val;
        if (r <= acc) { *out_token = pairs[i].idx; return; }
    }
    *out_token = pairs[cutoff - 1].idx;
}
