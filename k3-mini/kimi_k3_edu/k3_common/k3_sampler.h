/* k3_sampler.h — Logit sampling */
#ifndef K3_SAMPLER_H
#define K3_SAMPLER_H
#include "k3_common.h"

typedef struct {
    float temperature;
    int top_k;
    float top_p;
    int max_new_tokens;
    float repetition_penalty;
    int do_sample;
} SamplerConfig;

void sample_logits(float *logits, int vocab_size, const SamplerConfig *cfg,
                   const int *prev_tokens, int n_prev, K3Rng *rng, int *out_token);

#endif
