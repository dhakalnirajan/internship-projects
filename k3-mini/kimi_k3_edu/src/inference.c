int greedy_token(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* ============================================================================
 * GENERATION
 * ============================================================================ */

int generate(TransformerModel* model, Tokenizer* tokenizer,
             const char* prompt, const GenerationConfig* config,
             char* out_text, int out_max_len) {

    /* Encode prompt */
    int prompt_len;
    int* prompt_ids = tokenizer_encode(tokenizer, prompt, &prompt_len);
    if (!prompt_ids || prompt_len == 0) {
        free(prompt_ids);
        return -1;
    }

    /* Truncate if too long */
    if (prompt_len > model->max_seq_len) {
        prompt_len = model->max_seq_len;
    }

    /* Allocate token history */
    int max_total = prompt_len + config->max_new_tokens;
    int* tokens = (int*)malloc(max_total * sizeof(int));
    memcpy(tokens, prompt_ids, prompt_len * sizeof(int));
    free(prompt_ids);

    unsigned int rng_state = (unsigned int)time(NULL);
    int generated = 0;

    /* KV cache */
    KVCache* kv_cache = kv_cache_create(1, model->n_heads, model->max_seq_len, 
                                         model->d_model / model->n_heads);

    for (int i = 0; i < config->max_new_tokens; i++) {
        int current_len = prompt_len + i;

        /* Forward pass with current context */
        Tensor* logits = model_forward(model, tokens, 1, current_len, 0);

        /* Get logits for last position */
        float* last_logits = logits->data + (current_len - 1) * model->vocab_size;

        /* Copy for modification */
        float* mod_logits = (float*)malloc(model->vocab_size * sizeof(float));
        memcpy(mod_logits, last_logits, model->vocab_size * sizeof(float));

        /* Apply repetition penalty */
        apply_repetition_penalty(mod_logits, model->vocab_size, 
                                  tokens, current_len, config->repetition_penalty);

        /* Temperature scaling */
        apply_temperature(mod_logits, model->vocab_size, config->temperature);

        /* Top-k filtering */
        top_k_filter(mod_logits, model->vocab_size, config->top_k);

        /* Top-p filtering */
        top_p_filter(mod_logits, model->vocab_size, config->top_p);

        /* Sample or greedy */
        int next_token;
        if (config->do_sample && config->temperature > 0.0f) {
            next_token = sample_token(mod_logits, model->vocab_size, &rng_state);
        } else {
            next_token = greedy_token(mod_logits, model->vocab_size);
        }

        free(mod_logits);
        tensor_free(logits);

        /* Check for EOS */
        if (next_token == TOK_EOS) {
            break;
        }

        /* Check stop sequences */
        for (int s = 0; s < config->n_stop_sequences; s++) {
            /* Simplified: check if last token matches stop token */
            if (next_token == tokenizer_get_id(tokenizer, config->stop_sequences[s])) {
                goto generation_done;
            }
        }

        tokens[current_len] = next_token;
        generated++;

        /* Stream callback */
        if (config->stream_callback) {
            const char* token_text = tokenizer_get_token(tokenizer, next_token);
            config->stream_callback(token_text, config->stream_user_data);
        }
    }

generation_done:
    /* Decode generated tokens */
    char* decoded = tokenizer_decode(tokenizer, tokens + prompt_len, generated);
    int dec_len = strlen(decoded);
    if (dec_len >= out_max_len) dec_len = out_max_len - 1;
    memcpy(out_text, decoded, dec_len);
    out_text[dec_len] = '\0';
    free(decoded);

    free(tokens);
    kv_cache_free(kv_cache);

    return generated;
}

int generate_streaming(TransformerModel* model, Tokenizer* tokenizer,
                       const char* prompt, const GenerationConfig* config) {
    char buffer[65536];
    return generate(model, tokenizer, prompt, config, buffer, sizeof(buffer));
}

/* ============================================================================
 * CHAT / INFERENCE FORMAT
 * ============================================================================ */

char* format_chat_prompt(const ChatMessage* messages, int n_messages) {
    int size = 65536;
    char* result = (char*)malloc(size);
    result[0] = '\0';
    int pos = 0;

    for (int i = 0; i < n_messages; i++) {
        const char* role = messages[i].role;
        const char* content = messages[i].content;

        if (strcmp(role, "system") == 0) {
            pos += snprintf(result + pos, size - pos, 
                "%s%s\n%s%s\n", 
                SPECIAL_IM_START, SPECIAL_SYSTEM,
                content, SPECIAL_IM_END);
        } else if (strcmp(role, "user") == 0) {
            pos += snprintf(result + pos, size - pos,
                "%s%s\n%s%s\n",
                SPECIAL_IM_START, SPECIAL_USER,
                content, SPECIAL_IM_END);
        } else if (strcmp(role, "assistant") == 0) {
            pos += snprintf(result + pos, size - pos,
                "%s%s\n%s%s\n",
                SPECIAL_IM_START, SPECIAL_ASSISTANT,
                content, SPECIAL_IM_END);
        }
    }

    /* Add assistant prefix for generation */
    pos += snprintf(result + pos, size - pos,
        "%s%s\n", SPECIAL_IM_START, SPECIAL_ASSISTANT);

    return result;
}

char* chat_completion(TransformerModel* model, Tokenizer* tokenizer,
                      const ChatMessage* messages, int n_messages,
                      const GenerationConfig* config) {
    char* prompt = format_chat_prompt(messages, n_messages);
    char* response = (char*)malloc(65536);

    generate(model, tokenizer, prompt, config, response, 65536);

    free(prompt);
    return response;
}

/* ============================================================================
 * EVALUATION
 * ============================================================================ */

float evaluate_perplexity(TransformerModel* model, Tokenizer* tokenizer,
                          const char* text) {
    int len;
    int* ids = tokenizer_encode(tokenizer, text, &len);
    if (len < 2) {
        free(ids);
        return 0.0f;
    }

    float total_loss = 0.0f;
    int count = 0;

    /* Process in chunks */
    int chunk_size = model->max_seq_len;
    for (int start = 0; start < len - 1; start += chunk_size) {
        int end = start + chunk_size;
        if (end > len) end = len;
        int current_len = end - start;

        Tensor* logits = model_forward(model, ids + start, 1, current_len, 0);

        for (int s = 0; s < current_len - 1; s++) {
            float* logit = logits->data + s * model->vocab_size;
            int target = ids[start + s + 1];
            total_loss += cross_entropy_loss(logit, target, model->vocab_size);
            count++;
        }

        tensor_free(logits);
    }

    free(ids);

    float avg_loss = total_loss / count;
    return expf(avg_loss);
}

float evaluate_code_accuracy(TransformerModel* model, Tokenizer* tokenizer,
                              const char** prompts, const char** expected, int n_samples) {
    int correct = 0;
    GenerationConfig cfg = DEFAULT_GENERATION_CONFIG;
    cfg.max_new_tokens = 256;
    cfg.temperature = 0.1f; /* Low temp for deterministic code */

    for (int i = 0; i < n_samples; i++) {
        char generated[1024];
        generate(model, tokenizer, prompts[i], &cfg, generated, sizeof(generated));

        /* Simple exact match (in production, use AST comparison) */
        if (strstr(generated, expected[i]) != NULL) {
            correct++;
        }
    }

    return (float)correct / n_samples;
}

float benchmark_inference(TransformerModel* model, Tokenizer* tokenizer,
                          const char* prompt, int n_tokens) {
    GenerationConfig cfg = DEFAULT_GENERATION_CONFIG;
    cfg.max_new_tokens = n_tokens;
    cfg.do_sample = 0; /* Greedy for consistent timing */

    Timer timer;
    timer_start(&timer);

    char output[65536];
    generate(model, tokenizer, prompt, &cfg, output, sizeof(output));

    double elapsed = timer_elapsed(&timer);
    return n_tokens / elapsed;
}

/* ============================================================================
 * KV CACHE
 * ============================================================================ */

KVCache* kv_cache_create(int batch_size, int n_heads, int max_seq_len, int head_dim) {
    KVCache* cache = (KVCache*)calloc(1, sizeof(KVCache));
    cache->batch_size = batch_size;
    cache->n_heads = n_heads;
    cache->max_cache_len = max_seq_len;
    cache->head_dim = head_dim;
    cache->cache_len = 0;

    int k_shape[4] = {batch_size, n_heads, max_seq_len, head_dim};
    cache->k_cache = tensor_create(4, k_shape);
    cache->v_cache = tensor_create(4, k_shape);

    return cache;
}

void kv_cache_free(KVCache* cache) {
    if (!cache) return;
    tensor_free(cache->k_cache);
    tensor_free(cache->v_cache);
    free(cache);
}

void kv_cache_append(KVCache* cache, const Tensor* k, const Tensor* v, int seq_len) {
    if (cache->cache_len + seq_len > cache->max_cache_len) {
        log_warn("KV cache overflow");
        return;
    }

    /* Append new K, V to cache */
    size_t copy_size = cache->batch_size * cache->n_heads * seq_len * cache->head_dim;
    size_t offset = cache->cache_len * cache->head_dim;

    memcpy(cache->k_cache->data + offset, k->data, copy_size * sizeof(float));
    memcpy(cache->v_cache->data + offset, v->data, copy_size * sizeof(float));

    cache->cache_len += seq_len;
}

void kv_cache_clear(KVCache* cache) {
    cache->cache_len = 0;
}

/* ============================================================================
 * PROMPT TEMPLATES
 * ============================================================================ */

char* prompt_code(const char* language, const char* instruction) {
    int size = 4096;
    char* result = (char*)malloc(size);
    snprintf(result, size,
        "Write %s code to %s.\n\n"
        "```%s\n",
        language, instruction, language);
    return result;
}

char* prompt_translate(const char* text, const char* source_lang, const char* target_lang) {
    int size = 4096;
    char* result = (char*)malloc(size);
    snprintf(result, size,
        "Translate the following text from %s to %s:\n\n"
        "%s\n\n"
        "Translation:",
        source_lang, target_lang, text);
    return result;
}

char* prompt_summarize(const char* text) {
    int size = 4096;
    char* result = (char*)malloc(size);
    snprintf(result, size,
        "Summarize the following text:\n\n"
        "%s\n\n"
        "Summary:",
        text);
    return result;
}

char* prompt_qa(const char* context, const char* question) {
    int size = 4096;
    char* result = (char*)malloc(size);
    snprintf(result, size,
        "Context: %s\n\n"
        "Question: %s\n\n"
        "Answer:",
        context, question);
    return result;
}
