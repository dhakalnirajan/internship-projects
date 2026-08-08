/**
 * inference.h - Inference Engine
 * 
 * Supports both CPU (C) and GPU (CUDA) execution.
 * 
 * Features:
 *   - Temperature sampling
 *   - Top-k filtering
 *   - Top-p (nucleus) filtering
 *   - Efficient KV-cache management
 *   - Streaming generation
 *   - Batch inference
 *   - INT8/FP8 precision for weights/activations
 * 
 * Inspired by Kimi K3's inference optimizations:
 *   - KDA with prefill caching (contributed to vLLM)
 *   - Block AttnRes for efficient depth-wise attention
 *   - MXFP4 weights, MXFP8 activations
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "config.h"
#include "model.h"
#include "tokenizer.h"

/* ============================================================================
 * GENERATION CONFIG
 * ============================================================================ */

typedef struct {
    float temperature;      /* Sampling temperature (0 = greedy) */
    int top_k;              /* Top-k filtering (0 = disabled) */
    float top_p;            /* Top-p (nucleus) filtering (1.0 = disabled) */
    int max_new_tokens;     /* Maximum tokens to generate */
    int min_new_tokens;     /* Minimum tokens to generate */
    float repetition_penalty; /* Penalty for repeated tokens (1.0 = none) */
    int do_sample;          /* 0 = greedy, 1 = sampling */

    /* Stop sequences */
    char** stop_sequences;
    int n_stop_sequences;

    /* Streaming callback */
    void (*stream_callback)(const char* token_text, void* user_data);
    void* stream_user_data;
} GenerationConfig;

/* Default generation config */
extern const GenerationConfig DEFAULT_GENERATION_CONFIG;

/* ============================================================================
 * SAMPLING
 * ============================================================================ */

/* Apply temperature scaling to logits */
void apply_temperature(float* logits, int vocab_size, float temperature);

/* Top-k filtering: zero out logits not in top k */
void top_k_filter(float* logits, int vocab_size, int k);

/* Top-p (nucleus) filtering: keep smallest set with cumulative prob >= p */
void top_p_filter(float* logits, int vocab_size, float p);

/* Repetition penalty */
void apply_repetition_penalty(float* logits, int vocab_size, 
                               const int* prev_tokens, int n_prev, float penalty);

/* Sample from logits (with temperature, top-k, top-p applied) */
int sample_token(float* logits, int vocab_size, unsigned int* rng_state);

/* Greedy selection */
int greedy_token(const float* logits, int vocab_size);

/* ============================================================================
 * GENERATION
 * ============================================================================ */

/**
 * Generate text from a prompt.
 * 
 * @param model        The transformer model
 * @param tokenizer    The tokenizer
 * @param prompt       Input prompt text
 * @param config       Generation configuration
 * @param out_text     Output buffer (must be pre-allocated)
 * @param out_max_len  Maximum output buffer length
 * @return             Number of tokens generated, or -1 on error
 */
int generate(TransformerModel* model, Tokenizer* tokenizer,
             const char* prompt, const GenerationConfig* config,
             char* out_text, int out_max_len);

/**
 * Generate with streaming callback.
 * Same as generate() but calls stream_callback for each token.
 */
int generate_streaming(TransformerModel* model, Tokenizer* tokenizer,
                       const char* prompt, const GenerationConfig* config);

/**
 * Batch generation: generate for multiple prompts in parallel.
 */
int generate_batch(TransformerModel* model, Tokenizer* tokenizer,
                   const char** prompts, int n_prompts,
                   const GenerationConfig* config,
                   char** out_texts, int out_max_len);

/* ============================================================================
 * CHAT / INFERENCE FORMAT
 * ============================================================================ */

/**
 * Chat message structure for instruction-tuned inference.
 */
typedef struct {
    char* role;         /* "system", "user", or "assistant" */
    char* content;      /* Message content */
} ChatMessage;

/**
 * Format chat messages into a prompt string with special tokens.
 * 
 * Output format:
 *   <|im_start|>system
 *   You are a helpful assistant.<|im_end|>
 *   <|im_start|>user
 *   Hello!<|im_end|>
 *   <|im_start|>assistant
 */
char* format_chat_prompt(const ChatMessage* messages, int n_messages);

/**
 * Chat completion API (OpenAI-compatible style).
 */
char* chat_completion(TransformerModel* model, Tokenizer* tokenizer,
                      const ChatMessage* messages, int n_messages,
                      const GenerationConfig* config);

/* ============================================================================
 * EVALUATION
 * ============================================================================ */

/**
 * Compute perplexity on a text corpus.
 */
float evaluate_perplexity(TransformerModel* model, Tokenizer* tokenizer,
                          const char* text);

/**
 * Evaluate code generation accuracy.
 * Returns: accuracy score (0.0 - 1.0)
 */
float evaluate_code_accuracy(TransformerModel* model, Tokenizer* tokenizer,
                              const char** prompts, const char** expected, int n_samples);

/**
 * Benchmark inference speed (tokens/sec).
 */
float benchmark_inference(TransformerModel* model, Tokenizer* tokenizer,
                          const char* prompt, int n_tokens);

/* ============================================================================
 * GPU INFERENCE (CUDA)
 * ============================================================================ */

#ifdef USE_CUDA

/* CUDA inference context */
typedef struct {
    void* d_model_weights;      /* Device pointer to model weights */
    void* d_kv_cache;           /* Device KV cache */
    void* d_workspace;          /* Temporary workspace */
    size_t workspace_size;
    cudaStream_t stream;
} CudaInferenceContext;

/* Initialize CUDA inference */
CudaInferenceContext* cuda_inference_init(TransformerModel* model);

/* Free CUDA inference context */
void cuda_inference_free(CudaInferenceContext* ctx);

/* CUDA forward pass */
void cuda_model_forward(CudaInferenceContext* ctx, const int* token_ids,
                        int batch, int seq_len, float* out_logits);

/* CUDA generate single token */
int cuda_generate_token(CudaInferenceContext* ctx, const int* prev_tokens,
                        int n_prev, const GenerationConfig* config);

#endif /* USE_CUDA */

/* ============================================================================
 * QUANTIZED INFERENCE
 * ============================================================================ */

/* Load INT8 quantized model for fast CPU inference */
TransformerModel* model_load_quantized(const char* path, Tokenizer* tokenizer);

/* Quantize model weights to INT8 */
int model_quantize_weights(TransformerModel* model);

/* ============================================================================
 * PROMPT TEMPLATES
 * ============================================================================ */

/* Code generation prompt template */
char* prompt_code(const char* language, const char* instruction);

/* Translation prompt template */
char* prompt_translate(const char* text, const char* source_lang, const char* target_lang);

/* Summarization prompt template */
char* prompt_summarize(const char* text);

/* Question answering prompt template */
char* prompt_qa(const char* context, const char* question);

#endif /* INFERENCE_H */
