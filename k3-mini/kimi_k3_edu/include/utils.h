/**
 * utils.h - Common utilities and math operations
 */

#ifndef UTILS_H
#define UTILS_H

#include "config.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * MEMORY ALLOCATION
 * ============================================================================ */

/* Aligned allocation for SIMD */
void* aligned_alloc(size_t alignment, size_t size);
void aligned_free(void* ptr);

/* Safe realloc */
void* safe_realloc(void* ptr, size_t size);

/* ============================================================================
 * RANDOM NUMBER GENERATION
 * ============================================================================ */

/* XORShift RNG (fast, good quality) */
typedef struct {
    uint64_t state;
} XorShift64;

void xorshift_init(XorShift64* rng, uint64_t seed);
uint64_t xorshift_next(XorShift64* rng);
float xorshift_float(XorShift64* rng);  /* [0, 1) */
float xorshift_normal(XorShift64* rng); /* Standard normal */

/* Initialize RNG with time-based seed */
static inline void xorshift_init_time(XorShift64* rng) {
    xorshift_init(rng, (uint64_t)time(NULL) ^ (uint64_t)clock());
}

/* ============================================================================
 * MATH UTILITIES
 * ============================================================================ */

/* Fast approximations */
static inline float fast_exp(float x) {
    /* Approximate exp using Taylor or lookup table */
    return expf(x);
}

static inline float fast_tanh(float x) {
    return tanhf(x);
}

/* GELU activation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* SiLU (Swish) activation */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ReLU activation */
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* Softmax: in-place normalization */
void softmax(float* x, int n);

/* Log-softmax */
void log_softmax(float* x, int n);

/* Cross-entropy loss */
float cross_entropy_loss(const float* logits, int target, int vocab_size);

/* Layer normalization (mean + variance) */
void layer_norm(float* out, const float* in, const float* gamma, const float* beta,
                int n, float eps);

/* RMS normalization */
void rms_norm(float* out, const float* in, const float* weight, int n, float eps);

/* Matrix multiplication: C = A * B + C (GEMM) */
void matmul(float* C, const float* A, const float* B, int M, int N, int K);

/* Transposed matrix multiplication: C = A * B^T */
void matmul_bt(float* C, const float* A, const float* B, int M, int N, int K);

/* Matrix-vector multiplication: y = A * x */
void matvec(float* y, const float* A, const float* x, int M, int N);

/* ============================================================================
 * FP8 / INT8 QUANTIZATION UTILITIES
 * ============================================================================ */

/* Convert float to FP8 (E4M3 format) */
static inline fp8_t float_to_fp8(float x) {
    /* E4M3: 1 sign, 4 exponent, 3 mantissa bits */
    /* Max value ~448, min normal ~0.00195 */
    float abs_x = fabsf(x);
    if (abs_x > 448.0f) abs_x = 448.0f;
    if (abs_x < 0.00195f && abs_x > 0.0f) abs_x = 0.00195f;

    /* Simplified: scale to FP8 range */
    float scale = 448.0f;
    int sign = x < 0 ? 1 : 0;
    int val = (int)(abs_x / scale * 127.0f);
    if (val > 127) val = 127;
    return (fp8_t)((sign << 7) | val);
}

/* Convert FP8 to float */
static inline float fp8_to_float(fp8_t x) {
    int sign = (x >> 7) & 1;
    int val = x & 0x7F;
    float scale = 448.0f / 127.0f;
    float result = val * scale;
    return sign ? -result : result;
}

/* Quantize float array to INT8 */
void quantize_int8(i8* out, const float* in, int n, float* scale);

/* Dequantize INT8 to float */
void dequantize_int8(float* out, const i8* in, int n, float scale);

/* ============================================================================
 * STRING UTILITIES
 * ============================================================================ */

/* Read entire file into string */
char* read_file(const char* path, size_t* out_len);

/* Read directory contents (txt and markdown files) */
char** list_dataset_files(const char* dir, int* out_count);

/* Check if file has .txt or .md extension */
int is_text_file(const char* path);

/* UTF-8 string length in characters */
int utf8_strlen(const char* s);

/* UTF-8 next character */
const char* utf8_next(const char* s);

/* ============================================================================
 * TIMING
 * ============================================================================ */

typedef struct {
    clock_t start;
    double elapsed;
} Timer;

void timer_start(Timer* t);
double timer_elapsed(Timer* t);

/* ============================================================================
 * LOGGING
 * ============================================================================ */

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO  1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_ERROR 3

void log_set_level(int level);
void log_debug(const char* fmt, ...);
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

/* ============================================================================
 * JSON PARSING (minimal)
 * ============================================================================ */

/* Simple JSON value extraction */
float json_get_float(const char* json, const char* key, float default_val);
int json_get_int(const char* json, const char* key, int default_val);
char* json_get_string(const char* json, const char* key);

/* ============================================================================
 * HARDWARE DETECTION
 * ============================================================================ */

/* Get number of CPU cores */
int get_cpu_cores(void);

/* Get system RAM in MB */
size_t get_system_ram_mb(void);

/* Get GPU memory in MB (0 if no GPU) */
size_t get_gpu_memory_mb(void);

/* Get number of GPUs */
int get_gpu_count(void);

/* Set number of OpenMP/BLAS threads */
void set_num_threads(int n);

#endif /* UTILS_H */
