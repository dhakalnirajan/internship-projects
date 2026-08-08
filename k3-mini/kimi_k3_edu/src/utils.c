/**
 * utils.c - Common utilities implementation
 */

#include "utils.h"
#include <stdarg.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static int g_log_level = LOG_LEVEL_INFO;

/* ============================================================================
 * MEMORY ALLOCATION
 * ============================================================================ */

void* aligned_alloc(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void aligned_free(void* ptr) {
    free(ptr);
}

void* safe_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        log_error("Failed to allocate %zu bytes", size);
        exit(1);
    }
    return new_ptr;
}

/* ============================================================================
 * RANDOM NUMBER GENERATION
 * ============================================================================ */

void xorshift_init(XorShift64* rng, uint64_t seed) {
    rng->state = seed ? seed : 0x853c49e6748fea9bULL;
}

uint64_t xorshift_next(XorShift64* rng) {
    uint64_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng->state = x;
    return x;
}

float xorshift_float(XorShift64* rng) {
    /* Generate float in [0, 1) using 23 bits of precision */
    return (xorshift_next(rng) >> 11) * (1.0f / (1ULL << 53));
}

float xorshift_normal(XorShift64* rng) {
    /* Box-Muller transform */
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u1 = xorshift_float(rng);
    float u2 = xorshift_float(rng);

    /* Avoid log(0) */
    while (u1 <= 1e-7f) u1 = xorshift_float(rng);

    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * 3.14159265358979323846f * u2);
    float z1 = mag * sinf(2.0f * 3.14159265358979323846f * u2);

    spare = z1;
    has_spare = 1;
    return z0;
}

/* ============================================================================
 * MATH UTILITIES
 * ============================================================================ */

void softmax(float* x, int n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void log_softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }

    float log_sum = logf(sum);
    for (int i = 0; i < n; i++) {
        x[i] = x[i] - max_val - log_sum;
    }
}

float cross_entropy_loss(const float* logits, int target, int vocab_size) {
    /* logits are already log-softmaxed or we compute here */
    /* For numerical stability, find max */
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }

    float log_sum_exp = logf(sum_exp) + max_logit;
    return log_sum_exp - logits[target];
}

void layer_norm(float* out, const float* in, const float* gamma, const float* beta,
                int n, float eps) {
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = in[i] - mean;
        var += diff * diff;
    }
    var /= n;

    float inv_std = 1.0f / sqrtf(var + eps);

    /* Normalize and scale/shift */
    for (int i = 0; i < n; i++) {
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

void rms_norm(float* out, const float* in, const float* weight, int n, float eps) {
    /* RMSNorm: no mean subtraction, only RMS */
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += in[i] * in[i];
    }
    ss /= n;
    float inv_rms = 1.0f / sqrtf(ss + eps);

    for (int i = 0; i < n; i++) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

/* ============================================================================
 * MATRIX OPERATIONS
 * ============================================================================ */

void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    /* C[M,N] = A[M,K] * B[K,N] */
    /* Simple reference implementation - can be optimized with BLAS */

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_bt(float* C, const float* A, const float* B, int M, int N, int K) {
    /* C[M,N] = A[M,K] * B^T[N,K] = A[M,K] * B[K,N] but B is transposed */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void matvec(float* y, const float* A, const float* x, int M, int N) {
    /* y[M] = A[M,N] * x[N] */
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ============================================================================
 * QUANTIZATION
 * ============================================================================ */

void quantize_int8(i8* out, const float* in, int n, float* scale) {
    /* Find max abs value for scale */
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(in[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    *scale = max_abs / 127.0f;
    if (*scale < 1e-8f) *scale = 1e-8f;

    float inv_scale = 1.0f / *scale;
    for (int i = 0; i < n; i++) {
        int val = (int)roundf(in[i] * inv_scale);
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        out[i] = (i8)val;
    }
}

void dequantize_int8(float* out, const i8* in, int n, float scale) {
    for (int i = 0; i < n; i++) {
        out[i] = (float)in[i] * scale;
    }
}

/* ============================================================================
 * FILE UTILITIES
 * ============================================================================ */

char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        log_error("Cannot open file: %s", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buf = (char*)malloc(size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(buf, 1, size, f);
    buf[read] = '\0';
    fclose(f);

    if (out_len) *out_len = read;
    return buf;
}

char** list_dataset_files(const char* dir, int* out_count) {
    DIR* d = opendir(dir);
    if (!d) {
        log_error("Cannot open directory: %s", dir);
        *out_count = 0;
        return NULL;
    }

    /* First pass: count files */
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        int len = strlen(entry->d_name);
        if ((len > 4 && strcmp(entry->d_name + len - 4, ".txt") == 0) ||
            (len > 3 && strcmp(entry->d_name + len - 3, ".md") == 0) ||
            (len > 9 && strcmp(entry->d_name + len - 9, ".markdown") == 0)) {
            count++;
        }
    }
    rewinddir(d);

    /* Second pass: collect paths */
    char** files = (char**)malloc(count * sizeof(char*));
    int idx = 0;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        int len = strlen(entry->d_name);
        if ((len > 4 && strcmp(entry->d_name + len - 4, ".txt") == 0) ||
            (len > 3 && strcmp(entry->d_name + len - 3, ".md") == 0) ||
            (len > 9 && strcmp(entry->d_name + len - 9, ".markdown") == 0)) {
            files[idx] = (char*)malloc(strlen(dir) + strlen(entry->d_name) + 2);
            sprintf(files[idx], "%s/%s", dir, entry->d_name);
            idx++;
        }
    }
    closedir(d);

    *out_count = count;
    return files;
}

int is_text_file(const char* path) {
    int len = strlen(path);
    return (len > 4 && strcmp(path + len - 4, ".txt") == 0) ||
           (len > 3 && strcmp(path + len - 3, ".md") == 0) ||
           (len > 9 && strcmp(path + len - 9, ".markdown") == 0);
}

/* ============================================================================
 * UTF-8 UTILITIES
 * ============================================================================ */

int utf8_strlen(const char* s) {
    int count = 0;
    while (*s) {
        if ((*s & 0xC0) != 0x80) count++;
        s++;
    }
    return count;
}

const char* utf8_next(const char* s) {
    if (!*s) return s;
    int len = 1;
    unsigned char c = (unsigned char)*s;
    if (c >= 0xF0) len = 4;
    else if (c >= 0xE0) len = 3;
    else if (c >= 0xC0) len = 2;
    return s + len;
}

/* ============================================================================
 * TIMING
 * ============================================================================ */

void timer_start(Timer* t) {
    t->start = clock();
    t->elapsed = 0.0;
}

double timer_elapsed(Timer* t) {
    clock_t now = clock();
    t->elapsed = (double)(now - t->start) / CLOCKS_PER_SEC;
    return t->elapsed;
}

/* ============================================================================
 * LOGGING
 * ============================================================================ */

void log_set_level(int level) {
    g_log_level = level;
}

void log_debug(const char* fmt, ...) {
    if (g_log_level > LOG_LEVEL_DEBUG) return;
    va_list args;
    va_start(args, fmt);
    printf("[DEBUG] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void log_info(const char* fmt, ...) {
    if (g_log_level > LOG_LEVEL_INFO) return;
    va_list args;
    va_start(args, fmt);
    printf("[INFO] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void log_warn(const char* fmt, ...) {
    if (g_log_level > LOG_LEVEL_WARN) return;
    va_list args;
    va_start(args, fmt);
    printf("[WARN] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void log_error(const char* fmt, ...) {
    if (g_log_level > LOG_LEVEL_ERROR) return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

/* ============================================================================
 * JSON PARSING (minimal)
 * ============================================================================ */

float json_get_float(const char* json, const char* key, float default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return strtof(p, NULL);
}

int json_get_int(const char* json, const char* key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

char* json_get_string(const char* json, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return NULL;
    p++;
    const char* end = strchr(p, '"');
    if (!end) return NULL;
    int len = end - p;
    char* result = (char*)malloc(len + 1);
    strncpy(result, p, len);
    result[len] = '\0';
    return result;
}

/* ============================================================================
 * HARDWARE DETECTION
 * ============================================================================ */

int get_cpu_cores(void) {
    #ifdef _OPENMP
    return omp_get_num_procs();
    #else
    #ifdef __linux__
    return sysconf(_SC_NPROCESSORS_ONLN);
    #else
    return 4; /* Default fallback */
    #endif
    #endif
}

size_t get_system_ram_mb(void) {
    #ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram / (1024 * 1024);
    }
    #endif
    return 8192; /* Default 8GB */
}

size_t get_gpu_memory_mb(void) {
    #ifdef USE_CUDA
    /* CUDA implementation in cuda/ directory */
    extern size_t cuda_get_memory_mb(void);
    return cuda_get_memory_mb();
    #else
    return 0;
    #endif
}

int get_gpu_count(void) {
    #ifdef USE_CUDA
    extern int cuda_get_device_count(void);
    return cuda_get_device_count();
    #else
    return 0;
    #endif
}

void set_num_threads(int n) {
    #ifdef _OPENMP
    omp_set_num_threads(n);
    #endif
}
