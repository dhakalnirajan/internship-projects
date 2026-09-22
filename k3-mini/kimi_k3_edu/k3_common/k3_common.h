/* k3_common.h — Shared types, arena, and utilities */
#ifndef K3_COMMON_H
#define K3_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

/* ---------- Arena Allocator ---------- */
typedef struct {
    char *base;
    size_t used;
    size_t cap;
} Arena;

Arena *arena_new(size_t cap);
void  *arena_alloc(Arena *a, size_t size, size_t align);
void   arena_reset(Arena *a);
void   arena_free(Arena *a);

/* ---------- Errors ---------- */
#define K3_OK  0
#define K3_ERR -1
#define k3_panic(msg) do {                                              \
    fprintf(stderr, "[k3] panic at %s:%d: %s\n", __FILE__, __LINE__, msg); \
    exit(1);                                                            \
} while (0)

/* ---------- Math ---------- */
static inline float k3_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float k3_silu(float x)    { return x / (1.0f + expf(-x)); }
static inline float k3_rsqrt(float x)   { return 1.0f / sqrtf(x); }
static inline float k3_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static inline int   k3_max(int a, int b) { return a > b ? a : b; }
static inline int   k3_min(int a, int b) { return a < b ? a : b; }
static inline int   k3_clamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ---------- RNG ---------- */
typedef struct { uint64_t state; } K3Rng;
void   k3_rng_init(K3Rng *rng, uint64_t seed);
uint32_t k3_rng_u32(K3Rng *rng);
float  k3_rng_float(K3Rng *rng);

/* ---------- File ---------- */
long   k3_file_size(const char *path);
void  *k3_read_file(const char *path, size_t *out_len);

#endif
