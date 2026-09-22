/* k3_common.c */
#include "k3_common.h"

Arena *arena_new(size_t cap) {
    Arena *a = calloc(1, sizeof(Arena));
    a->base = calloc(1, cap);
    a->cap = cap;
    return a;
}

void *arena_alloc(Arena *a, size_t size, size_t align) {
    size_t off = (size_t)a->base + a->used;
    size_t pad = (align - (off & (align - 1))) & (align - 1);
    if (a->used + pad + size > a->cap) k3_panic("arena oom");
    void *p = a->base + a->used + pad;
    a->used += pad + size;
    memset(p, 0, size);
    return p;
}

void arena_reset(Arena *a) { a->used = 0; }
void arena_free(Arena *a)  { free(a->base); free(a); }

void k3_rng_init(K3Rng *rng, uint64_t seed) {
    rng->state = seed ? seed : 0x853c49e6748fea9bULL;
}

uint32_t k3_rng_u32(K3Rng *rng) {
    uint64_t x = rng->state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng->state = x;
    return (uint32_t)(x >> 32);
}

float k3_rng_float(K3Rng *rng) {
    return (k3_rng_u32(rng) & 0xFFFFFF) / 16777216.0f;
}

long k3_file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    return sz;
}

void *k3_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    if (out_len) *out_len = sz;
    return buf;
}
