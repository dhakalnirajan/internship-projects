/* k3_tensor.h — Minimal tensor library */
#ifndef K3_TENSOR_H
#define K3_TENSOR_H
#include "k3_common.h"

typedef struct {
    float *data;
    int    ndim;
    int    shape[4];
    int    stride[4];
    int    size;
} Tensor;

Tensor *tensor_new(int ndim, const int *shape);
Tensor *tensor_new_1d(int n);
Tensor *tensor_new_2d(int rows, int cols);
void    tensor_free(Tensor *t);

static inline float *tensor_at(const Tensor *t, int i, int j) {
    return t->data + i * t->stride[0] + j * t->stride[1];
}
static inline float *tensor_at3(const Tensor *t, int i, int j, int k) {
    return t->data + i*t->stride[0] + j*t->stride[1] + k*t->stride[2];
}

void tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C);
void tensor_matmul_tB(const Tensor *A, const Tensor *B, Tensor *C);
void tensor_add(const Tensor *a, const Tensor *b, Tensor *out);
void tensor_scale(Tensor *t, float s);
void tensor_copy(const Tensor *src, Tensor *dst);

void tensor_softmax_last(float *vec, int n);
void tensor_rmsnorm(const float *x, const float *weight, float eps,
                    float *out, int n);
void tensor_rope(float *q, float *k,
                 const float *sin_cache, const float *cos_cache,
                 int seq_len, int head_dim);

#endif
