/* k3_tensor.c */
#include "k3_tensor.h"

Tensor *tensor_new(int ndim, const int *shape) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->ndim = ndim;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        size *= shape[i];
    }
    t->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    t->size = size;
    t->data = calloc(size, sizeof(float));
    return t;
}

Tensor *tensor_new_1d(int n) { int s[] = {n}; return tensor_new(1, s); }
Tensor *tensor_new_2d(int r, int c) { int s[] = {r,c}; return tensor_new(2, s); }
void tensor_free(Tensor *t) { if (t) { free(t->data); free(t); } }

void tensor_matmul(const Tensor *A, const Tensor *B, Tensor *C) {
    int M = A->shape[0], K = A->shape[1], N = B->shape[1];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A->data[i * K + k] * B->data[k * N + j];
            C->data[i * N + j] = sum;
        }
    }
}

void tensor_matmul_tB(const Tensor *A, const Tensor *B, Tensor *C) {
    int M = A->shape[0], K = A->shape[1], N = B->shape[0];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A->data[i * K + k] * B->data[j * K + k];
            C->data[i * N + j] = sum;
        }
    }
}

void tensor_add(const Tensor *a, const Tensor *b, Tensor *out) {
    for (int i = 0; i < a->size; i++) out->data[i] = a->data[i] + b->data[i];
}

void tensor_scale(Tensor *t, float s) {
    for (int i = 0; i < t->size; i++) t->data[i] *= s;
}

void tensor_copy(const Tensor *src, Tensor *dst) {
    memcpy(dst->data, src->data, src->size * sizeof(float));
}

void tensor_softmax_last(float *vec, int n) {
    float mx = vec[0];
    for (int i = 1; i < n; i++) if (vec[i] > mx) mx = vec[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { vec[i] = expf(vec[i] - mx); sum += vec[i]; }
    for (int i = 0; i < n; i++) vec[i] /= sum;
}

void tensor_rmsnorm(const float *x, const float *weight, float eps,
                    float *out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float norm = k3_rsqrt(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * norm * weight[i];
}

void tensor_rope(float *q, float *k,
                 const float *sin_cache, const float *cos_cache,
                 int seq_len, int head_dim) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim; i += 2) {
            float sin = sin_cache[pos * (head_dim/2) + i/2];
            float cos = cos_cache[pos * (head_dim/2) + i/2];
            float q0 = q[pos * head_dim + i];
            float q1 = q[pos * head_dim + i + 1];
            q[pos * head_dim + i]     = q0 * cos - q1 * sin;
            q[pos * head_dim + i + 1] = q0 * sin + q1 * cos;
            float k0 = k[pos * head_dim + i];
            float k1 = k[pos * head_dim + i + 1];
            k[pos * head_dim + i]     = k0 * cos - k1 * sin;
            k[pos * head_dim + i + 1] = k0 * sin + k1 * cos;
        }
    }
}
