import numpy as np
from ..autograd import Function, Tensor
from .base import Layer


class _Conv1DFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel, bias, strides, padding):
        k, C, F = kernel.shape
        s = strides

        if padding == 'valid':
            p = 0
        elif padding == 'same':
            out_len = int(np.ceil(x.shape[1] / s))
            total_p = max(0, (out_len - 1) * s + k - x.shape[1])
            p = total_p // 2
        else:
            if isinstance(padding, int):
                p = padding
            else:
                p = 0

        ctx.save_for_backward(x, kernel, bias)
        ctx.strides = s
        ctx.padding = p
        ctx.input_shape = x.shape
        ctx.kernel_shape = kernel.shape

        N, L, C = x.shape
        x_pad = np.pad(x, ((0,0), (p,p), (0,0)), mode='constant')
        out_len = (L + 2*p - k) // s + 1

        patches = np.zeros((N, k*C, out_len), dtype=x.dtype)
        for n in range(N):
            for i in range(out_len):
                patch = x_pad[n, i*s:i*s+k, :]
                patches[n, :, i] = patch.ravel()
        ctx.patches = patches

        kernel_flat = kernel.reshape(F, -1)  # (F, k*C)
        out_flat = np.einsum('fk,nkw->nfw', kernel_flat, patches)
        out_flat += bias.reshape(1, F, 1)
        out = out_flat.transpose(0, 2, 1)  # (N, out_len, F)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, kernel, bias = ctx.saved_tensors
        s = ctx.strides
        p = ctx.padding
        k, C, F = ctx.kernel_shape
        N, L, C = ctx.input_shape
        patches = ctx.patches
        out_len = grad_output.shape[1]

        dbias = grad_output.sum(axis=(0, 1))

        grad_flat = grad_output.transpose(0, 2, 1)  # (N, F, out_len)
        dkernel = np.zeros((F, k*C), dtype=grad_output.dtype)
        for n in range(N):
            dkernel += grad_flat[n] @ patches[n].T
        dkernel = dkernel.reshape(k, C, F)

        kernel_flat = kernel.reshape(F, -1)
        grad_patches = np.zeros((N, k*C, out_len), dtype=grad_output.dtype)
        for n in range(N):
            grad_patches[n] = kernel_flat.T @ grad_flat[n]

        grad_input = np.zeros((N, L + 2*p, C), dtype=grad_output.dtype)
        for n in range(N):
            for i in range(out_len):
                patch = grad_patches[n, :, i].reshape(k, C)
                grad_input[n, i*s:i*s+k, :] += patch
        if p > 0:
            grad_input = grad_input[:, p:-p, :]

        return grad_input, dkernel, dbias, None, None


class Conv1D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel = None
        self.bias = None
        self.built = False
        self._ctx = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        k_shape = (self.kernel_size, in_channels, self.filters)
        limit = np.sqrt(6 / (in_channels + self.filters))
        self.kernel = Tensor(np.random.uniform(-limit, limit, k_shape), requires_grad=True)
        self.bias = Tensor(np.zeros(self.filters), requires_grad=True)
        self.built = True

    def forward(self, inputs):
        if not self.built:
            self.build(inputs.shape)
        out = _Conv1DFunction.apply(inputs, self.kernel, self.bias, self.strides, self.padding)
        if out._ctx is not None:
            self._ctx = out._ctx
        return out

    def backward(self, grad_output):
        if self._ctx is None:
            raise RuntimeError("Forward pass must be called before backward.")
        grad_input, dkernel, dbias, _, _ = _Conv1DFunction.backward(self._ctx, grad_output.data if hasattr(grad_output, 'data') else grad_output)
        if self.kernel.grad is None:
            self.kernel.grad = dkernel
        else:
            self.kernel.grad += dkernel
        if self.bias.grad is None:
            self.bias.grad = dbias
        else:
            self.bias.grad += dbias
        return Tensor(grad_input, requires_grad=False)

    def parameters(self):
        return [('kernel', self.kernel), ('bias', self.bias)]