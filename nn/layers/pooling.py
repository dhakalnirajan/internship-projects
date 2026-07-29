import numpy as np
from ..autograd import Function, Tensor
from .base import Layer


class _MaxPooling2DFunction(Function):
    @staticmethod
    def forward(ctx, x, pool_size, strides):
        N, H, W, C = x.shape
        ph, pw = pool_size
        sh, sw = strides

        out_h = (H - ph) // sh + 1
        out_w = (W - pw) // sw + 1

        out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)
        max_indices = np.zeros((N, out_h, out_w, C, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * sh
                        h_end = h_start + ph
                        w_start = j * sw
                        w_end = w_start + pw

                        patch = x[n, h_start:h_end, w_start:w_end, c]
                        out[n, i, j, c] = np.max(patch)
                        
                        max_idx = np.argmax(patch)
                        local_h, local_w = divmod(max_idx, pw)
                        max_indices[n, i, j, c] = [h_start + local_h, w_start + local_w]

        ctx.save_for_backward(x)
        ctx.max_indices = max_indices
        ctx.pool_size = pool_size
        ctx.strides = strides
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        max_indices = ctx.max_indices
        N, H, W, C = ctx.input_shape
        out_h, out_w = grad_output.shape[1:3]

        grad_input = np.zeros_like(x, dtype=grad_output.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_idx, w_idx = max_indices[n, i, j, c]
                        grad_input[n, h_idx, w_idx, c] += grad_output[n, i, j, c]

        return grad_input, None, None


class _AveragePooling2DFunction(Function):
    @staticmethod
    def forward(ctx, x, pool_size, strides):
        N, H, W, C = x.shape
        ph, pw = pool_size
        sh, sw = strides

        out_h = (H - ph) // sh + 1
        out_w = (W - pw) // sw + 1

        out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * sh
                        h_end = h_start + ph
                        w_start = j * sw
                        w_end = w_start + pw

                        patch = x[n, h_start:h_end, w_start:w_end, c]
                        out[n, i, j, c] = np.mean(patch)

        ctx.save_for_backward(x)
        ctx.pool_size = pool_size
        ctx.strides = strides
        ctx.input_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        pool_size = ctx.pool_size
        strides = ctx.strides
        N, H, W, C = ctx.input_shape
        ph, pw = pool_size
        sh, sw = strides
        out_h, out_w = grad_output.shape[1:3]

        grad_input = np.zeros_like(x, dtype=grad_output.dtype)
        pool_area = ph * pw

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * sh
                        h_end = h_start + ph
                        w_start = j * sw
                        w_end = w_start + pw

                        grad_input[n, h_start:h_end, w_start:w_end, c] += grad_output[n, i, j, c] / pool_area

        return grad_input, None, None


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2,2), strides=(2,2)):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs):
        input_data = inputs.data if hasattr(inputs, 'data') else inputs
        out_tensor = _MaxPooling2DFunction.apply(input_data, self.pool_size, self.strides)
        if hasattr(inputs, 'requires_grad') and inputs.requires_grad:
            out_tensor.requires_grad = True
        return out_tensor

    def backward(self, grad_output):
        if self._ctx is None:
            raise RuntimeError("Forward pass must be called before backward.")
        grad_input, _, _ = _MaxPooling2DFunction.backward(
            self._ctx, 
            grad_output.data if hasattr(grad_output, 'data') else grad_output
        )
        return Tensor(grad_input, requires_grad=False)

    def parameters(self):
        return []


class AveragePooling2D(Layer):
    def __init__(self, pool_size=(2,2), strides=(2,2)):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs):
        input_data = inputs.data if hasattr(inputs, 'data') else inputs
        out_tensor = _AveragePooling2DFunction.apply(input_data, self.pool_size, self.strides)
        if hasattr(inputs, 'requires_grad') and inputs.requires_grad:
            out_tensor.requires_grad = True
        return out_tensor

    def backward(self, grad_output):
        if self._ctx is None:
            raise RuntimeError("Forward pass must be called before backward.")
        grad_input, _, _ = _AveragePooling2DFunction.backward(
            self._ctx, 
            grad_output.data if hasattr(grad_output, 'data') else grad_output
        )
        return Tensor(grad_input, requires_grad=False)

    def parameters(self):
        return []