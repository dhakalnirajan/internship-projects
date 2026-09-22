import numpy as np
from ..autograd import Function, Tensor
from .base import Layer


def _windows(x, ph, pw, sh, sw):
    """Gather pooling windows as array of shape (N, C, out_h, out_w, ph, pw)."""
    N, H, W, C = x.shape
    out_h = (H - ph) // sh + 1
    out_w = (W - pw) // sw + 1
    r = np.arange(out_h) * sh
    c = np.arange(out_w) * sw
    rows = r[:, None] + np.arange(ph)[None, :]               # (out_h, ph)
    cols = c[:, None] + np.arange(pw)[None, :]               # (out_w, pw)
    win = x[:, rows[:, None, :, None], cols[None, :, None, :], :]
    # current shape: (N, out_h, out_w, ph, pw, C)
    return win.transpose(0, 5, 1, 2, 3, 4), out_h, out_w


class _MaxPooling2DFunction(Function):
    @staticmethod
    def forward(ctx, x, pool_size, strides):
        ph, pw = pool_size
        sh, sw = strides
        win, out_h, out_w = _windows(x, ph, pw, sh, sw)

        out = win.max(axis=(4, 5))  # (N, C, out_h, out_w)
        out = out.transpose(0, 2, 3, 1)  # -> NHWC
        win_flat = win.reshape(win.shape[:4] + (-1,))
        max_idx = win_flat.argmax(axis=-1)  # (N, C, out_h, out_w)
        ph_idx, pw_idx = np.unravel_index(max_idx.ravel(), (ph, pw))
        ph_idx = ph_idx.reshape(max_idx.shape)
        pw_idx = pw_idx.reshape(max_idx.shape)

        ctx.save_for_backward(x)
        ctx.pool_size = pool_size
        ctx.strides = strides
        ctx.input_shape = x.shape
        ctx.out_shape = (out_h, out_w)
        ctx.max_positions = (ph_idx, pw_idx)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        ph, pw = ctx.pool_size
        sh, sw = ctx.strides
        out_h, out_w = ctx.out_shape
        ph_idx, pw_idx = ctx.max_positions
        N, H, W, C = ctx.input_shape

        grad_input = np.zeros_like(x, dtype=grad_output.dtype)
        # positions of the max element for each window
        go = np.asarray(grad_output.data if hasattr(grad_output, 'data') else grad_output)
        go = go.transpose(0, 3, 1, 2)  # NHWC -> (N, C, out_h, out_w)
        rows = np.arange(out_h)[None, None, :, None] * sh + ph_idx   # (N, C, out_h, out_w)
        cols = np.arange(out_w)[None, None, None, :] * sw + pw_idx   # (N, C, out_h, out_w)
        n_idx = np.broadcast_to(np.arange(N)[:, None, None, None], rows.shape)
        c_idx = np.broadcast_to(np.arange(C)[None, :, None, None], rows.shape)
        np.add.at(grad_input, (n_idx, rows, cols, c_idx), go)
        return grad_input, None, None


class _AveragePooling2DFunction(Function):
    @staticmethod
    def forward(ctx, x, pool_size, strides):
        ph, pw = pool_size
        sh, sw = strides
        win, out_h, out_w = _windows(x, ph, pw, sh, sw)

        out = win.mean(axis=(4, 5))  # (N, C, out_h, out_w)
        out = out.transpose(0, 2, 3, 1)  # -> NHWC

        ctx.save_for_backward(x)
        ctx.pool_size = pool_size
        ctx.strides = strides
        ctx.input_shape = x.shape
        ctx.out_shape = (out_h, out_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        ph, pw = ctx.pool_size
        sh, sw = ctx.strides
        out_h, out_w = ctx.out_shape
        N, H, W, C = ctx.input_shape

        grad_input = np.zeros_like(x, dtype=grad_output.dtype)
        pool_area = ph * pw
        go = np.asarray(grad_output.data if hasattr(grad_output, 'data') else grad_output)
        for i in range(out_h):
            for j in range(out_w):
                gi_slice = go[:, i, j, :][:, None, None, :] / pool_area
                grad_input[:, i*sh:i*sh+ph, j*sw:j*sw+pw, :] += gi_slice
        return grad_input, None, None


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2,2), strides=(2,2)):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self._ctx = None

    def forward(self, inputs):
        input_data = inputs.data if hasattr(inputs, 'data') else inputs
        out_tensor = _MaxPooling2DFunction.apply(input_data, self.pool_size, self.strides)
        if hasattr(inputs, 'requires_grad') and inputs.requires_grad:
            out_tensor.requires_grad = True
        if out_tensor._ctx is not None:
            self._ctx = out_tensor._ctx
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
        self._ctx = None

    def forward(self, inputs):
        input_data = inputs.data if hasattr(inputs, 'data') else inputs
        out_tensor = _AveragePooling2DFunction.apply(input_data, self.pool_size, self.strides)
        if hasattr(inputs, 'requires_grad') and inputs.requires_grad:
            out_tensor.requires_grad = True
        if out_tensor._ctx is not None:
            self._ctx = out_tensor._ctx
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