import numpy as np
from ..function import Function
from ..tensor import Tensor


class Concat(Function):
    """Concatenate tensors along an axis (default: last axis, i.e. channels)."""

    @staticmethod
    def forward(ctx, *inputs):
        axis = inputs[-1] if isinstance(inputs[-1], int) else -1
        tensors = inputs[:-1] if isinstance(inputs[-1], int) else inputs
        ctx.axis = axis
        ctx.sizes = [t.shape[axis] for t in tensors]
        return np.concatenate([t.data if isinstance(t, Tensor) else t for t in tensors], axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        grads = []
        start = 0
        for size in ctx.sizes:
            end = start + size
            sl = [slice(None)] * grad_output.ndim
            sl[ctx.axis] = slice(start, end)
            grads.append(grad_output[tuple(sl)].copy())
            start = end
        return tuple(grads)


def concat(tensors, axis=-1):
    """Concatenate a list of Tensors along `axis`, supporting autograd."""
    tensors = list(tensors)
    return Concat.apply(*tensors, axis)
