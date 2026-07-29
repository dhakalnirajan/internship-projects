import numpy as np
from ..function import Function
from ..tensor import Tensor

class Transpose(Function):
    @staticmethod
    def forward(ctx, a, axes):
        ctx.save_for_backward(a)
        ctx.axes = axes
        return np.transpose(a, axes=axes)
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        if isinstance(a, Tensor) and a.requires_grad:
            axes = ctx.axes
            if axes is None:
                ga = np.transpose(grad_output)
            else:
                inv_axes = np.argsort(axes)
                ga = np.transpose(grad_output, axes=inv_axes)
            return ga, None
        return None, None
