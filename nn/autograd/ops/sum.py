import numpy as np
from ..function import Function
from ..tensor import Tensor

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(a, axis=axis, keepdims=keepdims)
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        if isinstance(a, Tensor) and a.requires_grad:
            if ctx.axis is None:
                ga = np.full_like(a_data, grad_output)
            elif ctx.keepdims:
                # keepdims=True means output has same ndim, just broadcast
                ga = np.broadcast_to(grad_output, a_data.shape).copy()
            else:
                ga = np.expand_dims(grad_output, axis=ctx.axis)
                ga = np.broadcast_to(ga, a_data.shape).copy()
            return ga, None, None
        return None, None, None
