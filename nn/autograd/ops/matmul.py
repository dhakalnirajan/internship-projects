import numpy as np
from ..function import Function
from ..tensor import Tensor

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.matmul(a, b)
    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        a, b = ctx.inputs
        # Ensure grad_output is at least 1-d for matmul
        g = np.atleast_1d(grad_output)
        ga = np.matmul(g, b_data.T) if isinstance(a, Tensor) and a.requires_grad else None
        gb = np.matmul(a_data.T, g) if isinstance(b, Tensor) and b.requires_grad else None
        return ga, gb
