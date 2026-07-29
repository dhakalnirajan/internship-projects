import numpy as np
from ..function import Function
from ..tensor import Tensor

class Max(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.maximum(a, b)
    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        a, b = ctx.inputs
        ga = grad_output * (a_data >= b_data) if isinstance(a, Tensor) and a.requires_grad else None
        gb = grad_output * (b_data > a_data) if isinstance(b, Tensor) and b.requires_grad else None
        return ga, gb
