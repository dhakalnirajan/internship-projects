import numpy as np
from ..function import Function
from ..tensor import Tensor

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.log(a)
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        ga = grad_output / a_data if isinstance(a, Tensor) and a.requires_grad else None
        return ga,
