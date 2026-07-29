import numpy as np
from ..function import Function
from ..tensor import Tensor

# float32 exp overflows above ~88.7
_EXP_MAX = 88.0

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        clamped = np.clip(a, None, _EXP_MAX)
        ctx.save_for_backward(clamped)
        return np.exp(clamped)
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        ga = grad_output * np.exp(a_data) if isinstance(a, Tensor) and a.requires_grad else None
        return ga,
