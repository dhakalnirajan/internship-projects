import numpy as np
from ..function import Function
from ..tensor import Tensor
from ._utils import unbroadcast

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a - b
    @staticmethod
    def backward(ctx, grad_output):
        a_data, b_data = ctx.saved_tensors
        a, b = ctx.inputs
        ga = grad_output if isinstance(a, Tensor) and a.requires_grad else None
        gb = -grad_output if isinstance(b, Tensor) and b.requires_grad else None
        if ga is not None:
            ga = unbroadcast(ga, a_data.shape)
        if gb is not None:
            gb = unbroadcast(gb, b_data.shape)
        return ga, gb
