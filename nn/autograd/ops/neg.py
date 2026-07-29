from ..function import Function
from ..tensor import Tensor

class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.inputs[0]
        ga = -grad_output if isinstance(a, Tensor) and a.requires_grad else None
        return ga,
