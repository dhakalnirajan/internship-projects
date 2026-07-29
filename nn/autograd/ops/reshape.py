from ..function import Function
from ..tensor import Tensor

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_for_backward(a)
        ctx.original_shape = a.shape
        return a.reshape(shape)
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        ga = grad_output.reshape(ctx.original_shape) if isinstance(a, Tensor) and a.requires_grad else None
        return ga, None
