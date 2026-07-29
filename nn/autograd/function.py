class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.inputs = None
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

class Function:
    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
    @classmethod
    def apply(cls, *args):
        from .tensor import Tensor
        requires_grad = any(isinstance(a, Tensor) and a.requires_grad for a in args)
        ctx = Context()
        data = cls.forward(ctx, *[a.data if isinstance(a, Tensor) else a for a in args])
        ctx.inputs = args
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            out._ctx = ctx
            out.grad_fn = cls.backward
        return out