import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self._ctx = None
        self.grad = None

    def backward(self, grad=None):
        """Perform backpropagation from this tensor."""
        # If grad is None and this is a scalar, use 1.0
        if grad is None:
            if self.data.size == 1:
                grad = np.array(1.0, dtype=np.float32)
            else:
                raise ValueError("grad must be provided for non-scalar tensors")
        
        # Accumulate gradient
        if self.grad is None:
            self.grad = grad.copy() if isinstance(grad, np.ndarray) else grad
        else:
            self.grad += grad
        
        # If this tensor was created by an operation, propagate backward
        # IMPORTANT: pass the incoming grad (increment), not self.grad (accumulated)
        if self.grad_fn is not None:
            grads = self.grad_fn(self._ctx, grad)
            
            # grads should be a tuple with same length as ctx.inputs
            if grads is not None:
                for i, (inp, g) in enumerate(zip(self._ctx.inputs, grads)):
                    if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                        # Convert g to numpy array if needed
                        if isinstance(g, Tensor):
                            g = g.data
                        # Propagate backward
                        inp.backward(g)

    def zero_grad(self):
        self.grad = None

    # Arithmetic operators
    def __add__(self, other):
        from .ops.add import Add
        return Add.apply(self, other)
    def __radd__(self, other):
        from .ops.add import Add
        return Add.apply(other, self)
    def __sub__(self, other):
        from .ops.sub import Sub
        return Sub.apply(self, other)
    def __rsub__(self, other):
        from .ops.sub import Sub
        return Sub.apply(other, self)
    def __mul__(self, other):
        from .ops.mul import Mul
        return Mul.apply(self, other)
    def __rmul__(self, other):
        from .ops.mul import Mul
        return Mul.apply(other, self)
    def __matmul__(self, other):
        from .ops.matmul import MatMul
        return MatMul.apply(self, other)
    def __rmatmul__(self, other):
        from .ops.matmul import MatMul
        return MatMul.apply(other, self)
    def __neg__(self):
        from .ops.neg import Neg
        return Neg.apply(self)
    def __truediv__(self, other):
        from .ops.div import Div
        return Div.apply(self, other)
    def __rtruediv__(self, other):
        from .ops.div import Div
        return Div.apply(other, self)

    # Utility methods
    def sum(self, axis=None, keepdims=False):
        from .ops.sum import Sum
        return Sum.apply(self, axis, keepdims)
    def exp(self):
        from .ops.exp import Exp
        return Exp.apply(self)
    def log(self):
        from .ops.log import Log
        return Log.apply(self)
    def reshape(self, *shape):
        from .ops.reshape import Reshape
        return Reshape.apply(self, shape)
    def transpose(self, axes=None):
        from .ops.transpose import Transpose
        return Transpose.apply(self, axes)
    def max(self, axis=None, keepdims=False):
        # For softmax stability
        from .ops.max_reduce import MaxReduce
        return MaxReduce.apply(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False):
        from .ops.sum import Sum
        s = Sum.apply(self, axis, True)
        if axis is None:
            n = self.data.size
        else:
            if isinstance(axis, int):
                n = self.data.shape[axis]
            elif isinstance(axis, tuple):
                n = 1
                for ax in axis:
                    n *= self.data.shape[ax]
            else:
                n = 1
        result = s / n
        if not keepdims and axis is not None:
            out_shape = list(result.shape)
            if isinstance(axis, int):
                out_shape.pop(axis)
            elif isinstance(axis, tuple):
                for ax in sorted(axis, reverse=True):
                    out_shape.pop(ax)
            result = result.reshape(out_shape)
        return result

    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def dtype(self):
        return self.data.dtype
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
