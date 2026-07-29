import numpy as np
from ..function import Function
from ..tensor import Tensor

class MaxReduce(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.shape = a.shape
        return np.max(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, grad_output):
        a_data = ctx.saved_tensors[0]
        a = ctx.inputs[0]
        if isinstance(a, Tensor) and a.requires_grad:
            # Gradient of max is 1 at the max position, 0 elsewhere
            axis = ctx.axis
            keepdims = ctx.keepdims
            
            # Find where max occurs
            if axis is None:
                # Global max
                max_val = np.max(a_data)
                mask = (a_data == max_val)
                # Count how many times max occurs
                count = np.sum(mask)
                if count > 1:
                    # Average gradient among ties
                    grad = grad_output / count
                else:
                    grad = grad_output
                ga = np.full_like(a_data, 0.0)
                ga[mask] = grad
            else:
                # Max along specific axis
                max_val = np.max(a_data, axis=axis, keepdims=True)
                mask = (a_data == max_val)
                # Expand grad_output to match a's shape
                if not keepdims:
                    grad_expanded = np.expand_dims(grad_output, axis=axis)
                else:
                    grad_expanded = grad_output
                # Broadcast to match shape
                grad_broadcasted = np.broadcast_to(grad_expanded, a_data.shape)
                # Count max occurrences along axis
                count = np.sum(mask, axis=axis, keepdims=True)
                count = np.broadcast_to(count, a_data.shape)
                ga = np.zeros_like(a_data)
                ga[mask] = grad_broadcasted[mask] / count[mask]
            
            return ga, None, None
        return None, None, None
