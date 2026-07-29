import numpy as np

def unbroadcast(grad, shape):
    """Sum out dimensions that were broadcast to match original shape.
    
    When numpy broadcasts, it adds size-1 dimensions on the left and
    stretches size-1 dims. The gradient must be reduced back to the
    original operand's shape by summing over those broadcast dims.
    """
    if not isinstance(grad, np.ndarray):
        grad = np.array(grad, dtype=np.float32)
    
    # If grad has fewer dims, pad with leading 1s (numpy adds them on left)
    while len(grad.shape) < len(shape):
        grad = np.expand_dims(grad, axis=0)
    
    # Sum over leading dims that don't exist in original shape
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    
    # Sum over dimensions that were broadcast (size 1 in original)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad.reshape(shape)
