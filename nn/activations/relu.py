from ..autograd.ops.max import Max

def relu(x):
    return Max.apply(x, 0.0)
