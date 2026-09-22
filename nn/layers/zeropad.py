import numpy as np
from .base import Layer
from ..autograd import Tensor


class ZeroPad2D(Layer):
    """Zero-padding layer for NHWC inputs: pads (top, bottom, left, right)."""

    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = padding  # (ph_top, ph_bottom, pw_left, pw_right)

    def forward(self, inputs):
        x = inputs.data if hasattr(inputs, 'data') else inputs
        pt, pb, pl, pr = self.padding
        out = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode='constant')
        return Tensor(out.astype(np.float32), requires_grad=False)

    def backward(self, grad_output):
        g = grad_output.data if hasattr(grad_output, 'data') else grad_output
        pt, pb, pl, pr = self.padding
        H, W = g.shape[1] - pt - pb, g.shape[2] - pl - pr
        return Tensor(g[:, pt:pt + H, pl:pl + W, :], requires_grad=False)

    def parameters(self):
        return []
