import numpy as np
from .base import Layer
from ..autograd import Tensor


class BatchNorm2D(Layer):
    """Batch normalization over NHWC inputs.

    Training mode normalizes with batch statistics and updates running
    averages; inference mode normalizes with the running statistics (the
    behavior needed for pretrained-weight inference). Backward propagation is
    not implemented — this layer is intended for the pretrained / inference
    path of the benchmark.
    """

    def __init__(self, channels=None, eps=1e-5, momentum=0.1):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def build(self, input_shape):
        c = self.channels if self.channels is not None else input_shape[-1]
        self.gamma = Tensor(np.ones(c, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(c, dtype=np.float32), requires_grad=True)
        self.running_mean = np.zeros(c, dtype=np.float32)
        self.running_var = np.ones(c, dtype=np.float32)
        self.built = True

    def forward(self, inputs, training=None):
        if not self.built:
            self.build(inputs.shape if hasattr(inputs, 'shape') else inputs.data.shape)
        x = inputs.data if hasattr(inputs, 'data') else inputs

        if training:
            mean = x.mean(axis=(0, 1, 2))
            var = x.var(axis=(0, 1, 2))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        out = (x - mean[None, None, None, :]) / np.sqrt(var[None, None, None, :] + self.eps)
        out = out * self.gamma.data[None, None, None, :] + self.beta.data[None, None, None, :]
        return Tensor(out.astype(np.float32), requires_grad=False)

    def parameters(self):
        if self.gamma is None:
            return []
        return [('gamma', self.gamma), ('beta', self.beta)]
