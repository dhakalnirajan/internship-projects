from .base import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def parameters(self):
        return []