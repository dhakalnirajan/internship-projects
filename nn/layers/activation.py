import numpy as np
from .base import Layer

class Activation(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs)
        return self.output

    def backward(self, grad_output):
        # Ensure grad_output is numpy array
        if not isinstance(grad_output, np.ndarray):
            grad_output = np.array(grad_output, dtype=np.float32)
        
        # Get numpy data from stored tensors
        inputs_data = self.inputs.data if not isinstance(self.inputs, np.ndarray) else self.inputs
        output_data = self.output.data if not isinstance(self.output, np.ndarray) else self.output
        
        # Compute gradient based on activation function type
        if self.activation.__name__ == 'relu':
            # ReLU derivative: 1 if x > 0, else 0
            grad_input = grad_output * (inputs_data > 0).astype(grad_output.dtype)
        elif self.activation.__name__ == 'sigmoid':
            # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
            s = output_data
            grad_input = grad_output * s * (1 - s)
        elif self.activation.__name__ == 'tanh':
            # Tanh derivative: 1 - tanh(x)^2
            t = output_data
            grad_input = grad_output * (1 - t * t)
        elif self.activation.__name__ == 'softmax':
            # Softmax is typically used with cross-entropy loss
            # The combined gradient is simply (softmax - target)
            # For standalone use, we approximate: softmax * (1 - softmax) for diagonal
            s = output_data
            grad_input = grad_output * s * (1 - s)
        else:
            # For custom activations, attempt to use autograd if available
            if hasattr(self.output, 'grad_fn') and self.output.grad_fn is not None:
                self.output.backward(grad_output)
                grad_input = self.inputs.grad if self.inputs.grad is not None else grad_output
            else:
                # Fallback: assume identity gradient (not ideal but prevents crash)
                grad_input = grad_output
        
        return grad_input
