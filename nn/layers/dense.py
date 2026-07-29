import numpy as np
from ..autograd import Tensor
from .base import Layer

class Dense(Layer):
    def __init__(self, units, input_dim=None, activation=None):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.activation = activation
        self.W = None
        self.b = None
        self.built = False
        self.inputs = None

    def build(self, input_shape):
        if self.input_dim is None:
            self.input_dim = input_shape[-1]
        limit = np.sqrt(6 / (self.input_dim + self.units))
        W_data = np.random.uniform(-limit, limit, (self.input_dim, self.units))
        b_data = np.zeros(self.units)
        self.W = Tensor(W_data, requires_grad=True)
        self.b = Tensor(b_data, requires_grad=True)
        self.built = True

    def forward(self, inputs):
        if not self.built:
            self.build(inputs.shape)
        self.inputs = inputs
        out = inputs @ self.W + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out

    def backward(self, grad_output):
        # If activation is applied, first backprop through it
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)
        
        # Ensure grad_output is numpy array for manual backprop
        if not isinstance(grad_output, np.ndarray):
            grad_output = np.array(grad_output, dtype=np.float32)
        
        # Get numpy data from inputs and weights
        inputs_data = self.inputs.data if not isinstance(self.inputs, np.ndarray) else self.inputs
        W_data = self.W.data
        
        # Compute gradients using numpy (all numpy operations)
        grad_W = inputs_data.T @ grad_output
        grad_b = grad_output.sum(axis=0)
        grad_input = grad_output @ W_data.T
        
        # Accumulate gradients for manual backprop
        if self.W.grad is None:
            self.W.grad = grad_W
        else:
            self.W.grad += grad_W
        if self.b.grad is None:
            self.b.grad = grad_b
        else:
            self.b.grad += grad_b
        
        return grad_input

    def parameters(self):
        return [('W', self.W), ('b', self.b)]
