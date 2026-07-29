import numpy as np
from .base import Layer
from ..autograd import Tensor

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, inputs, training=None):
        # Use provided training flag or instance attribute
        if training is not None:
            self.training = training
        
        # Store input shape for backward
        self.input_shape = inputs.shape
        
        # Get numpy data from input
        if isinstance(inputs, Tensor):
            inputs_data = inputs.data
        elif isinstance(inputs, np.ndarray):
            inputs_data = inputs
        else:
            inputs_data = np.array(inputs)
        
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs_data.shape) / (1 - self.rate)
            return Tensor(inputs_data * self.mask, requires_grad=False)
        else:
            return Tensor(inputs_data, requires_grad=False)

    def backward(self, grad_output):
        # Ensure grad_output is numpy array
        if isinstance(grad_output, Tensor):
            grad_output = grad_output.data
        elif not isinstance(grad_output, np.ndarray):
            grad_output = np.array(grad_output, dtype=np.float32)
        
        # Apply dropout mask (inverted dropout)
        if self.mask is not None:
            return grad_output * self.mask
        return grad_output
