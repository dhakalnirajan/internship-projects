import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {id(p): np.zeros_like(p.data) for _, p in params}

    def step(self):
        for _, p in self.params:
            if p.grad is None:
                continue
            v = self.velocities[id(p)]
            v = self.momentum * v - self.lr * p.grad
            p.data += v
            self.velocities[id(p)] = v
