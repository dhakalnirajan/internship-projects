import numpy as np
from .base import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {id(p): np.zeros_like(p.data) for _, p in params}
        self.v = {id(p): np.zeros_like(p.data) for _, p in params}

    def step(self):
        self.t += 1
        for _, p in self.params:
            if p.grad is None:
                continue
            m = self.m[id(p)]
            v = self.v[id(p)]
            g = p.grad
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[id(p)] = m
            self.v[id(p)] = v
