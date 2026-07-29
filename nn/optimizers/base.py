class Optimizer:
    def __init__(self, params):
        self.params = params
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        for _, p in self.params:
            p.zero_grad()
