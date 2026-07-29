import inspect

class Layer:
    def __init__(self):
        self.built = False
        self.training = True
    def __call__(self, inputs, **kwargs):
        # Only pass kwargs that forward() accepts
        sig = inspect.signature(self.forward)
        accepted = set(sig.parameters.keys()) - {'self'}
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return self.forward(inputs, **filtered)
    def forward(self, inputs, **kwargs):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError
    def parameters(self):
        return []
