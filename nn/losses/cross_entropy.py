from .base import Loss

class CrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        # Keep the Tensor on the left of operators: ndarray.__mul__(Tensor)
        # would otherwise create an object array and break the autograd chain.
        log_probs = (y_pred + 1e-8).log()
        loss = - (log_probs * y_true).sum(axis=-1).mean()
        return loss
