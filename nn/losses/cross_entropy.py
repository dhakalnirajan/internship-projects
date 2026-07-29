from .base import Loss

class CrossEntropy(Loss):
    def forward(self, y_true, y_pred):
        max_logits = y_pred - y_pred.max(axis=-1, keepdims=True)
        exp_logits = max_logits.exp()
        softmax = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        loss = - (y_true * (softmax + 1e-8).log()).sum(axis=-1).mean()
        return loss
