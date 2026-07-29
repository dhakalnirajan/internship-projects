from .base import Loss

class MSE(Loss):
    def forward(self, y_true, y_pred):
        diff = y_pred - y_true
        return (diff * diff).sum() / diff.data.size
