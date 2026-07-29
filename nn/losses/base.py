class Loss:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)
    def forward(self, y_true, y_pred):
        raise NotImplementedError
