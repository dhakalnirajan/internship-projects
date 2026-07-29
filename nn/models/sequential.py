import numpy as np
from ..autograd import Tensor

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.loss = None
        self.optimizer = None
        self.built = False

    def add(self, layer):
        self.layers.append(layer)
        self.built = False

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, inputs, training=True):
        if isinstance(inputs, np.ndarray):
            inputs = Tensor(inputs, requires_grad=False)
        x = inputs
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            if hasattr(layer, 'forward') and 'training' in layer.forward.__code__.co_varnames:
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs, training=False)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = Tensor(X, requires_grad=False)
        elif not isinstance(X, Tensor):
            X = Tensor(X, requires_grad=False)
        return self.forward(X, training=False)

    def fit(self, X, y, epochs=10, batch_size=32, verbose=1):
        if self.loss is None or self.optimizer is None:
            raise ValueError("Model must be compiled before fitting.")
        
        if isinstance(X, np.ndarray):
            X = Tensor(X, requires_grad=False)
        if isinstance(y, np.ndarray):
            y = Tensor(y, requires_grad=False)

        n_samples = X.shape[0]
        
        # Determine print frequency
        if verbose is True:
            freq = 1
        elif verbose is False or verbose == 0:
            freq = None
        elif isinstance(verbose, int) and verbose > 0:
            freq = verbose
        else:
            freq = 1  # fallback

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X.data[indices]
            y_shuffled = y.data[indices]
            
            epoch_loss = 0.0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_X = Tensor(X_shuffled[start:end], requires_grad=False)
                batch_y = Tensor(y_shuffled[start:end], requires_grad=False)

                y_pred = self.forward(batch_X, training=True)
                loss = self.loss.forward(batch_y, y_pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data.item() * (end - start)

            epoch_loss /= n_samples
            
            if freq and (epoch + 1) % freq == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params