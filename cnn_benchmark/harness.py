"""Shared training / evaluation harness for the CNN benchmark."""
import time

import numpy as np

from nn.losses import CrossEntropy
from nn.optimizers import Adam


def to_onehot(y, num_classes=10):
    return np.eye(num_classes, dtype=np.float32)[y]


def compile_model(model, lr=1e-3, sample=None):
    """Build optimizer. If `sample` (a numpy batch) is given, run one forward
    pass first so lazily-built layers create their parameters."""
    if sample is not None:
        model.forward(sample, training=True)
    loss = CrossEntropy()
    opt = Adam(model.parameters(), lr=lr)
    return loss, opt


def train_model(model, X_train, y_train, X_val=None, y_val=None,
                epochs=3, batch_size=64, lr=1e-3, verbose=True, max_batches=None):
    """Train `model`; returns history dict. max_batches caps batches/epoch (smoke tests)."""
    loss_fn, opt = compile_model(model, lr=lr, sample=X_train[:2])
    # labels for evaluate() must be integer class ids, not one-hot
    y_int = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    n = X_train.shape[0]
    history = {"loss": [], "acc": [], "val_acc": []}

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_ep, y_ep = X_train[perm], y_train[perm]
        ep_loss, seen, n_batches = 0.0, 0, 0

        t0 = time.time()
        for start in range(0, n, batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break
            end = min(start + batch_size, n)
            xb, yb = X_ep[start:end], y_ep[start:end]

            pred = model.forward(xb, training=True)
            loss = loss_fn.forward(yb, pred)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.data.item()) * (end - start)
            seen += end - start
            n_batches += 1
        dt = time.time() - t0

        ep_loss /= max(seen, 1)
        train_acc = evaluate(model, X_train[:2000], y_int[:2000])
        val_acc = None
        if X_val is not None:
            y_val_int = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
            val_acc = evaluate(model, X_val, y_val_int)

        history["loss"].append(ep_loss)
        history["acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            msg = (f"  epoch {epoch+1}/{epochs}  loss={ep_loss:.4f}  "
                   f"acc={train_acc:.4f}  ({dt:.1f}s)")
            if val_acc is not None:
                msg += f"  val_acc={val_acc:.4f}"
            print(msg)

    return history


def evaluate(model, X, y, batch_size=256):
    """Accuracy over (X, int labels y)."""
    correct = 0
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        pred = model.forward(xb, training=False)
        preds = np.argmax(pred.data if hasattr(pred, "data") else pred, axis=1)
        correct += int((preds == y[start:start + batch_size]).sum())
    return correct / X.shape[0]
