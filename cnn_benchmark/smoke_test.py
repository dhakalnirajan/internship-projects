"""Tiny end-to-end smoke test: 1 forward+backward step on every architecture."""
import numpy as np

from cnn_benchmark.architectures import build_models
from cnn_benchmark.harness import train_model, to_onehot

rng = np.random.default_rng(42)
X = rng.random((64, 28, 28, 1)).astype(np.float32)
y = rng.integers(0, 10, size=64)
y_oh = to_onehot(y)

for m in build_models():
    try:
        h = train_model(m, X, y_oh, epochs=1, batch_size=16, max_batches=2, verbose=False)
        status = "OK  loss=%.4f" % h["loss"][0]
    except Exception as e:
        status = "FAIL %s: %s" % (type(e).__name__, e)
    print(f"{m.name:<18} params={m.count_params():>9,}  {status}")
