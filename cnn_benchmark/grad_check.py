"""Quick gradient checks: compare numerical vs analytical gradients for Conv2D & pooling."""
import numpy as np

from nn.autograd import Tensor
from nn.layers import Conv2D, MaxPooling2D, AveragePooling2D


def num_grad(f, x, eps=1e-3):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check_conv2d():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 6, 6, 2)).astype(np.float32)
    layer = Conv2D(3, 3, padding="same")
    layer.build(x.shape)
    k, b = layer.kernel.data, layer.bias.data

    def loss():
        out = _Conv_forward(layer, x)
        return float(out.data.sum())

    def _Conv_forward(layer, x):
        from nn.layers.conv2d import _Conv2DFunction
        return _Conv2DFunction.apply(Tensor(x), layer.kernel, layer.bias,
                                     layer.strides, layer.padding)

    # analytical grads via function backward
    out = _Conv_forward(layer, x)
    gin, dk, db, _, _ = __import__("nn.layers.conv2d", fromlist=["_Conv2DFunction"])._Conv2DFunction.backward(out._ctx, np.ones_like(out.data))

    gk_num = num_grad(loss, k)
    gb_num = num_grad(loss, b)
    gx_num = num_grad(loss, x)

    def rel_err(a, b_):
        denom = max(np.abs(a).max(), np.abs(b_).max(), 1e-8)
        return np.abs(a - b_).max() / denom

    print("conv2d  dK rel_err:", rel_err(gk_num, dk))
    print("conv2d  db rel_err:", rel_err(gb_num, db))
    print("conv2d  dx rel_err:", rel_err(gx_num, gin))


def check_pooling():
    rng = np.random.default_rng(1)
    for name, cls in [("maxpool", MaxPooling2D), ("avgpool", AveragePooling2D)]:
        x = rng.normal(size=(2, 6, 6, 2)).astype(np.float32)

        # weighted-sum objective whose weights differ per output position,
        # so any positional mix-up in the backward shows up as a large error
        w = rng.normal(size=(2, 3, 3, 2)).astype(np.float32)

        def loss():
            from nn.layers.pooling import _MaxPooling2DFunction, _AveragePooling2DFunction
            fn = _MaxPooling2DFunction if name == "maxpool" else _AveragePooling2DFunction
            out = fn.apply(x, (2, 2), (2, 2))
            return float((out.data * w).sum())

        from nn.layers.pooling import _MaxPooling2DFunction, _AveragePooling2DFunction
        F = _MaxPooling2DFunction if name == "maxpool" else _AveragePooling2DFunction
        from nn.autograd.function import Context
        ctx = Context()
        o = F.forward(ctx, x, (2, 2), (2, 2))
        gin, _, _ = F.backward(ctx, w)
        gx_num = num_grad(loss, x)
        denom = max(np.abs(gx_num).max(), np.abs(gin).max(), 1e-8)
        print(f"{name}  dx rel_err:", np.abs(gx_num - gin).max() / denom)


if __name__ == "__main__":
    check_conv2d()
    check_pooling()
