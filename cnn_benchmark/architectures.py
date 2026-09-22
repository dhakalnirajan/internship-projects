"""8 classic CNN architectures built with the custom nn library.

Each model maps MNIST input (N, 28, 28, 1) -> 10-class logits/probs.
Where the original architecture used large input sizes (227/224), we adapt
channel counts/kernels to 28x28 while keeping the architectural *pattern*:
LeNet-5, AlexNet, VGG-16, ResNet-18, GoogLeNet (Inception v1),
MobileNetV1 (depthwise-separable as grouped convs), DenseNet-121 (reduced),
SqueezeNet 1.0 (fire modules).
"""
import numpy as np

from nn.activations import relu, softmax
from nn.autograd import Tensor
from nn.autograd import Tensor
from nn.autograd.ops import concat
from nn.layers import (
    AveragePooling2D, Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
)
from nn.models import Sequential


class Model:
    """Base wrapper providing parameter counting and shape-based naming."""

    def __init__(self, name):
        self.name = name

    def parameters(self):
        raise NotImplementedError

    def count_params(self):
        total = 0
        for _, p in self.parameters():
            if p is not None and p.data is not None:
                total += int(np.prod(p.data.shape))
        return total


class SequentialModel(Model):
    """Wraps a plain Sequential model."""

    def __init__(self, name, seq):
        super().__init__(name)
        self.seq = seq

    def forward(self, x, training=True):
        return self.seq.forward(x, training=training)

    def predict(self, X):
        return self.seq.predict(X)

    def parameters(self):
        return self.seq.parameters()


# --- Residual / Inception modules using the Concat op ------------------------

def _residual_block(x, filters, name=""):
    """ResNet basic block: conv-conv with identity shortcut + ReLU."""
    f1, f2 = filters
    identity = x
    # If channel mismatch, project with 1x1 conv (handled by caller for now)
    out = Conv2D(f1, 3, strides=(1, 1), padding="same")(x)
    out = relu(out)
    out = Conv2D(f2, 3, strides=(1, 1), padding="same")(out)
    out = concat([out, identity])  # concat-based "shortcut" (He's option is add, we use concat since simpler w/ autograd)
    out = relu(out)
    return out


class ResNetMini(Model):
    """ResNet-style network with residual blocks (concat shortcut)."""

    def __init__(self, num_classes=10):
        super().__init__("ResNet18-mini")
        self.num_classes = num_classes
        self.blocks = []
        layers = []

        x_shape = (None, 28, 28, 1)
        ch = 16
        self._cur = None

        self.stem = Conv2D(16, 3, strides=(1, 1), padding="same")
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.block1a = Conv2D(16, 3, padding="same")
        self.block1b = Conv2D(16, 3, padding="same")
        self.block2a = Conv2D(32, 3, padding="same")
        self.block2b = Conv2D(32, 3, padding="same")
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.block3a = Conv2D(64, 3, padding="same")
        self.block3b = Conv2D(64, 3, padding="same")
        self.gap = AveragePooling2D(pool_size=(7, 7), strides=(7, 7))
        self.fc = Dense(num_classes, activation=softmax)

    def _block(self, x, a, b):
        out = a(x)
        out = relu(out)
        out = b(out)
        out = concat([out, x])
        return relu(out)

    def forward(self, x, training=True):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        out = self.stem(x)
        out = relu(out)
        out = self.pool1(out)
        out = self._block(out, self.block1a, self.block1b)
        out = self.pool2(out)
        out = self._block(out, self.block2a, self.block2b)
        out = self.block3a(out)
        out = relu(out)
        out = self.block3b(out)
        out = self.gap(out)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def predict(self, X):
        return self.forward(X, training=False)

    def parameters(self):
        params = []
        for m in [self.stem, self.block1a, self.block1b, self.block2a,
                  self.block2b, self.block3a, self.block3b, self.fc]:
            params.extend(m.parameters())
        return params


def _inception(x, f1x1, f3x3, pool_proj):
    """Inception v1 module: parallel 1x1, 3x3, avgpool-1x1 branches."""
    b1 = Conv2D(f1x1, 1, padding="same")(x)
    b3 = Conv2D(f3x3, 3, padding="same")(x)
    bp = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")
    # NOTE: our AveragePooling2D doesn't support 'same' padding; emulate with 2x2/1x1
    bp = AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    bpc = Conv2D(pool_proj, 1, padding="same")(bp)
    return concat([b1, b3, bpc])


class GoogLeNetMini(Model):
    """GoogLeNet-style network with 2 inception modules (for 28x28 input)."""

    def __init__(self, num_classes=10):
        super().__init__("GoogLeNet-mini")
        self.stem = Conv2D(16, 3, padding="same")
        self.inc1 = None
        self.inc2 = None
        self._build(num_classes)

    def _build(self, num_classes):
        # We build the inception branches eagerly so parameters() works.
        class Inc:
            def __init__(self, f1, f3, fp):
                self.c1 = Conv2D(f1, 1, padding="same")
                self.c3 = Conv2D(f3, 3, padding="same")
                self.cp = Conv2D(fp, 1, padding="same")

            def apply(self, x):
                b1 = self.c1(x)
                b3 = self.c3(x)
                # zero-pad manually (AveragePooling2D has no padding arg)
                xp_data = x.data if hasattr(x, 'data') else x
                xp = Tensor(np.pad(xp_data, ((0,0),(1,1),(1,1),(0,0))), requires_grad=False)
                ap = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(xp)
                bpc = self.cp(ap)
                return concat([b1, b3, bpc])

            def parameters(self):
                return self.c1.parameters() + self.c3.parameters() + self.cp.parameters()

        self.inc1 = Inc(16, 24, 8)
        self.inc2 = Inc(32, 40, 12)
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.gap = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))
        self.fc = Dense(num_classes, activation=softmax)

    def forward(self, x, training=True):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        out = self.stem(x)
        out = relu(out)
        out = self.inc1.apply(out)
        out = self.pool(out)
        out = self.inc2.apply(out)
        out = self.gap(out)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def predict(self, X):
        return self.forward(X, training=False)

    def parameters(self):
        return (self.stem.parameters() + self.inc1.parameters()
                + self.inc2.parameters() + self.fc.parameters())


class DenseNetMini(Model):
    """DenseNet-style: each layer's output concatenated to the growing feature map."""

    def __init__(self, num_classes=10, growth=8):
        super().__init__("DenseNet-mini")
        self.growth = growth
        self.stem = Conv2D(16, 3, padding="same")
        self.convs = [Conv2D(growth, 3, padding="same") for _ in range(8)]
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.gap = AveragePooling2D(pool_size=(7, 7), strides=(7, 7))
        self.fc = Dense(num_classes, activation=softmax)

    def forward(self, x, training=True):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        out = self.stem(x)
        out = relu(out)
        for c in self.convs[:4]:
            new = relu(c(out))
            out = concat([out, new])
        out = self.pool(out)
        for c in self.convs[4:]:
            new = relu(c(out))
            out = concat([out, new])
        out = self.gap(out)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def predict(self, X):
        return self.forward(X, training=False)

    def parameters(self):
        params = self.stem.parameters()
        for c in self.convs:
            params += c.parameters()
        params += self.fc.parameters()
        return params


class SqueezeNetMini(Model):
    """SqueezeNet-style fire modules: squeeze(1x1) -> expand(1x1 + 3x3)."""

    def __init__(self, num_classes=10):
        super().__init__("SqueezeNet-mini")

        class Fire:
            def __init__(self, s, e1, e3):
                self.squeeze = Conv2D(s, 1, padding="same")
                self.e1 = Conv2D(e1, 1, padding="same")
                self.e3 = Conv2D(e3, 3, padding="same")

            def apply(self, x):
                s = relu(self.squeeze(x))
                return concat([relu(self.e1(s)), relu(self.e3(s))])

            def parameters(self):
                return (self.squeeze.parameters() + self.e1.parameters()
                        + self.e3.parameters())

        self.fire1 = Fire(8, 16, 16)
        self.fire2 = Fire(16, 32, 32)
        self.fire3 = Fire(24, 48, 48)
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.gap = AveragePooling2D(pool_size=(7, 7), strides=(7, 7))
        self.fc = Dense(num_classes, activation=softmax)
        self._Fire = Fire

    def forward(self, x, training=True):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        out = self.fire1.apply(x)
        out = self.pool(out)
        out = self.fire2.apply(out)
        out = self.fire3.apply(out)
        out = self.gap(out)
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)

    def __call__(self, x, training=True):
        return self.forward(x, training=training)

    def predict(self, X):
        return self.forward(X, training=False)

    def parameters(self):
        params = []
        for f in [self.fire1, self.fire2, self.fire3]:
            params += f.parameters()
        params += self.fc.parameters()
        return params


def build_lenet5():
    seq = Sequential([
        Conv2D(6, 5, padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(16, 5, padding="valid"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Flatten(),
        Dense(120, activation=relu),
        Dense(84, activation=relu),
        Dense(10, activation=softmax),
    ])
    return SequentialModel("LeNet-5", seq)


def build_alexnet():
    seq = Sequential([
        Conv2D(32, 5, strides=(1, 1), padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(64, 3, padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, 3, padding="same"), relu,
        Flatten(),
        Dense(256, activation=relu),
        Dropout(0.5),
        Dense(128, activation=relu),
        Dense(10, activation=softmax),
    ])
    return SequentialModel("AlexNet-mini", seq)


def build_vgg16():
    # VGG pattern: blocks of 3x3 convs, halving via pooling; channels double.
    seq = Sequential([
        Conv2D(16, 3, padding="same"), relu,
        Conv2D(16, 3, padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(32, 3, padding="same"), relu,
        Conv2D(32, 3, padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(64, 3, padding="same"), relu,
        Conv2D(64, 3, padding="same"), relu,
        MaxPooling2D((2, 2), (2, 2)),
        Conv2D(128, 3, padding="same"), relu,
        Flatten(),
        Dense(256, activation=relu),
        Dropout(0.5),
        Dense(10, activation=softmax),
    ])
    return SequentialModel("VGG16-mini", seq)


def build_mobilenet():
    # MobileNetV1 pattern emulated with 1x1 "pointwise" bottleneck convs
    # (true depthwise needs grouped conv; we approximate the bottleneck pattern).
    seq = Sequential([
        Conv2D(16, 3, strides=(2, 2), padding="same"), relu,   # 28 -> 13
        Conv2D(24, 3, strides=(1, 1), padding="same"), relu,   # 13 -> 11
        Conv2D(32, 3, strides=(2, 2), padding="same"), relu,   # 11 -> 5
        MaxPooling2D((2, 2), (2, 2)),                          # 5 -> 2
        Conv2D(48, 3, strides=(1, 1), padding="same"), relu,   # 2 -> 2
        Conv2D(64, 3, strides=(1, 1), padding="same"), relu,   # 2 -> 2
        Conv2D(96, 3, strides=(1, 1), padding="same"), relu,   # 2 -> 2
        Flatten(),
        Dense(128, activation=relu),
        Dense(10, activation=softmax),
    ])
    return SequentialModel("MobileNetV1-mini", seq)


def build_models():
    """Return list of all models for the benchmark."""
    return [
        build_lenet5(),
        build_alexnet(),
        build_vgg16(),
        ResNetMini(),
        GoogLeNetMini(),
        DenseNetMini(),
        SqueezeNetMini(),
        build_mobilenet(),
    ]
