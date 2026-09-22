"""Mirror architecture specs shared by the nn library and PyTorch.

Each spec below is a plain, ordered list of layer dicts describing ONE
architecture exactly once. Two builders consume the same spec:

  * ``build_nn_from_spec``  (this module)  -> custom ``nn`` layers
  * ``build_torch_from_spec`` (torch_bridge) -> equivalent ``torch.nn`` modules

Because both walk the same list in the same order, parameter shapes match 1:1
and a PyTorch ``state_dict`` can be converted and loaded into the ``nn``
implementation with zero guessing (see ``torch_bridge.convert_state_dict``).

Layer dict keys:
    conv  : {out, k, stride, pad, bias(optional, default True)}   NHWC conv
    bn    : {c}                                                   BatchNorm2D
    relu  : {}                                                    ReLU
    maxpool / avgpool : {k, stride, pad(optional)}                pooling
    flatten : {}
    dense : {out, activation(optional: 'relu'|'softmax'|None)}
    dropout : {p}   (inference no-op; kept so specs match torch modules)

The MNIST CNN is the "first MNIST model" path: a compact mirror that trains
in PyTorch in ~1 minute so its weights can be converted and benchmarked in nn
without any nn-side training.
"""

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

MNIST_CNN = {
    "name": "MNIST-CNN",
    "input": (28, 28, 1),
    "layers": [
        {"conv": {"out": 32, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"bn": {"c": 32}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"bn": {"c": 64}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"flatten": {}},
        {"dense": {"out": 128, "activation": "relu"}},
        {"dropout": {"p": 0.5}},
        {"dense": {"out": 10, "activation": "softmax"}},
    ],
}

LENET5 = {
    "name": "LeNet-5-full",
    "input": (32, 32, 1),   # classic input size; MNIST 28x28 is zero-padded
    "layers": [
        {"conv": {"out": 6, "k": 5, "stride": 1, "pad": 0}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 16, "k": 5, "stride": 1, "pad": 0}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"flatten": {}},
        {"dense": {"out": 120, "activation": "relu"}},
        {"dense": {"out": 84, "activation": "relu"}},
        {"dense": {"out": 10, "activation": "softmax"}},
    ],
}

ALEXNET_MNIST = {
    "name": "AlexNet-mirror",
    "input": (28, 28, 1),
    "layers": [
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 192, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 384, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"conv": {"out": 256, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"conv": {"out": 256, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"flatten": {}},
        {"dense": {"out": 1024, "activation": "relu"}},
        {"dropout": {"p": 0.5}},
        {"dense": {"out": 512, "activation": "relu"}},
        {"dense": {"out": 10, "activation": "softmax"}},
    ],
}

VGG16_MNIST = {
    "name": "VGG16-mirror",
    "input": (28, 28, 1),
    "layers": [
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"maxpool": {"k": 2, "stride": 2}},
         {"conv": {"out": 128, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 128, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"maxpool": {"k": 2, "stride": 2}},
         {"conv": {"out": 256, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 256, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 256, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"maxpool": {"k": 2, "stride": 2}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"maxpool": {"k": 2, "stride": 2}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"conv": {"out": 512, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
         {"maxpool": {"k": 2, "stride": 2}},
         {"flatten": {}},
         {"dense": {"out": 512, "activation": "relu"}},
         {"dropout": {"p": 0.5}},
         {"dense": {"out": 512, "activation": "relu"}},
         {"dense": {"out": 10, "activation": "softmax"}}
    ],
}

RESNET18_MNIST = {
    "name": "ResNet18-mirror",
    "input": (28, 28, 1),
    "layers": [
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1, "bias": False}}, {"relu": {}},
        {"bn": {"c": 64}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"flatten": {}},
        {"dense": {"out": 256, "activation": "relu"}},
        {"dense": {"out": 10, "activation": "softmax"}},
    ],
}

SQUEEZENET_MNIST = {
    "name": "SqueezeNet-mirror",
    "input": (28, 28, 1),
    "layers": [
        {"conv": {"out": 96, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"conv": {"out": 16, "k": 1, "stride": 1, "pad": 0}}, {"relu": {}},
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"conv": {"out": 16, "k": 1, "stride": 1, "pad": 0}}, {"relu": {}},
        {"conv": {"out": 64, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"conv": {"out": 32, "k": 1, "stride": 1, "pad": 0}}, {"relu": {}},
        {"conv": {"out": 128, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 32, "k": 1, "stride": 1, "pad": 0}}, {"relu": {}},
        {"conv": {"out": 128, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"conv": {"out": 64, "k": 1, "stride": 1, "pad": 0}}, {"relu": {}},
        {"conv": {"out": 200, "k": 3, "stride": 1, "pad": 1}}, {"relu": {}},
        {"maxpool": {"k": 2, "stride": 2}},
        {"flatten": {}},
        {"dense": {"out": 10, "activation": "softmax"}},
    ],
}

SPECS = {s["name"]: s for s in
         [MNIST_CNN, LENET5, ALEXNET_MNIST, VGG16_MNIST,
          RESNET18_MNIST, SQUEEZENET_MNIST]}


# ---------------------------------------------------------------------------
# nn builder
# ---------------------------------------------------------------------------

def _activation(name):
    from nn.activations import relu, softmax
    return {"relu": relu, "softmax": softmax}.get(name)


def build_nn_from_spec(spec, input_shape=None):
    """Build an nn SequentialModel from a spec. Requires a warm-up forward
    pass afterwards to materialise lazily-created parameters."""
    import numpy as np
    from nn.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D,
                           BatchNorm2D)
    from nn.models import Sequential
    from cnn_benchmark.architectures import SequentialModel

    layers = []
    for step in spec["layers"]:
        if "conv" in step:
            c = step["conv"]
            layers.append(Conv2D(c["out"], c["k"], strides=(c["stride"], c["stride"]),
                                 padding="same" if c.get("pad") else "valid"))
        elif "bn" in step:
            layers.append(BatchNorm2D(step["bn"]["c"]))
        elif "relu" in step:
            from nn.activations import relu
            layers.append(relu)
        elif "maxpool" in step:
            p = step["maxpool"]
            if p.get("pad"):
                from nn.layers import ZeroPad2D
                layers.append(ZeroPad2D(p["pad"]))
            layers.append(MaxPooling2D((p["k"], p["k"]), (p["stride"], p["stride"])))
        elif "avgpool" in step:
            from nn.layers import AveragePooling2D
            p = step["avgpool"]
            layers.append(AveragePooling2D((p["k"], p["k"]), (p["stride"], p["stride"])))
        elif "flatten" in step:
            layers.append(Flatten())
        elif "dense" in step:
            d = step["dense"]
            layers.append(Dense(d["out"], activation=_activation(d.get("activation"))))
        elif "dropout" in step:
            layers.append(Dropout(step["dropout"]["p"]))
        else:
            raise ValueError(f"unknown spec step: {step}")

    if input_shape is None:
        input_shape = spec["input"]
    from cnn_benchmark.architectures import SequentialModel as _SM
    seq = Sequential(layers)
    return _SM(spec["name"], seq)


if __name__ == "__main__":
    m = build_nn_from_spec(MNIST_CNN)
    import numpy as np
    from nn.autograd import Tensor
    m.forward(Tensor(np.random.rand(2, 28, 28, 1).astype(np.float32)), training=True)
    print(m.name, m.count_params(), "params")
