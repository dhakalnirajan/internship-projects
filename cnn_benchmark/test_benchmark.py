"""CI test suite for the CNN benchmark package.

Run via GitHub Actions (.github/workflows/tests.yml) with:
    python -m pytest cnn_benchmark/test_benchmark.py -q

Covers the pure-NumPy parts (no torch required) and, when torch IS available,
the PyTorch bridge and the pretrained-weights conversion path.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_benchmark.specs import (MNIST_CNN, SPECS, build_nn_from_spec)
from cnn_benchmark.report import generate_report


# --------------------------------------------------------------- nn library
def test_conv2d_true_cross_correlation():
    """Regression: kernel.reshape(F,-1) scrambled filters; must match a
    manual cross-correlation reference now."""
    from nn.layers.conv2d import _Conv2DFunction
    from nn.autograd import Tensor

    rng = np.random.default_rng(0)
    x = rng.random((2, 7, 7, 3)).astype(np.float32)
    w = rng.standard_normal((3, 3, 3, 8)).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    out = _Conv2DFunction.apply(Tensor(x), Tensor(w), Tensor(b), (1, 1), "same")

    kh, kw, C, F = w.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    ref = np.zeros((2, 7, 7, 8), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            for f in range(8):
                ref[:, i, j, f] = (xp[:, i:i+kh, j:j+kw, :] * w[:, :, :, f]).sum(axis=(1, 2, 3)) + b[f]
    assert np.abs(out.data - ref).max() < 1e-4


def test_gradient_checks():
    from cnn_benchmark.grad_check import check_conv2d, check_pooling
    check_conv2d()   # prints rel errors; grad check failure would show huge err
    check_pooling()


def test_batchnorm2d_inference():
    from nn.layers import BatchNorm2D
    bn = BatchNorm2D(4)
    bn.build((None, 6, 6, 4))
    x = np.random.rand(2, 6, 6, 4).astype(np.float32)
    out = bn.forward(x, training=False)
    assert out.data.shape == x.shape


def test_zeropad():
    from nn.layers import ZeroPad2D
    x = np.ones((1, 4, 4, 2), dtype=np.float32)
    out = ZeroPad2D(1).forward(x)
    assert out.data.shape == (1, 6, 6, 2)
    assert out.data[0, 0, :, :].max() == 0 and out.data[0, 1, 1, 0] == 1


# -------------------------------------------------------------- specs/nn
def test_spec_builds_and_counts():
    m = build_nn_from_spec(MNIST_CNN)
    m.forward(np.random.rand(2, 28, 28, 1).astype(np.float32), training=True)
    assert m.count_params() == 421834  # matches the PyTorch twin exactly


# ------------------------------------------------------- torch bridge (torch)
torch = pytest.importorskip("torch", reason="torch not installed")


def test_torch_twin_param_parity():
    from cnn_benchmark import torch_bridge as tb
    spec = MNIST_CNN
    tm = tb.build_torch_from_spec(spec)
    x = np.random.rand(2, 28, 28, 1).astype(np.float32)
    with torch.no_grad():
        tm(torch.from_numpy(x))
    n_torch = sum(p.numel() for p in tm.parameters())
    nn_m = build_nn_from_spec(spec)
    nn_m.forward(x, training=True)
    assert n_torch == nn_m.count_params()


def test_weight_conversion_roundtrip():
    """Same weights in both engines must give identical argmax predictions."""
    from cnn_benchmark import torch_bridge as tb
    spec = MNIST_CNN
    x = np.random.rand(4, 28, 28, 1).astype(np.float32)
    tm = tb.build_torch_from_spec(spec)
    with torch.no_grad():
        tm(torch.from_numpy(x))
    sd = tm.state_dict()

    nn_m = build_nn_from_spec(spec)
    nn_m.forward(x, training=True)
    tb.load_weights_into_nn(nn_m, sd, spec, verbose=False)
    tb.sync_running_stats(nn_m, sd)
    out_n = nn_m.forward(x, training=False)

    tm2 = tb.build_torch_from_spec(spec)
    with torch.no_grad():
        tm2(torch.from_numpy(x))
    tm2.load_state_dict(sd)
    tm2.eval()
    with torch.no_grad():
        logits = tm2(torch.from_numpy(x)).numpy()
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert (out_n.data.argmax(1) == p.argmax(1)).all()
    assert np.abs(out_n.data - p).max() < 0.05  # softmax-corrected agreement


def test_flatten_shape_helper():
    from cnn_benchmark.torch_bridge import _flatten_shape
    c, h, w = _flatten_shape(MNIST_CNN)
    assert (c, h, w) == (64, 7, 7)


def test_all_specs_build_in_both_engines():
    from cnn_benchmark import torch_bridge as tb
    for spec in SPECS.values():
        m = build_nn_from_spec(spec)
        x = np.random.rand(1, *spec["input"]).astype(np.float32)
        m.forward(x, training=True)
        tm = tb.build_torch_from_spec(spec)
        with torch.no_grad():
            tm(torch.from_numpy(x))
        assert m.count_params() == sum(p.numel() for p in tm.parameters())


# ----------------------------------------------------------------- report
def test_report_renders_and_conditionals():
    results = [{
        "model": "MNIST-CNN (nn, loaded weights)", "params": 421834,
        "test_acc": 0.95, "train_s": 0.0,
        "weights_source": "PyTorch twin, converted",
        "history": {"loss": [], "acc": [], "val_acc": []},
    }, {
        "model": "MNIST-CNN (PyTorch twin)", "params": 421834,
        "test_acc": 0.95, "train_s": 0.0, "lat_ms": 1.5, "imgs_s": 3000.0,
        "weights_source": "shared", "history": {"loss": [], "acc": [], "val_acc": []},
    }]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "REPORT.md")
        generate_report(results, settings={"mode": "pytorch"}, assets_dir=d,
                        out_path=out, title="T")
        text = open(out, encoding="utf-8").read()
    assert "nn versus PyTorch (loaded weights)" in text
    assert "torch twin" in text.lower() or "PyTorch" in text


def test_report_no_cross_section_for_scratch():
    results = [{"model": "LeNet-5", "params": 61706, "test_acc": 0.9,
                "train_s": 100.0, "history": {"loss": [0.5], "acc": [0.9], "val_acc": [0.9]}}]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "REPORT.md")
        generate_report(results, assets_dir=d, out_path=out, title="T")
        text = open(out, encoding="utf-8").read()
    assert "nn versus PyTorch" not in text
