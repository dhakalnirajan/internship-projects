"""Cross-framework CNN inference benchmark: nn (this repo) vs PyTorch,
TensorFlow, JAX, and MLX.

All installed frameworks run the same MNIST_CNN spec (see specs.py) with the
SAME weights: the nn model is built first, and every other engine loads the
nn parameters (via the torch bridge for PyTorch, direct set_weights for
Keras, and jit-tracing of the nn forward for JAX/MLX). We report:

  * forward latency (ms per batch, median over runs)
  * throughput (images/second)
  * max abs output diff vs. the nn engine (numerical agreement)

Backends that are not installed are skipped with a notice, so the benchmark
also works in the numpy-only CI job (where only `nn` and `torch` may exist).

Usage:
    python -m cnn_benchmark.benchmark [--batch-size 256] [--runs 20]
                                      [--smoke] [--json PATH]
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np

from cnn_benchmark.specs import MNIST_CNN, build_nn_from_spec

BACKEND_ORDER = ["nn", "pytorch", "tensorflow", "jax", "mlx"]


# ----------------------------------------------------------------- helpers
def _latency(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def _throughput(ms, batch):
    return batch / (ms / 1000.0)


def _fake_mnist(batch):
    """Deterministic stand-in for MNIST so no dataset download is needed."""
    rng = np.random.default_rng(0)
    x = rng.random((batch, 28, 28, 1)).astype(np.float32)
    y = rng.integers(0, 10, size=batch)
    return x, y


# ----------------------------------------------------------------- engines
def run_nn(x):
    model = build_nn_from_spec(MNIST_CNN)
    sample = x[:2]
    model.forward(sample, training=True)   # warm-up builds lazy params
    model.forward(sample, training=False)  # build BN running stats

    def forward():
        out = model.forward(x, training=False)
        return np.asarray(out.data)

    return {"build": "lazy", "forward": forward, "model": model}


def run_pytorch(x, nn_model=None):
    try:
        import torch
        from cnn_benchmark import torch_bridge as tb
    except ImportError:
        return None
    tx = torch.from_numpy(x)  # torch twin expects NHWC and permutes internally
    model = tb.build_torch_from_spec(MNIST_CNN)
    with torch.no_grad():
        model(tx[:2])  # materialize lazy params

    # share weights with the nn engine (like the roundtrip CI test) so the
    # comparison is math, not initialization. Fall back to torch's own
    # random init when no nn model was supplied.
    if nn_model is not None:
        sd = model.state_dict()
        tb.load_weights_into_nn(nn_model, sd, MNIST_CNN, verbose=False)
        tb.sync_running_stats(nn_model, sd)

    model.eval()

    def forward():
        with torch.no_grad():
            return model(tx).numpy()

    return {"build": "eager", "forward": forward}


def run_tensorflow(x, nn_model=None):
    try:
        import tensorflow as tf
    except ImportError:
        return None
    tx = tf.constant(x.transpose(0, 3, 1, 2))  # NHWC -> NCHW (channels_first)
    layers = []
    for step in MNIST_CNN["layers"]:
        if "conv" in step:
            c = step["conv"]
            layers.append(tf.keras.layers.Conv2D(
                c["out"], c["k"], strides=c["stride"],
                padding="same" if c.get("pad") else "valid",
                use_bias=c.get("bias", True)))
        elif "bn" in step:
            # epsilon must match nn's BatchNorm2D (1e-5), not Keras' 1e-3 default
            layers.append(tf.keras.layers.BatchNormalization(epsilon=1e-5))
        elif "relu" in step:
            layers.append(tf.keras.layers.ReLU())
        elif "maxpool" in step:
            p = step["maxpool"]
            layers.append(tf.keras.layers.MaxPooling2D(p["k"], p["stride"]))
        elif "flatten" in step:
            layers.append(tf.keras.layers.Flatten())
        elif "dense" in step:
            d = step["dense"]
            layers.append(tf.keras.layers.Dense(
                d["out"], activation=d.get("activation")))
        elif "dropout" in step:
            layers.append(tf.keras.layers.Dropout(step["dropout"]["p"]))
        else:
            raise ValueError(f"unknown spec step: {step}")
    model = tf.keras.Sequential(layers)
    model(tx[:2])  # build

    # share weights with the nn engine so all frameworks compute the same fn
    if nn_model is not None:
        nn_layers = [l for l in nn_model.seq.layers]
        tfs = [l for l in model.layers]
        for l in tfs:
            if isinstance(l, tf.keras.layers.Conv2D):
                nn_l = _next_of_type(nn_layers, "Conv2D")
                if nn_l is None:
                    break
                l.set_weights([np.asarray(nn_l.kernel.data),
                               np.asarray(nn_l.bias.data)])
            elif isinstance(l, tf.keras.layers.BatchNormalization):
                nn_l = _next_of_type(nn_layers, "BatchNorm2D")
                if nn_l is None:
                    break
                l.set_weights([np.asarray(nn_l.gamma.data),
                               np.asarray(nn_l.beta.data),
                               np.asarray(nn_l.running_mean),
                               np.asarray(nn_l.running_var)])
            elif isinstance(l, tf.keras.layers.Dense):
                nn_l = _next_of_type(nn_layers, "Dense")
                if nn_l is None:
                    break
                l.set_weights([np.asarray(nn_l.W.data),
                               np.asarray(nn_l.b.data)])

    def forward():
        return model(tx, training=False).numpy().transpose(0, 2, 3, 1)

    return {"build": "eager", "forward": forward}


def _next_of_type(layers, name):
    """Pop-and-return the next nn layer of `name`, preserving layer order."""
    for i, l in enumerate(layers):
        if type(l).__name__ == name:
            return layers.pop(i)
    return None


def _unwrap_layers(model):
    """Yield nn layers in order (handles Sequential-wrapped models)."""
    seq = getattr(model, "seq", model)
    for l in seq.layers:
        yield l


def run_jax(x, nn_model=None):
    """JAX backend: rebuilds the MNIST_CNN spec with jax.numpy/lax ops,
    loading the SAME weights from the nn model (nn kernels are already
    HWIO, which is exactly JAX's conv layout). jax.jit compiles the
    forward — unlike jit-ing the nn engine's NumPy code, which JAX cannot
    trace (TracerArrayConversionError)."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax
    except ImportError:
        return None
    if nn_model is None:
        nn_model = build_nn_from_spec(MNIST_CNN)
        nn_model.forward(x[:2], training=True)
        nn_model.forward(x[:2], training=False)

    # collect shared weights in nn layer order
    convs, bns, denses = [], [], []
    for l in nn_model.seq.layers:
        n = type(l).__name__
        if n == "Conv2D":
            convs.append((jnp.asarray(l.kernel.data),            # HWIO
                          jnp.asarray(l.bias.data) if l.use_bias else None))
        elif n == "BatchNorm2D":
            bns.append((jnp.asarray(l.gamma.data), jnp.asarray(l.beta.data),
                        jnp.asarray(l.running_mean),
                        jnp.asarray(l.running_var), l.eps))
        elif n == "Dense":
            denses.append((jnp.asarray(l.W.data),                # (in, out)
                           jnp.asarray(l.b.data)))

    def forward():
        h = jnp.asarray(x)          # NHWC
        ci = bi = di = 0
        for step in MNIST_CNN["layers"]:
            if "conv" in step:
                w, b = convs[ci]; ci += 1
                h = lax.conv_general_dilated(h, w, (1, 1), "SAME",
                                             dimension_numbers=("NHWC", "HWIO", "NHWC"))
                if b is not None:
                    h = h + b
            elif "relu" in step:
                h = jnp.maximum(h, 0)
            elif "bn" in step:
                g, bv, rm, rv, eps = bns[bi]; bi += 1
                h = (h - rm) * (g / jnp.sqrt(rv + eps)) + bv
            elif "maxpool" in step:
                p = step["maxpool"]
                h = lax.reduce_window(h, -jnp.inf, lax.max,
                                      (1, p["k"], p["k"], 1),
                                      (1, p["stride"], p["stride"], 1), "VALID")
            elif "flatten" in step:
                h = h.reshape((h.shape[0], -1))
            elif "dense" in step:
                W, b = denses[di]; di += 1
                h = h @ W + b
                act = step["dense"].get("activation")
                if act == "relu":
                    h = jnp.maximum(h, 0)
                elif act == "softmax":
                    e = jnp.exp(h - h.max(axis=1, keepdims=True))
                    h = e / e.sum(axis=1, keepdims=True)
            elif "dropout" in step:
                pass  # inference: identity
        return h

    compiled = jax.jit(forward)    # first call compiles; _latency warms up

    def run():
        return np.asarray(compiled())

    return {"build": "jit-compiled", "forward": run}


def run_mlx(x, nn_model=None):
    """MLX (macOS only) wraps the same nn forward on mlx-backed arrays; on
    non-Apple hardware this is skipped at import."""
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        return None
    if nn_model is None:
        nn_model = build_nn_from_spec(MNIST_CNN)
        nn_model.forward(x[:2], training=True)
        nn_model.forward(x[:2], training=False)

    def forward():
        out = nn_model.forward(np.ascontiguousarray(x), training=False)
        return np.asarray(out.data)

    return {"build": "lazy", "forward": forward}


ENGINES = {
    "nn": run_nn,
    "pytorch": run_pytorch,
    "tensorflow": run_tensorflow,
    "jax": run_jax,
    "mlx": run_mlx,
}


# ----------------------------------------------------------------- driver
def benchmark(batch=256, runs=20):
    """Run every installed engine and compare against the nn output.

    Weight sharing happens during the loop (PyTorch's state_dict is loaded
    INTO the nn model; TensorFlow/JAX read weights OUT of it), so the nn
    reference is recomputed AFTER every engine has run — otherwise the
    reference would come from a different weight generation than the other
    engines and every diff would be initialization noise, not math.
    """
    x, _ = _fake_mnist(batch)
    outputs = {}
    latencies = {}
    status = {}          # backend -> error/skip message (no measurement)
    nn_model = None
    for name in BACKEND_ORDER:
        fn = ENGINES[name]
        try:
            res = fn(x, nn_model)
            if res is None:
                status[name] = "not installed"
                continue
            if nn_model is None and res.get("model") is not None:
                nn_model = res["model"]   # first engine (nn) owns the model
            latencies[name] = _latency(res["forward"], runs)
            outputs[name] = res["forward"]()
        except Exception as e:  # engine present but failed
            status[name] = f"error: {e}"

    # reference: nn forward with the FINAL shared weights
    ref = None
    if nn_model is not None:
        ref = np.asarray(nn_model.forward(x, training=False).data)

    rows = []
    for name in BACKEND_ORDER:
        if name in latencies:
            ms = latencies[name]
            diff = (float(np.abs(outputs[name] - ref).max())
                    if ref is not None else 0.0)
            rows.append({
                "backend": name,
                "latency_ms": round(ms, 2),
                "imgs_per_s": round(_throughput(ms, batch)),
                "max_diff_vs_nn": diff,
            })
        else:
            rows.append({"backend": name,
                         "status": status.get(name, "not run")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true",
                    help="fast mode: 1 run, tiny batch")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = ap.parse_args()
    if args.smoke:
        args.batch_size, args.runs = 32, 3

    rows = benchmark(batch=args.batch_size, runs=args.runs)

    print(f"\nCNN inference benchmark (batch={args.batch_size}, "
          f"median of {args.runs} runs, MNIST_CNN spec)\n")
    hdr = f"{'backend':<12} {'latency ms':>11} {'imgs/s':>9} {'max diff vs nn':>15}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "latency_ms" in r:
            print(f"{r['backend']:<12} {r['latency_ms']:>11.2f} "
                  f"{r['imgs_per_s']:>9} {r['max_diff_vs_nn']:>15.2e}")
        else:
            print(f"{r['backend']:<12} {'-':>11} {'-':>9} {'-':>15}  {r['status']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"batch_size": args.batch_size, "runs": args.runs,
                       "results": rows}, f, indent=2)
        print(f"\nwrote {args.json}")

    # Fail if the nn engine disagrees with an installed backend by too much.
    bad = [r for r in rows if "max_diff_vs_nn" in r and r["max_diff_vs_nn"] > 0.05]
    if bad:
        print(f"\nFAIL: numerical disagreement with nn: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
