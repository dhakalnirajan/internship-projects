"""PyTorch bridge: build/train/convert — the pretrained-weights path.

Responsibilities:
  * ``build_torch_from_spec`` — mirror a spec as an equivalent torch.nn.Sequential
    (parameter shapes match ``build_nn_from_spec`` exactly).
  * ``train_torch_spec`` — quick GPU/CPU training of a torch twin (~1 min for
    MNIST-CNN), producing a state_dict with real learned weights.
  * ``convert_state_dict`` — flatten a torch state_dict into an ordered list of
    numpy arrays that lines up layer-by-layer with the nn model's parameters.
  * ``load_weights_into_nn`` — copy converted arrays into the nn model's
    Tensor parameters (kernel/weights transposed torch OIHW -> nn HWIO etc).
  * ``torch_latency`` — inference/train-step latency for the PyTorch twin, so
    the report can compare frameworks on the same weights.
  * ``torchvision_reference`` — pretrained torchvision models as an extra
    ImageNet-scale inference-speed reference table.

torch is imported lazily everywhere so the pure-NumPy benchmark keeps working
without it installed.
"""
import os
import time

import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def _torch():
    import torch
    import torch.nn as tnn
    return torch, tnn


# ---------------------------------------------------------------------------
# torch builder from spec
# ---------------------------------------------------------------------------

def build_torch_from_spec(spec):
    """Mirror a spec as torch.nn.Sequential. Order/shapes match build_nn_from_spec.

    NOTE: uses lazy conv/linear modules — run one forward pass before
    ``load_state_dict``/``parameters()`` so the params are materialised.
    """
    torch, tnn = _torch()

    class _Twin(tnn.Module):
        def __init__(self):
            super().__init__()
            mods = []
            for step in spec["layers"]:
                if "conv" in step:
                    c = step["conv"]
                    mods.append(tnn.LazyConv2d(c["out"], c["k"],
                                               stride=c["stride"], padding=c.get("pad", 0),
                                               bias=c.get("bias", True)))
                elif "bn" in step:
                    mods.append(tnn.BatchNorm2d(step["bn"]["c"]))
                elif "relu" in step:
                    mods.append(tnn.ReLU())
                elif "maxpool" in step:
                    p = step["maxpool"]
                    mods.append(tnn.MaxPool2d(p["k"], p["stride"], p.get("pad", 0)))
                elif "avgpool" in step:
                    p = step["avgpool"]
                    mods.append(tnn.AvgPool2d(p["k"], p["stride"]))
                elif "flatten" in step:
                    mods.append(tnn.Flatten())
                elif "dense" in step:
                    mods.append(_LazyDense(step["dense"]["out"]))
                elif "dropout" in step:
                    mods.append(tnn.Dropout(step["dropout"]["p"]))
                else:
                    raise ValueError(f"unknown spec step: {step}")
            self.net = tnn.Sequential(*mods)

        def forward(self, x):
            # torch is NCHW; specs are written for NHWC-style mirrors
            x = x.permute(0, 3, 1, 2).contiguous()
            return self.net(x)

    class _LazyDense(tnn.Module):
        """Dense with lazy input dim (mirrors nn's lazy Dense build)."""
        def __init__(self, out):
            super().__init__()
            self.out = out
            self.fc = None
            self.act_name = None

        def build(self, in_features):
            self.fc = tnn.Linear(in_features, self.out)

        def forward(self, x):
            if self.fc is None:
                self.build(x.shape[-1])
            return self.fc(x)

    # attach activations to dense layers: specs put activations inside dense,
    # torch twin applies ReLU / softmax after the linear
    twin = _Twin()
    # post-process: replace "dense w/ activation" by linear+activation module pair
    net = twin.net
    new_mods = []
    i = 0
    spec_steps = [s for s in spec["layers"]]
    dense_iter = iter([s["dense"] for s in spec_steps if "dense" in s])
    for m in net:
        if isinstance(m, _LazyDense):
            new_mods.append(m)
            d = next(dense_iter)
            if d.get("activation") == "relu":
                new_mods.append(tnn.ReLU())
        else:
            new_mods.append(m)
    twin.net = tnn.Sequential(*new_mods)
    return twin


# ---------------------------------------------------------------------------
# quick training of the torch twin
# ---------------------------------------------------------------------------

def train_torch_spec(spec, X_train, y_train, X_val=None, y_val=None,
                     epochs=1, batch_size=128, lr=1e-3, device=None, verbose=True):
    """Train the torch twin on MNIST data (NHWC float [0,1], int labels).

    Returns (state_dict, history). ~1 min for MNIST-CNN on free Colab CPU
    for one epoch.
    """
    torch, tnn = _torch()
    import torch.nn.functional as F
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_torch_from_spec(spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "val_acc": []}

    model.train()
    n = X_train.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n)
        ep_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).long().to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(idx)
            seen += len(idx)
        history["loss"].append(ep_loss / max(seen, 1))

        val_acc = None
        if X_val is not None and len(X_val):
            val_acc = _torch_eval(model, X_val, y_val, device)
            history["val_acc"].append(val_acc)
        if verbose:
            msg = f"  [torch] epoch {ep+1}/{epochs} loss={ep_loss/max(seen,1):.4f}"
            if val_acc is not None:
                msg += f" val_acc={val_acc:.4f}"
            print(msg)
    return model.state_dict(), model, history, device


def _torch_eval(model, X, y, device, batch=512):
    torch, _ = _torch()
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[s:s + batch]).to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            correct += int((pred == y[s:s + batch]).sum())
    return correct / X.shape[0]


# ---------------------------------------------------------------------------
# state_dict -> nn converter
# ---------------------------------------------------------------------------

def convert_state_dict(state_dict, spec):
    """Return the state_dict as an ordered list of (key, numpy array) pairs.

    ``load_weights_into_nn`` matches arrays to nn parameters by shape/layout
    (OIHW->HWIO, (out,in)->(in,out)), so this converter only needs to keep the
    state_dict order intact.
    """
    return [(k, v.detach().cpu().float().numpy() if hasattr(v, "detach") else v)
            for k, v in state_dict.items()]


def load_weights_into_nn(nn_model, state_dict, spec, verbose=True):
    """Load torch weights into a built nn SequentialModel.

    Walks the state_dict keys in order against the nn model's parameter list
    (built layer order), matching shapes (transposing where layouts differ):
      torch Conv2d weight (out,in,kh,kw)   -> nn kernel (kh,kw,in,out)
      torch Linear weight (out,in)         -> nn W (in,out)
      flat arrays                          -> nn bias / BN gamma / beta
    """
    sd = [(k, (v.detach().cpu().float().numpy() if hasattr(v, "detach") else v))
          for k, v in state_dict.items()]

    params = nn_model.parameters()  # list of (name, Tensor)
    param_iter = iter(params)

    def next_param():
        try:
            return next(param_iter)
        except StopIteration:
            return (None, None)

    used, skipped = 0, []
    p_name, p_tensor = next_param()

    # flattened feature-map shape (C, H, W) computed from the spec, used to
    # permute the first Linear's input blocks from torch (C,H,W) to nn (H,W,C)
    flat_c, flat_h, flat_w = _flatten_shape(spec)

    for key, arr in sd:
        if p_name is None:
            skipped.append(key)
            continue
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue   # BN buffers are synced separately via sync_running_stats
        arr = np.asarray(arr)

        p_shape = p_tensor.data.shape
        if arr.shape == p_shape:
            p_tensor.data[...] = arr
            used += 1
            p_name, p_tensor = next_param()
        elif arr.ndim == 4 and p_shape == (arr.shape[2], arr.shape[3], arr.shape[1], arr.shape[0]):
            # torch OIHW -> nn HWIO
            p_tensor.data[...] = arr.transpose(2, 3, 1, 0)
            used += 1
            p_name, p_tensor = next_param()
        elif arr.ndim == 2 and p_shape == (arr.shape[1], arr.shape[0]):
            # torch (out,in) -> nn (in,out). If this linear consumes the first
            # flattened feature map, permute input blocks (C,H,W) -> (H,W,C).
            w = arr.T  # (in,out)
            n_in = arr.shape[1]
            if flat_c and n_in == flat_c * flat_h * flat_w and flat_c > 1:
                w = w.reshape(flat_c, flat_h, flat_w, -1)          # (C,H,W,out)
                w = w.transpose(1, 2, 0, 3).reshape(n_in, -1)      # -> (H,W,C,out)
                flat_c = 0   # only the first Linear consumes the flattened map
            p_tensor.data[...] = w
            used += 1
            p_name, p_tensor = next_param()
        elif arr.ndim == 1 and p_shape == arr.shape:
            p_tensor.data[...] = arr
            used += 1
            p_name, p_tensor = next_param()
        else:
            skipped.append(f"{key} {arr.shape} != {p_shape}")
            # do not advance nn param — try next torch key against same param
    if verbose:
        print(f"  loaded {used} params into nn, unmatched torch keys: "
              f"{len(skipped)}{': ' + ', '.join(skipped[:4]) if skipped else ''}")
    return used, skipped


def _flatten_shape(spec):
    """(C, H, W) of the feature map right before the first Flatten, or
    (None, None, None) when the spec has no conv/pool chain before flatten."""
    h, w, c = spec.get("input", (28, 28, 1))
    seen_flatten = False
    for step in spec["layers"]:
        if "conv" in step:
            c = step["conv"]["out"]
            p = step["conv"].get("pad", 0)
            s = step["conv"]["stride"]
            k = step["conv"]["k"]
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        elif "maxpool" in step:
            p = step["maxpool"].get("pad", 0)
            k = step["maxpool"]["k"]
            s = step["maxpool"]["stride"]
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
        elif "avgpool" in step:
            k = step["avgpool"]["k"]
            s = step["avgpool"]["stride"]
            h = (h - k) // s + 1
            w = (w - k) // s + 1
        elif "flatten" in step:
            seen_flatten = True
            break
    if not seen_flatten:
        return (None, None, None)
    return (c, h, w)


def sync_running_stats(nn_model, state_dict):
    """Copy BN running_mean/var buffers from a state_dict (torch tensors or
    numpy arrays) into the nn model's BatchNorm2D layers (matched in order)."""
    def _np(v):
        return v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v)
    means = [_np(v) for k, v in state_dict.items() if k.endswith("running_mean")]
    varis = [_np(v) for k, v in state_dict.items() if k.endswith("running_var")]
    bns = [L for L in _iter_layers(nn_model) if type(L).__name__ == "BatchNorm2D"]
    for bn, m, v in zip(bns, means, varis):
        bn.running_mean = m.copy()
        bn.running_var = v.copy()
    return len(bns)


def _iter_layers(model):
    if hasattr(model, "seq"):
        yield from model.seq.layers
    else:
        yield model


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def save_weights(state_dict, name):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, f"{name}.npz")
    np.savez_compressed(path, **{k: v.detach().cpu().float().numpy()
                                 if hasattr(v, "detach") else v
                                 for k, v in state_dict.items()})
    print(f"  weights saved -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def load_weights(name):
    path = os.path.join(WEIGHTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return dict(np.load(path))


# ---------------------------------------------------------------------------
# PyTorch-side latency benchmark (same protocol as the nn one)
# ---------------------------------------------------------------------------

def torch_latency(model, X_test, device="cpu"):
    """Inference latency (single image), batch-256 throughput and train-step
    latency for the torch twin. Returns dict matching the nn latency keys."""
    torch, _ = _torch()
    model = model.to(device)
    model.eval()

    def _stats(fn, warmup=5, runs=30):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            ts = np.empty(runs)
            for i in range(runs):
                t0 = time.perf_counter()
                fn()
                ts[i] = (time.perf_counter() - t0) * 1000.0
        return float(ts.mean()), float(np.percentile(ts, 95))

    xb1 = torch.from_numpy(X_test[:1]).to(device)
    xb256 = torch.from_numpy(X_test[:256]).to(device)
    xb64 = torch.from_numpy(X_test[:64]).to(device)

    mean_ms, p95 = _stats(lambda: model(xb1))
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            model(xb256)
    batch_ms = (time.perf_counter() - t0) / 5.0 * 1000.0

    # train step (batch 64)
    import torch.nn.functional as F
    yb = torch.zeros(64, dtype=torch.long, device=device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        loss = F.cross_entropy(model(xb64), yb)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pass
    for _ in range(2):
        step()
    ts = np.empty(10)
    for i in range(10):
        t0 = time.perf_counter()
        step()
        ts[i] = (time.perf_counter() - t0) * 1000.0

    return {"lat_ms": round(mean_ms, 3), "p95_ms": round(p95, 3),
            "imgs_s": round(256.0 / (batch_ms / 1000.0), 1),
            "step_ms": round(float(ts.mean()), 2)}


# ---------------------------------------------------------------------------
# torchvision pretrained reference (ImageNet)
# ---------------------------------------------------------------------------

def torchvision_reference(names=("alexnet", "vgg16", "resnet18", "squeezenet1_0"),
                          input_size=224, device="cpu", batch=64):
    """Inference latency of torchvision *pretrained* models on ImageNet-scale
    input. These are a reference point only: their accuracy is ImageNet's, not
    MNIST-comparable. Returns list of dicts."""
    import torch
    rows = []
    x = torch.randn(1, 3, input_size, input_size, device=device)
    xb = torch.randn(batch, 3, input_size, input_size, device=device)
    for name in names:
        try:
            import torchvision.models as tvm
            weights = "IMAGENET1K_V1"
            try:
                m = getattr(tvm, name)(weights=weights)
            except TypeError:
                m = getattr(tvm, name)(pretrained=True)
            m = m.to(device).eval()
            with torch.no_grad():
                for _ in range(3):
                    m(x)
                t0 = time.perf_counter()
                for _ in range(10):
                    m(x)
                lat = (time.perf_counter() - t0) / 10.0 * 1000.0
                t0 = time.perf_counter()
                for _ in range(3):
                    m(xb)
                batch_ms = (time.perf_counter() - t0) / 3.0 * 1000.0
            n_params = sum(p.numel() for p in m.parameters())
            rows.append({"model": f"tv-{name}", "params": n_params,
                         "lat_ms": round(lat, 2),
                         "imgs_s": round(batch / (batch_ms / 1000.0), 1),
                         "input": f"3x{input_size}x{input_size} (ImageNet)"})
            print(f"  tv-{name:<16} lat={lat:.2f}ms  {batch / (batch_ms / 1000.0):,.0f} img/s")
        except Exception as e:
            print(f"  tv-{name}: skipped ({type(e).__name__}: {e})")
    return rows
