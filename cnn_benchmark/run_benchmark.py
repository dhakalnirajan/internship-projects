"""CNN Benchmark: compare 8 architectures on MNIST.

Usage:
    python -m cnn_benchmark.run_benchmark [--smoke] [--mode scratch|pretrained|pytorch]

Modes:
    scratch    train every architecture from random init (the original benchmark)
    pretrained quick-train a PyTorch twin of MNIST-CNN, convert its weights and
               load them into the nn implementation, then benchmark inference
               (no nn-side training) — plus optional torchvision ImageNet reference
    pytorch    benchmark the PyTorch twins alongside the nn implementations

Results are saved to cnn_benchmark/results.json and printed as a table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from cnn_benchmark.architectures import build_models
from cnn_benchmark.harness import evaluate, to_onehot, train_model
from cnn_benchmark.report import generate_report

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_FILE = os.path.join(HERE, "..", "mnist.npz")


def load_mnist():
    data = np.load(MNIST_FILE)
    X_train = data["x_train"][:50000].astype(np.float32) / 255.0
    y_train = data["y_train"][:50000].astype(np.int32)
    X_val = data["x_train"][50000:].astype(np.float32) / 255.0
    y_val = data["y_train"][50000:].astype(np.int32)
    X_test = data["x_test"].astype(np.float32) / 255.0
    y_test = data["y_test"].astype(np.int32)
    return (X_train.reshape(-1, 28, 28, 1), y_train,
            X_val.reshape(-1, 28, 28, 1), y_val,
            X_test.reshape(-1, 28, 28, 1), y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: 1 epoch, 30 batches per epoch, 2000 train samples")
    parser.add_argument("--mode", choices=["scratch", "pretrained", "pytorch"],
                        default="scratch",
                        help="scratch: train nn from random init; pretrained: load "
                             "converted PyTorch weights into nn and benchmark inference; "
                             "pytorch: also benchmark the PyTorch twins")
    args = parser.parse_args()

    if args.mode in ("pretrained", "pytorch"):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise SystemExit("mode '%s' requires PyTorch: pip install torch" % args.mode)
        run_pretrained_or_pytorch(args)
        return

    run_scratch(args)


def run_scratch(args):

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    y_train_oh = to_onehot(y_train)
    y_val_oh = to_onehot(y_val)

    if args.smoke:
        X_train, y_train_oh = X_train[:2000], y_train_oh[:2000]
        epochs, batch_size, max_batches = 1, 64, 30
    else:
        epochs, batch_size, max_batches = 5, 64, None

    results = []
    models = build_models()
    for model in models:
        print(f"\n=== {model.name} ===")

        # warm-up forward pass to build layers (params are created lazily)
        t0 = time.time()
        model.forward(X_train[:2], training=True)
        build_s = time.time() - t0

        n_params = model.count_params()

        t0 = time.time()
        history = train_model(
            model, X_train, y_train_oh,
            X_val=X_val[:2000], y_val=y_val[:2000],
            epochs=epochs, batch_size=batch_size, max_batches=max_batches,
        )
        train_s = time.time() - t0

        n_eval = 2000 if args.smoke else X_test.shape[0]
        test_acc = evaluate(model, X_test[:n_eval], y_test[:n_eval])
        results.append({
            "model": model.name,
            "params": n_params,
            "build_s": round(build_s, 2),
            "train_s": round(train_s, 2),
            "final_loss": history["loss"][-1],
            "val_acc": history["val_acc"][-1],
            "test_acc": test_acc,
            "history": history,
        })
        print(f"  params={n_params:,}  test_acc={test_acc:.4f}")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'Model':<18}{'Params':>10}{'Train(s)':>10}{'Val acc':>10}{'Test acc':>10}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: -r["test_acc"]):
        print(f"{r['model']:<18}{r['params']:>10,}{r['train_s']:>10.1f}"
              f"{r['val_acc']:>10.4f}{r['test_acc']:>10.4f}")
    print("=" * 78)

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {out}")

    # dynamic Markdown report (sections appear only when the data supports them)
    report_path = generate_report(
        results,
        settings={
            "mode": "smoke" if args.smoke else "full",
            "SUBSET_TRAIN": int(X_train.shape[0]),
            "EPOCHS": epochs,
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": 1e-3,
            "dataset": "MNIST (mnist.npz)",
        },
        assets_dir=os.path.join(HERE, "assets"),
        out_path=os.path.join(HERE, "REPORT.md"),
    )
    print(f"Report written to {report_path}")


def run_pretrained_or_pytorch(args):
    """Pretrained / cross-framework path.

    1. Quick-train a PyTorch twin of the MNIST-CNN mirror spec (1 epoch is
       enough for ~98% on MNIST; the torch training itself is NOT part of the
       nn benchmark — it only produces the weights).
    2. Convert its state_dict and load it into the nn implementation.
    3. Benchmark nn inference-only (latency, throughput, accuracy) — compared
       against the same-weight PyTorch twin in --mode pytorch.
    4. Optionally add a torchvision pretrained (ImageNet) reference table.
    """
    from cnn_benchmark.specs import MNIST_CNN, SPECS, build_nn_from_spec
    from cnn_benchmark import torch_bridge as tb

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    if args.smoke:
        X_train, y_train = X_train[:4000], y_train[:4000]

    spec = MNIST_CNN
    name = spec["name"]
    print(f"\n=== {name} (pretrained-weights path) ===")

    # cached weights? otherwise quick-train the torch twin
    try:
        state_dict = tb.load_weights(name)
        print(f"  loaded cached weights: cnn_benchmark/weights/{name}.npz")
        import torch as _torch
        torch_model = tb.build_torch_from_spec(spec)
        with _torch.no_grad():
            torch_model(_torch.from_numpy(X_test[:2]))   # materialise lazy params
        torch_model.load_state_dict(
            {k: _torch.from_numpy(v) for k, v in state_dict.items()})
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    except FileNotFoundError:
        print("  no cached weights — quick-training the PyTorch twin "
              "(1 epoch, ~1 min on Colab CPU)...")
        state_dict, torch_model, hist, device = tb.train_torch_spec(
            spec, X_train, y_train, X_val=X_val[:2000], y_val=y_val[:2000],
            epochs=1, batch_size=128)
        tb.save_weights(state_dict, name)

    # build the nn mirror and load the converted weights
    nn_model = build_nn_from_spec(spec)
    nn_model.forward(X_test[:2], training=True)   # materialise lazy params
    used, skipped = tb.load_weights_into_nn(nn_model, state_dict, spec)
    tb.sync_running_stats(nn_model, state_dict)

    # inference-only benchmark of the nn implementation
    n_params = nn_model.count_params()
    test_acc = evaluate(nn_model, X_test, y_test)
    print(f"  nn inference accuracy: {test_acc:.4f}  ({n_params:,} params)")

    results = [{
        "model": f"{name} (nn, loaded weights)",
        "params": n_params,
        "test_acc": test_acc,
        "train_s": 0.0,          # inference-only: no nn-side training
        "weights_source": f"PyTorch twin, converted ({used} tensors)",
        "history": {"loss": [], "acc": [], "val_acc": []},
    }]

    # cross-check: same weights on the PyTorch twin must give the same accuracy
    import torch
    torch_model = torch_model.to(device).eval()
    torch_acc = tb._torch_eval(torch_model, X_test, y_test, device)
    print(f"  torch twin accuracy:   {torch_acc:.4f}  (weights cross-check)")
    results[0]["torch_acc"] = torch_acc

    # latency on both frameworks
    from cnn_benchmark.harness import compile_model  # noqa: F401  (import check)

    # nn latency
    def _lat_stats(fn, warmup=5, runs=30):
        for _ in range(warmup):
            fn()
        ts = np.empty(runs)
        for i in range(runs):
            t0 = time.perf_counter()
            fn()
            ts[i] = (time.perf_counter() - t0) * 1000.0
        return float(ts.mean()), float(np.percentile(ts, 95))

    xb1, xb256 = X_test[:1], X_test[:256]
    mean_ms, p95 = _lat_stats(lambda: nn_model.forward(xb1, training=False))
    t0 = time.perf_counter()
    for _ in range(5):
        nn_model.forward(xb256, training=False)
    batch_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    results[0].update({"lat_ms": round(mean_ms, 3), "p95_ms": round(p95, 3),
                       "imgs_s": round(256.0 / (batch_ms / 1000.0), 1)})
    print(f"  nn latency: {mean_ms:.2f}ms/img, "
          f"{results[0]['imgs_s']:,.0f} img/s @ batch256")

    # pytorch comparison mode: same measurements on the torch twin
    if args.mode == "pytorch":
        tlat = tb.torch_latency(torch_model, X_test, device=device)
        results.append({
            "model": f"{name} (PyTorch twin)",
            "params": n_params,
            "test_acc": torch_acc,
            "train_s": 0.0,
            "weights_source": "shared with nn (same converted weights)",
            "history": {"loss": [], "acc": [], "val_acc": []},
            **tlat,
        })
        print(f"  torch latency: {tlat['lat_ms']:.2f}ms/img, "
              f"{tlat['imgs_s']:,.0f} img/s @ batch256")

        # extra torch twins for a broader cross-framework table (quick: inference only)
        for spec_name in ["LeNet-5-full", "AlexNet-mirror"]:
            sp = SPECS[spec_name]
            print(f"\n=== {spec_name} (PyTorch quick-train + nn convert) ===")
            sd2, tm2, _, dev2 = tb.train_torch_spec(
                sp, X_train, y_train, X_val=X_val[:2000], y_val=y_val[:2000],
                epochs=1, batch_size=128)
            tb.save_weights(sd2, spec_name)
            nn2 = build_nn_from_spec(sp)
            nn2.forward(X_test[:2], training=True)
            tb.load_weights_into_nn(nn2, sd2, sp)
            tb.sync_running_stats(nn2, sd2)
            acc2 = evaluate(nn2, X_test[:2000], y_test[:2000])
            acc2t = tb._torch_eval(tm2.to(dev2).eval(), X_test[:2000], y_test[:2000], dev2)
            lat2, p952 = _lat_stats(lambda: nn2.forward(xb1, training=False))
            tlat2 = tb.torch_latency(tm2, X_test[:512], device=dev2)
            results.append({"model": f"{spec_name} (nn, loaded weights)",
                            "params": nn2.count_params(), "test_acc": acc2,
                            "torch_acc": acc2t, "lat_ms": round(lat2, 3),
                            "p95_ms": round(p952, 3), "train_s": 0.0,
                            "weights_source": "PyTorch twin, converted",
                            "history": {"loss": [], "acc": [], "val_acc": []}})
            results.append({"model": f"{spec_name} (PyTorch twin)",
                            "params": nn2.count_params(), "test_acc": acc2t,
                            "lat_ms": tlat2["lat_ms"], "p95_ms": tlat2["p95_ms"],
                            "imgs_s": tlat2["imgs_s"], "step_ms": tlat2["step_ms"],
                            "train_s": 0.0, "weights_source": "shared",
                            "history": {"loss": [], "acc": [], "val_acc": []}})

    # torchvision pretrained reference (ImageNet-scale; accuracy not comparable)
    try:
        tv_rows = tb.torchvision_reference(device="cuda" if torch.cuda.is_available() else "cpu")
        if tv_rows:
            with open(os.path.join(HERE, "torchvision_reference.json"), "w") as f:
                json.dump(tv_rows, f, indent=2)
    except ImportError:
        print("\ntorchvision not installed — skipping pretrained reference")

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved results to {out}")

    report_path = generate_report(
        results,
        settings={
            "mode": args.mode,
            "SUBSET_TRAIN": int(X_train.shape[0]) if not args.smoke else 4000,
            "EPOCHS": 1,
            "BATCH_SIZE": 128,
            "LEARNING_RATE": 1e-3,
            "dataset": "MNIST (mnist.npz) — inference-only for loaded weights",
        },
        assets_dir=os.path.join(HERE, "assets"),
        out_path=os.path.join(HERE, "REPORT.md"),
    )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
