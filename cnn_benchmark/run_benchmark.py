"""CNN Benchmark: compare 8 architectures on MNIST.

Usage:
    python -m cnn_benchmark.run_benchmark [--smoke]

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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
