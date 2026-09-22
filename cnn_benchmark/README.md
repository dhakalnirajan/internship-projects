# CNN Benchmark — 8 Architectures on MNIST (custom nn/ library)

Compares classic CNN architectures implemented **from scratch** with your
NumPy autograd library. All models train on the same MNIST split and are
compared by accuracy, parameter count, and training time.

> Note: true *pretrained* weights (ImageNet) exist only in PyTorch/TF format
> and cannot be loaded into a custom NumPy autograd engine, so this benchmark
> trains each architecture from scratch. The architectural *patterns*
> (residuals, inception, fire modules, dense connections) are preserved.

## Models

| Model | Key idea reproduced |
|---|---|
| LeNet-5 | original 1998 CNN |
| AlexNet-mini | big convs + dropout FC head |
| VGG16-mini | stacked 3x3 convs, double-then-pool |
| ResNet18-mini | residual blocks (concat shortcut) |
| GoogLeNet-mini | inception modules (parallel 1x1/3x3/pool) |
| DenseNet-mini | dense concatenation of all features |
| SqueezeNet-mini | fire modules (squeeze 1x1 → expand) |
| MobileNetV1-mini | bottleneck conv chain |

## Run

```bash
# quick correctness check (gradient checks + tiny train step on all models)
python -m cnn_benchmark.grad_check
python -m cnn_benchmark.smoke_test

# full benchmark (laptop-safe now; for zero-risk use Colab)
python -m cnn_benchmark.run_benchmark

# tiny smoke benchmark
python -m cnn_benchmark.run_benchmark --smoke

# pretrained-weights path: quick-train a PyTorch twin of MNIST-CNN,
# convert its weights into nn, benchmark nn inference-only (needs torch)
python -m cnn_benchmark.run_benchmark --mode pretrained

# same + LeNet-5-full / AlexNet-mirror twins, same-weights nn-vs-PyTorch
# latency comparison, and a torchvision pretrained (ImageNet) reference
# (needs torch + torchvision)
python -m cnn_benchmark.run_benchmark --mode pytorch
```

## Weight loading / converting (`torch_bridge.py`)

The `pretrained` and `pytorch` modes never train the nn side. Instead:

1. `specs.py` defines each mirror architecture **once** as a plain layer list;
   `build_nn_from_spec` builds it in nn and `build_torch_from_spec` builds the
   identical twin in PyTorch (parameter counts match exactly).
2. The PyTorch twin is quick-trained on MNIST (~1 min for MNIST-CNN) or its
   weights are reused from `cnn_benchmark/weights/*.npz`.
3. `load_weights_into_nn` converts the state_dict into the nn library:
   torch OIHW kernels → nn HWIO, transposed Linear weights, BatchNorm running
   stats via `sync_running_stats`, and a flatten-order permutation (torch
   flattens NCHW, nn flattens NHWC) on the first Linear after a Flatten.
4. Accuracy cross-check: the loaded nn model reproduces the PyTorch twin's
   test accuracy exactly (same weights, same data).
5. `pytorch` mode additionally benchmarks both engines on the same weights
   (inference latency, throughput, train-step) and pulls torchvision
   pretrained ImageNet models as a speed reference table.

The `REPORT.md` gains a conditional **`nn versus PyTorch (loaded weights)`**
section whenever cross-framework results are present in `results.json`.

Outputs: `cnn_benchmark/results.json` + summary table + **`cnn_benchmark/REPORT.md`** —
a dynamically generated Markdown report (run setup, methodology, ranked comparison,
per-model traces, plus a per-model section with accuracy trends and full 10x10
confusion matrices). Every section is conditional on the actual
output: latency tables only appear if the latency fields exist, divergence/collapsed-
run notes only appear if a model actually diverged, parameter columns only if the
counts were recorded, chart sections only for images that were saved (into
`cnn_benchmark/assets/` by the Colab notebook).

For Google Colab, open `cnn_benchmark/colab_cnn_benchmark.ipynb` — its final cells
save every chart to `cnn_benchmark/assets/`, generate `REPORT.md`, preview it in the
notebook, and download the report + charts + results as a zip.

## Library fixes included (these caused your laptop freeze)

1. **Conv2D im2col/col2im vectorized** — was O(N·L) Python loops per batch,
   now single vectorized gather + `np.add.at` scatter (~100x faster).
2. **Pooling vectorized** — max/avg pooling loops replaced with window
   gathering; max-argmax backward uses scatter-add.
3. **Autograd chain bug fixed** — `Dropout` and pooling returned
   `requires_grad=False` tensors, silently cutting gradients to earlier
   layers (your previous MNIST runs likely never trained the conv layers).
4. **CrossEntropy fixed** — `ndarray * Tensor` produced an object array and
   broke backprop; the Tensor now stays on the left of operators.
5. **Tensor.mean(axis=None)** returns a true numpy scalar while preserving
   the autograd graph.
6. **Conv 'same' padding math fixed** for stride > 1 (out-of-bounds index).
