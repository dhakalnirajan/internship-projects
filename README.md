# Internship Projects

This repository contains the source code for the projects completed during my internship.

## Projects

1. **Custom Neural Network Library** (`nn/`)
   Built from scratch using NumPy. It features a custom reverse-mode automatic
   differentiation engine (Autograd) inspired by PyTorch, supporting core autograd
   operations, custom implementations of dense, dropout, flatten, pooling, and
   convolutional layers (1D and 2D), sequential models, Adam and SGD optimizers,
   and standard activation functions including ReLU, sigmoid, tanh, and softmax.
   Convolution and pooling are vectorized (im2col/col2im), and gradient checks
   validate the analytic gradients against numerical differentiation.

2. **CNN Benchmark** (`cnn_benchmark/`)
   Compares 8 classic CNN architectures — LeNet-5, AlexNet-mini, VGG16-mini,
   ResNet18-mini, GoogLeNet-mini, DenseNet-mini, SqueezeNet-mini and
   MobileNetV1-mini — trained from scratch on MNIST with the custom `nn/`
   library, plus an inference-latency / throughput benchmark.

   ```bash
   python -m cnn_benchmark.grad_check        # gradient correctness
   python -m cnn_benchmark.smoke_test        # one tiny train step per model
   python -m cnn_benchmark.run_benchmark     # full benchmark
   ```

   Each run writes `results.json` **and a dynamically generated
   `REPORT.md`** — a Markdown report documenting what was run, how it was
   run, a ranked comparison table, training traces, and captioned charts.
   Every section of the report is conditional on the actual output:
   latency tables only if latency fields exist, divergence / collapsed-run
   notes only if a model actually diverged, parameter columns only if the
   counts were recorded, and chart sections only for images that were
   saved. A per-model section shows each architecture's accuracy trend per
   epoch and its full 10×10 confusion matrix whenever those were recorded.

   For Google Colab, open `cnn_benchmark/colab_cnn_benchmark.ipynb`: it
   clones the repo, runs gradient checks + smoke tests + the full
   benchmark, saves every chart into `cnn_benchmark/assets/`, generates
   and previews `REPORT.md`, and downloads the report, charts and results
   as a zip.

3. **Pothole Detection** (`pothole-detection/`)
   A computer vision application built using YOLOv8 for detecting potholes.
   The project pipeline covers dataset preparation, model training,
   evaluation, image inference, real-time detection, and model export
   workflows.

4. **Data Structures in C** (`data_structures/`)
   Implementations of a B+ tree (`bplus_tree.c`) and a B-tree (`btree.c`).

5. **Kimi K3 Mini Transformer** (`k3-mini/`)
   A dense transformer (Kimi K3) implemented in C, with an educational
   variant (`kimi_k3_edu`) that includes a live training monitor
   (metrics server + web dashboard) and hardware detection utilities.

## Data

- `mnist.npz` — MNIST dataset used by the `nn/` tests and the CNN benchmark.
- `Autism emotion recogition dataset.zip` — dataset for emotion recognition.
- `mnist_test.py` — MNIST smoke test for the `nn/` library.
