# CNN Benchmark Report

> Generated 2026-10-06 12:17:03 UTC · 8 architectures · best test accuracy **79.3%** (GoogLeNet-mini)

## Contents

- [Overview and settings](#overview-and-settings)
- [What was compared](#what-was-compared)
- [How it was run](#how-it-was-run)
- [Results](#results)
- [Head-to-head comparison](#head-to-head-comparison)
- [Training traces](#training-traces)
- [Charts](#charts)
- [Per-model confusion and accuracy trends](#per-model-confusion-and-accuracy-trends)
- [Notes and caveats](#notes-and-caveats)
- [Artifacts](#artifacts)

## Overview and settings

- **Generated:** 2026-10-06 12:17:03 UTC
- **Python:** 3.13.15 · **NumPy:** 2.1.3
- **Platform:** Linux-6.6.122+-x86_64-with-glibc2.39
- **Models benchmarked:** 8
- **Training samples:** 20,000
- **Epochs:** 5
- **Batch size:** 64
- **Learning rate:** 0.01
- **Dataset:** MNIST - 60k train / 10k test, 28x28x1, 10 classes
- **Optimizer:** Adam (lr from cell 3)
- **Loss:** cross-entropy

## What was compared

Eight classic CNN architectures rebuilt **from scratch** on the custom NumPy autograd library (`nn/`), all on the same data split and the same training harness — so differences reflect the architectural *patterns*, not the plumbing.

| Model | Key idea reproduced | Params |
|---|---|---|
| GoogLeNet-mini | inception modules (parallel 1x1 / 3x3 / pool branches) | 36,974 |
| DenseNet-mini | dense concatenation of every preceding feature map | 28,778 |
| SqueezeNet-mini | fire modules: 1x1 squeeze -> 1x1 + 3x3 expand | 24,066 |
| ResNet18-mini | residual blocks with shortcut connections | 97,802 |
| LeNet-5 | the original 1998 CNN (conv -> pool -> fully-connected) | 61,706 |
| AlexNet-mini | big convs + dropout fully-connected head | 1,733,258 |
| VGG16-mini | stacked 3x3 convs, channels double while pooling halves | 443,386 |
| MobileNetV1-mini | depthwise-separable style bottleneck conv chain | 219,570 |

## How it was run

- **Library:** `nn/` — a from-scratch NumPy autograd engine (vectorized im2col `Conv2D`, vectorized pooling, reverse-mode autograd). Gradient checks (analytic vs numerical) and a one-step smoke test on every architecture ran before the benchmark.
- **Data:** MNIST images scaled to [0, 1] and shaped `(N, 28, 28, 1)`, split into train / held-out validation / test as stored in `mnist.npz`.
- **Training:** Adam (lr=0.01, mini-batches of 64, 5 epoch(s), 20,000 training samples) minimising cross-entropy; each model gets one warm-up forward pass to materialise lazily-built layers before its parameters are counted and the timed training loop starts.
- **Evaluation:** validation accuracy after every epoch on a held-out slice; the headline number is final test accuracy on the test split.
- **Latency protocol:** single-image inference timed with warm-up + 30 runs (mean and p95 reported), batch-256 throughput averaged over 5 passes, and a full train step (forward + backward, batch 64).

## Results

| # | Model | Params | Final loss | Val acc | Test acc | Train (s) | Infer (ms) | p95 (ms) | Throughput (img/s) | Train step (ms) | Δ vs best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GoogLeNet-mini | 36,974 | 2.008 | 81.0% | 79.3% | 427.7 | 7.29 | 8.84 | 59 | 1101.8 | &mdash; |
| 2 | DenseNet-mini | 28,778 | 1.027 | 70.4% | 66.6% | 634.0 | 15.93 | 18.79 | 40 | 1509.4 | -12.7 pp |
| 3 | SqueezeNet-mini | 24,066 | 1.676 | 58.4% | 54.5% | 275.0 | 5.04 | 5.24 | 92 | 675.6 | -24.8 pp |
| 4 | ResNet18-mini | 97,802 | 1.975 | 41.3% | 36.3% | 255.2 | 8.09 | 14.41 | 103 | 566.0 | -43.0 pp |
| 5 | LeNet-5 | 61,706 | 12.552 | 29.4% | 30.2% | 33.2 | 1.75 | 2.02 | 982 | 101.0 | -49.1 pp |
| 6 | AlexNet-mini | 1,733,258 | 16.290 | 9.3% | 11.6% | 549.7 | 8.98 | 12.86 | 45 | 1913.6 | -67.7 pp |
| 7 | VGG16-mini | 443,386 | 16.675 | 9.6% | 9.4% | 465.4 | 5.97 | 7.09 | 56 | 1451.9 | -69.9 pp |
| 8 | MobileNetV1-mini | 219,570 | 16.521 | 10.8% | 8.9% | 120.7 | 3.91 | 4.47 | 241 | 477.1 | -70.4 pp |

*Ranked by test accuracy. Δ is the gap to the best model in percentage points (pp).*

## Head-to-head comparison

- **Accuracy leader:** **GoogLeNet-mini** at 79.3% test accuracy, +12.7 pp ahead of DenseNet-mini; MobileNetV1-mini is last at 8.9% (70.4 pp behind the winner).
- **Learned vs. didn't:** 5 of 8 models clearly learned the task (>15% test acc); AlexNet-mini, VGG16-mini, MobileNetV1-mini finished at chance level (10.0%).
- **Convergence:** lowest final loss DenseNet-mini (1.027) vs highest VGG16-mini (16.675); loss *rose* during training for AlexNet-mini, VGG16-mini — those diverged.
- **Cost:** LeNet-5 trained fastest (33.2s), DenseNet-mini slowest (634.0s) — a 19× spread for identical data and epochs.
- **Most efficient:** LeNet-5 buys the most accuracy per training second (0.0091 acc/s).
- **Size:** SqueezeNet-mini is smallest (24,066 params), AlexNet-mini largest (1,733,258); DenseNet-mini extracts the most accuracy per parameter.
- **Latency:** LeNet-5 serves one image in 1.75ms on average, DenseNet-mini needs 15.93ms (9× apart); best throughput LeNet-5 at 982 img/s (batch 256).

## Training traces

| Model | Epochs | Loss (start → end) | Val acc (start → end) | Test acc |
|---|---|---|---|---|
| GoogLeNet-mini | 5 | 3.439 → 2.008 (-42%) | 64.3% → 81.0% | 79.3% |
| DenseNet-mini | 5 | 5.712 → 1.027 (-82%) | 15.7% → 70.4% | 66.6% |
| SqueezeNet-mini | 5 | 2.274 → 1.676 (-26%) | 34.9% → 58.4% | 54.5% |
| ResNet18-mini | 5 | 7.462 → 1.975 (-74%) | 17.7% → 41.3% | 36.3% |
| LeNet-5 | 5 | 14.177 → 12.552 (-11%) | 10.4% → 29.4% | 30.2% |
| AlexNet-mini | 5 | 15.494 → 16.290 (+5%) | 10.2% → 9.3% | 11.6% |
| VGG16-mini | 5 | 16.437 → 16.675 (+1%) | 10.2% → 9.6% | 9.4% |
| MobileNetV1-mini | 5 | 16.570 → 16.521 (-0%) | 10.8% → 10.8% | 8.9% |

## Charts

### Accuracy cost and convergence

Four views of the same race: test accuracy, parameter count, training time, and validation accuracy per epoch. **GoogLeNet-mini** leads at 79.3%; the field spans 8.9%–79.3%. Size ranges from SqueezeNet-mini (24,066 params) to AlexNet-mini (1,733,258 params). Slowest to train: DenseNet-mini (634.0s).

![Accuracy cost and convergence](assets/charts_bars.png)

### Training loss curves

Cross-entropy per epoch on a log scale. **DenseNet-mini** converges lowest at 1.027. Loss *increased* over training for AlexNet-mini, VGG16-mini — those runs diverged.

![Training loss curves](assets/charts_loss_curves.png)

### Latency and throughput

**LeNet-5** answers one image in 1.75ms on average; **DenseNet-mini** is slowest at 15.93ms — a 9× spread. At batch 256, LeNet-5 pushes the most images/s (982). Cheapest training step: LeNet-5 (101.0ms).

![Latency and throughput](assets/charts_latency.png)

### Accuracy versus cost

Accuracy plotted against model size and against training time — the top-left/top-right corners are the efficient frontier. Parameter counts span 24,066–1,733,258. Most accuracy per training second: **LeNet-5** (0.0091 acc/s); overall accuracy leader is **GoogLeNet-mini**.

![Accuracy versus cost](assets/charts_pareto.png)

### Confusion matrices

Confusion matrices for the top 4 models by test accuracy — GoogLeNet-mini (79.3%), DenseNet-mini (66.6%), SqueezeNet-mini (54.5%), ResNet18-mini (36.3%). Off-diagonal cells show where each model confuses digits.

![Confusion matrices](assets/charts_confusion.png)

### Per-class accuracy

Per-digit accuracy of the winning model, **GoogLeNet-mini** (overall 79.3%). Bars reveal which digits are individually hard — 4/7/9 are classically the messy ones on MNIST.

![Per-class accuracy](assets/charts_per_class.png)

### Sample predictions

Random test images with GoogLeNet-mini's predictions — green = correct, red = wrong (overall 79.3%).

![Sample predictions](assets/charts_samples.png)

### Misclassifications

Examples GoogLeNet-mini gets wrong — useful for judging whether errors look genuinely ambiguous (4 vs 9) or systematic.

![Misclassifications](assets/charts_mistakes.png)

### charts accuracy bar

![charts accuracy bar](assets/charts_accuracy_bar.png)

### charts accuracy vs latency

![charts accuracy vs latency](assets/charts_accuracy_vs_latency.png)

### charts confusion DenseNet mini

![charts confusion DenseNet mini](assets/charts_confusion_DenseNet_mini.png)

### charts confusion GoogLeNet mini

![charts confusion GoogLeNet mini](assets/charts_confusion_GoogLeNet_mini.png)

### charts confusion ResNet18 mini

![charts confusion ResNet18 mini](assets/charts_confusion_ResNet18_mini.png)

### charts confusion SqueezeNet mini

![charts confusion SqueezeNet mini](assets/charts_confusion_SqueezeNet_mini.png)

### charts latency only

![charts latency only](assets/charts_latency_only.png)

### charts parameters bar

![charts parameters bar](assets/charts_parameters_bar.png)

### charts pareto size only

![charts pareto size only](assets/charts_pareto_size_only.png)

### charts pareto time only

![charts pareto time only](assets/charts_pareto_time_only.png)

### charts throughput only

![charts throughput only](assets/charts_throughput_only.png)

### charts traintime bar

![charts traintime bar](assets/charts_traintime_bar.png)

### charts validation curves

![charts validation curves](assets/charts_validation_curves.png)

## Per-model confusion and accuracy trends

Per model: how accuracy moved epoch over epoch (classified from the actual deltas) and — where confusion matrices were recorded — the full 10×10 confusion matrix with the dominant error pair and hardest digit.

### GoogLeNet-mini

**Accuracy trend:** train 63.6% → 81.5%, val 64.3% → 81.0% over 5 epochs — **strongly improving** (+16.7 pp), still climbing at the final epoch.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **81.6%** accuracy (8,156/10,000 test images); most confounded pair **4 → 9** (744 images); hardest digit **4** (recall 0.0%), easiest 0 (97.6%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 956 | 0 | 3 | 0 | 0 | 0 | 9 | 2 | 10 | 0 |
| 1 | 0 | 1101 | 18 | 2 | 0 | 0 | 2 | 1 | 10 | 1 |
| 2 | 4 | 0 | 981 | 1 | 0 | 1 | 8 | 14 | 19 | 4 |
| 3 | 4 | 0 | 109 | 781 | 0 | 22 | 0 | 38 | 52 | 4 |
| 4 | 1 | 1 | 26 | 0 | 0 | 1 | 84 | 58 | 67 | 744 |
| 5 | 13 | 1 | 23 | 42 | 0 | 701 | 8 | 6 | 85 | 13 |
| 6 | 12 | 5 | 21 | 0 | 0 | 7 | 895 | 9 | 8 | 1 |
| 7 | 0 | 4 | 41 | 0 | 0 | 0 | 0 | 958 | 4 | 21 |
| 8 | 8 | 4 | 39 | 5 | 0 | 3 | 1 | 23 | 888 | 3 |
| 9 | 6 | 4 | 10 | 5 | 0 | 2 | 1 | 61 | 25 | 895 |

### DenseNet-mini

**Accuracy trend:** train 17.2% → 70.8%, val 15.7% → 70.4% over 5 epochs — **strongly improving** (+54.8 pp), still climbing at the final epoch.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **69.3%** accuracy (6,928/10,000 test images); most confounded pair **4 → 9** (483 images); hardest digit **5** (recall 25.0%), easiest 1 (93.4%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 908 | 1 | 0 | 0 | 5 | 0 | 7 | 9 | 5 | 45 |
| 1 | 4 | 1060 | 4 | 3 | 1 | 1 | 1 | 0 | 3 | 58 |
| 2 | 28 | 1 | 873 | 0 | 8 | 0 | 20 | 56 | 25 | 21 |
| 3 | 53 | 21 | 95 | 527 | 0 | 0 | 1 | 52 | 59 | 202 |
| 4 | 3 | 16 | 8 | 0 | 428 | 0 | 14 | 12 | 18 | 483 |
| 5 | 63 | 67 | 10 | 95 | 5 | 223 | 22 | 32 | 161 | 214 |
| 6 | 86 | 38 | 8 | 0 | 20 | 0 | 728 | 10 | 6 | 62 |
| 7 | 0 | 37 | 26 | 0 | 1 | 0 | 0 | 804 | 16 | 144 |
| 8 | 31 | 64 | 21 | 7 | 9 | 1 | 5 | 20 | 454 | 362 |
| 9 | 15 | 33 | 2 | 0 | 3 | 0 | 0 | 18 | 15 | 923 |

### SqueezeNet-mini

**Accuracy trend:** train 35.9% → 62.4%, val 34.9% → 58.4% over 5 epochs — **strongly improving** (+23.4 pp), peaked at epoch 4.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **60.3%** accuracy (6,032/10,000 test images); most confounded pair **4 → 9** (387 images); hardest digit **4** (recall 32.8%), easiest 2 (93.7%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 388 | 4 | 350 | 0 | 0 | 1 | 0 | 123 | 51 | 63 |
| 1 | 0 | 939 | 12 | 0 | 0 | 0 | 0 | 132 | 5 | 47 |
| 2 | 0 | 4 | 967 | 1 | 0 | 0 | 0 | 44 | 13 | 3 |
| 3 | 1 | 3 | 254 | 413 | 0 | 2 | 0 | 194 | 88 | 55 |
| 4 | 0 | 27 | 90 | 0 | 322 | 0 | 3 | 82 | 71 | 387 |
| 5 | 3 | 13 | 116 | 43 | 0 | 403 | 8 | 87 | 114 | 105 |
| 6 | 4 | 30 | 297 | 0 | 0 | 15 | 380 | 53 | 63 | 116 |
| 7 | 0 | 21 | 32 | 0 | 0 | 0 | 0 | 926 | 30 | 19 |
| 8 | 0 | 13 | 52 | 13 | 0 | 1 | 0 | 116 | 564 | 215 |
| 9 | 0 | 5 | 18 | 5 | 1 | 0 | 0 | 182 | 68 | 730 |

### ResNet18-mini

**Accuracy trend:** train 19.7% → 43.0%, val 17.7% → 41.3% over 5 epochs — **strongly improving** (+23.6 pp), still climbing at the final epoch.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **41.5%** accuracy (4,150/10,000 test images); most confounded pair **9 → 4** (405 images); hardest digit **6** (recall 7.0%), easiest 1 (92.4%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 216 | 32 | 124 | 15 | 395 | 121 | 24 | 26 | 1 | 26 |
| 1 | 0 | 1049 | 7 | 4 | 2 | 21 | 0 | 47 | 0 | 5 |
| 2 | 17 | 46 | 346 | 166 | 195 | 183 | 32 | 5 | 35 | 7 |
| 3 | 6 | 30 | 44 | 360 | 71 | 384 | 2 | 52 | 29 | 32 |
| 4 | 23 | 9 | 91 | 3 | 549 | 160 | 42 | 23 | 7 | 75 |
| 5 | 11 | 58 | 82 | 87 | 84 | 508 | 3 | 20 | 22 | 17 |
| 6 | 21 | 103 | 381 | 2 | 242 | 89 | 67 | 0 | 17 | 36 |
| 7 | 1 | 70 | 29 | 39 | 182 | 96 | 0 | 499 | 5 | 107 |
| 8 | 10 | 53 | 27 | 22 | 159 | 332 | 15 | 2 | 275 | 79 |
| 9 | 3 | 26 | 34 | 16 | 405 | 129 | 4 | 98 | 13 | 281 |

### LeNet-5

**Accuracy trend:** train 10.9% → 30.4%, val 10.4% → 29.4% over 5 epochs — **strongly improving** (+19.0 pp), still climbing at the final epoch.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **30.3%** accuracy (3,029/10,000 test images); most confounded pair **7 → 9** (897 images); hardest digit **0** (recall 0.0%), easiest 1 (99.3%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 4 | 0 | 0 | 0 | 0 | 482 | 0 | 0 | 494 |
| 1 | 0 | 1127 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| 2 | 0 | 344 | 0 | 0 | 0 | 0 | 484 | 0 | 0 | 204 |
| 3 | 0 | 377 | 0 | 0 | 0 | 0 | 237 | 0 | 0 | 396 |
| 4 | 0 | 43 | 0 | 0 | 0 | 0 | 90 | 0 | 0 | 849 |
| 5 | 0 | 153 | 0 | 0 | 0 | 0 | 318 | 0 | 0 | 421 |
| 6 | 0 | 4 | 0 | 0 | 0 | 0 | 945 | 0 | 0 | 9 |
| 7 | 0 | 128 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 897 |
| 8 | 0 | 361 | 0 | 0 | 0 | 0 | 169 | 0 | 0 | 444 |
| 9 | 0 | 48 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 957 |

### AlexNet-mini

**Accuracy trend:** train 11.0% → 9.9%, val 10.2% → 9.3% over 5 epochs — **plateaued** (-0.9 pp), peaked at epoch 1.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **10.3%** accuracy (1,032/10,000 test images); most confounded pair **1 → 2** (1135 images); hardest digit **0** (recall 0.0%), easiest 2 (100.0%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 980 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1135 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1032 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 1010 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 982 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 892 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 958 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 1028 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 974 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 0 | 0 | 1009 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### VGG16-mini

**Accuracy trend:** train 10.0% → 10.5%, val 10.2% → 9.6% over 5 epochs — **plateaued** (-0.6 pp), peaked at epoch 1.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **10.1%** accuracy (1,009/10,000 test images); most confounded pair **1 → 9** (1135 images); hardest digit **0** (recall 0.0%), easiest 9 (100.0%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 980 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1135 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1032 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1010 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 982 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 892 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 958 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1028 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 974 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1009 |

### MobileNetV1-mini

**Accuracy trend:** train 8.6% → 8.6%, val 10.8% → 10.8% over 5 epochs — **plateaued** (+0.0 pp), peaked at epoch 1.

**Confusion:** Rows are the true digit, columns the prediction; the diagonal reproduces **9.7%** accuracy (974/10,000 test images); most confounded pair **1 → 8** (1135 images); hardest digit **0** (recall 0.0%), easiest 8 (100.0%).

| true \ pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 980 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1135 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1032 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1010 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 982 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 892 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 958 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1028 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 974 | 0 |
| 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1009 | 0 |

## Notes and caveats

- **From scratch, not pretrained:** ImageNet weights exist only in PyTorch/TF formats and cannot be loaded into a pure-NumPy autograd engine, so every architecture trains from random init; the comparison is about architectural *patterns*, not pretrained accuracy.
- **Collapsed runs:** AlexNet-mini, VGG16-mini, MobileNetV1-mini ended at chance level. On 28×28 inputs over few epochs this usually means the optimizer never got traction (learning rate, depth, initialization) rather than that the pattern is wrong — see the loss curves above.
- **Diverging runs:** loss increased for AlexNet-mini, VGG16-mini; lower the learning rate or train longer before trusting their numbers.
- **Reproducibility:** `python -m cnn_benchmark.run_benchmark` (or the Colab notebook, cells in order) with the settings listed above regenerates these results; all numbers here are computed from `results.json`, never hard-coded.

## Artifacts

- `REPORT.md` — this report
- `results.json` — raw metrics behind every table and caption
- `assets/charts_accuracy_bar.png` — chart image
- `assets/charts_accuracy_vs_latency.png` — chart image
- `assets/charts_bars.png` — chart image
- `assets/charts_confusion.png` — chart image
- `assets/charts_confusion_DenseNet_mini.png` — chart image
- `assets/charts_confusion_GoogLeNet_mini.png` — chart image
- `assets/charts_confusion_ResNet18_mini.png` — chart image
- `assets/charts_confusion_SqueezeNet_mini.png` — chart image
- `assets/charts_latency.png` — chart image
- `assets/charts_latency_only.png` — chart image
- `assets/charts_loss_curves.png` — chart image
- `assets/charts_mistakes.png` — chart image
- `assets/charts_parameters_bar.png` — chart image
- `assets/charts_pareto.png` — chart image
- `assets/charts_pareto_size_only.png` — chart image
- `assets/charts_pareto_time_only.png` — chart image
- `assets/charts_per_class.png` — chart image
- `assets/charts_samples.png` — chart image
- `assets/charts_throughput_only.png` — chart image
- `assets/charts_traintime_bar.png` — chart image
- `assets/charts_validation_curves.png` — chart image
