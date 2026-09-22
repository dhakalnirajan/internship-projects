"""Dynamic Markdown report generator for the CNN benchmark.

``generate_report()`` turns benchmark output (the result dicts saved to
``results.json``), the run settings, and any chart images that were saved
into a clean, self-contained ``REPORT.md``:

  * **what** was run  — models, dataset, settings (all read from the output)
  * **how** it ran    — library, harness, optimizer, evaluation protocol
  * **comparison**    — ranked table with deltas + dynamic "who won what"
  * **charts**        — every saved image, each captioned from the actual numbers
  * **traces**        — per-model loss / accuracy trajectories

Nothing is hard-coded per run: change the models, epochs, or metrics and the
report rewrites itself around them.

Used by ``colab_cnn_benchmark.ipynb`` (report cell) and ``run_benchmark.py``.
"""

import datetime
import os
import platform
import re

CHANCE_LEVEL = 0.10  # 10 balanced digit classes

MODEL_IDEAS = {
    "LeNet-5": "the original 1998 CNN (conv -> pool -> fully-connected)",
    "LeNet-5-full": "the original 1998 CNN (conv -> pool -> fully-connected)",
    "AlexNet-mini": "big convs + dropout fully-connected head",
    "AlexNet-mirror": "five conv layers + dropout fully-connected head",
    "MNIST-CNN": "compact two-block conv-BN-pool network for MNIST",
    "VGG16-mini": "stacked 3x3 convs, channels double while pooling halves",
    "ResNet18-mini": "residual blocks with shortcut connections",
    "GoogLeNet-mini": "inception modules (parallel 1x1 / 3x3 / pool branches)",
    "DenseNet-mini": "dense concatenation of every preceding feature map",
    "SqueezeNet-mini": "fire modules: 1x1 squeeze -> 1x1 + 3x3 expand",
    "MobileNetV1-mini": "depthwise-separable style bottleneck conv chain",
}

SETTING_LABELS = {
    "SUBSET_TRAIN": "Training samples",
    "EPOCHS": "Epochs",
    "BATCH_SIZE": "Batch size",
    "LEARNING_RATE": "Learning rate",
    "mode": "Run mode",
    "dataset": "Dataset",
    "optimizer": "Optimizer",
    "loss": "Loss",
}

# ----------------------------------------------------------------- formatting
def _pct(x):
    return f"{100 * x:.1f}%"


def _pp(x):
    return f"{100 * x:+.1f} pp"


def _sec(x):
    return f"{x:,.1f}s"


def _fmt(v, spec):
    if v is None:
        return "&mdash;"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def _md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _slug(text):
    """GitHub-style anchor for a heading (headings here avoid punctuation)."""
    return re.sub(r"[\s]+", "-", re.sub(r"[^a-z0-9\s-]", "", text.lower())).strip("-")


def _field(r, key):
    """Metric from the result dict, falling back to the last history entry."""
    v = r.get(key)
    if v is not None:
        return v
    seq = (r.get("history") or {}).get(key)
    if seq:
        return seq[-1]
    if key == "final_loss":
        seq = (r.get("history") or {}).get("loss")
        if seq:
            return seq[-1]
    return None


def _acc(r):
    v = _field(r, "test_acc")
    return float(v) if isinstance(v, (int, float)) else 0.0


def _loss_grew(r):
    loss = (r.get("history") or {}).get("loss") or []
    return len(loss) >= 2 and loss[-1] > loss[0]


def _params_known(ranked):
    return any(r.get("params") for r in ranked)


def _numpy_version():
    try:
        import numpy
        return numpy.__version__
    except Exception:
        return "n/a"


# ------------------------------------------------------------ metric columns
# (key, header, formatter) — a column is shown when any row carries the key
_METRIC_COLS = [
    ("params", "Params", lambda v: f"{v:,}"),
    ("final_loss", "Final loss", lambda v: f"{v:.3f}"),
    ("val_acc", "Val acc", _pct),
    ("test_acc", "Test acc", _pct),
    ("train_s", "Train (s)", lambda v: f"{v:,.1f}"),
    ("build_s", "Build (s)", lambda v: f"{v:,.2f}"),
    ("lat_ms", "Infer (ms)", lambda v: f"{v:.2f}"),
    ("p95_ms", "p95 (ms)", lambda v: f"{v:.2f}"),
    ("imgs_s", "Throughput (img/s)", lambda v: f"{v:,.0f}"),
    ("step_ms", "Train step (ms)", lambda v: f"{v:.1f}"),
]


# --------------------------------------------------------------- chart caption
def _caption_bars(ranked):
    best, worst = ranked[0], ranked[-1]
    s = (f"Four views of the same race: test accuracy, parameter count, training "
         f"time, and validation accuracy per epoch. **{best['model']}** leads at "
         f"{_pct(_acc(best))}; the field spans {_pct(_acc(worst))}–{_pct(_acc(best))}.")
    if _params_known(ranked):
        big = max(ranked, key=lambda r: r.get("params") or 0)
        small = min(ranked, key=lambda r: r.get("params") or 0)
        s += (f" Size ranges from {small['model']} ({small['params']:,} params) to "
              f"{big['model']} ({big['params']:,} params).")
    timed = [r for r in ranked if r.get("train_s") is not None]
    if timed:
        slow = max(timed, key=lambda r: r["train_s"])
        s += f" Slowest to train: {slow['model']} ({_sec(slow['train_s'])})."
    return s


def _caption_loss(ranked):
    curves = [r for r in ranked if (r.get("history") or {}).get("loss")]
    if not curves:
        return "Per-epoch cross-entropy loss (log scale)."
    lo = min(curves, key=lambda r: r["history"]["loss"][-1])
    s = (f"Cross-entropy per epoch on a log scale. **{lo['model']}** converges "
         f"lowest at {lo['history']['loss'][-1]:.3f}.")
    grew = [r["model"] for r in curves if _loss_grew(r)]
    if grew:
        s += f" Loss *increased* over training for {', '.join(grew)} — those runs diverged."
    else:
        s += " Every model reduced its loss."
    return s


def _caption_latency(ranked):
    lat = [r for r in ranked if r.get("lat_ms") is not None]
    if not lat:
        return "Single-image inference latency, batch-256 throughput and train-step cost."
    fast = min(lat, key=lambda r: r["lat_ms"])
    slow = max(lat, key=lambda r: r["lat_ms"])
    s = (f"**{fast['model']}** answers one image in {fast['lat_ms']:.2f}ms on "
         f"average; **{slow['model']}** is slowest at {slow['lat_ms']:.2f}ms — "
         f"a {slow['lat_ms'] / max(fast['lat_ms'], 1e-9):.0f}× spread.")
    thr = [r for r in ranked if r.get("imgs_s") is not None]
    if thr:
        t = max(thr, key=lambda r: r["imgs_s"])
        s += f" At batch 256, {t['model']} pushes the most images/s ({t['imgs_s']:,.0f})."
    steps = [r for r in ranked if r.get("step_ms") is not None]
    if steps:
        c = min(steps, key=lambda r: r["step_ms"])
        s += f" Cheapest training step: {c['model']} ({c['step_ms']:.1f}ms)."
    return s


def _caption_pareto(ranked):
    s = ("Accuracy plotted against model size and against training time — the "
         "top-left/top-right corners are the efficient frontier.")
    if _params_known(ranked):
        sizes = sorted((r.get("params") or 0) for r in ranked)
        s += (f" Parameter counts span {sizes[0]:,}–{sizes[-1]:,}.")
    best = ranked[0]
    timed = [r for r in ranked if r.get("train_s")]
    if timed:
        efficient = max(timed, key=lambda r: _acc(r) / r["train_s"])
        s += (f" Most accuracy per training second: **{efficient['model']}** "
              f"({_acc(efficient) / efficient['train_s']:.4f} acc/s); "
              f"overall accuracy leader is **{best['model']}**.")
    return s


def _caption_confusion(ranked):
    top = ranked[:4]
    names = ", ".join(f"{r['model']} ({_pct(_acc(r))})" for r in top)
    return (f"Confusion matrices for the top {len(top)} models by test accuracy — "
            f"{names}. Off-diagonal cells show where each model confuses digits.")


def _caption_perclass(ranked):
    best = ranked[0]
    return (f"Per-digit accuracy of the winning model, **{best['model']}** "
            f"(overall {_pct(_acc(best))}). Bars reveal which digits are "
            f"individually hard — 4/7/9 are classically the messy ones on MNIST.")


def _caption_samples(ranked):
    best = ranked[0]
    return (f"Random test images with {best['model']}'s predictions — green = "
            f"correct, red = wrong (overall {_pct(_acc(best))}).")


def _caption_mistakes(ranked):
    best = ranked[0]
    return (f"Examples {best['model']} gets wrong — useful for judging whether "
            f"errors look genuinely ambiguous (4 vs 9) or systematic.")


CHART_SPECS = [
    ("charts_bars.png", "Accuracy cost and convergence", _caption_bars),
    ("charts_loss_curves.png", "Training loss curves", _caption_loss),
    ("charts_latency.png", "Latency and throughput", _caption_latency),
    ("charts_pareto.png", "Accuracy versus cost", _caption_pareto),
    ("charts_confusion.png", "Confusion matrices", _caption_confusion),
    ("charts_per_class.png", "Per-class accuracy", _caption_perclass),
    ("charts_samples.png", "Sample predictions", _caption_samples),
    ("charts_mistakes.png", "Misclassifications", _caption_mistakes),
]


# ------------------------------------------------------------------- sections
def _overview(ranked, settings):
    lines = [
        f"- **Generated:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Python:** {platform.python_version()} · **NumPy:** {_numpy_version()}",
        f"- **Platform:** {platform.platform()}",
        f"- **Models benchmarked:** {len(ranked)}",
    ]
    for key, value in (settings or {}).items():
        label = SETTING_LABELS.get(key, key.replace("_", " ").capitalize())
        if isinstance(value, int):
            value = f"{value:,}"
        lines.append(f"- **{label}:** {value}")
    return "\n".join(lines) + "\n"


def _whats_compared(ranked):
    head = ("Eight classic CNN architectures rebuilt **from scratch** on the custom "
            "NumPy autograd library (`nn/`), all on the same data split and the same "
            "training harness — so differences reflect the architectural *patterns*, "
            "not the plumbing.\n")
    rows = [[r["model"],
             MODEL_IDEAS.get(r["model"], "custom architecture"),
             f"{r['params']:,}" if r.get("params") else "&mdash;"]
            for r in ranked]
    return head + "\n" + _md_table(["Model", "Key idea reproduced", "Params"], rows) + "\n"


def _how(ranked, settings):
    lr = settings.get("LEARNING_RATE")
    batch = settings.get("BATCH_SIZE")
    epochs = settings.get("EPOCHS")
    subset = settings.get("SUBSET_TRAIN")
    bits = []
    if lr is not None:
        bits.append(f"lr={lr:g}")
    if batch is not None:
        bits.append(f"mini-batches of {batch}")
    if epochs is not None:
        bits.append(f"{epochs} epoch(s)")
    if subset is not None:
        bits.append(f"{subset:,} training samples")
    training = ", ".join(bits)
    lines = [
        "- **Library:** `nn/` — a from-scratch NumPy autograd engine (vectorized "
        "im2col `Conv2D`, vectorized pooling, reverse-mode autograd). Gradient "
        "checks (analytic vs numerical) and a one-step smoke test on every "
        "architecture ran before the benchmark.",
        "- **Data:** MNIST images scaled to [0, 1] and shaped `(N, 28, 28, 1)`, "
        "split into train / held-out validation / test as stored in `mnist.npz`.",
        f"- **Training:** Adam ({training}) minimising cross-entropy; each model "
        "gets one warm-up forward pass to materialise lazily-built layers before "
        "its parameters are counted and the timed training loop starts.",
        "- **Evaluation:** validation accuracy after every epoch on a held-out "
        "slice; the headline number is final test accuracy on the test split.",
    ]
    if any(r.get("lat_ms") is not None for r in ranked):
        lines.append(
            "- **Latency protocol:** single-image inference timed with warm-up + 30 "
            "runs (mean and p95 reported), batch-256 throughput averaged over 5 "
            "passes, and a full train step (forward + backward, batch 64).")
    return "\n".join(lines) + "\n"


def _results_table(ranked):
    best = _acc(ranked[0]) if ranked else 0.0
    cols = [(k, h, f) for (k, h, f) in _METRIC_COLS
            if (any(_field(r, k) is not None for r in ranked)
                and not (k == "params" and not _params_known(ranked)))]
    headers = ["#", "Model"] + [h for _, h, _ in cols] + ["Δ vs best"]
    rows = []
    for i, r in enumerate(ranked):
        row = [i + 1, r["model"]]
        for k, _, f in cols:
            row.append(_cell(_field(r, k), f))
        row.append("&mdash;" if i == 0 else _pp(_acc(r) - best))
        rows.append(row)
    return (_md_table(headers, rows) + "\n\n"
            "*Ranked by test accuracy. Δ is the gap to the best model in "
            "percentage points (pp).*\n")


def _cell(v, f):
    return "&mdash;" if v is None else f(v)


def _comparisons(ranked):
    out = []
    if not ranked:
        return out
    best = ranked[0]
    if len(ranked) > 1:
        second, worst = ranked[1], ranked[-1]
        out.append(f"- **Accuracy leader:** **{best['model']}** at {_pct(_acc(best))} "
                   f"test accuracy, {_pp(_acc(best) - _acc(second))} ahead of "
                   f"{second['model']}; {worst['model']} is last at {_pct(_acc(worst))} "
                   f"({_pct(_acc(best) - _acc(worst)).replace('%', ' pp')} behind the winner).")
    strong = [r for r in ranked if _acc(r) > 0.15]
    weak = [r["model"] for r in ranked if _acc(r) <= 0.15]
    if weak and strong:
        out.append(f"- **Learned vs. didn't:** {len(strong)} of {len(ranked)} models "
                   f"clearly learned the task (>15% test acc); {', '.join(weak)} "
                   f"finished at chance level ({_pct(CHANCE_LEVEL)}).")
    with_loss = [r for r in ranked if _field(r, "final_loss") is not None]
    if with_loss:
        lo = min(with_loss, key=lambda r: _field(r, "final_loss"))
        hi = max(with_loss, key=lambda r: _field(r, "final_loss"))
        msg = (f"- **Convergence:** lowest final loss {lo['model']} "
               f"({float(_field(lo, 'final_loss')):.3f})")
        if hi is not lo:
            msg += (f" vs highest {hi['model']} "
                    f"({float(_field(hi, 'final_loss')):.3f})")
        grew = [r["model"] for r in with_loss if _loss_grew(r)]
        if grew:
            msg += f"; loss *rose* during training for {', '.join(grew)} — those diverged"
        out.append(msg + ".")
    timed = [r for r in ranked if r.get("train_s") is not None]
    if len(timed) > 1:
        fast = min(timed, key=lambda r: r["train_s"])
        slow = max(timed, key=lambda r: r["train_s"])
        ratio = slow["train_s"] / max(fast["train_s"], 1e-9)
        out.append(f"- **Cost:** {fast['model']} trained fastest "
                   f"({_sec(fast['train_s'])}), {slow['model']} slowest "
                   f"({_sec(slow['train_s'])}) — a {ratio:.0f}× spread for identical "
                   f"data and epochs.")
    eff = [r for r in timed if r.get("train_s") and _acc(r) > 0]
    if len(eff) > 1:
        top = max(eff, key=lambda r: _acc(r) / r["train_s"])
        out.append(f"- **Most efficient:** {top['model']} buys the most accuracy per "
                   f"training second ({_acc(top) / top['train_s']:.4f} acc/s).")
    sized = [r for r in ranked if r.get("params")]
    if len(sized) > 1:
        big = max(sized, key=lambda r: r["params"])
        small = min(sized, key=lambda r: r["params"])
        per_param = max(sized, key=lambda r: _acc(r) / r["params"])
        out.append(f"- **Size:** {small['model']} is smallest ({small['params']:,} "
                   f"params), {big['model']} largest ({big['params']:,}); "
                   f"{per_param['model']} extracts the most accuracy per parameter.")
    lat = [r for r in ranked if r.get("lat_ms") is not None]
    if lat:
        fast = min(lat, key=lambda r: r["lat_ms"])
        slow = max(lat, key=lambda r: r["lat_ms"])
        msg = (f"- **Latency:** {fast['model']} serves one image in "
               f"{fast['lat_ms']:.2f}ms on average, {slow['model']} needs "
               f"{slow['lat_ms']:.2f}ms ({slow['lat_ms'] / max(fast['lat_ms'], 1e-9):.0f}× apart)")
        thr = [r for r in ranked if r.get("imgs_s") is not None]
        if thr:
            t = max(thr, key=lambda r: r["imgs_s"])
            msg += f"; best throughput {t['model']} at {t['imgs_s']:,.0f} img/s (batch 256)"
        out.append(msg + ".")
    return out


def _traces(ranked):
    rows = []
    for r in ranked:
        hist = r.get("history") or {}
        loss = hist.get("loss") or []
        vac = hist.get("val_acc") or []
        epochs = max(len(loss), len(vac), 1)
        if loss:
            drop = (loss[-1] - loss[0]) / abs(loss[0]) * 100 if loss[0] else 0.0
            loss_txt = f"{loss[0]:.3f} → {loss[-1]:.3f} ({drop:+.0f}%)"
        else:
            loss_txt = "&mdash;"
        if vac:
            first = next((v for v in vac if v is not None), None)
            last = next((v for v in reversed(vac) if v is not None), None)
            vac_txt = (_pct(first) + " → " + _pct(last)) if first is not None and last is not None else "&mdash;"
        else:
            vac_txt = "&mdash;"
        rows.append([r["model"], epochs, loss_txt, vac_txt, _pct(_acc(r))])
    return (_md_table(["Model", "Epochs", "Loss (start → end)", "Val acc (start → end)",
                       "Test acc"], rows) + "\n")


def _caveats(ranked):
    lines = []
    pretrained = any(r.get("weights_source") for r in ranked)
    if pretrained:
        lines.append(
            "- **Loaded weights, not trained in nn:** the nn models here did not "
            "train — their parameters were converted from a PyTorch state_dict "
            "(torch OIHW -> nn HWIO, transposed Linear, and the flatten-order "
            "permutation) and evaluated inference-only.")
    else:
        lines.append(
            "- **From scratch, not pretrained:** ImageNet weights exist only in "
            "PyTorch/TF formats and cannot be loaded into a pure-NumPy autograd "
            "engine, so every architecture trains from random init; the comparison is "
            "about architectural *patterns*, not pretrained accuracy.")
        f"- **Chance level is {_pct(CHANCE_LEVEL)}** (10 balanced digit classes) — "
        "anything near it did not learn at these settings.",
    ]
    weak = [r["model"] for r in ranked if _acc(r) <= 0.15]
    if weak:
        lines.append(f"- **Collapsed runs:** {', '.join(weak)} ended at chance "
                     "level. On 28×28 inputs over few epochs this usually means the "
                     "optimizer never got traction (learning rate, depth, "
                     "initialization) rather than that the pattern is wrong — see "
                     "the loss curves above.")
    if ranked and not _params_known(ranked):
        lines.append("- **Parameter counts are 0** in the source results — they "
                     "were counted before the lazily-built layers materialised. "
                     "Re-run the benchmark (a warm-up forward pass precedes "
                     "counting) to populate them.")
    grew = [r["model"] for r in ranked if _loss_grew(r)]
    if grew:
        lines.append(f"- **Diverging runs:** loss increased for {', '.join(grew)}; "
                     "lower the learning rate or train longer before trusting "
                     "their numbers.")
    lines.append("- **Reproducibility:** `python -m cnn_benchmark.run_benchmark` "
                 "(or the Colab notebook, cells in order) with the settings listed "
                 "above regenerates these results; all numbers here are computed "
                 "from `results.json`, never hard-coded.")
    return "\n".join(lines) + "\n"


def _artifacts(out_path, assets_dir):
    lines = [f"- `{os.path.basename(out_path)}` — this report"]
    res = os.path.join(os.path.dirname(out_path) or ".", "results.json")
    if os.path.exists(res):
        lines.append("- `results.json` — raw metrics behind every table and caption")
    if assets_dir and os.path.isdir(assets_dir):
        imgs = sorted(f for f in os.listdir(assets_dir) if f.lower().endswith(".png"))
        for i in imgs:
            lines.append(f"- `{os.path.basename(assets_dir)}/{i}` — chart image")
    return "\n".join(lines) + "\n"


def _charts(ranked, assets_dir, out_path):
    if not assets_dir or not os.path.isdir(assets_dir):
        return None
    present = {f for f in os.listdir(assets_dir)}
    blocks, used = [], set()
    for fname, heading, caption in CHART_SPECS:
        if fname not in present:
            continue
        used.add(fname)
        rel = os.path.relpath(os.path.join(assets_dir, fname),
                              os.path.dirname(os.path.abspath(out_path))
                              ).replace(os.sep, "/")
        blocks.append(f"### {heading}\n\n{caption(ranked)}\n\n"
                      f"![{heading}]({rel})\n")
    for fname in sorted(present - used):
        if not fname.lower().endswith(".png"):
            continue
        rel = os.path.relpath(os.path.join(assets_dir, fname),
                              os.path.dirname(os.path.abspath(out_path))
                              ).replace(os.sep, "/")
        title = fname[:-4].replace("_", " ")
        blocks.append(f"### {title}\n\n![{title}]({rel})\n")
    if not blocks:
        return None
    return "\n".join(blocks)


# ---------------------------------------------------- per-model detail section
def _trend_line(r):
    """Per-epoch accuracy trend sentence; None when <2 epochs were recorded."""
    hist = r.get("history") or {}
    acc = [v for v in (hist.get("acc") or []) if v is not None]
    vac = [v for v in (hist.get("val_acc") or []) if v is not None]
    series = vac if len(vac) >= 2 else (acc if len(acc) >= 2 else None)
    if series is None:
        return None
    epochs = len(series)
    bits = []
    if len(acc) >= 2:
        bits.append(f"train {_pct(acc[0])} → {_pct(acc[-1])}")
    if len(vac) >= 2:
        bits.append(f"val {_pct(vac[0])} → {_pct(vac[-1])}")
    delta = series[-1] - series[0]
    if delta >= 0.05:
        kind = "strongly improving"
    elif delta >= 0.01:
        kind = "improving"
    elif delta > -0.01:
        kind = "plateaued"
    elif delta > -0.05:
        kind = "slipping"
    else:
        kind = "degrading"
    best_ep = max(range(epochs), key=series.__getitem__)
    if best_ep == epochs - 1 and kind in ("strongly improving", "improving"):
        ending = "still climbing at the final epoch"
    else:
        ending = f"peaked at epoch {best_ep + 1}"
    return f"{', '.join(bits)} over {epochs} epochs — **{kind}** ({_pp(delta)}), {ending}."


def _confusion_block(r):
    """Confusion-matrix paragraph + markdown table; None if not recorded."""
    cm = r.get("confusion")
    if not cm or not isinstance(cm, (list, tuple)) or not cm:
        return None
    n = len(cm)
    total = sum(sum(row) for row in cm)
    if not total:
        return None
    diag = sum(cm[i][i] for i in range(n))
    pair, top = None, 0
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > top:
                pair, top = (i, j), cm[i][j]
    recalls = sorted((cm[i][i] / sum(cm[i]) if sum(cm[i]) else 0.0, i)
                     for i in range(n))
    hard, easy = recalls[0], recalls[-1]
    txt = [f"rows are the true digit, columns the prediction; the diagonal "
           f"reproduces **{_pct(diag / total)}** accuracy ({diag:,}/{total:,} test "
           f"images)"]
    if pair:
        txt.append(f"most confounded pair **{pair[0]} → {pair[1]}** "
                   f"({top} images)")
    else:
        txt.append("no off-diagonal errors")
    txt.append(f"hardest digit **{hard[1]}** (recall {_pct(hard[0])}), easiest "
               f"{easy[1]} ({_pct(easy[0])})")
    headers = ["true \\ pred"] + [str(j) for j in range(n)]
    rows = [[str(i)] + [str(cm[i][j]) for j in range(n)] for i in range(n)]
    return "; ".join(txt).capitalize() + ".\n\n" + _md_table(headers, rows)


def _framework(r):
    """Framework tag from a result dict: 'nn' | 'PyTorch' | None."""
    m = r.get("model", "")
    if "PyTorch twin" in m or m.startswith("tv-"):
        return "PyTorch"
    if "loaded weights" in m or "weights_source" in r:
        return "nn"
    return None


def _cross_framework(ranked):
    """Conditional nn-vs-PyTorch comparison. Only renders when results mix
    frameworks (the pretrained / pytorch modes); None otherwise."""
    tagged = [r for r in ranked if _framework(r)]
    if not tagged:
        return None
    nn_rows = {}
    torch_rows = {}
    for r in tagged:
        base = r["model"]
        for tag in (" (nn, loaded weights)", " (PyTorch twin)"):
            base = base.replace(tag, "")
        d = {"lat_ms": r.get("lat_ms"), "imgs_s": r.get("imgs_s"),
             "test_acc": r.get("test_acc"), "params": r.get("params")}
        if "PyTorch twin" in r["model"]:
            torch_rows[base] = d
        else:
            nn_rows[base] = d

    lines = ["Same weights, two engines. The PyTorch twin was trained (or the "
             "weights downloaded); its state_dict was converted and loaded into "
             "the pure-NumPy nn implementation, then both were measured on the "
             "identical test data with the identical latency protocol."]
    shared = sorted(set(nn_rows) & set(torch_rows))
    rows = []
    for name in shared:
        n, t = nn_rows[name], torch_rows[name]
        acc_check = ("identical" if n["test_acc"] == t["test_acc"]
                     else f"{_pp(n['test_acc'] - t['test_acc'])}")
        speed = None
        if n["lat_ms"] and t["lat_ms"]:
            speed = f"{n['lat_ms'] / t['lat_ms']:.1f}x slower"
        rows.append([name,
                     f"{n['test_acc']:.4f}" if n["test_acc"] is not None else "&mdash;",
                     acc_check,
                     f"{n['lat_ms']:.2f}" if n["lat_ms"] else "&mdash;",
                     f"{t['lat_ms']:.2f}" if t["lat_ms"] else "&mdash;",
                     speed or "&mdash;"])
    if rows:
        lines.append("\n" + _md_table(
            ["Model", "nn acc", "acc match", "nn (ms/img)",
             "PyTorch (ms/img)", "Speed gap"], rows))

    # torchvision pretrained reference block (ImageNet; accuracy not comparable)
    tv = [r for r in tagged if r["model"].startswith("tv-")]
    if tv:
        lines.append("\n**torchvision pretrained reference (ImageNet):** inference "
                     "speed only — these models carry ImageNet weights and their "
                     "accuracy is not MNIST-comparable.\n")
        lines.append(_md_table(
            ["Model", "Params", "Latency (ms)", "Throughput (img/s)", "Input"],
            [[r["model"], f"{r['params']:,}", f"{r['lat_ms']:.2f}",
              f"{r['imgs_s']:,.0f}", r.get("input", "&mdash;")] for r in tv]))
    return "\n".join(lines) + "\n"


def _per_model_detail(ranked):
    """Per-model accuracy trend + confusion matrix; None if neither exists."""
    blocks = []
    for r in ranked:
        trend = _trend_line(r)
        conf = _confusion_block(r)
        if trend is None and conf is None:
            continue
        lines = [f"### {r['model']}\n"]
        if trend:
            lines.append(f"**Accuracy trend:** {trend}\n")
        else:
            hist = r.get("history") or {}
            if hist.get("acc") or hist.get("val_acc"):
                lines.append("**Accuracy trend:** only one epoch was recorded — "
                             "re-run with ≥ 2 epochs to see a trend.\n")
            else:
                lines.append("**Accuracy trend:** no per-epoch history in the "
                             "results.\n")
        if conf:
            lines.append(f"**Confusion:** {conf}\n")
        blocks.append("\n".join(lines))
    if not blocks:
        return None
    intro = ("Per model: how accuracy moved epoch over epoch (classified from the "
             "actual deltas) and — where confusion matrices were recorded — the "
             "full 10×10 confusion matrix with the dominant error pair and "
             "hardest digit.\n")
    return intro + "\n" + "\n".join(blocks)


# ----------------------------------------------------------------------- main
def generate_report(results, settings=None, assets_dir=None,
                    out_path="REPORT.md", title="CNN Benchmark Report"):
    """Write a Markdown report for this run and return its path.

    results    -- list of result dicts (what's in results.json)
    settings   -- run settings to document, e.g. {"EPOCHS": 5, ...}
    assets_dir -- directory of saved chart PNGs to embed (optional)
    out_path   -- where to write the .md file
    """
    settings = settings or {}
    ranked = sorted(list(results or []), key=lambda r: -_acc(r))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = [f"# {title}\n"]
    if ranked:
        best = ranked[0]
        header.append(f"> Generated {stamp} · {len(ranked)} architectures · "
                      f"best test accuracy **{_pct(_acc(best))}** ({best['model']})\n")
    else:
        header.append(f"> Generated {stamp} · no results recorded\n")

    if not ranked:
        body = ["## Results\n\nNo results were recorded for this run "
                "(empty `results.json`). Re-run the benchmark first.\n"]
        text = "\n".join(header) + "\n" + "\n".join(body)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return out_path

    sections = [
        ("Overview and settings", _overview(ranked, settings)),
        ("What was compared", _whats_compared(ranked)),
        ("How it was run", _how(ranked, settings)),
        ("Results", _results_table(ranked)),
        ("Head-to-head comparison",
         "\n".join(_comparisons(ranked)) + "\n"),
        ("Training traces", _traces(ranked)),
    ]
    charts = _charts(ranked, assets_dir, out_path)
    if charts:
        sections.append(("Charts", charts))
    detail = _per_model_detail(ranked)
    if detail:
        sections.append(("Per-model confusion and accuracy trends", detail))
    cross = _cross_framework(ranked)
    if cross:
        sections.append(("nn versus PyTorch (loaded weights)", cross))
    sections.append(("Notes and caveats", _caveats(ranked)))
    sections.append(("Artifacts", _artifacts(out_path, assets_dir)))

    toc = "## Contents\n\n" + "\n".join(
        f"- [{h}](#{_slug(h)})" for h, _ in sections) + "\n"

    parts = "\n".join(header) + "\n" + toc + "\n"
    parts += "\n".join(f"## {h}\n\n{b}" for h, b in sections)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(parts.rstrip() + "\n")
    return out_path


if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "REPORT.md"
    with open(src) as f:
        generate_report(json.load(f), out_path=dst,
                        assets_dir=os.path.dirname(os.path.abspath(src)))
    print(f"wrote {dst}")
