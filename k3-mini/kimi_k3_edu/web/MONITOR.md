# K3-Edu Live Training Monitor

A TensorBoard-style live dashboard for K3-Edu training, in the spirit of
[Xiaomi MiMo's RL training view](https://mimo.xiaomi.com/rl/): while training
runs, a browser page shows which dataset is being trained on, live loss /
perplexity curves, LR schedule, throughput, GPU/RAM gauges, running cost, ETA,
and checkpoint events.

```
┌─────────────┐   appends     ┌──────────────────┐   HTTP poll     ┌───────────────┐
│ C trainer   │  1 JSON line  │ scripts/          │  every 2s      │ web/           │
│ train_main  ├──────────────►│ monitor_server.py ◄───────────────┤ dashboard.html │
│ (metrics.c) │  per step     │ (tails jsonl)     │  /api/metrics  │ (canvas charts)│
└─────────────┘               └──────────────────┘                 └───────────────┘
        runs/run_<timestamp>/metrics.jsonl
```

## Usage

```bash
# 1. start training (creates runs/run_<timestamp>/metrics.jsonl)
./build/train_base --dataset datasets/train/ --tokenizer tokenizer.bin

# 2. in another terminal, start the dashboard (auto-picks the newest run)
python scripts/monitor_server.py                 # opens http://localhost:8000

# or watch a specific run / port
python scripts/monitor_server.py --run runs/run_20260922_101500 --port 8000
```

You can start the monitor **before or during** training — the page updates
every 2 s and picks up newly appended lines incrementally (byte-offset
polling, no re-download).

## What the dashboard shows

| Panel | Contents |
|---|---|
| KPI strip | step, train/val loss (with per-step delta), val perplexity, tokens/s, **cost ($ + $/hr)**, ETA |
| Progress | overall % through `total_steps`, current phase (base/inst) |
| Data Streaming | dataset dir, train/val file + token counts, batch × seq_len, epoch, samples, **batch load latency**, **which file is being streamed right now** + per-file progress bar |
| Model/Config | model name, parameter count, LR, grad norm, best val loss, GPU step time |
| Hardware | **Real detected values, not assumptions**: device name, CUDA/driver, CPU model + threads + util, GPU temp, GPU power, disk R/W, host/OS, disk free — plus GPU-util / VRAM / RAM donut gauges (turn pink when hot) |
| Loss chart | **linear/log toggle**, **step-wise or time-wise x-axis**, **EMA smoothing slider**, **hover tooltip + crosshair** showing per-step loss/ppl/lr/throughput/cost/file |
| Throughput & LR | tokens/s curve (with its own linear/log toggle, hoverable) + LR schedule |
| Cost | total spent, billable elapsed, hourly rate, projected total, cost per 1M tokens, budget consumed |
| Feed | checkpoint saves and run events (newest first) |

All icons are Font Awesome 6 (loaded from CDN); no emojis are used.

## metrics.jsonl schema (one JSON object per line)

Meta line (once per run):
```json
{"t":1690000000.0,"event":"meta","model_name":"K3-Edu-200M","n_params":200000000,"config":{...}}
```

Step line (every `log_every` steps):
```json
{"t":1690000001.0,"step":1000,"phase":"base","loss":3.21,"val_loss":3.05,
 "ppl":24.8,"val_ppl":21.1,"lr":0.0003,"grad_norm":0.87,
 "tokens":123456789,"tokens_per_s":4321.5,"samples":90000,"epoch":2,
 "batch_size":4,"seq_len":8192,
 "dataset":"datasets/train","dataset_files":42,"dataset_tokens":5000000000,
 "val_files":4,"val_tokens":250000000,
 "file_pos":17,"file_name":"shard_017.txt","file_progress":0.42,
 "elapsed_s":3600.5,"eta_s":7200.0,"progress":0.0067,
 "hw":{"device":"NVIDIA A10G","cuda":12.1,"driver":"535.104",
       "gpu_util":96.0,"gpu_mem":21.5,"gpu_mem_total":24.0,
       "gpu_temp":71.0,"gpu_power":240.0,"gpu_power_limit":300.0,
       "cpu":"AMD EPYC 7R32","cpu_threads":16,"cpu_util":64.0,
       "ram_used":38.2,"ram_total":64.0,
       "disk_read_mbs":812.0,"disk_write_mbs":4.0,
       "batch_load_ms":12.5,"gpu_step_ms":810.0},
 "cost_usd":0.42,"gpu_hourly_usd":2.14,"checkpoint":"checkpoints/step_001000.bin"}
```

Untracked `hw` fields are simply omitted from the JSON; the dashboard renders
em-dashes for missing values, so CPU-only or minimal instrumented runs work.

Event lines: `{"t":...,"event":"finished|aborted","detail":"..."}`

## Wiring into other training loops

`train_main.c` contains a reference loop that calls `metrics_emit_step()` every
N steps. If you train through `cuda/train_gpu.cu` or a custom loop, do the same
three things:

```c
#include "metrics.h"

MetricsEmitter* me = metrics_open("runs", run_id);              // once
metrics_emit_meta(me, model_name, n_params, config_json);       // once
metrics_emit_step(me, ...);                                     // every log_every steps
metrics_close(me);                                              // at the end
```

Untracked values (e.g. no GPU on CPU-only runs) can be passed as `0` — the UI
renders them as em-dashes. `metrics_emit_step` never blocks training: it is a
single `fprintf` + `fflush` to an append-only file.

## Cost estimation

Cost = `elapsed_s / 3600 × gpu_hourly_usd`. The hourly rate is currently
hardcoded at `$2.14` (a typical cloud A10G/L4 price) in `train_main.c` — set it
to your actual rate (0 for local hardware you own).

## API reference

| Endpoint | Description |
|---|---|
| `GET /` | dashboard page |
| `GET /api/runs` | list runs under `runs/`, newest first |
| `GET /api/metrics` | all metric lines as a JSON array |
| `GET /api/metrics?after=N` | only lines appended after byte offset N (efficient polling) |
| `GET /api/summary` | latest step, meta, run state (`finished` flag) |

Only Python's standard library is used — no pip installs.

## Single-instance guarantee

Only one monitor server can run at a time. On startup it writes a lock file
(`runs/.monitor.lock` with its PID + port):

- A second instance **refuses to start** and prints the existing server's PID:
  `A monitor server is already running (PID 1234). Only one instance is allowed.`
- The lock is verified against a live process (and an actual HTTP response) —
  a stale lock left by a crashed or killed server is detected and removed
  automatically, so a restart always works.
- The lock is released on clean shutdown (Ctrl+C) via `atexit`.

## Hardware detection (real values, no assumptions)

The Hardware panel is populated by `scripts/detect_hardware.py`, which the
server queries on `/api/hw`:

- **Static info** (detected once at server start, cached to `runs/hw_static.json`):
  GPU name via `nvidia-smi` (or pynvml), driver version, CUDA version, VRAM
  total, CPU model (wmic/registry on Windows, /proc/cpuinfo on Linux), thread
  count, RAM total, disk free, hostname, OS.
- **Dynamic samples** (fresh every 2 s poll): GPU util/temp/power/VRAM from
  `nvidia-smi --query-gpu`, CPU util (psutil or /proc/stat delta), RAM used,
  disk read/write rates.

Verified on the dev machine: it correctly detected `NVIDIA GeForce RTX 2060`,
`Intel i5-10300H (8T)`, `7.8 GB RAM`, CUDA 12.7, driver 566.36.

Fields that cannot be measured on a given machine are simply omitted and the
dashboard shows a dash for them — it never invents numbers.
