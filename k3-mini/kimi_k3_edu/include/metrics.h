/**
 * metrics.h - Lightweight JSONL metrics emitter for the live web dashboard.
 *
 * The trainer appends one JSON object per line to runs/<run_id>/metrics.jsonl.
 * A companion Python server (scripts/monitor_server.py) tails this file and
 * serves it to web/dashboard.html, which renders live charts (like
 * https://mimo.xiaomi.com/rl/).
 *
 * Emitted line format (single line, no newlines inside):
 * {"t": 1690000000.123, "step": 1000, "phase": "base",
 *  "loss": 3.21, "val_loss": 3.05, "ppl": 24.8, "val_ppl": 21.1,
 *  "lr": 0.0003, "grad_norm": 0.87, "tokens": 123456789,
 *  "tokens_per_s": 4321.5, "samples": 90000, "epoch": 2,
 *  "batch_size": 4, "seq_len": 8192,
 *  "dataset": "datasets/train", "dataset_files": 42, "dataset_tokens": 5.0e9,
 *  "val_files": 4, "val_tokens": 2.5e8,
 *  "file_pos": 17,                 -- index of file currently being streamed
 *  "file_name": "wiki_000.txt",
 *  "file_progress": 0.42,          -- fraction of current file consumed
 *  "elapsed_s": 3600.5, "eta_s": 7200.0, "progress": 0.0067,
 *  "mem_gb": 12.3,
 *  "hw": {                          -- hardware block (all optional)
 *    "device": "NVIDIA A10G", "cuda": 12.1, "driver": "535.104",
 *    "gpu_util": 96.0, "gpu_mem_gb": 21.5, "gpu_mem_total_gb": 24.0,
 *    "gpu_temp_c": 71.0, "gpu_power_w": 240.0, "gpu_power_limit_w": 300.0,
 *    "cpu_name": "AMD EPYC 7R32", "cpu_threads": 16, "cpu_util": 64.0,
 *    "ram_total_gb": 64.0,
 *    "disk_read_mbs": 812.0, "disk_write_mbs": 4.0,
 *    "batch_load_ms": 12.5,         -- dataloader time per batch
 *    "gpu_step_ms": 810.0           -- compute time per step
 *  },
 *  "cost_usd": 0.42, "gpu_hourly_usd": 2.14,
 *  "checkpoint": "checkpoints/step_001000.bin"}
 *
 * A "meta" line is emitted once at run start:
 * {"t": ..., "event": "meta", "run_id": "...", "model_name": "...",
 *  "n_params": ..., "config": {...}}
 */

#ifndef METRICS_H
#define METRICS_H

#include <stddef.h>

typedef struct MetricsEmitter MetricsEmitter;

/* Open (or reopen) the metrics file for a run. run_dir example: "runs/run_20260922_101500". */
MetricsEmitter* metrics_open(const char* run_dir, const char* run_id);

/* Emit the one-time meta line describing model + config. */
int metrics_emit_meta(MetricsEmitter* me, const char* model_name,
                      size_t n_params, const char* config_json);

/* Hardware snapshot for the dashboard's Hardware panels. Pass 0/NULL for
 * values not tracked on this machine. */
typedef struct {
    const char* device;          /* e.g. "NVIDIA A10G" or "CPU-only" */
    float cuda_version;
    const char* driver;
    float gpu_util;              /* percent */
    float gpu_mem_gb;
    float gpu_mem_total_gb;
    float gpu_temp_c;
    float gpu_power_w;
    float gpu_power_limit_w;
    const char* cpu_name;
    int cpu_threads;
    float cpu_util;              /* percent */
    float ram_used_gb;
    float ram_total_gb;
    float disk_read_mbs;
    float disk_write_mbs;
    float batch_load_ms;         /* dataloader latency per batch */
    float gpu_step_ms;           /* compute latency per step */
} MetricsHardware;

/* Data-streaming snapshot: what the dataloader is currently feeding. */
typedef struct {
    const char* dataset_dir;
    int dataset_files;
    double dataset_tokens;
    int val_files;
    double val_tokens;
    int file_pos;                /* index of file currently streaming */
    const char* file_name;
    float file_progress;         /* 0..1 within current file */
} MetricsData;

/* Emit a training-step metric line. NAN / -1 / NULL fields may be passed for
 * values not tracked. */
int metrics_emit_step(MetricsEmitter* me,
                      long step, const char* phase,
                      float loss, float val_loss,
                      float lr, float grad_norm,
                      double tokens_total, double tokens_per_s,
                      double samples_total, int epoch,
                      int batch_size, int seq_len,
                      const MetricsData* data,
                      double elapsed_s, double eta_s, float progress,
                      const MetricsHardware* hw,
                      float cost_usd, float gpu_hourly_usd,
                      const char* checkpoint_path);

/* Emit a run-finished / aborted event. */
int metrics_emit_event(MetricsEmitter* me, const char* event, const char* detail);

/* Emit a benchmark evaluation point (e.g. after each val pass):
 * benchmark: "DeepSWE v1.1", score: 72.57. Rendered on the benchmarks cards. */
int metrics_emit_benchmark(MetricsEmitter* me, const char* name,
                           long step, float score);

/* Emit a data-source sampling snapshot (like MiMo's dynamic sampler):
 * call once per step with per-source accepted/judged counts. */
int metrics_emit_source(MetricsEmitter* me, long step,
                        const char* source_name, const char* category,
                        int accepted, int target, int judged, int in_flight);

void metrics_close(MetricsEmitter* me);

/* Convenience: estimate cost from elapsed time and hourly GPU rate. */
float metrics_cost_usd(double elapsed_s, float gpu_hourly_usd);

#endif /* METRICS_H */
