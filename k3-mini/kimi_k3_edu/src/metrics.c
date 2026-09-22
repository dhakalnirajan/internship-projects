/**
 * metrics.c - JSONL metrics emitter (see metrics.h).
 *
 * Design goals:
 *  - Never crash or slow down training: all writes are best-effort,
 *    flushed immediately so the dashboard server can tail the file.
 *  - One JSON object per line (JSON Lines format) - trivially parseable
 *    and appendable from any language.
 */

#include "metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#define MKDIR(p) mkdir(p, 0755)
#endif

struct MetricsEmitter {
    FILE* fp;
    char path[512];
};

static double now_unix(void) {
    return (double)time(NULL);
}

static void ensure_dir(const char* dir) {
    MKDIR(dir); /* ok if exists */
}

/* Write a float, mapping NaN to null so JSON stays valid. */
static void put_float(FILE* fp, float v) {
    if (isnan(v) || isinf(v)) fputs("null", fp);
    else fprintf(fp, "%.6g", (double)v);
}

MetricsEmitter* metrics_open(const char* run_dir, const char* run_id) {
    ensure_dir("runs");
    ensure_dir(run_dir);

    MetricsEmitter* me = (MetricsEmitter*)calloc(1, sizeof(MetricsEmitter));
    if (!me) return NULL;

    snprintf(me->path, sizeof(me->path), "%s/metrics.jsonl", run_dir);
    me->fp = fopen(me->path, "a");
    if (!me->fp) {
        free(me);
        return NULL;
    }
    (void)run_id;
    return me;
}

int metrics_emit_meta(MetricsEmitter* me, const char* model_name,
                      size_t n_params, const char* config_json) {
    if (!me || !me->fp) return -1;
    fprintf(me->fp,
            "{\"t\":%.3f,\"event\":\"meta\",\"model_name\":\"%s\",\"n_params\":%zu,"
            "\"config\":%s}\n",
            now_unix(), model_name ? model_name : "unknown", n_params,
            config_json ? config_json : "{}");
    fflush(me->fp);
    return 0;
}

/* Helper: write one hw field as  {"key":value,  or  {"key":value}  */
static void hw_f(FILE* fp, const char* key, float v, float total, int* first) {
    if (v <= 0.0f && total <= 0.0f) return; /* untracked -> omit */
    if (!*first) fputc(',', fp);
    *first = 0;
    if (total > 0.0f && v <= total) {
        fprintf(fp, "\"%s\":%.6g,\"%s_used\":%.6g", key, (double)total, key, (double)v);
    } else {
        fprintf(fp, "\"%s\":%.6g", key, (double)v);
    }
}

static void hw_s(FILE* fp, const char* key, const char* val, int* first) {
    if (!val || !val[0]) return;
    if (!*first) fputc(',', fp);
    *first = 0;
    fprintf(fp, "\"%s\":\"%s\"", key, val);
}

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
                      const char* checkpoint_path) {
    if (!me || !me->fp) return -1;

    float ppl = (loss > 0.0f && !isnan(loss)) ? expf(loss) : 0.0f;
    float val_ppl = (val_loss > 0.0f && !isnan(val_loss)) ? expf(val_loss) : 0.0f;

    fprintf(me->fp, "{\"t\":%.3f,\"step\":%ld,\"phase\":\"%s\",\"loss\":",
            now_unix(), step, phase ? phase : "base");
    put_float(me->fp, loss);
    fputs(",\"val_loss\":", me->fp);
    put_float(me->fp, val_loss);
    fputs(",\"ppl\":", me->fp);
    put_float(me->fp, ppl);
    fputs(",\"val_ppl\":", me->fp);
    put_float(me->fp, val_ppl);
    fputs(",\"lr\":", me->fp);
    put_float(me->fp, lr);
    fputs(",\"grad_norm\":", me->fp);
    put_float(me->fp, grad_norm);
    fprintf(me->fp,
            ",\"tokens\":%.0f,\"tokens_per_s\":%.1f,\"samples\":%.0f,\"epoch\":%d,"
            "\"batch_size\":%d,\"seq_len\":%d",
            tokens_total, tokens_per_s, samples_total, epoch,
            batch_size, seq_len);

    /* ---- data streaming block ---- */
    if (data) {
        fprintf(me->fp,
                ",\"dataset\":\"%s\",\"dataset_files\":%d,\"dataset_tokens\":%.0f"
                ",\"val_files\":%d,\"val_tokens\":%.0f"
                ",\"file_pos\":%d,\"file_name\":\"%s\",\"file_progress\":%.4f",
                data->dataset_dir ? data->dataset_dir : "",
                data->dataset_files, data->dataset_tokens,
                data->val_files, data->val_tokens,
                data->file_pos,
                data->file_name ? data->file_name : "",
                (double)data->file_progress);
    }

    fprintf(me->fp,
            ",\"elapsed_s\":%.1f,\"eta_s\":%.1f,\"progress\":%.6f",
            elapsed_s, eta_s, (double)progress);

    /* ---- hardware block (only tracked fields are emitted) ---- */
    if (hw) {
        int first = 1;
        fputs(",\"hw\":{", me->fp);
        hw_s(me->fp, "device", hw->device, &first);
        if (hw->cuda_version > 0.0f) {
            if (!first) fputc(',', me->fp);
            first = 0;
            fprintf(me->fp, "\"cuda\":%.1f", (double)hw->cuda_version);
        }
        hw_s(me->fp, "driver", hw->driver, &first);
        hw_f(me->fp, "gpu_util", hw->gpu_util, 100.0f, &first);
        hw_f(me->fp, "gpu_mem", hw->gpu_mem_gb, hw->gpu_mem_total_gb, &first);
        hw_f(me->fp, "gpu_temp", hw->gpu_temp_c, 0.0f, &first);
        hw_f(me->fp, "gpu_power", hw->gpu_power_w, hw->gpu_power_limit_w, &first);
        hw_s(me->fp, "cpu", hw->cpu_name, &first);
        hw_f(me->fp, "cpu_threads", (float)hw->cpu_threads, 0.0f, &first);
        hw_f(me->fp, "cpu_util", hw->cpu_util, 100.0f, &first);
        hw_f(me->fp, "ram_used_gb", hw->ram_used_gb, hw->ram_total_gb, &first);
        hw_f(me->fp, "disk_read_mbs", hw->disk_read_mbs, 0.0f, &first);
        hw_f(me->fp, "disk_write_mbs", hw->disk_write_mbs, 0.0f, &first);
        hw_f(me->fp, "batch_load_ms", hw->batch_load_ms, 0.0f, &first);
        hw_f(me->fp, "gpu_step_ms", hw->gpu_step_ms, 0.0f, &first);
        fputc('}', me->fp);
    }

    fprintf(me->fp,
            ",\"cost_usd\":%.4f,\"gpu_hourly_usd\":%.2f",
            (double)cost_usd, (double)gpu_hourly_usd);

    if (checkpoint_path && checkpoint_path[0]) {
        fprintf(me->fp, ",\"checkpoint\":\"%s\"", checkpoint_path);
    }
    fputs("}\n", me->fp);
    fflush(me->fp);
    return 0;
}

int metrics_emit_event(MetricsEmitter* me, const char* event, const char* detail) {
    if (!me || !me->fp) return -1;
    fprintf(me->fp, "{\"t\":%.3f,\"event\":\"%s\",\"detail\":\"%s\"}\n",
            now_unix(), event ? event : "event", detail ? detail : "");
    fflush(me->fp);
    return 0;
}

int metrics_emit_benchmark(MetricsEmitter* me, const char* name,
                           long step, float score) {
    if (!me || !me->fp) return -1;
    fprintf(me->fp,
            "{\"t\":%.3f,\"event\":\"benchmark\",\"name\":\"%s\","
            "\"step\":%ld,\"score\":", now_unix(), name ? name : "bench", step);
    put_float(me->fp, score);
    fputs("}\n", me->fp);
    fflush(me->fp);
    return 0;
}

int metrics_emit_source(MetricsEmitter* me, long step,
                        const char* source_name, const char* category,
                        int accepted, int target, int judged, int in_flight) {
    if (!me || !me->fp) return -1;
    fprintf(me->fp,
            "{\"t\":%.3f,\"event\":\"source\",\"step\":%ld,"
            "\"source\":\"%s\",\"category\":\"%s\","
            "\"accepted\":%d,\"target\":%d,\"judged\":%d,\"in_flight\":%d}\n",
            now_unix(), step,
            source_name ? source_name : "dataset", category ? category : "general",
            accepted, target, judged, in_flight);
    fflush(me->fp);
    return 0;
}

void metrics_close(MetricsEmitter* me) {
    if (!me) return;
    if (me->fp) fclose(me->fp);
    free(me);
}

float metrics_cost_usd(double elapsed_s, float gpu_hourly_usd) {
    return (float)((elapsed_s / 3600.0) * (double)gpu_hourly_usd);
}
