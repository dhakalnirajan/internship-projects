#!/usr/bin/env python3
"""
detect_hardware.py - Real hardware detection for the K3-Edu dashboard.

The monitor server calls this so the Hardware panel shows the actual machine
the trainer runs on -- no hardcoded assumptions.

    static_info()    -> dict    device, CPU model, cores, RAM total, disk
                                  total, CUDA/driver versions, etc.
                                  Fetched once per run and cached.
    sample_dynamic() -> dict    gpu util/temp/power/VRAM-used, CPU util,
                                  RAM used, disk R/W rates, per poll.

Detection order for the GPU: nvidia-smi (works on any NVIDIA card) ->
pynvml -> torch (only if already installed) -> CPU-only fallback.
Everything degrades gracefully: a field that can't be measured is simply
absent, and the dashboard renders a dash for it.

Only stdlib is required; psutil is used when available for CPU/RAM/disk.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import platform

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(cmd, timeout=5):
    """Run a command, return stdout or None."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, encoding="utf-8", errors="replace")
        if out.returncode == 0:
            return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_first(paths):
    for p in paths:
        v = _read_int(p)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# GPU detection (nvidia-smi first: no Python deps needed)
# ---------------------------------------------------------------------------

def _tofloat(v):
    """nvidia-smi prints [N/A] / [Not Supported] for unavailable fields."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _nvidia_smi_query(fields):
    """Query nvidia-smi; returns list of per-GPU dicts or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _run([exe, "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    gpus = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(fields, vals)))
    return gpus


_NVML = None
_NVML_TRIED = False
def _pynvml():
    global _NVML, _NVML_TRIED
    if not _NVML_TRIED:
        _NVML_TRIED = True
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:
            _NVML = None
    return _NVML


def gpu_static():
    g = _nvidia_smi_query(["name", "driver_version", "memory.total",
                           "compute_cap", "power.limit"])
    if g:
        d = g[0]
        mem_total = _tofloat(d.get("memory.total"))
        pwr = _tofloat(d.get("power.limit"))
        return {
            "device": d.get("name"),
            "driver": d.get("driver_version"),
            "gpu_mem_total_gb": round(mem_total / 1024, 1) if mem_total else None,
            "cuda": _cuda_version_from_smi(),
            "gpu_power_limit_w": pwr,
            "gpu_count": len(g),
        }
    nv = _pynvml()
    if nv:
        try:
            h = nv.nvmlDeviceGetHandleByIndex(0)
            name = nv.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            drv = nv.nvmlSystemGetDriverVersion()
            if isinstance(drv, bytes):
                drv = drv.decode()
            return {"device": name, "driver": drv,
                    "gpu_mem_total_gb": round(mem.total / 1024**3, 1),
                    "gpu_count": nv.nvmlDeviceGetCount()}
        except Exception:
            pass
    return {"device": "CPU", "gpu_count": 0}


def _cuda_version_from_smi():
    out = _run(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            return float(m.group(1))
    # fall back to nvcc
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release ([\d.]+)", out)
        if m:
            return float(m.group(1))
    return None


def gpu_dynamic():
    """GPU util %, temp C, power W, VRAM used GB. Values absent if unmeasurable."""
    fields = ["utilization.gpu", "temperature.gpu", "power.draw",
              "memory.used", "clocks.sm"]
    g = _nvidia_smi_query(fields)
    if g:
        d = g[0]
        def f(key, scale=1.0):
            v = _tofloat(d.get(key))
            return round(v * scale, 1) if v is not None else None
        mem = _tofloat(d.get("memory.used"))
        return {
            "gpu_util": f("utilization.gpu"),
            "gpu_temp": f("temperature.gpu"),
            "gpu_power_w": f("power.draw"),
            "gpu_mem_gb": round(mem / 1024, 2) if mem is not None else None,
        }
    nv = _pynvml()
    if nv:
        try:
            h = nv.nvmlDeviceGetHandleByIndex(0)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            pw = nv.nvmlDeviceGetPowerUsage(h)
            res = {"gpu_util": float(util.gpu),
                   "gpu_temp": float(temp),
                   "gpu_mem_gb": round(mem.used / 1024**3, 2)}
            try:
                res["gpu_power_w"] = round(pw / 1000.0, 1)
            except Exception:
                pass
            return res
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# CPU / RAM / disk
# ---------------------------------------------------------------------------

def cpu_name():
    if platform.system() == "Windows":
        out = _run(["wmic", "cpu", "get", "name"], timeout=8)
        if out:
            lines = [l.strip() for l in out.splitlines() if l.strip()][1:]
            if lines:
                return lines[0]
        # registry fallback
        out = _run(["reg", "query",
                    r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                    "/v", "ProcessorNameString"])
        if out:
            m = re.search(r"REG_SZ\s+(.+)", out)
            if m:
                return m.group(1).strip()
        return platform.processor() or None
    # Linux: /proc/cpuinfo; macOS: sysctl
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out:
        return out.strip()
    return platform.processor() or None


def cpu_count():
    return os.cpu_count() or (os.sysconf("SC_NPROCESSORS_ONLN")
                              if hasattr(os, "sysconf") else None)


def ram_total_gb():
    # try /proc/meminfo (Linux) first: exact
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except (OSError, ValueError):
        pass
    psutil = _psutil()
    if psutil:
        return round(psutil.virtual_memory().total / 1024**3, 1)
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return round(int(out.strip()) / 1024**3, 1)
            except ValueError:
                pass
    if platform.system() == "Windows":
        out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"], timeout=8)
        if out:
            lines = [l.strip() for l in out.splitlines() if l.strip().isdigit()]
            if lines:
                return round(int(lines[0]) / 1024**3, 1)
    return None


_ps = None
def _psutil():
    global _ps
    if _ps is None:
        try:
            import psutil
            _ps = psutil
        except ImportError:
            _ps = False
    return _ps or None


# previous CPU sample for util delta
_prev_cpu = {"time": None, "idle": None, "total": None}

def cpu_util():
    p = _psutil()
    if p:
        return round(p.cpu_percent(interval=None), 1)
    # /proc/stat delta (Linux, no deps)
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        prev = _prev_cpu
        if prev["time"] is not None:
            dt = time.time() - prev["time"]
            if dt > 0:
                u = 100.0 * (1 - (idle - prev["idle"]) / (total - prev["total"]))
                _prev_cpu.update(time=time.time(), idle=idle, total=total)
                return round(max(0.0, min(100.0, u)), 1)
        _prev_cpu.update(time=time.time(), idle=idle, total=total)
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    return None


def ram_used_gb():
    p = _psutil()
    if p:
        return round(p.virtual_memory().used / 1024**3, 2)
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.strip().split()[0])
            used_kb = info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))
            return round(used_kb / 1024 / 1024, 2)
    except (OSError, KeyError, ValueError):
        pass
    return None


# disk rate sampling: read /proc/diskdata or use psutil io counters
_prev_disk = {"time": None, "read": None, "write": None}

def disk_rates_mbs():
    p = _psutil()
    if p:
        try:
            io = p.disk_io_counters()
            now = time.time()
            prev = _prev_disk
            res = {}
            if prev["time"] and now > prev["time"]:
                dt = now - prev["time"]
                res["disk_read_mbs"] = round((io.read_bytes - prev["read"]) / dt / 1024 / 1024, 1)
                res["disk_write_mbs"] = round((io.write_bytes - prev["write"]) / dt / 1024 / 1024, 1)
            _prev_disk.update(time=now, read=io.read_bytes, write=io.write_bytes)
            return res
        except Exception:
            pass
    # Linux: /proc/diskstats sectors (512B each), sum all real devices
    try:
        reads = writes = 0
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 9 and not parts[2].startswith(("loop", "ram")):
                    reads += int(parts[5]) * 512
                    writes += int(parts[9]) * 512
        now = time.time()
        prev = _prev_disk
        res = {}
        if prev["time"] and now > prev["time"]:
            dt = now - prev["time"]
            res["disk_read_mbs"] = round((reads - prev["read"]) / dt / 1024 / 1024, 1)
            res["disk_write_mbs"] = round((writes - prev["write"]) / dt / 1024 / 1024, 1)
        _prev_disk.update(time=now, read=reads, write=writes)
        return res
    except (OSError, ValueError, IndexError):
        pass
    return {}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def static_info():
    """Everything that doesn't change during a run. Fetch once, cache."""
    info = {
        "hostname": platform.node() or None,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_name": cpu_name(),
        "cpu_threads": cpu_count(),
        "ram_total_gb": ram_total_gb(),
        "disk_free_gb": shutil.disk_usage(os.getcwd()).free / 1024**3
                         if os.path.exists(os.getcwd()) else None,
        "timestamp": time.time(),
    }
    if platform.system() == "Windows":
        info["os"] = platform.system() + " " + platform.release()
    info.update(gpu_static())
    return {k: v for k, v in info.items() if v is not None}


def sample_dynamic():
    """Everything that changes per poll. Cheap; called on every /api/hw poll."""
    res = {}
    res.update(gpu_dynamic())
    cu = cpu_util()
    if cu is not None:
        res["cpu_util"] = cu
    ru = ram_used_gb()
    if ru is not None:
        res["ram_used_gb"] = ru
    res.update(disk_rates_mbs())
    return res


if __name__ == "__main__":
    print("== static ==")
    print(json.dumps(static_info(), indent=2))
    print("== dynamic (2 samples, 1s apart) ==")
    print(json.dumps(sample_dynamic(), indent=2))
    time.sleep(1)
    print(json.dumps(sample_dynamic(), indent=2))
